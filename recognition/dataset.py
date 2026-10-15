import json
import re
import zipfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# 这些默认路径沿用你 notebook 中的路径。
# 使用其他目录时，可以在运行 train.py / predict.py 时传入命令行参数。
DEFAULT_ZIP_PATH = Path(
    "/content/drive/MyDrive/COMP3710/"
    "Final_Project_dataset/ADNI.zip"
)

DEFAULT_WORK_DIR = Path("/content/comp3710_adni")

DEFAULT_MANIFEST_PATH = Path(
    "/content/drive/MyDrive/COMP3710/"
    "Final_Project_dataset/patient_level_manifest.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/COMP3710/"
    "Final_Project_dataset/convnext_results"
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms():
    """
    返回训练集和评估集的图像预处理。

    MRI 灰度图复制成 3 个通道，以适配预训练 ConvNeXt。
    验证集和测试集不使用随机增强。
    """
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),

        transforms.RandomResizedCrop(
            size=224,
            scale=(0.90, 1.00),
            ratio=(0.95, 1.05),
        ),

        transforms.RandomRotation(degrees=5),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(236),
        transforms.CenterCrop(224),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    return train_transform, evaluation_transform


def extract_dataset(
    zip_path: Path,
    work_dir: Path,
) -> tuple[Path, Path]:
    """如果数据尚未解压，则从 zip 文件解压 ADNI 数据。"""
    zip_path = Path(zip_path)
    work_dir = Path(work_dir)

    data_root = work_dir / "ADNI"
    image_root = data_root / "AD_NC"
    metadata_path = data_root / "meta_data_with_label.json"

    if image_root.exists() and metadata_path.exists():
        print("Dataset already extracted.")
        return image_root, metadata_path

    if not zip_path.exists():
        raise FileNotFoundError(
            f"找不到数据压缩包：{zip_path}"
        )

    work_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, mode="r") as zip_file:
        # 忽略 macOS 压缩包中可能附带的辅助文件。
        valid_members = [
            name
            for name in zip_file.namelist()
            if not name.startswith("__MACOSX/")
        ]

        zip_file.extractall(
            work_dir,
            members=valid_members,
        )

    if not image_root.exists():
        raise FileNotFoundError(
            f"解压后找不到图像目录：{image_root}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"解压后找不到 metadata：{metadata_path}"
        )

    print("Dataset extracted successfully.")
    return image_root, metadata_path


def build_patient_level_manifest(
    image_root: Path,
    metadata_path: Path,
    manifest_path: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """
    根据 metadata 建立图像 manifest，并按患者划分 train/validation/test。

    同一患者的图像只会进入一个 split，以避免患者级数据泄漏。
    """
    image_root = Path(image_root)
    metadata_path = Path(metadata_path)
    manifest_path = Path(manifest_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    subject_pattern = re.compile(r"(\d{3}_S_\d{4})")
    scan_pattern = re.compile(r"\b(I\d{3,})\b", re.IGNORECASE)

    # 建立 scan ID 到 subject ID 的映射。
    scan_to_subject = {}
    conflicting_scans = set()

    for scan_id, paths_dict in metadata.items():
        clean_scan_id = scan_id.strip().upper()

        subject_ids = set()

        if isinstance(paths_dict, dict):
            path_values = paths_dict.values()
        else:
            path_values = []

        for path_str in path_values:
            if isinstance(path_str, str):
                for subject_id in subject_pattern.findall(path_str):
                    subject_ids.add(subject_id)

        if len(subject_ids) == 1:
            subject_id = next(iter(subject_ids))
            old_subject = scan_to_subject.get(clean_scan_id)

            if old_subject is not None and old_subject != subject_id:
                conflicting_scans.add(clean_scan_id)
            else:
                scan_to_subject[clean_scan_id] = subject_id

        elif len(subject_ids) > 1:
            conflicting_scans.add(clean_scan_id)

    print("Metadata type:", type(metadata))
    print("Scan-to-subject mappings:", len(scan_to_subject))
    print("Conflicting scan IDs:", len(conflicting_scans))

    # 扫描图像目录。
    image_paths = sorted(
        path
        for path in image_root.rglob("*")
        if path.suffix.lower() in {".jpeg", ".jpg", ".png"}
    )

    rows = []
    unmatched_files = []

    for image_path in image_paths:
        name_stem = image_path.stem
        scan_parts = name_stem.split("_")
        scan_id_candidate = (
            scan_parts[0].upper()
            if scan_parts
            else ""
        )

        scan_id = None

        # 首先匹配文件名中的第一个部分。
        if scan_id_candidate in scan_to_subject:
            scan_id = scan_id_candidate

        # 备用方式：从文件名中寻找 I 开头的 scan ID。
        if scan_id is None:
            scan_match = scan_pattern.search(image_path.name)

            if scan_match:
                backup_id = scan_match.group(1).upper()

                if backup_id in scan_to_subject:
                    scan_id = backup_id

        # 再尝试给数字 scan ID 加上 I 前缀。
        if scan_id is None:
            prefixed_id = f"I{scan_id_candidate}"

            if prefixed_id in scan_to_subject:
                scan_id = prefixed_id

        if scan_id is None:
            unmatched_files.append(str(image_path))
            continue

        class_name = image_path.parent.name
        original_split = image_path.parent.parent.name

        if class_name not in {"AD", "NC"}:
            continue

        rows.append({
            "filepath": str(image_path),
            "label_name": class_name,
            "label": 1 if class_name == "AD" else 0,
            "scan_id": scan_id,
            "subject_id": scan_to_subject[scan_id],
            "original_split": original_split,
        })

    manifest = pd.DataFrame(rows)

    print("\nTotal image files:", len(image_paths))
    print("Matched images:", len(manifest))
    print("Unmatched images:", len(unmatched_files))

    if unmatched_files:
        print("\nFirst unmatched files:")
        for filename in unmatched_files[:5]:
            print(filename)

    if len(manifest) != len(image_paths):
        raise AssertionError(
            "部分图片无法与 metadata 匹配，"
            "请检查 scan ID 和图片文件名。"
        )

    if manifest.empty:
        raise ValueError("没有从图像目录中找到可用图像。")

    if manifest["subject_id"].isna().any():
        raise AssertionError(
            "部分图片没有匹配到 subject_id。"
        )

    # 一个患者不能同时拥有 AD 和 NC 标签。
    labels_per_subject = (
        manifest.groupby("subject_id")["label"].nunique()
    )
    conflicting_subjects = labels_per_subject[
        labels_per_subject > 1
    ]

    if not conflicting_subjects.empty:
        raise AssertionError(
            f"发现 {len(conflicting_subjects)} 个患者"
            "同时具有 AD 和 NC 标签。"
        )

    # 每位患者保留一行，再进行分层抽样。
    subject_table = (
        manifest[["subject_id", "label"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    train_subjects, temporary_subjects = train_test_split(
        subject_table,
        test_size=0.30,
        random_state=seed,
        stratify=subject_table["label"],
    )

    validation_subjects, test_subjects = train_test_split(
        temporary_subjects,
        test_size=0.50,
        random_state=seed,
        stratify=temporary_subjects["label"],
    )

    subject_to_split = {}

    for subject_id in train_subjects["subject_id"]:
        subject_to_split[subject_id] = "train"

    for subject_id in validation_subjects["subject_id"]:
        subject_to_split[subject_id] = "validation"

    for subject_id in test_subjects["subject_id"]:
        subject_to_split[subject_id] = "test"

    manifest["split"] = manifest["subject_id"].map(
        subject_to_split
    )

    # 检查不同 split 之间没有患者重叠。
    train_ids = set(
        manifest.loc[
            manifest["split"] == "train",
            "subject_id",
        ]
    )
    validation_ids = set(
        manifest.loc[
            manifest["split"] == "validation",
            "subject_id",
        ]
    )
    test_ids = set(
        manifest.loc[
            manifest["split"] == "test",
            "subject_id",
        ]
    )

    if not train_ids.isdisjoint(validation_ids):
        raise AssertionError("train 和 validation 患者发生重叠。")

    if not train_ids.isdisjoint(test_ids):
        raise AssertionError("train 和 test 患者发生重叠。")

    if not validation_ids.isdisjoint(test_ids):
        raise AssertionError("validation 和 test 患者发生重叠。")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    print("\nImage counts:")
    print(pd.crosstab(manifest["split"], manifest["label_name"]))

    print("\nSubject counts:")
    subject_summary = (
        manifest[["subject_id", "label_name", "split"]]
        .drop_duplicates()
    )
    print(
        pd.crosstab(
            subject_summary["split"],
            subject_summary["label_name"],
        )
    )

    print("\nTotal subjects:", manifest["subject_id"].nunique())
    print("Manifest saved to:", manifest_path)

    return manifest


def load_and_validate_manifest(
    manifest_path: Path,
    expected_rows: Optional[int] = 30520,
) -> pd.DataFrame:
    """读取 manifest 并检查列、标签、划分及图像路径。"""
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"找不到 manifest：{manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)

    # 删除可能由旧 CSV 保存过程产生的多余索引列。
    manifest = manifest.drop(
        columns=["index", "Unnamed: 0"],
        errors="ignore",
    )

    required_columns = {
        "filepath",
        "label_name",
        "label",
        "scan_id",
        "subject_id",
        "original_split",
        "split",
    }

    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"manifest 缺少列：{sorted(missing_columns)}"
        )

    if expected_rows is not None and len(manifest) != expected_rows:
        raise AssertionError(
            f"manifest 行数应为 {expected_rows}，"
            f"实际为 {len(manifest)}。"
        )

    if not manifest["filepath"].is_unique:
        raise AssertionError("manifest 中 filepath 有重复。")

    if set(manifest["label"].astype(int)) != {0, 1}:
        raise AssertionError(
            "manifest 的 label 应同时包含 0 和 1。"
        )

    expected_splits = {"train", "validation", "test"}

    if set(manifest["split"]) != expected_splits:
        raise AssertionError(
            f"split 应为 {expected_splits}。"
        )

    missing_files = [
        path
        for path in manifest["filepath"]
        if not Path(path).exists()
    ]

    print("Manifest rows:", len(manifest))
    print("Unique scans:", manifest["scan_id"].nunique())
    print("Unique subjects:", manifest["subject_id"].nunique())
    print("Missing files:", len(missing_files))

    if missing_files:
        print("First missing files:", missing_files[:5])
        raise FileNotFoundError(
            "manifest 中有图片路径不存在。"
            "请确认数据已解压，且路径与当前 Colab 环境一致。"
        )

    return manifest


class ADNISliceDataset(Dataset):
    """按需读取单张 MRI 切片，不会一次性把所有图像载入内存。"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
    ):
        self.dataframe = (
            dataframe.reset_index(drop=True)
            .copy()
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = row["filepath"]
        label = int(row["label"])
        subject_id = str(row["subject_id"])

        with Image.open(image_path) as image:
            image = image.convert("L")

            if self.transform is not None:
                image = self.transform(image)

        return image, label, subject_id


def create_dataloaders(
    manifest: pd.DataFrame,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
):
    """根据 manifest 创建 train、validation、test DataLoader。"""
    train_transform, evaluation_transform = get_transforms()

    train_dataframe = (
        manifest[manifest["split"] == "train"]
        .reset_index(drop=True)
    )
    validation_dataframe = (
        manifest[manifest["split"] == "validation"]
        .reset_index(drop=True)
    )
    test_dataframe = (
        manifest[manifest["split"] == "test"]
        .reset_index(drop=True)
    )

    train_dataset = ADNISliceDataset(
        train_dataframe,
        transform=train_transform,
    )
    validation_dataset = ADNISliceDataset(
        validation_dataframe,
        transform=evaluation_transform,
    )
    test_dataset = ADNISliceDataset(
        test_dataframe,
        transform=evaluation_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Testing images:", len(test_dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Testing batches:", len(test_loader))

    return train_loader, validation_loader, test_loader


def save_batch_preview(
    train_loader: DataLoader,
    output_path: Path,
) -> None:
    """保存一个训练 batch 的图像预览，用于检查预处理结果。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images, labels, subject_ids = next(iter(train_loader))

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    figure, axes = plt.subplots(2, 4, figsize=(12, 6))

    for index, axis in enumerate(axes.flat):
        display_image = images[index].cpu() * std + mean
        display_image = display_image.clamp(0, 1)

        axis.imshow(
            display_image[0],
            cmap="gray",
        )

        class_name = "AD" if int(labels[index]) == 1 else "NC"

        axis.set_title(
            f"{class_name}\n{subject_ids[index]}"
        )
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("Batch preview saved to:", output_path)


def prepare_data(
    zip_path: Path = DEFAULT_ZIP_PATH,
    work_dir: Path = DEFAULT_WORK_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    rebuild_manifest: bool = False,
    expected_rows: Optional[int] = 30520,
    seed: int = 42,
) -> pd.DataFrame:
    """
    准备数据并返回已验证的 manifest。

    如果 manifest 已存在，默认读取并检查。
    如果 manifest 不存在，自动生成患者级 manifest。
    设置 rebuild_manifest=True 可以重新生成划分。
    """
    image_root, metadata_path = extract_dataset(
        zip_path=zip_path,
        work_dir=work_dir,
    )

    manifest_path = Path(manifest_path)

    if rebuild_manifest or not manifest_path.exists():
        build_patient_level_manifest(
            image_root=image_root,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            seed=seed,
        )

    return load_and_validate_manifest(
        manifest_path=manifest_path,
        expected_rows=expected_rows,
    )
