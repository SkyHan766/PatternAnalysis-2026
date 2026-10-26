"""
dataset.py — ADNI MRI 切片数据集与患者级安全划分

数据说明（学校 ADNI 包，/home/groups/comp3710/ADNI）：
- 预处理后的结构 MRI 切片，AD / CN 两类（代码中 NC 与 CN 同义）
- 原始 Train/Test 文件夹划分存在 216 名患者重叠，不满足患者隔离要求，
  因此本模块提供按患者（patient）为单位重新划分的能力。
- 若已有一份 manifest csv（含 path/label/patient_id/scan_id/split 字段），
  可直接加载使用。

manifest 约定字段：
  path        图像文件的绝对或相对路径
  label       'NC' 或 'AD'
  patient_id  患者唯一编号（同一患者所有扫描/切片共享）
  scan_id     扫描唯一编号（同一次扫描的所有切片共享，介于患者与切片之间）
  split       'train' / 'val' / 'test'
"""
import os
import json
import zipfile
from pathlib import Path
import re
import random
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

try:
    from torchvision import transforms
except ImportError:  # 无 torchvision 时的最小 fallback
    transforms = None


# 与之前 Colab 代码一致的压缩包、解压及结果目录。
DEFAULT_ZIP_PATH = "/content/drive/MyDrive/COMP3710/Final_Project_dataset/ADNI.zip"
DEFAULT_DATA_ROOT = "/content/comp3710_adni/ADNI"
DEFAULT_OUTPUT_DIR = (
    "/content/drive/MyDrive/COMP3710/Final_Project_dataset/convnext_results_4/"
    "convnext_narrow_b64_fromkimi"
)


def ensure_data_root(data_root=DEFAULT_DATA_ROOT, zip_path=DEFAULT_ZIP_PATH):
    """复用已解压数据；缺失时将学校 ADNI.zip 解压至数据根目录的父目录。"""
    root = Path(data_root).expanduser().resolve()
    image_root = root / "AD_NC" if (root / "AD_NC").is_dir() else root
    metadata_candidates = [root / "meta_data_with_label.json",
                           root.parent / "meta_data_with_label.json"]
    if ((image_root / "train").is_dir() and (image_root / "test").is_dir()
            and any(p.is_file() for p in metadata_candidates)):
        print(f"Dataset already extracted: {root}")
        return str(root)

    archive = Path(zip_path).expanduser()
    if not archive.is_file() and str(archive) == DEFAULT_ZIP_PATH:
        alternative = archive.with_name("ADNI(2).zip")
        if alternative.is_file():
            archive = alternative
    if not archive.is_file():
        raise FileNotFoundError(
            f"数据目录不完整：{root}；压缩包不存在：{archive}。"
            "请先挂载 Google Drive，或通过 --zip_path 指定压缩包。"
        )
    # 学校压缩包内包含 ADNI/AD_NC 和 ADNI/meta_data_with_label.json。
    if root.name != "ADNI":
        raise FileNotFoundError(
            f"自定义数据目录尚未准备好：{root}。自动解压要求 data_root 末尾为 ADNI。"
        )
    destination = root.parent
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.filename.startswith("__MACOSX/")]
        for member in members:
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError(f"Invalid zip member: {member.filename}")
        z.extractall(destination, members=members)
    if not ((root / "AD_NC" / "train").is_dir()
            and (root / "AD_NC" / "test").is_dir()
            and (root / "meta_data_with_label.json").is_file()):
        raise FileNotFoundError(f"解压后未找到预期的 AD_NC 和 metadata：{root}")
    print(f"Dataset extracted: {root}")
    return str(root)


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")

# ADNI 文件名中常见的患者编号格式，例如 006_S_4153 / 037_S_4430
_ADNI_SUBJECT_RE = re.compile(r"(\d{3}_S_\d{4})")


# --------------------------------------------------------------------------- #
# 从文件名解析患者 / 扫描编号
# --------------------------------------------------------------------------- #
def parse_patient_scan_id(filename, metadata=None):
    """
    从 ADNI 切片文件名解析 (patient_id, scan_id)。
    兼容形如 '006_S_4153_MR_..._br_raw_xxx.png' 的命名：
      - patient_id = ADNI 受试者编号 (e.g. 006_S_4153)
      - scan_id    = 受试者编号 + 文件名中扫描/序列相关的前缀字段
    学校的数字文件名（如 1003730_100.jpeg）通过 metadata 映射患者。
    无法确定患者时直接报错，不能把每张切片当成独立患者。
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    short = re.fullmatch(r"(\d+)_(\d+)", stem)
    if short:
        scan_id = short.group(1)
        record = (metadata or {}).get(scan_id)
        if not isinstance(record, dict):
            raise ValueError(f"扫描 {scan_id} 缺少 metadata，无法确定患者编号。")
        subjects = {
            match.group(1)
            for value in record.values() if isinstance(value, str)
            for match in _ADNI_SUBJECT_RE.finditer(value)
        }
        if len(subjects) != 1:
            raise ValueError(f"扫描 {scan_id} 的患者编号缺失或不一致：{subjects}")
        return subjects.pop(), scan_id
    m = _ADNI_SUBJECT_RE.search(stem)
    if m:
        patient_id = m.group(1)
        rest = stem[m.end():].lstrip("_").split("_")
        scan_tokens = [t for t in rest if t][:4]  # 取扫描相关的前几个字段
        scan_id = "_".join([patient_id] + scan_tokens) if scan_tokens else patient_id
        return patient_id, scan_id
    raise ValueError(f"无法从文件名确定患者/扫描编号：{filename}")


# --------------------------------------------------------------------------- #
# 扫描磁盘构建 manifest，并按患者重新划分 train/val/test
# --------------------------------------------------------------------------- #
def build_manifest(
    data_root,
    out_csv=None,
    test_ratio=0.20,
    val_ratio=0.10,
    seed=42,
    folder_split=True,
    metadata_path=None,
):
    """
    扫描 data_root 下所有图像，生成含患者/扫描/划分信息的 manifest。

    folder_split=True  （默认）期望目录结构为
        data_root/{train,test}/{AD,CN|NC}/xxx.png
        此时所有文件先纳入统计，再按患者重新划分，避免患者泄漏。
    folder_split=False 期望目录结构为 data_root/{AD,CN}/xxx.png。
    也支持 data_root/AD_NC/{train,test}/{AD,NC}/xxx.jpeg。
    数字文件名使用 meta_data_with_label.json 映射患者；可传入 metadata_path，
    默认在 data_root 或其父目录寻找该文件。

    划分方式：按患者分组、按标签分层，随机打乱后按比例切分，
    保证同一患者的所有切片只出现在一个 split 中。
    """
    data_root = os.path.abspath(data_root)
    if os.path.isdir(os.path.join(data_root, "AD_NC")):
        data_root = os.path.join(data_root, "AD_NC")
    if metadata_path is None:
        for directory in (data_root, os.path.dirname(data_root)):
            candidate = os.path.join(directory, "meta_data_with_label.json")
            if os.path.isfile(candidate):
                metadata_path = candidate
                break
    metadata = None
    if metadata_path is not None:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

    rows = []
    if folder_split:
        split_dirs = [d for d in os.listdir(data_root)
                      if os.path.isdir(os.path.join(data_root, d))]
    else:
        split_dirs = ["."]

    for split_name in split_dirs:
        split_dir = os.path.join(data_root, split_name)
        for label in ("AD", "CN", "NC"):
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue
            label_norm = "NC" if label in ("CN", "NC") else "AD"
            for dirpath, _, files in os.walk(label_dir):
                for f in sorted(files):
                    if f.lower().endswith(IMG_EXTS):
                        path = os.path.join(dirpath, f)
                        pid, sid = parse_patient_scan_id(f, metadata)
                        rows.append({
                            "path": path,
                            "label": label_norm,
                            "patient_id": pid,
                            "scan_id": sid,
                            "orig_split": split_name,
                        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"在 {data_root} 下未找到任何图像，请检查路径与目录结构。")

    # ---- 按患者重新划分（患者隔离，训练/测试无重叠） ----
    # 必须在去重前检查，否则 drop_duplicates 会隐藏标签冲突。
    if (df.groupby("patient_id")["label"].nunique() > 1).any():
        raise ValueError("同一患者出现不同类别标签，请检查数据。")
    rng = random.Random(seed)
    patients = (df[["patient_id", "label"]]
                .drop_duplicates("patient_id")
                .reset_index(drop=True))
    # 同一患者只应有一个标签
    assert patients["patient_id"].is_unique

    assign = {}
    for label in ("NC", "AD"):
        group = patients[patients["label"] == label]["patient_id"].tolist()
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * test_ratio)) if n > 2 else 0
        n_val = max(1, round(n * val_ratio)) if n - n_test > 2 else 0
        for i, pid in enumerate(group):
            if i < n_test:
                assign[pid] = "test"
            elif i < n_test + n_val:
                assign[pid] = "val"
            else:
                assign[pid] = "train"

    df["split"] = df["patient_id"].map(assign)
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"manifest 已保存至 {out_csv}，共 {len(df)} 张切片，"
              f"{df['patient_id'].nunique()} 名患者。")
    return df


def load_manifest(csv_path, data_root=None):
    df = pd.read_csv(csv_path)
    need = {"path", "label", "patient_id", "scan_id", "split"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"manifest 缺少字段: {missing}")
    df["label"] = df["label"].replace({"CN": "NC"})
    # 仅适配图片路径，不改变标签、患者或 split。
    def resolve_image_path(value):
        path = Path(value)
        if path.is_file():
            return str(path)
        if data_root is not None:
            root = Path(data_root)
            image_root = root / "AD_NC" if (root / "AD_NC").is_dir() else root
            if "AD_NC" in path.parts:
                candidate = image_root.joinpath(*path.parts[path.parts.index("AD_NC") + 1:])
            elif not path.is_absolute():
                candidate = root / path
            else:
                candidate = path
            if candidate.is_file():
                return str(candidate)
        candidate = Path(csv_path).resolve().parent / path
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"manifest 中的图片不存在：{value}")
    df["path"] = df["path"].map(resolve_image_path)
    return df


# --------------------------------------------------------------------------- #
# 图像读取
# --------------------------------------------------------------------------- #
def load_image(path):
    if path.lower().endswith(".npy"):
        import numpy as np
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[None, ...]  # HW -> 1HW
        return torch.from_numpy(arr).float()
    with Image.open(path) as im:
        im = im.convert("L")  # 灰度
        arr = torch.from_numpy(
            __import__("numpy").array(im, dtype="float32") / 255.0
        )
    return arr.unsqueeze(0)  # 1 x H x W


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def build_transform(img_size=224, train=True):
    """训练/验证共用预处理；训练时加入轻度数据增强。"""

    if transforms is not None:
        ops = [
            transforms.Resize((img_size, img_size))
        ]

        if train:
            ops += [
                transforms.RandomHorizontalFlip(p=0.5),

                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                ),

                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                ),
            ]

        ops += [
            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.5],
                std=[0.5],
            ),
        ]

        return transforms.Compose(ops)

    # 没有 torchvision 时的备用处理
    def _tf(img):
        x = load_image(img)

        x = torch.nn.functional.interpolate(
            x[None],
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )[0]

        return (x - 0.5) / 0.5

    return _tf


class ADNISliceDataset(Dataset):
    """返回 (image_tensor, label, meta_dict)。meta 含 path/patient/scan，供评估聚合。"""

    def __init__(self, dataframe, img_size=224, train=True):
        self.df = dataframe.reset_index(drop=True)
        self.tf = build_transform(img_size, train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if transforms is not None:
            with Image.open(row["path"]) as im:
                x = self.tf(im.convert("L"))
        else:
            x = self.tf(row["path"])
        y = 0 if row["label"] == "NC" else 1
        meta = {
            "path": row["path"],
            "patient_id": row["patient_id"],
            "scan_id": row["scan_id"],
        }
        return x, y, meta


def split_dataframe(df):
    """返回 train/val/test 三个子集 DataFrame。"""
    out = {}
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        out[s] = sub.reset_index(drop=True)
        print(f"split={s:5s}: {len(sub):6d} 张切片, "
              f"{sub['scan_id'].nunique():4d} 次扫描, "
              f"{sub['patient_id'].nunique():4d} 名患者")
    # 患者隔离检查
    tr = set(out["train"]["patient_id"]) | set(out["val"]["patient_id"])
    te = set(out["test"]["patient_id"])
    overlap = tr & te
    print(f"train/val 与 test 的患者重叠数: {len(overlap)}（应为 0）")
    if overlap or (set(out["train"]["patient_id"]) &
                   set(out["val"]["patient_id"])):
        raise ValueError("train/val/test 之间存在患者重叠。")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--zip_path", default=DEFAULT_ZIP_PATH)
    ap.add_argument("--out_csv", default="manifest.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metadata_path", default=None)
    args = ap.parse_args()
    args.data_root = ensure_data_root(args.data_root, args.zip_path)
    df = build_manifest(args.data_root, args.out_csv, seed=args.seed,
                        metadata_path=args.metadata_path)
    split_dataframe(df)
