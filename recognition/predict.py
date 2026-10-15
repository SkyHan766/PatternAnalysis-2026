import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import SequentialSampler

from dataset import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WORK_DIR,
    DEFAULT_ZIP_PATH,
    create_dataloaders,
    prepare_data,
)
from modules import build_model, get_device


def select_best_checkpoint(
    model,
    device,
    output_dir: Path,
) -> tuple[Path, dict]:
    """
    在分类头阶段和微调阶段的最佳 checkpoint 中，
    选择 validation loss 较低者。
    """
    candidates = [
        output_dir / "convnext_head_best.pt",
        output_dir / "convnext_finetune_best.pt",
    ]

    for path in candidates:
        if not path.exists():
            raise FileNotFoundError(
                f"找不到模型权重：{path}。"
                "请先运行 train.py。"
            )

    checkpoint_infos = []

    for path in candidates:
        info = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        print(
            "Checkpoint:",
            path.name,
            "| stage:",
            info["stage"],
            "| epoch:",
            info["epoch"],
            "| validation loss:",
            info["validation_loss"],
        )

        checkpoint_infos.append((path, info))

    selected_path, selected_info = min(
        checkpoint_infos,
        key=lambda item: item[1]["validation_loss"],
    )

    model.load_state_dict(selected_info["model_state"])
    model.to(device)
    model.eval()

    print("\nSelected:", selected_path.name)
    print("Stage:", selected_info["stage"])
    print("Epoch:", selected_info["epoch"])
    print(
        "Saved validation loss:",
        selected_info["validation_loss"],
    )

    return selected_path, selected_info


def collect_predictions(
    model,
    loader,
    device,
) -> pd.DataFrame:
    """对一个 split 的所有图像预测 AD 概率。"""
    if not isinstance(loader.sampler, SequentialSampler):
        raise AssertionError(
            "评估 DataLoader 必须 shuffle=False，"
            "以保证预测顺序和 manifest 一致。"
        )

    rows = loader.dataset.dataframe[
        ["filepath", "label", "scan_id", "subject_id", "split"]
    ].reset_index(drop=True).copy()

    probabilities = []
    observed_labels = []
    observed_subjects = []

    model.eval()

    with torch.inference_mode():
        for images, labels, subject_ids in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)

            prob_ad = torch.softmax(
                logits.float(),
                dim=1,
            )[:, 1]

            probabilities.extend(
                prob_ad.cpu().tolist()
            )
            observed_labels.extend(
                labels.tolist()
            )
            observed_subjects.extend(
                list(subject_ids)
            )

    if len(probabilities) != len(rows):
        raise AssertionError(
            "预测数量和 manifest 行数不一致。"
        )

    if observed_labels != rows["label"].astype(int).tolist():
        raise AssertionError(
            "DataLoader 标签顺序和 manifest 不一致。"
        )

    if observed_subjects != rows["subject_id"].astype(str).tolist():
        raise AssertionError(
            "DataLoader 患者顺序和 manifest 不一致。"
        )

    rows["prob_ad"] = probabilities
    rows["pred_label"] = (
        rows["prob_ad"] >= 0.5
    ).astype(int)

    return rows


def aggregate_predictions(
    image_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    从切片级预测聚合到扫描级和患者级。

    先平均同一扫描切片的 AD 概率，
    再平均同一患者不同扫描的 AD 概率。
    """
    slices_per_scan = image_rows.groupby(
        ["subject_id", "scan_id"]
    )["label"].nunique()

    if not (slices_per_scan == 1).all():
        raise AssertionError(
            "同一扫描中出现了不一致的标签。"
        )

    scan_rows = (
        image_rows.groupby(
            ["subject_id", "scan_id"],
            as_index=False,
        )
        .agg(
            label=("label", "first"),
            prob_ad=("prob_ad", "mean"),
        )
    )

    scans_per_subject = scan_rows.groupby(
        "subject_id"
    )["label"].nunique()

    if not (scans_per_subject == 1).all():
        raise AssertionError(
            "同一患者的不同扫描出现了不一致的标签。"
        )

    patient_rows = (
        scan_rows.groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            label=("label", "first"),
            prob_ad=("prob_ad", "mean"),
        )
    )

    for table in (scan_rows, patient_rows):
        table["pred_label"] = (
            table["prob_ad"] >= 0.5
        ).astype(int)

    return scan_rows, patient_rows


def calculate_metrics(
    rows: pd.DataFrame,
) -> tuple[dict, object]:
    """计算准确率、平衡准确率、AD recall、AD F1 和 ROC-AUC。"""
    truth = rows["label"].astype(int)
    prediction = rows["pred_label"].astype(int)

    metrics = {
        "samples": len(rows),
        "accuracy": accuracy_score(
            truth,
            prediction,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            truth,
            prediction,
        ),
        "ad_recall": recall_score(
            truth,
            prediction,
            pos_label=1,
            zero_division=0,
        ),
        "ad_f1": f1_score(
            truth,
            prediction,
            pos_label=1,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            truth,
            rows["prob_ad"],
        ),
    }

    matrix = confusion_matrix(
        truth,
        prediction,
        labels=[0, 1],
    )

    return metrics, matrix


def evaluate_split(
    model,
    loader,
    split_name: str,
    device,
    output_dir: Path,
) -> pd.DataFrame:
    """评估一个 split，并保存图片级、扫描级和患者级预测 CSV。"""
    image_rows = collect_predictions(
        model=model,
        loader=loader,
        device=device,
    )

    if set(image_rows["split"]) != {split_name}:
        raise AssertionError(
            f"DataLoader 中的数据并非全部属于 {split_name}。"
        )

    scan_rows, patient_rows = aggregate_predictions(
        image_rows
    )

    result_rows = []

    levels = [
        ("image", image_rows),
        ("scan", scan_rows),
        ("patient", patient_rows),
    ]

    for level, rows in levels:
        metrics, matrix = calculate_metrics(rows)

        result_rows.append({
            "split": split_name,
            "level": level,
            **metrics,
        })

        print(f"\n{split_name} — {level}")
        print(
            "Confusion matrix "
            "(rows=actual, columns=predicted; NC, AD):"
        )
        print(matrix)

        rows.to_csv(
            output_dir / f"{split_name}_{level}_predictions.csv",
            index=False,
        )

    summary = pd.DataFrame(result_rows)

    print(f"\n{split_name} metrics:")
    print(summary.round(4).to_string(index=False))

    summary.to_csv(
        output_dir / f"{split_name}_metrics.csv",
        index=False,
    )

    return summary


def save_result_figures(output_dir: Path) -> None:
    """保存训练曲线和测试集图片级/患者级混淆矩阵。"""
    history_path = output_dir / "training_history.csv"

    if history_path.exists():
        history_df = pd.read_csv(history_path)

        epochs = range(1, len(history_df) + 1)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(
            epochs,
            history_df["train_loss"],
            marker="o",
            label="Train",
        )
        axes[0].plot(
            epochs,
            history_df["validation_loss"],
            marker="o",
            label="Validation",
        )
        axes[0].set(
            title="Loss by epoch",
            xlabel="Epoch (1–2 head; 3–8 fine-tuning)",
            ylabel="Loss",
        )
        axes[0].legend()

        axes[1].plot(
            epochs,
            history_df["train_accuracy"],
            marker="o",
            label="Train",
        )
        axes[1].plot(
            epochs,
            history_df["validation_accuracy"],
            marker="o",
            label="Validation",
        )
        axes[1].set(
            title="Accuracy by epoch",
            xlabel="Epoch (1–2 head; 3–8 fine-tuning)",
            ylabel="Accuracy",
        )
        axes[1].legend()

        for axis in axes:
            axis.axvline(
                2.5,
                color="gray",
                linestyle="--",
                alpha=0.7,
            )
            axis.grid(alpha=0.25)

        figure.tight_layout()
        figure.savefig(
            output_dir / "training_curves.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)

    # 测试集预测 CSV 存在时，生成图像级和患者级混淆矩阵。
    image_csv = output_dir / "test_image_predictions.csv"
    patient_csv = output_dir / "test_patient_predictions.csv"

    if image_csv.exists() and patient_csv.exists():
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))

        for axis, level, csv_path in [
            (axes[0], "image", image_csv),
            (axes[1], "patient", patient_csv),
        ]:
            predictions = pd.read_csv(csv_path)

            matrix = confusion_matrix(
                predictions["label"],
                predictions["pred_label"],
                labels=[0, 1],
            )

            ConfusionMatrixDisplay(
                confusion_matrix=matrix,
                display_labels=["NC", "AD"],
            ).plot(
                ax=axis,
                cmap="Blues",
                colorbar=False,
                values_format="d",
            )

            axis.set_title(f"Test: {level} level")

        figure.tight_layout()
        figure.savefig(
            output_dir / "test_confusion_matrices.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)

    print("Figures saved to:", output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained ADNI ConvNeXt model."
    )

    parser.add_argument(
        "--zip_path",
        type=Path,
        default=DEFAULT_ZIP_PATH,
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--expected_rows",
        type=int,
        default=30520,
        help="不检查行数时传入 -1。",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test", "both"],
        default="both",
        help="选择评估 validation、test 或两者。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    expected_rows = (
        None
        if args.expected_rows < 0
        else args.expected_rows
    )

    device = get_device(require_cuda=True)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = prepare_data(
        zip_path=args.zip_path,
        work_dir=args.work_dir,
        manifest_path=args.manifest_path,
        rebuild_manifest=False,
        expected_rows=expected_rows,
    )

    _train_loader, validation_loader, test_loader = (
        create_dataloaders(
            manifest=manifest,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )

    model = build_model(
        num_classes=2,
        pretrained=False,
    )

    select_best_checkpoint(
        model=model,
        device=device,
        output_dir=args.output_dir,
    )

    if args.split in {"validation", "both"}:
        evaluate_split(
            model=model,
            loader=validation_loader,
            split_name="validation",
            device=device,
            output_dir=args.output_dir,
        )

    if args.split in {"test", "both"}:
        evaluate_split(
            model=model,
            loader=test_loader,
            split_name="test",
            device=device,
            output_dir=args.output_dir,
        )

    save_result_figures(args.output_dir)


if __name__ == "__main__":
    main()
