import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from dataset import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WORK_DIR,
    DEFAULT_ZIP_PATH,
    create_dataloaders,
    prepare_data,
    save_batch_preview,
)
from modules import (
    CLASS_TO_INDEX,
    build_model,
    check_model_output,
    get_device,
    set_seed,
)


def run_one_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer=None,
):
    """
    运行一个训练或验证 epoch。

    optimizer 不为 None 时执行训练；
    optimizer 为 None 时只进行验证。
    """
    is_training = optimizer is not None

    if is_training:
        model.train()
        description = "Train"
    else:
        model.eval()
        description = "Validation"

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    progress = tqdm(
        loader,
        desc=description,
        leave=False,
    )

    for images, labels, _subject_ids in progress:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_images += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (
            (logits.argmax(dim=1) == labels)
            .sum()
            .item()
        )

    if total_images == 0:
        raise ValueError("DataLoader 没有提供任何图像。")

    return {
        "loss": total_loss / total_images,
        "accuracy": total_correct / total_images,
    }


def train_stage(
    stage_name,
    num_epochs,
    model,
    train_loader,
    validation_loader,
    device,
    criterion,
    optimizer,
    checkpoint_path,
    output_dir,
    history,
):
    """训练一个阶段，并保存该阶段验证损失最低的 checkpoint。"""
    best_validation_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
        )

        validation_metrics = run_one_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            criterion=criterion,
            optimizer=None,
        )

        row = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }

        history.append(row)

        # 每个 epoch 更新 CSV，减少运行中断导致记录丢失的风险。
        pd.DataFrame(history).to_csv(
            output_dir / "training_history.csv",
            index=False,
        )

        improved = (
            validation_metrics["loss"] < best_validation_loss
        )

        if improved:
            best_validation_loss = validation_metrics["loss"]

            torch.save(
                {
                    "stage": stage_name,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "validation_loss": validation_metrics["loss"],
                    "validation_accuracy": (
                        validation_metrics["accuracy"]
                    ),
                    "class_to_index": CLASS_TO_INDEX,
                },
                checkpoint_path,
            )

        saved_text = " | saved best" if improved else ""

        print(
            f"{stage_name} | "
            f"epoch {epoch}/{num_epochs} | "
            f"train loss {train_metrics['loss']:.4f}, "
            f"acc {train_metrics['accuracy']:.4f} | "
            f"val loss {validation_metrics['loss']:.4f}, "
            f"acc {validation_metrics['accuracy']:.4f}"
            f"{saved_text}"
        )

    print("Best checkpoint:", checkpoint_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ConvNeXt-Tiny on ADNI AD/NC images."
    )

    parser.add_argument(
        "--zip_path",
        type=Path,
        default=DEFAULT_ZIP_PATH,
        help="ADNI.zip 的路径。",
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="数据解压目录。",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="患者级 manifest CSV 路径。",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="模型权重和训练结果的保存目录。",
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
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--expected_rows",
        type=int,
        default=30520,
        help="预期 manifest 行数；不检查时传入 -1。",
    )
    parser.add_argument(
        "--rebuild_manifest",
        action="store_true",
        help="强制重新建立患者级 manifest。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    expected_rows = (
        None
        if args.expected_rows < 0
        else args.expected_rows
    )

    set_seed(args.seed)

    device = get_device(require_cuda=True)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 若 manifest 不存在，prepare_data 会从 metadata 建立它。
    manifest = prepare_data(
        zip_path=args.zip_path,
        work_dir=args.work_dir,
        manifest_path=args.manifest_path,
        rebuild_manifest=args.rebuild_manifest,
        expected_rows=expected_rows,
        seed=args.seed,
    )

    train_loader, validation_loader, _test_loader = (
        create_dataloaders(
            manifest=manifest,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    )

    save_batch_preview(
        train_loader=train_loader,
        output_path=args.output_dir / "sample_batch.png",
    )

    model = build_model(
        num_classes=2,
        pretrained=True,
    ).to(device)

    check_model_output(
        model=model,
        device=device,
        input_size=224,
    )

    criterion = nn.CrossEntropyLoss()
    history = []

    # 阶段一：冻结特征提取器，只训练新的分类层。
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier[-1].parameters():
        parameter.requires_grad = True

    head_optimizer = torch.optim.AdamW(
        model.classifier[-1].parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    head_checkpoint = (
        args.output_dir / "convnext_head_best.pt"
    )

    train_stage(
        stage_name="classifier_head",
        num_epochs=2,
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        criterion=criterion,
        optimizer=head_optimizer,
        checkpoint_path=head_checkpoint,
        output_dir=args.output_dir,
        history=history,
    )

    # 阶段二：载入第一阶段最佳权重，再微调整个模型。
    head_info = torch.load(
        head_checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(head_info["model_state"])

    for parameter in model.parameters():
        parameter.requires_grad = True

    finetune_optimizer = torch.optim.AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": 1e-5,
            },
            {
                "params": model.classifier.parameters(),
                "lr": 1e-4,
            },
        ],
        weight_decay=1e-4,
    )

    finetune_checkpoint = (
        args.output_dir / "convnext_finetune_best.pt"
    )

    train_stage(
        stage_name="full_finetune",
        num_epochs=6,
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        criterion=criterion,
        optimizer=finetune_optimizer,
        checkpoint_path=finetune_checkpoint,
        output_dir=args.output_dir,
        history=history,
    )

    print("\nTraining completed.")
    print("Training history:", args.output_dir / "training_history.csv")
    print("Head checkpoint:", head_checkpoint)
    print("Fine-tuning checkpoint:", finetune_checkpoint)


if __name__ == "__main__":
    main()
