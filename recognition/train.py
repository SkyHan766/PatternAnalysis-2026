import argparse
import time
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
    count_parameters,
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
    """Run one complete training epoch or validation pass."""
    is_training = optimizer is not None
    model.train(is_training)
    description = "Train" if is_training else "Validation"

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    progress = tqdm(loader, desc=description, leave=False)

    for images, labels, _subject_ids in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
            (logits.argmax(dim=1) == labels).sum().item()
        )

    if total_images == 0:
        raise ValueError("DataLoader returned no images.")

    return {
        "loss": total_loss / total_images,
        "accuracy": total_correct / total_images,
    }


def train_model(
    model_name,
    model,
    train_loader,
    validation_loader,
    device,
    output_dir,
    epochs,
    learning_rate,
    weight_decay,
):
    """Train all model parameters from scratch and save best validation loss."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "training_history.csv"
    history = []
    best_validation_loss = float("inf")

    for epoch in range(1, epochs + 1):
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

        history.append({
            "model": model_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        })
        pd.DataFrame(history).to_csv(history_path, index=False)

        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = validation_metrics["loss"]
            torch.save(
                {
                    "model_name": model_name,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "validation_loss": validation_metrics["loss"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "class_to_index": CLASS_TO_INDEX,
                    "pretrained": False,
                    "parameter_count": count_parameters(model),
                    "learning_rate": learning_rate,
                },
                checkpoint_path,
            )

        saved_text = " | saved best" if improved else ""
        print(
            f"{model_name} | epoch {epoch}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f}, "
            f"acc {train_metrics['accuracy']:.4f} | "
            f"val loss {validation_metrics['loss']:.4f}, "
            f"acc {validation_metrics['accuracy']:.4f}"
            f"{saved_text}"
        )

    return checkpoint_path, history_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a from-scratch ConvNeXt-Tiny or parameter-matched "
            "plain CNN on ADNI AD/NC images."
        )
    )
    parser.add_argument(
        "--model",
        choices=["convnext", "baseline"],
        default="convnext",
        help="Architecture to train.",
    )
    parser.add_argument("--zip_path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--work_dir", type=Path, default=DEFAULT_WORK_DIR)
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--expected_rows",
        type=int,
        default=30520,
        help="Set to -1 to disable the manifest row-count check.",
    )
    parser.add_argument(
        "--rebuild_manifest",
        action="store_true",
        help="Recreate the patient-level manifest and split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    expected_rows = None if args.expected_rows < 0 else args.expected_rows

    set_seed(args.seed)
    device = get_device(require_cuda=True)

    # Separate folders prevent one model's files from overwriting the other.
    model_output_dir = args.output_dir / args.model
    model_output_dir.mkdir(parents=True, exist_ok=True)

    manifest = prepare_data(
        zip_path=args.zip_path,
        work_dir=args.work_dir,
        manifest_path=args.manifest_path,
        rebuild_manifest=args.rebuild_manifest,
        expected_rows=expected_rows,
        seed=args.seed,
    )

    train_loader, validation_loader, _test_loader = create_dataloaders(
        manifest=manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    save_batch_preview(
        train_loader,
        model_output_dir / "sample_batch.png",
    )

    # Both architectures are initialized randomly and fully trained from epoch 1.
    model = build_model(model_name=args.model, num_classes=2).to(device)
    model_parameters = count_parameters(model)
    other_model_name = "baseline" if args.model == "convnext" else "convnext"
    # Count the comparison model without changing the RNG state used for training.
    with torch.random.fork_rng(devices=[]):
        reference_model = build_model(model_name=other_model_name, num_classes=2)
        reference_parameters = count_parameters(reference_model)
        del reference_model
    parameter_gap_percent = (
        abs(model_parameters - reference_parameters)
        / max(model_parameters, reference_parameters)
        * 100.0
    )
    print(f"Model: {args.model}")
    print(f"Trainable parameters: {model_parameters:,}")
    print(
        f"{other_model_name} comparison parameters: "
        f"{reference_parameters:,} ({parameter_gap_percent:.3f}% gap)"
    )
    if parameter_gap_percent > 1.0:
        raise RuntimeError(
            "The two model parameter counts differ by more than 1%; "
            "review the baseline architecture before benchmarking."
        )
    print(f"Optimizer: AdamW | learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs} | pretrained weights: none")

    check_model_output(model, device, input_size=224)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()
    checkpoint_path, history_path = train_model(
        model_name=args.model,
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        output_dir=model_output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time

    print("\nTraining completed.")
    print(f"Training time: {elapsed_seconds / 60:.2f} minutes")
    if device.type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"Peak allocated GPU memory: {peak_memory_gb:.2f} GB")

    print("Training history:", history_path)
    print("Best checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
