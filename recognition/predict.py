"""
Evaluate a trained narrow ConvNeXt on train, validation, or test data.

Report nine metrics, confusion matrices, sample counts, and GPU resources.
Scan and patient results average slice AD probabilities within each group.
Save eval_report.txt, eval_report.json, and predictions_<split>.csv.

Example:
  python predict.py --checkpoint checkpoints/best_model.pth --split test
"""
import os
import json
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from modules import ConvNeXtNarrow
from dataset import (load_manifest, ADNISliceDataset, ensure_data_root,
                     DEFAULT_DATA_ROOT, DEFAULT_ZIP_PATH)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)


# --------------------------------------------------------------------------- #
# Compute nine evaluation metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_prob_ad):
    """y_prob_ad contains one AD probability per sample."""
    y_pred = (y_prob_ad >= 0.5).astype(int)
    y_true = np.asarray(y_true).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    nc_prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    nc_rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    ad_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    ad_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ad_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob_ad)
    except ValueError:  # Undefined AUC, e.g. only one class is present.
        auc = float("nan")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "nc_precision": nc_prec,
        "nc_recall": nc_rec,
        "ad_precision": ad_prec,
        "ad_recall": ad_rec,
        "macro_f1": macro_f1,
        "ad_f1": ad_f1,
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        # Rows: actual [NC, AD]; columns: predicted [NC, AD].
        "n_samples": len(y_true),
    }


def aggregate_level(df, prob_col, key):
    """Average slice AD probabilities by scan or patient, then threshold at 0.5."""
    grp = df.groupby(key).agg(
        y_true=("y_true", "first"),
        prob=("prob_ad", "mean"),
    )
    return compute_metrics(grp["y_true"].values, grp["prob"].values)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    probs, labels, metas = [], [], []
    n_imgs, total_time = 0, 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    for x, y, meta in loader:
        x = x.to(device, non_blocking=True)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        logits = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - t0
        p = torch.softmax(logits, dim=1)[:, 1]  # AD probability
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
        n_imgs += x.size(0)
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    # Collect path, patient, and scan metadata.
    for k in ("path", "patient_id", "scan_id"):
        metas.append(meta_list(loader, k))
    per_img_ms = total_time / max(n_imgs, 1) * 1000
    peak_mem_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2
                   if device == "cuda" else 0.0)
    return probs, labels, metas, per_img_ms, peak_mem_mb


def meta_list(loader, key):
    """Read metadata in dataset order; inference uses shuffle=False."""
    ds = loader.dataset
    return [ds.df.iloc[i][key] for i in range(len(ds))]


# --------------------------------------------------------------------------- #
# Format the evaluation report
# --------------------------------------------------------------------------- #
METRIC_KEYS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("nc_precision", "NC precision"),
    ("nc_recall", "NC recall"),
    ("ad_precision", "AD precision"),
    ("ad_recall", "AD recall"),
    ("macro_f1", "Macro-F1"),
    ("ad_f1", "AD F1"),
    ("roc_auc", "ROC-AUC"),
]


def print_level(name, m, lines):
    lines.append(f"\n===== {name} =====")
    lines.append(f"Samples: {m['n_samples']}")
    for k, label in METRIC_KEYS:
        lines.append(f"  {label:18s}: {m[k]:.4f}")
    cm = m["confusion_matrix"]
    lines.append("  Confusion matrix (rows=actual NC,AD; columns=predicted NC,AD):")
    lines.append(f"    NC: {cm[0]}")
    lines.append(f"    AD: {cm[1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default=None, help="Defaults to manifest.csv beside the checkpoint.")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--zip_path", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", "--output_dir", dest="out_dir", default=None, help="Output directory; defaults to the checkpoint directory.")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Model ----------------
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = ConvNeXtNarrow(
        in_channels=1, num_classes=2,
        base_channels=ckpt_args.get("base_channels", 32),
        depths=(3, 3, 9, 3),
        drop_path_rate=ckpt_args.get("drop_path_rate", 0.2),
    ).to(device)
    if "ema_state_dict" in ckpt:
        inference_state_dict = ckpt["ema_state_dict"]
        weight_source = "ema"
    else:
        inference_state_dict = ckpt["state_dict"]
        weight_source = ckpt.get("weight_source", "model")
    model.load_state_dict(inference_state_dict)
    n_params = model.num_parameters()

    # ---------------- Data ----------------
    data_root = ensure_data_root(
        args.data_root or ckpt_args.get("data_root", DEFAULT_DATA_ROOT),
        args.zip_path or ckpt_args.get("zip_path", DEFAULT_ZIP_PATH),
    )
    if args.manifest is None:
        args.manifest = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "manifest.csv")
    df = load_manifest(args.manifest, data_root=data_root)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No samples with split={args.split} in the manifest.")
    ds = ADNISliceDataset(df, img_size=ckpt_args.get("img_size", 224), train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # ---------------- Inference ----------------
    probs, labels, (paths, pids, sids), per_img_ms, peak_mem = infer(
        model, loader, device
    )

    import pandas as pd
    rec = pd.DataFrame({
        "path": paths, "patient_id": pids, "scan_id": sids,
        "y_true": labels, "prob_ad": probs,
    })

    # ---------------- Metrics at three evaluation levels ----------------
    img_m = compute_metrics(rec["y_true"].values, rec["prob_ad"].values)
    scan_m = aggregate_level(rec, "prob_ad", "scan_id")
    pat_m = aggregate_level(rec, "prob_ad", "patient_id")

    lines = []
    lines.append(f"checkpoint: {args.checkpoint}")
    lines.append(f"weight source: {weight_source}")
    if weight_source == "ema":
        lines.append(f"EMA decay: {ckpt.get('ema_decay', ckpt_args.get('ema_decay', 'unknown'))}")
    lines.append(f"split: {args.split} | device: {device}")
    lines.append(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
    lines.append(f"Inference time: {per_img_ms:.2f} ms/slices")
    lines.append(f"Peak allocated GPU memory: {peak_mem:.1f} MB")
    print_level("Image level (individual slices)", img_m, lines)
    print_level("Scan level (mean slice probability per scan)", scan_m, lines)
    print_level("Patient level (mean slice probability per patient)", pat_m, lines)

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, "eval_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(out_dir, "eval_report.json"), "w") as f:
        json.dump({
            "image_level": img_m,
            "scan_level": scan_m,
            "patient_level": pat_m,
            "resources": {
                "parameters": n_params,
                "inference_ms_per_image": per_img_ms,
                "peak_gpu_memory_mb": peak_mem,
                "device": device,
                "n_images": len(rec),
                "n_scans": rec["scan_id"].nunique(),
                "n_patients": rec["patient_id"].nunique(),
            },
            "weight_source": weight_source,
            "ema_decay": ckpt.get("ema_decay", ckpt_args.get("ema_decay")),
        }, f, indent=2)
    rec.to_csv(os.path.join(out_dir, f"predictions_{args.split}.csv"), index=False)
    print(f"\nReports saved to {out_dir}/eval_report.txt / .json，"
          f"Slice predictions: predictions_{args.split}.csv")


if __name__ == "__main__":
    main()
