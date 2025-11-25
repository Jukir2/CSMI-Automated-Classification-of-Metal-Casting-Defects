#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_final.py

Final version for the project.

Goals:
- MobileNetV3-Small feature extraction for casting defect classification
- Fair train metric (train_f1 computed in eval mode)
- Epoch-0 evaluation for full learning curves
- Minimal/standard/heavy augmentation presets
- Calibration metrics (Brier, ECE, MCE) + Expected Cost
"""

import argparse, math, time, json, random, sys, csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models, utils as vutils

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, roc_curve, auc, brier_score_loss,
    precision_score, recall_score, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score
)

from copy import deepcopy
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from tqdm import tqdm


console = Console()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# 1. Config & utilities
# --------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


@dataclass
class RunCfg:
    data: Path
    model_type: str = "mobilenet"

    epochs: int = 30
    batch_size: int = 64
    img_size: int = 224

    lr: float = 2e-4
    min_lr: float = 1e-6
    wd: float = 1e-4

    seed: int = 1337
    num_workers: int = 0
    patience: int = 5
    warmup_epochs: int = 5
    clip_norm: float = 1.0

    outdir: Path = Path("runs_final")
    log_interval: int = 10
    no_bars: bool = False
    min_delta: float = 0.0

    aug_level: str = "standard"
    dropout_rate: float = 0.5

    tune_for: str = "macro_f1"

    optimizer: str = "adamw"
    momentum: float = 0.9


# --------------------------------------------------------------------------
# 2. Model & data
# --------------------------------------------------------------------------

def build_mobilenet(num_classes: int = 2, dropout_rate: float = 0.5):
    m = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    for p in m.parameters():
        p.requires_grad = False

    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_f, num_classes),
    )
    for p in m.classifier.parameters():
        p.requires_grad = True
    return m


def build_model(cfg: RunCfg, num_classes: int):
    if cfg.model_type != "mobilenet":
        raise ValueError(
            f"Unsupported model_type: {cfg.model_type}. Only 'mobilenet' is supported."
        )
    console.print(
        f"[Model] Building MobileNetV3-Small "
        f"(feature extraction, dropout={cfg.dropout_rate})"
    )
    return build_mobilenet(num_classes=num_classes, dropout_rate=cfg.dropout_rate)


def build_dataloaders(cfg: RunCfg):
    plain_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    standard_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    heavy_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if cfg.aug_level == "minimal":
        tfms_train = plain_tfms
    elif cfg.aug_level == "standard":
        tfms_train = standard_tfms
    elif cfg.aug_level == "heavy":
        tfms_train = heavy_tfms
    else:
        console.print(
            f"[Warning] Unknown aug_level='{cfg.aug_level}', falling back to 'standard'.",
            style="yellow",
        )
        tfms_train = standard_tfms

    console.print(f"[Data] Train augmentation: [bold]{cfg.aug_level}[/bold]")

    ds_tr = datasets.ImageFolder(cfg.data / "train", transform=tfms_train)
    ds_va = datasets.ImageFolder(cfg.data / "val",   transform=plain_tfms)
    ds_te = datasets.ImageFolder(cfg.data / "test",  transform=plain_tfms)

    dl_tr = DataLoader(
        ds_tr, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0
    )
    dl_va = DataLoader(
        ds_va, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0
    )
    dl_te = DataLoader(
        ds_te, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0
    )
    return (ds_tr, ds_va, ds_te), (dl_tr, dl_va, dl_te)


# --------------------------------------------------------------------------
# 3. Metrics & plots
# --------------------------------------------------------------------------

def unnormalize_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, -1, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1, -1, 1, 1)
    return (x * std + mean).clamp(0, 1)


def calculate_ece(y_true, y_prob, n_bins: int = 10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[1:-1])

    ece = 0.0
    deltas = []

    for i in range(n_bins):
        in_bin = (bin_indices == i)
        n_in_bin = np.sum(in_bin)
        if n_in_bin > 0:
            avg_conf = np.mean(y_prob[in_bin])
            avg_acc  = np.mean(y_true[in_bin])
            delta = abs(avg_conf - avg_acc)
            ece += delta * (n_in_bin / len(y_true))
            deltas.append(delta)
        else:
            deltas.append(0.0)

    mce = float(np.max(deltas)) if deltas else 0.0
    return ece, mce


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_device: str,
    num_classes: int,
    cfg: RunCfg,
    bars_enabled: bool
) -> Dict[str, Any]:

    model.eval()
    total_loss = 0.0
    y_true, y_pred_argmax, y_prob = [], [], []

    pbar = tqdm(
        dataloader, desc="Eval", leave=False,
        dynamic_ncols=True, disable=not bars_enabled
    )
    with torch.no_grad():
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(amp_device, enabled=(amp_device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            probs = torch.softmax(logits, dim=1)

            y_true.extend(y.detach().cpu().tolist())
            y_pred_argmax.extend(torch.argmax(logits, 1).detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)

    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    probs_ok = y_prob_np[:, 1]

    brier = brier_score_loss(y_true_np, probs_ok)
    ece, mce = calculate_ece(y_true_np, probs_ok)

    f1_argmax = f1_score(y_true, y_pred_argmax, average="macro", zero_division=0)
    f1_best_thr = f1_argmax
    best_thr = 0.5

    if num_classes == 2:
        arr_p = probs_ok
        arr_y = y_true_np

        if cfg.tune_for == "def_front_f1":
            best_f1t = f1_score(arr_y, y_pred_argmax, pos_label=0, zero_division=0)
        else:
            best_f1t = f1_argmax

        for t in np.linspace(0.1, 0.9, 81):
            preds = (arr_p >= t).astype(int)
            if cfg.tune_for == "def_front_f1":
                metric_t = f1_score(arr_y, preds, pos_label=0, zero_division=0)
            else:
                metric_t = f1_score(arr_y, preds, average="macro", zero_division=0)

            if metric_t > best_f1t:
                best_f1t = metric_t
                best_thr = t

        f1_best_thr = best_f1t

    y_pred_thr = (probs_ok >= best_thr).astype(int)

    cm = confusion_matrix(y_true_np, y_pred_thr, labels=[0, 1])
    real_tn, real_fp, real_fn, real_tp = cm.ravel()

    TP = real_tn
    FP = real_fn
    TN = real_tp
    FN = real_fp

    pos_label = 0 if dataloader.dataset.classes[0] == "def_front" else 1

    precision = precision_score(y_true_np, y_pred_thr, pos_label=pos_label, zero_division=0)
    recall_tpr = recall_score(y_true_np, y_pred_thr, pos_label=pos_label, zero_division=0)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0

    acc_thr = accuracy_score(y_true_np, y_pred_thr)
    bal_acc_thr = balanced_accuracy_score(y_true_np, y_pred_thr)
    mcc_thr = matthews_corrcoef(y_true_np, y_pred_thr)
    f1_thr = f1_score(y_true_np, y_pred_thr, average="macro", zero_division=0)
    g_mean = math.sqrt(recall_tpr * specificity)
    auc_roc = roc_auc_score(y_true_np, probs_ok)
    auc_pr = average_precision_score(y_true_np, probs_ok)

    error_rate = 1.0 - accuracy_score(y_true_np, np.array(y_pred_argmax))
    expected_cost = error_rate

    return {
        "loss": avg_loss,
        "f1_argmax": f1_argmax,
        "f1_best_thr": f1_best_thr,
        "threshold": best_thr,

        "brier": brier,
        "ece": ece,
        "mce": mce,

        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Precision": precision, "Recall (TPR)": recall_tpr,
        "Specificity (TNR)": specificity, "FPR": fpr, "FNR": fnr, "NPV": npv,
        "Error Rate": error_rate,

        "Accuracy": acc_thr,
        "Balanced Acc": bal_acc_thr,
        "MCC": mcc_thr,
        "F1 Score": f1_thr,
        "G-mean": g_mean,
        "AUC PR": auc_pr,
        "AUC ROC": auc_roc,
        "Expected Cost": expected_cost,

        "y_true": y_true_np,
        "y_prob": y_prob_np
    }


def show_dataset_overview(ds_tr, ds_va, ds_te, class_names: List[str]):
    def count_by_class(ds):
        counts = [0] * len(class_names)
        for _, y in ds.samples:
            counts[y] += 1
        return counts

    counts_tr = count_by_class(ds_tr)
    counts_va = count_by_class(ds_va)
    counts_te = count_by_class(ds_te)

    table = Table(title="Dataset overview", box=box.SIMPLE_HEAVY)
    table.add_column("Split", justify="left")
    for c in class_names:
        table.add_column(c, justify="right")
    table.add_column("Total", justify="right")

    for name, counts in [("Train", counts_tr), ("Val", counts_va), ("Test", counts_te)]:
        table.add_row(name, *[str(c) for c in counts], str(sum(counts)))

    console.print(table)


def sample_grid(run_dir: Path, dl: DataLoader, title: str = "train_samples"):
    try:
        x, _ = next(iter(dl))
    except Exception as e:
        console.print(f"[Warning] Could not draw sample grid: {e}", style="yellow")
        return

    import matplotlib.pyplot as plt
    plt.switch_backend("agg")

    x = unnormalize_tensor(x)
    grid = vutils.make_grid(x[:32], nrow=8)
    npimg = grid.cpu().numpy().transpose(1, 2, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(npimg)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    out_path = run_dir / f"{title}.png"
    plt.savefig(out_path)
    plt.close()
    console.print(f"[Saved] Sample grid -> {out_path}")


def lr_schedule_lambda(epoch, total_epochs, warm, base_lr, min_lr):
    if epoch < warm:
        return (epoch + 1) / max(1, warm)
    t = (epoch - warm) / max(1, total_epochs - warm)
    cos_factor = 0.5 * (1 + math.cos(math.pi * t))
    return (min_lr / base_lr) + (1 - (min_lr / base_lr)) * cos_factor


def plot_lr_curve(run_dir: Path, cfg: RunCfg):
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")

    lrs = [
        lr_schedule_lambda(e, cfg.epochs, cfg.warmup_epochs, cfg.lr, cfg.min_lr) * cfg.lr
        for e in range(cfg.epochs)
    ]
    plt.figure(figsize=(6, 3))
    plt.plot(range(1, cfg.epochs + 1), lrs)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("LR schedule (warmup + cosine)")
    plt.tight_layout()
    plt.savefig(run_dir / "lr_schedule.png")
    plt.close()


def plot_learning_curves(run_dir: Path, hist: Dict[str, List[float]]):
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")

    epochs = range(0, len(hist["train_loss"]))

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, hist["train_loss"], label="train_loss (eval)")
    plt.plot(epochs, hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curves.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, hist["train_f1_argmax"], label="train_f1 (eval)")
    plt.plot(epochs, hist["val_f1_argmax"], label="val_f1 (argmax)")
    if "val_f1_best_thr" in hist:
        plt.plot(epochs, hist["val_f1_best_thr"], label="val_f1 (@best_thr)")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1 curves")
    plt.ylim(0.0, 1.01)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "f1_curves.png")
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(epochs, hist["train_loss"], linestyle="-", label="train_loss (eval)")
    ax1.plot(epochs, hist["val_loss"], linestyle="--", label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(epochs, hist["train_f1_argmax"], linestyle="-", label="train_f1 (eval)")
    ax2.plot(epochs, hist["val_f1_argmax"], linestyle="--", label="val_f1 (argmax)")
    if "val_f1_best_thr" in hist:
        ax2.plot(epochs, hist["val_f1_best_thr"], linestyle=":", label="val_f1 (@best_thr)")
    ax2.set_ylabel("F1")
    ax2.set_ylim(0.0, 1.01)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(run_dir / "learning_curves.png")
    plt.close(fig)


def plot_confusions(run_dir: Path, cm: np.ndarray, class_names: List[str]):
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    im0 = ax[0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax[0].set_title("Confusion matrix (counts)")
    fig.colorbar(im0, ax=ax[0], shrink=0.8)
    ax[0].set_xticks(range(len(class_names)))
    ax[0].set_xticklabels(class_names, rotation=45, ha="right")
    ax[0].set_yticks(range(len(class_names)))
    ax[0].set_yticklabels(class_names)
    for (i, j), v in np.ndenumerate(cm):
        ax[0].text(j, i, int(v), ha="center", va="center",
                   color="white" if v > cm.max() / 2 else "black")

    cmn = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    im1 = ax[1].imshow(cmn, vmin=0, vmax=1, cmap=plt.cm.Blues)
    ax[1].set_title("Confusion matrix (normalized)")
    fig.colorbar(im1, ax=ax[1], shrink=0.8)
    ax[1].set_xticks(range(len(class_names)))
    ax[1].set_xticklabels(class_names, rotation=45, ha="right")
    ax[1].set_yticks(range(len(class_names)))
    ax[1].set_yticklabels(class_names)
    for (i, j), v in np.ndenumerate(cmn):
        ax[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                   color="white" if v > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrices.png")
    plt.close()


def plot_pr_roc(run_dir: Path, y_true, y_prob, class_names: List[str]):
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")

    if len(class_names) != 2:
        console.print("[Plots] Skipping PR/ROC (non-binary problem).", style="yellow")
        return

    y = np.array(y_true)
    p1 = np.array(y_prob)[:, 1]

    prec, rec, _ = precision_recall_curve(y, p1)
    plt.figure(figsize=(6, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curve")
    plt.tight_layout()
    plt.savefig(run_dir / "pr_curve.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y, p1, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "roc_curve.png")
    plt.close()


def reliability_diagram(run_dir: Path, y_true, y_prob):
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")

    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)[:, 1]

    bins = np.linspace(0, 1, 11)
    idx = np.digitize(y_prob_np, bins) - 1

    accs, confs = [], []
    for b in range(10):
        m = (idx == b)
        confs.append((bins[b] + bins[b + 1]) / 2)
        accs.append(np.mean(y_true_np[m]) if m.sum() else 0.0)

    ece, mce = calculate_ece(y_true_np, y_prob_np, n_bins=10)

    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], "--", label="Ideal calibration")
    plt.bar(confs, np.array(accs), width=0.08, alpha=0.6, align="center", label="Model")
    plt.xlabel("Confidence for class 1")
    plt.ylabel("Empirical accuracy")
    plt.title(f"Reliability diagram (ECE≈{ece:.4f}, MCE≈{mce:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "reliability.png")
    plt.close()


def plot_top_errors(
    run_dir: Path,
    dl_te: DataLoader,
    test_results: Dict[str, Any],
    k: int = 24
):
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
    import math

    try:
        y_true = test_results["y_true"]
        y_prob = test_results["y_prob"]
        dataset = dl_te.dataset

        y_pred = np.argmax(y_prob, axis=1)
        error_mask = (y_true != y_pred)
        error_indices = np.where(error_mask)[0]

        if len(error_indices) == 0:
            console.print("[Info] No test errors to plot.", style="green")
            return

        error_confidences = y_prob[error_mask].max(axis=1)
        sorted_errors = sorted(
            zip(error_confidences, error_indices),
            key=lambda x: x[0],
            reverse=True
        )

        num_images = min(k, len(sorted_errors))
        top_k = sorted_errors[:num_images]

        images_unnorm = []
        for _, idx in top_k:
            img_tensor, _ = dataset[idx]
            img_unnorm = unnormalize_tensor(img_tensor.unsqueeze(0))[0]
            images_unnorm.append(img_unnorm)

        num_cols = 8
        grid = vutils.make_grid(images_unnorm, nrow=num_cols, padding=2)
        npimg = grid.cpu().numpy().transpose(1, 2, 0)

        num_rows = math.ceil(num_images / num_cols)
        plt.figure(figsize=(num_cols * 2.5, num_rows * 2.5))
        plt.imshow(npimg)
        plt.axis("off")
        plt.title(f"Top-{num_images} errors (highest confidence)", fontsize=16)
        plt.tight_layout()
        out_path = run_dir / "top-errors.png"
        plt.savefig(out_path)
        plt.close()
        console.print(f"[Saved] Top errors grid -> {out_path}")

    except Exception as e:
        console.print(f"[Error] Could not generate top-errors.png: {e}", style="red")


def save_history(run_dir: Path, hist: Dict[str, List[float]]):
    keys = list(hist.keys())
    if not keys:
        console.print("[Error] Empty history, nothing to save.", style="red")
        return

    num_epochs = len(hist[keys[0]])

    try:
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch"] + keys)
            for i in range(num_epochs):
                w.writerow([i] + [hist[k][i] for k in keys])

        with (run_dir / "history.jsonl").open("w", encoding="utf-8") as f:
            for i in range(num_epochs):
                entry = {"epoch": i, **{k: hist[k][i] for k in keys}}
                f.write(json.dumps(entry) + "\n")
    except Exception as e:
        console.print(f"[Error] Could not save history: {e}", style="red")


def generate_summary_table(
    train_results: Dict[str, Any],
    val_results: Dict[str, Any],
    test_results: Dict[str, Any],
    best_threshold: float,
    cfg: RunCfg
):
    console.print(Panel.fit(
        "Generating summary tables (train/val/test)",
        title="Final report", style="blue"
    ))

    tables_data = {}
    for split_name, results in [("Train", train_results), ("Valid", val_results), ("Test", test_results)]:
        tables_data[split_name] = {
            "TP": results["TP"], "FP": results["FP"], "TN": results["TN"], "FN": results["FN"],
            "Precision": results["Precision"], "Recall (TPR)": results["Recall (TPR)"],
            "Specificity (TNR)": results["Specificity (TNR)"],
            "FPR": results["FPR"], "FNR": results["FNR"], "NPV": results["NPV"],
            "Error Rate": results["Error Rate"],
            "Accuracy": results["Accuracy"], "Balanced Acc": results["Balanced Acc"],
            "MCC": results["MCC"], "F1 Score": results["F1 Score"], "G-mean": results["G-mean"],
            "AUC PR": results["AUC PR"], "AUC ROC": results["AUC ROC"],
            "Brier Score": results["brier"], "ECE": results["ece"],
            "MCE": results["mce"],
            "Expected Cost": results["Expected Cost"],
        }

    table1 = Table(
        title=f"Thresholded metrics (model={cfg.model_type}, thr={best_threshold:.4f})",
        box=box.ROUNDED
    )
    table1.add_column("Metric", style="cyan")
    table1.add_column("Train", justify="right")
    table1.add_column("Valid", justify="right")
    table1.add_column("Test",  justify="right")

    metrics_1 = [
        "TP", "FP", "TN", "FN", "Precision", "Recall (TPR)", "Specificity (TNR)",
        "FPR", "FNR", "NPV", "Error Rate"
    ]
    for m in metrics_1:
        is_int = m in ["TP", "FP", "TN", "FN"]
        def fmt(v):
            return str(int(v)) if is_int else f"{v:.4f}"
        table1.add_row(m, fmt(tables_data["Train"][m]), fmt(tables_data["Valid"][m]), fmt(tables_data["Test"][m]))

    table2 = Table(
        title=f"Aggregate & calibration metrics (model={cfg.model_type})",
        box=box.ROUNDED
    )
    table2.add_column("Metric", style="cyan")
    table2.add_column("Train", justify="right")
    table2.add_column("Valid", justify="right")
    table2.add_column("Test",  justify="right")

    metrics_2 = [
        "Accuracy", "Balanced Acc", "MCC", "F1 Score", "G-mean",
        "AUC PR", "AUC ROC", "Brier Score", "ECE", "MCE", "Expected Cost"
    ]
    for m in metrics_2:
        table2.add_row(
            m,
            f"{tables_data['Train'][m]:.4f}",
            f"{tables_data['Valid'][m]:.4f}",
            f"{tables_data['Test'][m]:.4f}",
        )

    console.print(table1)
    console.print(table2)


# --------------------------------------------------------------------------
# 4. Training loop
# --------------------------------------------------------------------------

def run_experiment(cfg: RunCfg, run_dir: Path) -> Dict[str, Any]:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(enabled=(amp_device == "cuda"))
    bars_enabled = (not cfg.no_bars) and sys.stdout.isatty()

    console.print(Panel.fit(
        f"Experiment start\n"
        f"- Model: {cfg.model_type}\n"
        f"- Optimizer: {cfg.optimizer}\n"
        f"- Aug: {cfg.aug_level}\n"
        f"- Tune for: {cfg.tune_for}\n"
        f"- Run dir: {run_dir}",
        title="New run", style="green"
    ))

    console.print(Panel.fit(
        f"Device: {device} "
        f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})\n"
        f"AMP: {'ON' if amp_device == 'cuda' else 'OFF'}",
        title="Hardware", style="cyan"
    ))

    (run_dir / "hparams.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8"
    )

    (ds_tr, ds_va, ds_te), (dl_tr, dl_va, dl_te) = build_dataloaders(cfg)
    class_names = ds_tr.classes
    num_classes = len(class_names)

    console.print(Panel.fit(f"Classes: {class_names}", title="Dataset", style="magenta"))
    show_dataset_overview(ds_tr, ds_va, ds_te, class_names)
    sample_grid(run_dir, dl_tr, title="train_samples")

    model = build_model(cfg, num_classes).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[Model] Trainable parameters: {trainable_params:,}")

    if cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.wd
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: lr_schedule_lambda(
            e, cfg.epochs, cfg.warmup_epochs, cfg.lr, cfg.min_lr
        )
    )
    plot_lr_curve(run_dir, cfg)

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1_argmax": [], "val_f1_argmax": [],
        "val_f1_best_thr": [], "lrs": [],
        "val_brier": [], "val_ece": [],
        "val_mce": [], "val_expected_cost": []
    }

    console.print("[Info] Epoch-0 evaluation...")
    epoch0 = evaluate(model, dl_va, criterion, device, amp_device, num_classes, cfg, bars_enabled)

    history["train_loss"].append(float("nan"))
    history["train_f1_argmax"].append(float("nan"))
    history["val_loss"].append(epoch0["loss"])
    history["val_f1_argmax"].append(epoch0["f1_argmax"])
    history["val_f1_best_thr"].append(epoch0["f1_best_thr"])
    history["val_brier"].append(epoch0["brier"])
    history["val_ece"].append(epoch0["ece"])
    history["val_mce"].append(epoch0["mce"])
    history["val_expected_cost"].append(epoch0["Expected Cost"])
    history["lrs"].append(optimizer.param_groups[0]["lr"])

    console.print(
        f"[Epoch 00] Val loss={epoch0['loss']:.4f}, "
        f"Val F1@thr={epoch0['f1_best_thr']:.4f}"
    )

    best_metric, best_state, best_epoch = -1.0, None, 0
    best_threshold = 0.5
    patience_left = cfg.patience

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss_acum = 0.0
        y_true_tr, y_pred_tr = [], []

        pbar = tqdm(
            dl_tr, desc=f"Epoch {epoch}/{cfg.epochs} [Train]",
            leave=False, dynamic_ncols=True, disable=not bars_enabled
        )

        for i, (x, y) in enumerate(pbar, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(amp_device, enabled=(amp_device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if cfg.clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.clip_norm
                )

            scaler.step(optimizer)
            scaler.update()

            tr_loss_acum += loss.item() * x.size(0)

            model.eval()
            with torch.no_grad():
                y_true_tr.extend(y.detach().cpu().tolist())
                with torch.amp.autocast(amp_device, enabled=(amp_device == "cuda")):
                    logits_eval = model(x)
                y_pred_tr.extend(torch.argmax(logits_eval, 1).detach().cpu().tolist())
            model.train()

            if bars_enabled and (i % cfg.log_interval == 0):
                pbar.set_postfix(loss=f"{(tr_loss_acum / (i * cfg.batch_size)):.4f}")

        train_loss = tr_loss_acum / len(dl_tr.dataset)
        train_f1 = f1_score(y_true_tr, y_pred_tr, average="macro", zero_division=0)

        val_res = evaluate(model, dl_va, criterion, device, amp_device, num_classes, cfg, bars_enabled)

        history["train_loss"].append(train_loss)
        history["train_f1_argmax"].append(train_f1)
        history["val_loss"].append(val_res["loss"])
        history["val_f1_argmax"].append(val_res["f1_argmax"])
        history["val_f1_best_thr"].append(val_res["f1_best_thr"])
        history["val_brier"].append(val_res["brier"])
        history["val_ece"].append(val_res["ece"])
        history["val_mce"].append(val_res["mce"])
        history["val_expected_cost"].append(val_res["Expected Cost"])
        history["lrs"].append(optimizer.param_groups[0]["lr"])

        console.print(
            f"[Epoch {epoch:02d}] "
            f"Train loss={train_loss:.4f}, Val loss={val_res['loss']:.4f} | "
            f"Train F1={train_f1:.4f} (eval), Val F1={val_res['f1_argmax']:.4f} (argmax) | "
            f"Val F1@thr={val_res['f1_best_thr']:.4f} (thr={val_res['threshold']:.2f})"
        )

        metric_to_check = val_res["f1_best_thr"]
        improved = metric_to_check > best_metric + cfg.min_delta

        if improved:
            best_metric = metric_to_check
            best_epoch = epoch
            best_threshold = val_res["threshold"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "best_model.pt")
            patience_left = cfg.patience
            console.print(f" -> [BEST] New best val F1={best_metric:.4f} at epoch {epoch}.")
        else:
            patience_left -= 1
            console.print(f" -> [ES] No improvement (patience {patience_left}/{cfg.patience}).")

        scheduler.step()

        if patience_left <= 0:
            console.print(f"[EarlyStopping] Stopping at epoch {epoch}.")
            break

    console.print(
        f"\n[Test] Loading best model from epoch {best_epoch} "
        f"(best val F1={best_metric:.4f})"
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        console.print("[Warning] best_state missing, using last weights.", style="yellow")

    model.to(device)
    model.eval()

    console.print(f"[Test] Using threshold={best_threshold:.4f} tuned on val ({cfg.tune_for}).")

    test_res = evaluate(model, dl_te, criterion, device, amp_device, num_classes, cfg, bars_enabled)
    y_true_te = test_res["y_true"]
    y_prob_te = test_res["y_prob"]
    y_pred_te = (y_prob_te[:, 1] >= best_threshold).astype(int)

    rep = classification_report(
        y_true_te, y_pred_te, target_names=class_names,
        digits=4, zero_division=0
    )

    console.print(Panel.fit(
        f"Test F1 (macro) @ thr={best_threshold:.2f}: {test_res['F1 Score']:.4f}\n"
        f"Brier: {test_res['brier']:.4f} | ECE: {test_res['ece']:.4f} | "
        f"MCE: {test_res['mce']:.4f} | Exp. Cost: {test_res['Expected Cost']:.4f}\n\n"
        + rep,
        title="Test report", style="green"
    ))

    console.print("[Info] Computing summary tables for all splits...")
    train_res = evaluate(model, dl_tr, criterion, device, amp_device, num_classes, cfg, bars_enabled)
    val_res2  = evaluate(model, dl_va, criterion, device, amp_device, num_classes, cfg, bars_enabled)

    generate_summary_table(
        train_results=train_res,
        val_results=val_res2,
        test_results=test_res,
        best_threshold=best_threshold,
        cfg=cfg
    )

    cm = confusion_matrix(y_true_te, y_pred_te)
    plot_confusions(run_dir, cm, class_names)
    plot_pr_roc(run_dir, y_true_te, y_prob_te, class_names)
    reliability_diagram(run_dir, y_true_te, y_prob_te)
    plot_top_errors(run_dir, dl_te, test_res, k=24)
    plot_learning_curves(run_dir, history)
    save_history(run_dir, history)

    console.print(Panel.fit(
        f"Run finished.\nArtifacts saved in:\n{run_dir}",
        title="Done", style="blue"
    ))

    return {
        "model": cfg.model_type,
        "aug": cfg.aug_level,
        "dropout": cfg.dropout_rate,
        "params": trainable_params,
        "best_epoch": best_epoch,
        "best_val_f1": best_metric,
        "test_f1": test_res["F1 Score"],
        "test_brier": test_res["brier"],
        "test_ece": test_res["ece"],
        "test_mce": test_res["mce"],
        "test_expected_cost": test_res["Expected Cost"],
    }


# --------------------------------------------------------------------------
# 5. CLI
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Realistic casting defect classification (MobileNetV3-Small, feature extraction)."
    )
    ap.add_argument(
        "--data", type=Path, default=Path("casting_data_fixed4"),
        help="Dataset root (train/val/test)."
    )
    ap.add_argument(
        "--outdir", type=Path, default="runs_final_v4",
        help="Output directory for artifacts."
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument(
        "--aug-level", type=str, default="standard",
        choices=["minimal", "standard", "heavy"]
    )
    ap.add_argument(
        "--model", type=str, default="mobilenet",
        choices=["mobilenet"]
    )
    ap.add_argument(
        "--tune-for", type=str, default="macro_f1",
        choices=["macro_f1", "def_front_f1"]
    )
    ap.add_argument("--dropout-rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-bars", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    cfg = RunCfg(
        data=args.data,
        outdir=args.outdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        aug_level=args.aug_level,
        dropout_rate=args.dropout_rate,
        seed=args.seed,
        num_workers=args.num_workers,
        no_bars=args.no_bars,
        tune_for=args.tune_for,
        model_type=args.model,
        optimizer="adamw",
        warmup_epochs=5,
    )

    current_time = time.strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{cfg.model_type}_{cfg.optimizer}_{cfg.aug_level}_dr{cfg.dropout_rate}_{current_time}"
    run_dir = cfg.outdir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        f"Launching experiment\n"
        f"- Model: {cfg.model_type}\n"
        f"- Aug: {cfg.aug_level}\n"
        f"- LR: {cfg.lr}\n"
        f"- Output: {run_dir}",
        title="Run configuration",
        style="bold green"
    ))

    try:
        res = run_experiment(cfg, run_dir)

        summary = Table(title="Final run summary (test metrics)", box=box.ROUNDED)
        summary.add_column("Model", style="cyan")
        summary.add_column("Augmentation", style="magenta")
        summary.add_column("Trainable params", justify="right")
        summary.add_column("Best epoch (val)", justify="right")
        summary.add_column("Best Val F1", style="yellow")
        summary.add_column("Test F1", style="bold green")
        summary.add_column("Test ECE", style="red")
        summary.add_column("Test MCE", style="red")
        summary.add_column("Test Brier", style="red")
        summary.add_column("Test Exp. Cost", style="red")

        summary.add_row(
            res["model"],
            res["aug"],
            f"{res['params']:,}",
            str(res["best_epoch"]),
            f"{res['best_val_f1']:.4f}",
            f"{res['test_f1']:.4f}",
            f"{res['test_ece']:.4f}",
            f"{res['test_mce']:.4f}",
            f"{res['test_brier']:.4f}",
            f"{res['test_expected_cost']:.4f}",
        )

        console.print(summary)

    except Exception as e:
        console.print(f"[CRITICAL ERROR] Run failed: {e}", style="bold red")
        console.print_exception()

    console.print(f"\n[Done] Artifacts stored under: [bold]{cfg.outdir}/[/]")


if __name__ == "__main__":
    main()
