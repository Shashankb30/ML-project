"""Shared helpers: split creation/loading, metrics, plotting, result IO."""
import csv
import json
from pathlib import Path

import numpy as np

NUM_CLASSES = 6  # bins holding 0..5 items
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def make_splits(data_dir, seed=42, val=0.2, test=0.1):
    """Create a stratified 70/20/10 train/val/test split -> data_dir/splits.csv."""
    from sklearn.model_selection import train_test_split

    data_dir = Path(data_dir)
    with open(data_dir / "labels.csv") as f:
        rows = list(csv.DictReader(f))
    ids = [r["id"] for r in rows]
    y = [int(r["quantity"]) for r in rows]
    tr, rest, _, y_rest = train_test_split(
        ids, y, test_size=val + test, stratify=y, random_state=seed)
    va, te = train_test_split(
        rest, test_size=test / (val + test), stratify=y_rest, random_state=seed)
    label = dict(zip(ids, y))
    with open(data_dir / "splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "quantity", "split"])
        for name, group in (("train", tr), ("val", va), ("test", te)):
            for i in group:
                w.writerow([i, label[i], name])
    return {"train": len(tr), "val": len(va), "test": len(te)}


def load_split(data_dir, split):
    """Return (image_paths, labels) for split in {'train','val','test'}."""
    data_dir = Path(data_dir)
    paths, labels = [], []
    with open(data_dir / "splits.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                paths.append(str(data_dir / "images" / f"{r['id']}.jpg"))
                labels.append(int(r["quantity"]))
    return paths, np.array(labels)


def compute_metrics(y_true, y_pred):
    """Accuracy, RMSE of predicted count, per-class accuracy, confusion matrix."""
    from sklearn.metrics import confusion_matrix

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    per_class = [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0
                 for i in range(NUM_CLASSES)]
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
    }


def save_confusion(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted count")
    ax.set_ylabel("True count")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_result(name, payload, results_dir=None):
    """Write results/<name>.json (and a confusion-matrix PNG if present)."""
    results_dir = Path(results_dir or RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{name}.json", "w") as f:
        json.dump(payload, f, indent=2)
    if "confusion_matrix" in payload:
        save_confusion(payload["confusion_matrix"], results_dir / f"{name}_confusion.png",
                       f"{payload.get('model', name)} (test)")
