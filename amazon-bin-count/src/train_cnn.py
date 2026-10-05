"""Transfer-learning CNN (ResNet18/34/50) that classifies a bin photo into 0..5 items.

Phase 1: train only the new fully-connected head (backbone frozen).
Phase 2: unfreeze everything with differential learning rates
         (early layers lr/9, later layers lr/3, head lr) and cosine annealing.
Augmentation: random horizontal flip (+ mild colour jitter). Optional flip TTA at test time.

    python src/train_cnn.py --data data --arch resnet34 --optim adam --lr 1e-3 --epochs 12
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

from common import NUM_CLASSES, ROOT, compute_metrics, load_split, save_result

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def eval_transform(size):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


def train_transform(size):
    return T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.2, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])


class BinDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), int(self.labels[i])


def build_model(arch, pretrained=True, num_classes=NUM_CLASSES):
    ctor = getattr(models, arch)
    weights = models.get_model_weights(arch).DEFAULT if pretrained else None
    m = ctor(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def param_groups(m, lr):
    early = [p for mod in (m.conv1, m.bn1, m.layer1, m.layer2) for p in mod.parameters()]
    late = [p for mod in (m.layer3, m.layer4) for p in mod.parameters()]
    return [{"params": early, "lr": lr / 9}, {"params": late, "lr": lr / 3},
            {"params": m.fc.parameters(), "lr": lr}]


def make_optimizer(name, groups, wd):
    if name == "adam":
        return torch.optim.Adam(groups, weight_decay=wd)
    return torch.optim.SGD(groups, momentum=0.9, weight_decay=wd)


@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device)
        p = model(x).softmax(1)
        if tta:
            p = (p + model(torch.flip(x, dims=[3])).softmax(1)) / 2
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def run_epochs(model, opt, sched, epochs, loaders, device, history, ckpt, best, arch, phase):
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        sched.step()
        probs, yv = predict(model, loaders["val"], device)
        pred = probs.argmax(1)
        vacc = float((pred == yv).mean())
        vrmse = float(np.sqrt(np.mean((pred - yv) ** 2)))
        history.append({"phase": phase, "train_loss": tot / n, "train_acc": correct / n,
                        "val_acc": vacc, "val_rmse": vrmse})
        print(f"[{phase}] epoch {ep + 1}/{epochs} loss={tot / n:.3f} "
              f"train_acc={correct / n:.3f} val_acc={vacc:.3f} val_rmse={vrmse:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if vacc > best[0]:
            best[0] = vacc
            torch.save({"arch": arch, "state_dict": model.state_dict()}, ckpt)


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot([h["train_loss"] for h in history])
    ax[0].set_title("Training loss")
    ax[1].plot([h["train_acc"] for h in history], label="train")
    ax[1].plot([h["val_acc"] for h in history], label="val")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    for a in ax:
        a.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--arch", default="resnet34", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--optim", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--head-epochs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10, help="full-network epochs")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = get_device()
    name = a.name or f"{a.arch}_{a.optim}"
    print(f"device={device} run={name}")

    tr_p, ytr = load_split(a.data, "train")
    va_p, yva = load_split(a.data, "val")
    te_p, yte = load_split(a.data, "test")
    mk = lambda p, y, tf, sh: DataLoader(BinDataset(p, y, tf), batch_size=a.batch, shuffle=sh,
                                         num_workers=a.workers)
    loaders = {"train": mk(tr_p, ytr, train_transform(a.img), True),
               "val": mk(va_p, yva, eval_transform(a.img), False),
               "test": mk(te_p, yte, eval_transform(a.img), False)}

    model = build_model(a.arch, pretrained=not a.no_pretrained).to(device)
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt = ckpt_dir / f"{name}.pt"
    history, best = [], [-1.0]

    # Phase 1: head only
    if a.head_epochs > 0:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        opt = make_optimizer(a.optim, [{"params": model.fc.parameters(), "lr": a.lr}], a.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.head_epochs)
        run_epochs(model, opt, sched, a.head_epochs, loaders, device, history, ckpt, best,
                   a.arch, "head")

    # Phase 2: whole network, differential LRs + cosine annealing
    for p in model.parameters():
        p.requires_grad = True
    opt = make_optimizer(a.optim, param_groups(model, a.lr), a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(a.epochs, 1))
    run_epochs(model, opt, sched, a.epochs, loaders, device, history, ckpt, best, a.arch, "full")

    # Final test evaluation with the best-validation checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    probs, yt = predict(model, loaders["test"], device)
    probs_tta, _ = predict(model, loaders["test"], device, tta=True)
    res = compute_metrics(yt, probs.argmax(1))
    tta = compute_metrics(yt, probs_tta.argmax(1))
    res.update(model=f"{a.arch} ({a.optim})", val_accuracy=best[0],
               test_accuracy=res.pop("accuracy"), test_rmse=res.pop("rmse"),
               test_accuracy_tta=tta["accuracy"], test_rmse_tta=tta["rmse"],
               params={k: v for k, v in vars(a).items() if k not in ("results_dir",)},
               history=history)
    save_result(f"cnn_{name}", res, a.results_dir)
    out_dir = Path(a.results_dir) if a.results_dir else ROOT / "results"
    plot_history(history, out_dir / f"cnn_{name}_curves.png")
    print(f"TEST acc={res['test_accuracy']:.3f} rmse={res['test_rmse']:.3f} "
          f"| with flip-TTA acc={tta['accuracy']:.3f}")


if __name__ == "__main__":
    main()
