"""Live demo: bin-inventory reconciliation.

Single image:
    python src/demo.py --checkpoint checkpoints/resnet34_adam.pt --image bin.jpg --recorded 3

Batch demo on random test-split bins (records are SIMULATED: 70% correct, 30% off by +/-1):
    python src/demo.py --checkpoint checkpoints/resnet34_adam.pt --data data --sample 12
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import ROOT, load_split
from train_cnn import build_model, eval_transform, get_device


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    m = build_model(ck["arch"], pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


@torch.no_grad()
def predict_image(model, path, device, size=224):
    x = eval_transform(size)(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    p = (model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)) / 2
    return p[0].cpu().numpy()


def verdict(pred, recorded):
    if recorded is None:
        return "no recorded count supplied"
    return "OK - matches record" if pred == recorded else \
        f"VARIANCE - record says {recorded}, photo suggests {pred}: flag for recount"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image")
    ap.add_argument("--recorded", type=int)
    ap.add_argument("--data", default="data")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    device = get_device()
    model = load_model(a.checkpoint, device)

    if a.image:
        p = predict_image(model, a.image, device)
        pred = int(p.argmax())
        print(f"Predicted items in bin: {pred} (confidence {p[pred]:.0%})")
        print("Class probabilities:", {i: round(float(v), 3) for i, v in enumerate(p)})
        print("Reconciliation:", verdict(pred, a.recorded))
        return

    if a.sample:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rnd = random.Random(a.seed)
        paths, labels = load_split(a.data, "test")
        idx = rnd.sample(range(len(paths)), min(a.sample, len(paths)))
        cols = 4
        rows = int(np.ceil(len(idx) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
        axes = np.atleast_1d(axes).ravel()
        flagged = 0
        for ax, i in zip(axes, idx):
            true = int(labels[i])
            recorded = true if rnd.random() < 0.7 else max(0, min(5, true + rnd.choice([-1, 1])))
            p = predict_image(model, paths[i], device)
            pred = int(p.argmax())
            v = verdict(pred, recorded)
            flagged += v.startswith("VARIANCE")
            print(f"{Path(paths[i]).name:>12} true={true} recorded={recorded} pred={pred}  {v}")
            ax.imshow(Image.open(paths[i]))
            ax.set_title(f"true {true} | rec {recorded} | pred {pred}", fontsize=9,
                         color="red" if v.startswith("VARIANCE") else "green")
            ax.axis("off")
        for ax in axes[len(idx):]:
            ax.axis("off")
        fig.tight_layout()
        out = ROOT / "results" / "demo_grid.png"
        fig.savefig(out, dpi=130)
        print(f"\n{flagged}/{len(idx)} bins flagged for recount. Grid saved to {out}")
        return
    ap.error("provide --image or --sample N")


if __name__ == "__main__":
    main()
