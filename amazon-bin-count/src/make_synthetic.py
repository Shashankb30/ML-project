"""Generate a tiny synthetic 'bin' dataset (k bright boxes on a dark bin, k=0..5).

Used ONLY for smoke-testing the pipeline offline. Never report these numbers.

    python src/make_synthetic.py --n 600 --out data_synth
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from common import make_splits


def render(k, rng, size=224):
    img = rng.normal(40, 10, (size, size, 3)).clip(0, 255)
    for _ in range(k):
        w, h = rng.integers(30, 55, 2)
        x, y = rng.integers(0, size - w), rng.integers(0, size - h)
        img[y:y + h, x:x + w] = rng.normal(190, 15, 3)
    return Image.fromarray(img.astype(np.uint8))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(a.n):
        k = int(i % 6)
        render(k, rng, a.size).save(out / "images" / f"{i}.jpg", quality=92)
        rows.append((i, k))
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "quantity"])
        w.writerows(rows)
    make_splits(out, seed=a.seed)
    return out


if __name__ == "__main__":
    main()
