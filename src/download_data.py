"""Download a subset of the Amazon Bin Image Dataset (public S3 bucket).

Keeps only bins with <= --max-qty items, resizes images to --size x --size,
writes data/images/<id>.jpg, data/labels.csv and a stratified data/splits.csv.

    python src/download_data.py --n 20000 --out data
"""
import argparse
import csv
import io
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

from common import make_splits

BASE = "https://aft-vbi-pds.s3.amazonaws.com"
TOTAL_IMAGES = 535234


def http_get(url, tries=3):
    for t in range(tries):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                return r.content
            if r.status_code in (403, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(1 + t)
    return None


def fetch_one(i, img_dir, max_qty, size):
    meta = http_get(f"{BASE}/metadata/{i}.json")
    if meta is None:
        return None
    qty = json.loads(meta).get("EXPECTED_QUANTITY")
    if qty is None or qty > max_qty:
        return None
    raw = http_get(f"{BASE}/bin-images/{i}.jpg")
    if raw is None:
        return None
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((size, size), Image.BILINEAR)
    img.save(img_dir / f"{i}.jpg", quality=90)
    return i, qty


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="images to keep")
    ap.add_argument("--max-qty", type=int, default=5)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    out = Path(a.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ids = list(range(1, TOTAL_IMAGES + 1))
    random.Random(a.seed).shuffle(ids)

    kept, pos = [], 0
    with ThreadPoolExecutor(a.workers) as ex:
        while len(kept) < a.n and pos < len(ids):
            batch = ids[pos:pos + 500]
            pos += 500
            kept += [r for r in ex.map(
                lambda i: fetch_one(i, img_dir, a.max_qty, a.size), batch) if r]
            print(f"scanned {pos} bins, kept {len(kept)}/{a.n}", flush=True)
    kept = kept[:a.n]

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "quantity"])
        w.writerows(kept)
    print("splits:", make_splits(out, seed=a.seed))


if __name__ == "__main__":
    main()
