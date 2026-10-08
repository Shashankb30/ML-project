"""Probabilistic baseline: raw pixels -> StandardScaler -> PCA -> Gaussian Naive Bayes."""

import argparse
import time

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from common import compute_metrics, load_split, save_result


def load_features(paths, size):
    return np.stack([
        np.asarray(
            Image.open(p).convert("RGB").resize((size, size)),
            dtype=np.float32
        ).ravel() / 255.0
        for p in paths
    ])


def main(argv=None):

    ap = argparse.ArgumentParser()

    ap.add_argument("--data", default="data")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--pca", type=int, default=200)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)

    a = ap.parse_args(argv)

    # ---------------------------------------------------------
    # Load the SAME train / validation / test splits
    # ---------------------------------------------------------

    tr_p, ytr = load_split(a.data, "train")
    va_p, yva = load_split(a.data, "val")
    te_p, yte = load_split(a.data, "test")

    # ---------------------------------------------------------
    # Create the SAME image features as baseline_svm.py
    # ---------------------------------------------------------

    t0 = time.time()

    Xtr = load_features(tr_p, a.size)
    Xva = load_features(va_p, a.size)
    Xte = load_features(te_p, a.size)

    # ---------------------------------------------------------
    # SAME StandardScaler
    # ---------------------------------------------------------

    scaler = StandardScaler().fit(Xtr)

    # ---------------------------------------------------------
    # SAME PCA
    # ---------------------------------------------------------

    n_comp = min(
        a.pca,
        Xtr.shape[0] - 1,
        Xtr.shape[1]
    )

    pca = PCA(
        n_components=n_comp,
        svd_solver="randomized",
        random_state=a.seed
    )

    Ztr = pca.fit_transform(
        scaler.transform(Xtr)
    )

    Zva = pca.transform(
        scaler.transform(Xva)
    )

    Zte = pca.transform(
        scaler.transform(Xte)
    )

    print(
        f"features ready in {time.time() - t0:.0f}s, "
        f"PCA dims={n_comp}, "
        f"explained var={pca.explained_variance_ratio_.sum():.2f}"
    )

    # ---------------------------------------------------------
    # Gaussian Naive Bayes
    # ---------------------------------------------------------

    print("Training Gaussian Naive Bayes...")

    model = GaussianNB()

    model.fit(Ztr, ytr)

    # Validation
    val_pred = model.predict(Zva)
    val_acc = (val_pred == yva).mean()

    # Test
    test_pred = model.predict(Zte)

    res = compute_metrics(
        yte,
        test_pred
    )

    res.update(
        model="Gaussian Naive Bayes + PCA",
        val_accuracy=float(val_acc),
        test_accuracy=res.pop("accuracy"),
        test_rmse=res.pop("rmse"),
        params={
            "pca": n_comp,
            "size": a.size
        }
    )

    save_result(
        "gaussian_nb_pca",
        res,
        a.results_dir
    )

    print(
        f"Gaussian NB  "
        f"val={val_acc:.3f} "
        f"test={res['test_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()