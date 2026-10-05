"""Classical baselines: raw pixels -> StandardScaler -> PCA -> {Logistic Regression, RBF-SVM}.

Hyper-parameters are chosen on the validation split; the test split is touched once.

    python src/baseline_svm.py --data data --size 32 --pca 200
"""
import argparse
import time

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import compute_metrics, load_split, save_result


def load_features(paths, size):
    return np.stack([
        np.asarray(Image.open(p).convert("RGB").resize((size, size)), dtype=np.float32).ravel() / 255.0
        for p in paths])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--size", type=int, default=32, help="downsample side (pixels)")
    ap.add_argument("--pca", type=int, default=200)
    ap.add_argument("--max-train", type=int, default=8000, help="cap train size for SVC speed")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    tr_p, ytr = load_split(a.data, "train")
    va_p, yva = load_split(a.data, "val")
    te_p, yte = load_split(a.data, "test")
    if len(tr_p) > a.max_train:
        idx = rng.choice(len(tr_p), a.max_train, replace=False)
        tr_p, ytr = [tr_p[i] for i in idx], ytr[idx]

    t0 = time.time()
    Xtr, Xva, Xte = (load_features(p, a.size) for p in (tr_p, va_p, te_p))
    scaler = StandardScaler().fit(Xtr)
    n_comp = min(a.pca, Xtr.shape[0] - 1, Xtr.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=a.seed)
    Ztr = pca.fit_transform(scaler.transform(Xtr))
    Zva, Zte = (pca.transform(scaler.transform(X)) for X in (Xva, Xte))
    print(f"features ready in {time.time() - t0:.0f}s, PCA dims={n_comp}, "
          f"explained var={pca.explained_variance_ratio_.sum():.2f}")

    # ---- Logistic regression -------------------------------------------------
    best = None
    for C in (0.01, 0.1, 1.0):
        m = LogisticRegression(C=C, max_iter=500).fit(Ztr, ytr)
        acc = (m.predict(Zva) == yva).mean()
        if best is None or acc > best[0]:
            best = (acc, C, m)
    val_acc, C, model = best
    res = compute_metrics(yte, model.predict(Zte))
    res.update(model="Logistic Regression + PCA", val_accuracy=float(val_acc),
               test_accuracy=res.pop("accuracy"), test_rmse=res.pop("rmse"),
               params={"C": C, "pca": n_comp, "size": a.size})
    save_result("logreg_pca", res, a.results_dir)
    print(f"LogReg  val={val_acc:.3f} test={res['test_accuracy']:.3f}")

    # ---- RBF SVM -------------------------------------------------------------
    best = None
    for C in (1, 10, 100):
        for gamma in ("scale", 1e-3, 1e-4):
            m = SVC(kernel="rbf", C=C, gamma=gamma).fit(Ztr, ytr)
            acc = (m.predict(Zva) == yva).mean()
            print(f"  SVC C={C} gamma={gamma} val={acc:.3f}")
            if best is None or acc > best[0]:
                best = (acc, C, gamma, m)
    val_acc, C, gamma, model = best
    res = compute_metrics(yte, model.predict(Zte))
    res.update(model="SVM (RBF) + PCA", val_accuracy=float(val_acc),
               test_accuracy=res.pop("accuracy"), test_rmse=res.pop("rmse"),
               params={"C": C, "gamma": str(gamma), "pca": n_comp, "size": a.size})
    save_result("svm_rbf_pca", res, a.results_dir)
    print(f"SVM     val={val_acc:.3f} test={res['test_accuracy']:.3f}")


if __name__ == "__main__":
    main()
