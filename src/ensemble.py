"""
Soft Voting Ensemble for the Amazon Bin Image classification project.

Combines:
1. Logistic Regression
2. RBF-SVM
3. Gaussian Naive Bayes

All three models use the same RGB -> StandardScaler -> PCA
feature pipeline used by the classical baselines.
"""

import time

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import ROOT, compute_metrics, load_split, save_result


def load_features(paths, size=32):
    """Load images as flattened RGB features."""
    X = []

    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size))
        X.append(
            np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
        )

    return np.asarray(X, dtype=np.float32)


def main():
    t0 = time.time()

    data_dir = ROOT / "data"

    # Load the existing train/validation/test splits.
    train_paths, y_train = load_split(data_dir, "train")
    val_paths, y_val = load_split(data_dir, "val")
    test_paths, y_test = load_split(data_dir, "test")

    # Same image feature pipeline as the classical models.
    X_train = load_features(train_paths)
    X_val = load_features(val_paths)
    X_test = load_features(test_paths)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # PCA feature reduction.
    pca = PCA(
        n_components=200,
        svd_solver="randomized",
        random_state=42
    )

    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    print(
        f"features ready in {time.time() - t0:.0f}s, "
        f"PCA dims={X_train.shape[1]}, "
        f"explained var={pca.explained_variance_ratio_.sum():.2f}"
    )

    # ---------------------------------------------------------
    # Component 1: Logistic Regression
    # ---------------------------------------------------------
    logreg = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=42
    )

    # ---------------------------------------------------------
    # Component 2: RBF-SVM
    # probability=True is required for soft voting.
    # ---------------------------------------------------------
    svm = SVC(
        C=100,
        gamma=1e-4,
        kernel="rbf",
        probability=True,
        random_state=42
    )

    # ---------------------------------------------------------
    # Component 3: Gaussian Naive Bayes
    # ---------------------------------------------------------
    gnb = GaussianNB()

    print("Training Logistic Regression...")
    logreg.fit(X_train, y_train)

    print("Training RBF-SVM...")
    svm.fit(X_train, y_train)

    print("Training Gaussian Naive Bayes...")
    gnb.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Soft Voting on validation set
    # Equal weight: 1/3 for each model.
    # ---------------------------------------------------------
    val_probs = (
        logreg.predict_proba(X_val)
        + svm.predict_proba(X_val)
        + gnb.predict_proba(X_val)
    ) / 3.0

    val_pred = val_probs.argmax(axis=1)

    val_metrics = compute_metrics(
        y_val,
        val_pred
    )

    # ---------------------------------------------------------
    # Soft Voting on test set
    # ---------------------------------------------------------
    test_probs = (
        logreg.predict_proba(X_test)
        + svm.predict_proba(X_test)
        + gnb.predict_proba(X_test)
    ) / 3.0

    test_pred = test_probs.argmax(axis=1)

    test_metrics = compute_metrics(
        y_test,
        test_pred
    )

    print(
        f"Soft Voting Ensemble "
        f"val={val_metrics['accuracy']:.3f} "
        f"test={test_metrics['accuracy']:.3f} "
        f"rmse={test_metrics['rmse']:.3f}"
    )

    # ---------------------------------------------------------
    # Save result
    # ---------------------------------------------------------
    result = {
        "model": "Soft Voting Ensemble",
        "val_accuracy": float(val_metrics["accuracy"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_rmse": float(test_metrics["rmse"]),
        "params": {
            "components": [
                "Logistic Regression",
                "RBF-SVM",
                "Gaussian Naive Bayes"
            ],
            "pca_components": 200,
            "voting": "soft",
            "weights": [
                1 / 3,
                1 / 3,
                1 / 3
            ],
            "svm_C": 100,
            "svm_gamma": 1e-4
        }
    }

    save_result(
        "ensemble_soft_voting",
        result
    )

    print(
        f"total time={time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()