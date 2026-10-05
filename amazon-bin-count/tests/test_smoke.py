"""Offline smoke tests on synthetic data (NOT real results)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import make_synthetic  # noqa: E402
import baseline_svm  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    make_synthetic.main(["--n", "240", "--out", str(d), "--size", "64"])
    return d


def test_baseline_runs(synth, tmp_path):
    baseline_svm.main(["--data", str(synth), "--size", "16", "--pca", "30",
                       "--results-dir", str(tmp_path)])
    res = json.load(open(tmp_path / "svm_rbf_pca.json"))
    assert 0.0 <= res["test_accuracy"] <= 1.0
    assert len(res["confusion_matrix"]) == 6


def test_cnn_one_epoch(synth, tmp_path):
    pytest.importorskip("torch")
    import train_cnn
    train_cnn.main(["--data", str(synth), "--arch", "resnet18", "--no-pretrained",
                    "--head-epochs", "1", "--epochs", "1", "--img", "64", "--batch", "16",
                    "--workers", "0", "--name", "smoke", "--results-dir", str(tmp_path)])
    assert (tmp_path / "cnn_smoke.json").exists()
