# Amazon Bin Inventory Reconciliation — Counting Items in a Bin from a Photo

**Course:** UE24CS352A – Machine Learning (Mini-Project) · **Problem #65**
**Team:** _<Name 1 – SRN>_, _<Name 2 – SRN>_ · **Section:** _<section>_

Amazon fulfilment-centre bins sometimes hold a different number of items than the system
records. Given only a photo of a bin, we predict how many items (0–5) it contains. If the
prediction disagrees with the recorded count, the bin is flagged for a recount.

We compare a classical pipeline (raw pixels → PCA → Logistic Regression / RBF-SVM) with
transfer-learning CNNs (ResNet-18/34/50 pre-trained on ImageNet).

## Repository layout

```
src/
  common.py          splits, metrics, confusion-matrix plots, result IO
  download_data.py   downloads a subset of the Amazon Bin Image Dataset (public S3)
  baseline_svm.py    PCA + Logistic Regression + RBF-SVM baselines
  train_cnn.py       ResNet transfer learning (head-only phase, then full fine-tune)
  demo.py            live demo: predict count + flag inventory variance
  make_synthetic.py  tiny synthetic data, ONLY for offline smoke tests
tests/test_smoke.py  end-to-end smoke tests on synthetic data
report/              build_report.py (write-up PDF), build_deck.js (slides)
results/             metrics JSONs, confusion matrices, training curves (generated)
```

## 1. Setup

Python 3.10+ is required. A GPU is strongly recommended for the CNN (Google Colab/Kaggle is fine).

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the data

The [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/) is public
(no AWS account needed). This downloads a random subset, keeps bins with ≤ 5 items,
resizes to 224×224, and writes a stratified 70/20/10 train/val/test split:

```bash
python src/download_data.py --n 20000 --out data
```

Outputs `data/images/*.jpg`, `data/labels.csv`, `data/splits.csv`. Use a smaller `--n`
(e.g. 5000) if you are short on time or compute.

## 3. Train and evaluate

```bash
# Classical baselines (CPU, minutes)
python src/baseline_svm.py --data data

# CNNs (GPU recommended). Each run writes results/cnn_<name>.json and checkpoints/<name>.pt
python src/train_cnn.py --data data --arch resnet18 --optim sgd  --lr 0.01  --epochs 10
python src/train_cnn.py --data data --arch resnet34 --optim sgd  --lr 0.01  --epochs 10
python src/train_cnn.py --data data --arch resnet34 --optim adam --lr 0.001 --epochs 10
```

Useful flags: `--img 128` (faster on CPU), `--head-epochs`, `--batch`, `--wd`, `--no-pretrained`.

## 4. Live demo

```bash
# one photo, with the count the system *thinks* is in the bin
python src/demo.py --checkpoint checkpoints/resnet34_adam.pt --image path/to/bin.jpg --recorded 3

# batch: random test bins with simulated inventory records (30% deliberately wrong)
python src/demo.py --checkpoint checkpoints/resnet34_adam.pt --data data --sample 12
```

The batch mode saves `results/demo_grid.png` (green = records agree, red = flagged for recount).

## 5. Tests

```bash
pytest -q        # builds a tiny synthetic dataset; the CNN test needs torch
```

## 6. Build the write-up and slides from your real results

```bash
python report/build_report.py --team "Name1 (SRN1), Name2 (SRN2)" --repo https://github.com/<user>/<repo>
cd report && npm install pptxgenjs && node build_deck.js     # produces report/slides.pptx
```

Both read `results/*.json`, so the tables always match the code you ran.

## Method summary

- **Preprocessing:** resize 224×224, ImageNet mean/std normalisation; train-time horizontal flip + mild colour jitter.
- **Baselines:** pixels (downsampled) → StandardScaler → PCA → Logistic Regression / RBF-SVM; C and γ chosen on the validation split.
- **CNN:** ImageNet-pretrained ResNet, new 6-way head, cross-entropy. Phase 1 trains the head only; phase 2 fine-tunes all layers with differential learning rates (lr/9, lr/3, lr) and cosine annealing. Best epoch chosen on validation accuracy; flip test-time augmentation reported separately.
- **Metrics:** accuracy, RMSE of the predicted count, per-class accuracy, confusion matrix. The test split is evaluated once per run.

## Reference

Bertorello, Sripada, Dendumrongsup — *Amazon Inventory Reconciliation Using AI* (Stanford). Used as a
reference point for expected accuracy ranges, not as a source of code.
Dataset: Amazon Bin Image Dataset, AWS Open Data Registry.
