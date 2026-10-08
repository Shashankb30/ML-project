# Amazon Bin Inventory Reconciliation — Counting Items in a Bin from a Photo

**Course:** UE24CS352A – Machine Learning (Mini-Project) · **Problem #65**

**Team:**
- **Shashank B Jain – PES1UG24CS611**
- **Sachin B R – PES1UG25CS839**

Amazon fulfilment-centre bins can sometimes contain a different number of items than the inventory system records. Given only a photo of a bin, this project predicts how many items (0–5) are present.

If the predicted count differs from the recorded inventory count, the bin is **flagged for manual recount**. The system does not automatically modify the inventory record.

The project compares classical machine-learning approaches, a probabilistic model, an ensemble model, and a transfer-learning CNN.

## Repository Layout

```text
src/
  common.py          Data loading, splits, metrics and result utilities
  download_data.py   Dataset download and preprocessing
  baseline_svm.py    PCA + Logistic Regression + RBF-SVM
  probabilistic.py   PCA + Gaussian Naive Bayes
  ensemble.py        Soft Voting Ensemble
  train_cnn.py       ResNet transfer learning
  demo.py            Prediction + inventory discrepancy demo
  make_synthetic.py  Synthetic data for offline smoke tests

tests/
  test_smoke.py      End-to-end smoke tests

report/
  build_report.py    Report generation
  build_deck.js      Presentation generation

data/
  images/            Bin images
  metadata/          Original JSON metadata
  labels.csv         Image IDs and quantity labels
  splits.csv         Train/validation/test split information

results/
  metrics JSON files and generated evaluation results
```

## 1. Setup

Python 3.10+ is required.

A GPU is recommended for CNN training, although the project can also run on CPU with longer training time.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Dataset

The project uses the **Amazon Bin Image Dataset**.

The dataset contains bin images along with metadata. The `EXPECTED_QUANTITY` field from the metadata is used as the ground-truth inventory count.

For this mini-project, the available data was filtered to six classes:

```text
0, 1, 2, 3, 4, 5
```

The current dataset contains:

- **6,798 usable images**
- **6 quantity classes**
- Quantity range: **0–5 items**
- **70% training**
- **20% validation**
- **10% testing**

The train/validation/test split is stored in:

```text
data/splits.csv
```

Labels are stored in:

```text
data/labels.csv
```

Example:

```text
id,quantity
00001,3
00002,1
00003,4
```

## 3. Models Implemented

The project evaluates four types of approaches.

### 3.1 Logistic Regression

Image pixels are converted into numerical features and reduced using PCA.

```text
RGB Image
    ↓
Resize to 32×32
    ↓
Flatten pixels
    ↓
StandardScaler
    ↓
PCA (200 components)
    ↓
Logistic Regression
```

### 3.2 RBF-SVM

The same PCA-based feature representation is used with an RBF-kernel Support Vector Machine.

```text
RGB Image
    ↓
Resize to 32×32
    ↓
Flatten pixels
    ↓
StandardScaler
    ↓
PCA (200 components)
    ↓
RBF-SVM
```

Different values of `C` and `gamma` were evaluated using the validation split.

### 3.3 Gaussian Naive Bayes

A probabilistic classification approach was added using Gaussian Naive Bayes.

```text
RGB Image
    ↓
Resize to 32×32
    ↓
Flatten pixels
    ↓
StandardScaler
    ↓
PCA (200 components)
    ↓
Gaussian Naive Bayes
```

This model predicts the probability of each inventory-count class and selects the most probable class.

### 3.4 Soft Voting Ensemble

A soft-voting ensemble combines the probability outputs of:

- Logistic Regression
- RBF-SVM
- Gaussian Naive Bayes

The three probability distributions are averaged to obtain the final prediction.

```text
                 ┌─ Logistic Regression ─┐
                 │                       │
Image → PCA ─────┼─ RBF-SVM ────────────┼→ Average Probabilities → Prediction
                 │                       │
                 └─ Gaussian NB ────────┘
```

Equal voting weights are used for the three models.

### 3.5 ResNet-34

A pretrained ResNet-34 model is used for transfer learning.

```text
Input Image
    ↓
Resize 224×224
    ↓
ImageNet Normalization
    ↓
Pretrained ResNet-34
    ↓
6-Class Classification Head
    ↓
Predicted Quantity (0–5)
```

Training uses two phases:

1. Train the classification head.
2. Fine-tune the complete network.

The CNN uses data augmentation including horizontal flipping and mild colour jitter.

## 4. Training and Evaluation

### Classical Models

```bash
python src/baseline_svm.py
```

This evaluates Logistic Regression and RBF-SVM.

### Gaussian Naive Bayes

```bash
python src/probabilistic.py
```

### Soft Voting Ensemble

```bash
python src/ensemble.py
```

### ResNet

```bash
python src/train_cnn.py --arch resnet34 --optim adam --lr 0.001 --epochs 10
```

The best CNN checkpoint is selected using validation accuracy.

## 5. Results

The models were evaluated using the same dataset split.

| Model | Validation Accuracy | Test Accuracy | Test RMSE |
|---|---:|---:|---:|
| Logistic Regression + PCA | 27.0% | 27.5% | — |
| RBF-SVM + PCA | 29.3% | 25.3% | — |
| Gaussian Naive Bayes + PCA | 23.3% | 21.9% | — |
| Soft Voting Ensemble | 23.7% | 23.2% | 1.813 |
| **ResNet-34 + Adam** | **37.2%** | **39.0%** | **1.219** |

For ResNet-34, flip test-time augmentation produced:

```text
Test Accuracy with flip-TTA: 39.1%
```

### Best Performing Model

The **ResNet-34 + Adam** model achieved the highest test accuracy:

**39.0%**

with a test RMSE of:

**1.219**

The soft-voting ensemble did not outperform the individual ResNet-34 model.

## 6. Demo

The demo predicts the inventory quantity from a bin image and compares the prediction with the recorded inventory count.

Example:

```bash
python src/demo.py --checkpoint checkpoints/resnet34_adam.pt --image path/to/bin.jpg --recorded 3
```

The workflow is:

```text
Bin Image
    ↓
ResNet-34
    ↓
Predicted Quantity
    ↓
Compare with Recorded Quantity
    ↓
Same → OK
Different → Flag for Manual Recount
```

The system **does not automatically change the inventory record**.

## 7. Testing

The project includes automated smoke tests using a small synthetic dataset.

Run:

```bash
python -m pytest tests/test_smoke.py
```

Current result:

```text
2 passed in 18.91s
```

Therefore:

**2/2 automated smoke tests passed.**

## 8. Method Summary

### Preprocessing

For classical models:

```text
RGB image → 32×32 → Flatten → StandardScaler → PCA
```

For ResNet:

```text
RGB image → 224×224 → ImageNet normalization
```

### Classical Models

- Logistic Regression
- RBF-SVM
- Gaussian Naive Bayes

### Ensemble

Soft Voting combines the probability outputs of the three classical models using equal weights.

### CNN

ImageNet-pretrained ResNet-34 with a new six-class classification head.

Training uses:

- Cross-entropy loss
- Adam optimizer
- Data augmentation
- Differential learning rates
- Cosine annealing
- Validation-based checkpoint selection

### Evaluation Metrics

The project uses:

- Accuracy
- RMSE
- Per-class performance
- Confusion matrix

## 9. Project Findings

The experiments show that:

1. Classical PCA-based models provide a simple baseline for inventory classification.
2. Gaussian Naive Bayes achieved lower accuracy than Logistic Regression and SVM on this dataset.
3. The soft-voting ensemble did not improve over the individual classical models.
4. ResNet-34 achieved the best test accuracy among the evaluated models.
5. A difference between predicted and recorded inventory is used to flag a bin for **manual recount**.

## 10. Reference

Bertorello, Sripada, Dendumrongsup — *Amazon Inventory Reconciliation Using AI* (Stanford).

The paper is used as a reference for the problem and methodology.

Dataset:

**Amazon Bin Image Dataset — AWS Open Data Registry**
