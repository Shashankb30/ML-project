"""Build the 2-page write-up PDF from results/*.json.

    python report/build_report.py --team "Name1 (SRN1), Name2 (SRN2)" --repo https://github.com/u/r
Run it again after every experiment; tables are generated from the JSON files, never typed by hand.
"""
import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
TBD = "TBD"


def load_results():
    out = []
    for f in sorted(glob.glob(str(ROOT / "results" / "*.json"))):
        d = json.load(open(f))
        if "test_accuracy" in d:
            d["_file"] = Path(f).stem
            out.append(d)
    return out


def split_counts(data_dir):
    p = Path(data_dir) / "splits.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    return Counter(r["split"] for r in rows), Counter(int(r["quantity"]) for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="[Name 1 (SRN)], [Name 2 (SRN)]")
    ap.add_argument("--repo", default="[GitHub repository URL]")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "report" / "writeup.pdf"))
    a = ap.parse_args()

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.8, leading=11.2, spaceAfter=3)
    h = ParagraphStyle("h", parent=ss["Heading2"], fontSize=10.5, spaceBefore=5, spaceAfter=2,
                       textColor=colors.HexColor("#1F3A5F"))
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=15, spaceAfter=2)
    sub = ParagraphStyle("s", parent=body, alignment=1, textColor=colors.HexColor("#444444"))
    note = ParagraphStyle("n", parent=body, backColor=colors.HexColor("#FFF4CC"), borderPadding=3)
    P = lambda t, s=body: Paragraph(t, s)

    results = load_results()
    sc = split_counts(a.data)
    story = [P("Counting Items in Amazon Fulfilment-Centre Bins from a Photo", title),
             P(f"UE24CS352A Machine Learning &middot; Mini-Project &middot; Problem #65 &middot; {a.team}", sub),
             P(f"Code: {a.repo}", sub)]

    story += [P("1. Problem statement", h), P(
        "Items in Amazon bins are sometimes misplaced, so the recorded inventory differs from the real "
        "contents. Given one colour photo of a bin, we predict the number of items it holds "
        "(0 to 5, a 6-class problem; random-guess baseline &asymp; 16.7%). If the prediction disagrees "
        "with the recorded count, the bin is flagged for a recount, which supports inventory reconciliation.")]

    if sc:
        splits, classes = sc
        n = sum(splits.values())
        cls = ", ".join(f"{k}: {classes[k]}" for k in sorted(classes))
        ds = (f"We use a random subset of <b>{n:,}</b> images ({splits['train']:,} train / {splits['val']:,} val / "
              f"{splits['test']:,} test, stratified 70/20/10). Class counts (items: images) &mdash; {cls}. ")
    else:
        ds = ("We use a random subset of <b>[N]</b> images, stratified 70/20/10 into train/val/test "
              "(<i>fill in by running the pipeline and re-building this PDF</i>). ")
    story += [P("2. Dataset", h), P(
        "<b>Amazon Bin Image Dataset</b> (AWS Open Data): 535,234 photos of bins in operating fulfilment "
        "centres, each with JSON metadata whose <i>EXPECTED_QUANTITY</i> field is our label. "
        + ds +
        "We keep bins with at most 5 items, resize to 224&times;224 and normalise. Images are noisy: tape and "
        "plastic wrap occlude items, items overlap, and lighting varies, so some bins are hard even for humans.")]

    story += [P("3. Approach", h), P(
        "<b>Classical baseline.</b> Downsampled pixels &rarr; standardisation &rarr; PCA &rarr; Logistic "
        "Regression and an RBF-kernel SVM; C and &gamma; are tuned on the validation split. "
        "<b>Deep model.</b> ImageNet-pretrained ResNets (18/34) with a new 6-way softmax head, trained with "
        "cross-entropy. Phase 1 trains only the head; phase 2 fine-tunes the whole network with "
        "differential learning rates (lr/9 early layers, lr/3 later layers, lr for the head) and cosine "
        "annealing, comparing SGD and Adam. Augmentation: horizontal flips and mild colour jitter; "
        "flip test-time augmentation (TTA) is reported separately. The model with the best validation "
        "accuracy is evaluated once on the held-out test split.")]

    story += [P("4. Implementation overview", h), P(
        "Python, scikit-learn and PyTorch/torchvision. <font face='Courier'>download_data.py</font> fetches and "
        "filters the subset and builds the splits; <font face='Courier'>baseline_svm.py</font> and "
        "<font face='Courier'>train_cnn.py</font> train and evaluate, writing metrics (accuracy, RMSE of the "
        "predicted count, per-class accuracy, confusion matrix) to <font face='Courier'>results/</font>; "
        "<font face='Courier'>demo.py</font> loads a checkpoint, predicts the count for a photo and flags any "
        "mismatch with the recorded count. A smoke-test suite runs the pipeline on synthetic data. "
        "This PDF is generated from the result files, so every number below comes from the code in the repository.")]

    # ---- results ---------------------------------------------------------------
    story.append(P("5. Results (held-out test split)", h))
    rows = [["Model", "Val acc", "Test acc", "Test RMSE", "Test acc (flip-TTA)"]]
    if results:
        for r in sorted(results, key=lambda r: -r["test_accuracy"]):
            rows.append([r["model"], f"{r.get('val_accuracy', float('nan')):.3f}",
                         f"{r['test_accuracy']:.3f}", f"{r['test_rmse']:.2f}",
                         f"{r['test_accuracy_tta']:.3f}" if "test_accuracy_tta" in r else "-"])
    else:
        rows += [[m, TBD, TBD, TBD, TBD] for m in
                 ("Logistic Regression + PCA", "SVM (RBF) + PCA", "ResNet-18", "ResNet-34")]
    t = Table(rows, colWidths=[62 * mm, 24 * mm, 24 * mm, 24 * mm, 36 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.3), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story.append(t)
    story.append(Spacer(1, 3))
    if results:
        best = max(results, key=lambda r: r["test_accuracy"])
        png = ROOT / "results" / f"{best['_file']}_confusion.png"
        if png.exists():
            story.append(Image(str(png), width=52 * mm, height=46 * mm))
            story.append(P(f"Normalised test confusion matrix of the best model ({best['model']}).",
                           ParagraphStyle("c", parent=body, fontSize=7.5)))
    else:
        story.append(P("<b>Results pending:</b> run the training scripts, then re-run "
                       "<font face='Courier'>python report/build_report.py</font> to fill this table.", note))
    story.append(P(
        "<i>Reference point from the literature (Bertorello et al., Stanford): on a similar 0&ndash;5 task, "
        "SVMs reached about 26&ndash;32% and ResNet-34 about 56% accuracy. These are context only, not our results.</i>",
        ParagraphStyle("r", parent=body, fontSize=8)))

    # ---- conclusions -----------------------------------------------------------
    story.append(P("6. Conclusions", h))
    if results:
        cnn = [r for r in results if r["_file"].startswith("cnn_")]
        cls_ = [r for r in results if not r["_file"].startswith("cnn_")]
        best = max(results, key=lambda r: r["test_accuracy"])
        txt = (f"The best model was <b>{best['model']}</b> with test accuracy <b>{best['test_accuracy']:.1%}</b> "
               f"and RMSE {best['test_rmse']:.2f} items (random guessing: 16.7%). ")
        if cnn and cls_:
            bc, bs = max(cnn, key=lambda r: r["test_accuracy"]), max(cls_, key=lambda r: r["test_accuracy"])
            d = bc["test_accuracy"] - bs["test_accuracy"]
            txt += (f"The best CNN ({bc['test_accuracy']:.1%}) is {abs(d) * 100:.1f} percentage points "
                    f"{'above' if d >= 0 else 'below'} the best classical model ({bs['test_accuracy']:.1%}). ")
        pc = best["per_class_accuracy"]
        txt += (f"Per-class accuracy ranges from {min(pc):.0%} (class {pc.index(min(pc))}) to "
                f"{max(pc):.0%} (class {pc.index(max(pc))}). ")
        story.append(P(txt + "Remaining errors are concentrated in neighbouring counts, consistent with occlusion by "
                       "tape and overlapping items. <i>[Review this paragraph against your confusion matrix and add "
                       "your own observations.]</i> Future work: removing unreadable images, class weighting, "
                       "training on more data, and ensembling."))
    else:
        story.append(P("<i>To be written once results exist.</i> Expected discussion: CNN vs classical models, "
                       "effect of optimiser/fine-tuning, per-class behaviour and its causes, limitations "
                       "(occlusion, label noise), and future work.", note))

    SimpleDocTemplate(a.out, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                      topMargin=13 * mm, bottomMargin=12 * mm,
                      title="Amazon Bin Inventory Reconciliation - Write-up").build(story)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
