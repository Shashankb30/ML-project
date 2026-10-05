// Builds slides.pptx from results/*.json.   Usage: npm install pptxgenjs && node build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const TEAM = process.env.TEAM || "[Name 1 (SRN)] · [Name 2 (SRN)]";
const REPO = process.env.REPO || "[GitHub repository URL]";
const NAVY = "1F3A5F", ORANGE = "E07B00", LIGHT = "F4F6FA", GREY = "5B6573", WHITE = "FFFFFF";

// ---- load results ----------------------------------------------------------
const resDir = path.join(ROOT, "results");
const results = fs.existsSync(resDir)
  ? fs.readdirSync(resDir).filter(f => f.endsWith(".json")).map(f => {
      const d = JSON.parse(fs.readFileSync(path.join(resDir, f)));
      d._file = f.replace(".json", ""); return d;
    }).filter(d => d.test_accuracy !== undefined)
  : [];
results.sort((a, b) => b.test_accuracy - a.test_accuracy);

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "Amazon Bin Inventory Reconciliation";

pres.defineSlideMaster({
  title: "CONTENT", background: { color: WHITE },
  objects: [
    { rect: { x: 0, y: 0, w: 10, h: 0.9, fill: { color: NAVY } } },
    { rect: { x: 0, y: 0.9, w: 10, h: 0.05, fill: { color: ORANGE } } },
  ],
  slideNumber: { x: 9.2, y: 5.25, color: GREY, fontSize: 10 },
});

function content(title) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(title, { x: 0.5, y: 0.1, w: 9, h: 0.7, fontFace: "Calibri", fontSize: 26, bold: true,
    color: WHITE, valign: "middle", isTextBox: true, margin: 0 });
  return s;
}
const bullets = (items, opt = {}) => items.map((t, i) => ({
  text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8, ...opt } }));
const body = { fontFace: "Calibri", fontSize: 17, color: "222222", valign: "top", isTextBox: true };

// 1. Title -------------------------------------------------------------------
let s = pres.addSlide();
s.background = { color: NAVY };
s.addShape(pres.ShapeType.rect, { x: 0.6, y: 2.35, w: 1.2, h: 0.06, fill: { color: ORANGE } });
s.addText("Counting Items in Amazon Bins from a Photo", { x: 0.6, y: 0.9, w: 8.8, h: 1.4,
  fontFace: "Calibri", fontSize: 38, bold: true, color: WHITE, valign: "bottom", isTextBox: true, margin: 0 });
s.addText("Inventory reconciliation with PCA+SVM and ResNet transfer learning", { x: 0.6, y: 2.55, w: 8.8, h: 0.5,
  fontFace: "Calibri", fontSize: 18, color: "D6DEEB", isTextBox: true, margin: 0 });
s.addText(`UE24CS352A Machine Learning · Mini-Project · Problem #65\n${TEAM}`, { x: 0.6, y: 4.1, w: 8.8, h: 0.8,
  fontFace: "Calibri", fontSize: 14, color: "D6DEEB", isTextBox: true, margin: 0 });

// 2. Problem -----------------------------------------------------------------
s = content("Problem: does the bin match the record?");
s.addText(bullets([
  "Items get misplaced in Amazon fulfilment-centre bins, so recorded inventory ≠ real contents",
  "Input: one colour photo of a bin",
  "Output: number of items in the bin (0–5), a 6-class classification task",
  "If prediction ≠ recorded count, flag the bin for recount",
  "Random-guess baseline ≈ 16.7%",
]), { ...body, x: 0.6, y: 1.3, w: 8.8, h: 3.8 });

// 3. Dataset -----------------------------------------------------------------
s = content("Dataset: Amazon Bin Image Dataset");
s.addText(bullets([
  "535,234 bin photos from operating fulfilment centres (AWS Open Data)",
  "Label = EXPECTED_QUANTITY from each image's JSON metadata",
  "We keep bins with ≤ 5 items, resized to 224×224",
  "Stratified 70 / 20 / 10 train / val / test split",
  "Hard cases: tape and wrap occlude items, items overlap, varied lighting",
]), { ...body, x: 0.6, y: 1.3, w: 5.6, h: 3.8 });
s.addShape(pres.ShapeType.roundRect, { x: 6.5, y: 1.4, w: 3.0, h: 3.2, fill: { color: LIGHT }, rectRadius: 0.1 });
s.addText([
  { text: "Our subset", options: { bold: true, breakLine: true, fontSize: 16, color: NAVY } },
  { text: "N images: [fill in]", options: { breakLine: true } },
  { text: "Train / val / test: [fill in]", options: { breakLine: true } },
  { text: "Seed 42, stratified", options: {} },
], { x: 6.7, y: 1.6, w: 2.6, h: 2.8, fontFace: "Calibri", fontSize: 14, color: "222222", valign: "top", isTextBox: true, paraSpaceAfter: 6 });

// 4. Approach ----------------------------------------------------------------
s = content("Approach: two model families, same split");
const boxes = [["Photo", "224×224, normalise"], ["Classical", "Pixels → PCA → LogReg / RBF-SVM"],
  ["Deep", "ResNet (ImageNet) → 6-way softmax"], ["Evaluate", "Acc · RMSE · per-class · confusion"]];
boxes.forEach(([h, t], i) => {
  const x = 0.5 + i * 2.35;
  s.addShape(pres.ShapeType.roundRect, { x, y: 1.5, w: 2.1, h: 1.7, fill: { color: i === 0 ? NAVY : LIGHT },
    line: { color: NAVY, width: 1 }, rectRadius: 0.1 });
  s.addText([{ text: h, options: { bold: true, fontSize: 17, breakLine: true } }, { text: t, options: { fontSize: 12 } }],
    { x: x + 0.1, y: 1.6, w: 1.9, h: 1.5, fontFace: "Calibri", align: "center", valign: "middle",
      color: i === 0 ? WHITE : NAVY, isTextBox: true });
  if (i < 3) s.addText("→", { x: x + 2.08, y: 2.05, w: 0.3, h: 0.5, fontSize: 20, color: ORANGE, align: "center", isTextBox: true, margin: 0 });
});
s.addText(bullets([
  "Test split is used once per final model; hyper-parameters are chosen on validation",
  "Compare optimisers (SGD vs Adam) and depth (ResNet-18 vs 34)",
]), { ...body, fontSize: 15, x: 0.6, y: 3.6, w: 8.8, h: 1.5 });

// 5. Models ------------------------------------------------------------------
s = content("Models in detail");
[["Classical baseline", ["Downsample pixels, standardise", "PCA to ~200 dimensions", "Logistic Regression", "RBF-SVM, C and γ tuned on val"]],
 ["ResNet transfer learning", ["Phase 1: train new head only", "Phase 2: fine-tune all layers", "Differential LRs: lr/9, lr/3, lr", "Cosine annealing, flip + jitter augmentation", "Flip test-time augmentation"]]]
  .forEach(([h, items], i) => {
    const x = 0.5 + i * 4.6;
    s.addShape(pres.ShapeType.roundRect, { x, y: 1.25, w: 4.4, h: 3.8, fill: { color: LIGHT }, rectRadius: 0.1 });
    s.addText(h, { x: x + 0.25, y: 1.35, w: 3.9, h: 0.5, fontFace: "Calibri", fontSize: 19, bold: true, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(bullets(items), { ...body, fontSize: 15, x: x + 0.25, y: 1.95, w: 3.9, h: 2.9, margin: 0 });
  });

// 6. Implementation ----------------------------------------------------------
s = content("Implementation");
s.addText(bullets([
  "download_data.py: fetch + filter + split the public S3 data",
  "baseline_svm.py: PCA + LogReg + RBF-SVM",
  "train_cnn.py: ResNet fine-tuning, metrics, curves, checkpoints",
  "demo.py: predict count and flag variance vs recorded count",
  "tests/: end-to-end smoke tests on synthetic data",
  "Every result is written to results/*.json; the write-up and these slides are generated from them",
]), { ...body, fontSize: 16, x: 0.6, y: 1.3, w: 8.8, h: 3.4 });
s.addText(`Repo: ${REPO}`, { x: 0.6, y: 4.8, w: 8.8, h: 0.4, fontFace: "Consolas", fontSize: 12, color: GREY, isTextBox: true, margin: 0 });

// 7. Results -----------------------------------------------------------------
s = content("Results (held-out test split)");
const hdr = ["Model", "Val acc", "Test acc", "Test RMSE", "Flip-TTA acc"]
  .map(t => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "center" } }));
let rows = [hdr];
if (results.length) {
  results.forEach(r => rows.push([r.model, (r.val_accuracy ?? NaN).toFixed(3), r.test_accuracy.toFixed(3),
    r.test_rmse.toFixed(2), r.test_accuracy_tta !== undefined ? r.test_accuracy_tta.toFixed(3) : "-"]
    .map((t, i) => ({ text: t, options: { align: i ? "center" : "left" } }))));
} else {
  ["Logistic Regression + PCA", "SVM (RBF) + PCA", "ResNet-18", "ResNet-34"].forEach(m =>
    rows.push([m, "TBD", "TBD", "TBD", "TBD"].map((t, i) => ({ text: t,
      options: { align: i ? "center" : "left", color: i ? ORANGE : "222222", italic: i > 0 } }))));
}
const tableH = Math.min(0.4 * rows.length, 2.6);
s.addTable(rows, { x: 0.5, y: 1.25, w: results.length ? 5.9 : 9, colW: results.length ? [2.5, 0.85, 0.85, 0.85, 0.85] : [3.6, 1.35, 1.35, 1.35, 1.35],
  fontFace: "Calibri", fontSize: 12, color: "222222", border: { type: "solid", pt: 0.5, color: "B8C0CC" }, rowH: 0.4 });
const best = results[0];
const png = best && path.join(resDir, `${best._file}_confusion.png`);
if (png && fs.existsSync(png)) {
  s.addImage({ path: png, x: 6.7, y: 1.2, w: 2.9, h: 2.58 });
  s.addText(`Best: ${best.model}`, { x: 6.7, y: 3.8, w: 2.9, h: 0.3, fontSize: 11, color: GREY, align: "center", isTextBox: true, margin: 0 });
}
s.addText(results.length
  ? `Best model: ${best.model}, ${(best.test_accuracy * 100).toFixed(1)}% accuracy vs 16.7% random guessing.`
  : "Run the training scripts and rebuild this deck to fill the table (values must come from results/*.json).",
  { x: 0.5, y: 4.35, w: 9, h: 0.7, fontFace: "Calibri", fontSize: 14, bold: true, color: results.length ? NAVY : ORANGE, isTextBox: true, margin: 0 });
s.addText("Literature reference (Stanford paper, similar 0–5 task): SVM ≈ 26–32%, ResNet-34 ≈ 56%. Context only.",
  { x: 0.5, y: 4.95, w: 8.2, h: 0.3, fontFace: "Calibri", fontSize: 10, italic: true, color: GREY, isTextBox: true, margin: 0 });

// 8. Demo --------------------------------------------------------------------
s = content("Live demo: flag inventory variance");
s.addText(bullets([
  "Load the best checkpoint (ResNet, flip-TTA)",
  "Input: a bin photo plus the count the system records",
  "Output: predicted count, confidence, and OK / VARIANCE verdict",
  "Batch mode: random test bins with simulated records (30% wrong) → green / red grid",
]), { ...body, x: 0.6, y: 1.3, w: 8.8, h: 2.4 });
s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 3.7, w: 8.8, h: 1.2, fill: { color: "1E1E1E" }, rectRadius: 0.08 });
s.addText("python src/demo.py --checkpoint checkpoints/<run>.pt --data data --sample 12", { x: 0.8, y: 3.8, w: 8.4, h: 1.0,
  fontFace: "Consolas", fontSize: 12, color: "9CDCFE", valign: "middle", isTextBox: true, margin: 0 });

// 9. Conclusions -------------------------------------------------------------
s = content("Conclusions and future work");
const conc = results.length
  ? [`Best: ${best.model}, ${(best.test_accuracy * 100).toFixed(1)}% test accuracy, RMSE ${best.test_rmse.toFixed(2)}`,
     "[Add your own takeaways: CNN vs classical, effect of fine-tuning/optimiser, hardest classes]"]
  : ["[Write conclusions after results are in: CNN vs classical, optimiser effect, hardest classes]"];
s.addText(bullets(conc), { ...body, x: 0.6, y: 1.2, w: 8.8, h: 1.5 });
s.addText("Future work", { x: 0.6, y: 2.85, w: 8.8, h: 0.4, fontFace: "Calibri", fontSize: 18, bold: true, color: NAVY, isTextBox: true, margin: 0 });
s.addText(bullets(["Remove unreadable (heavily occluded) images", "Class-weighted loss for weak classes",
  "Train on more of the 535K images", "Ensemble models"]), { ...body, fontSize: 15, x: 0.6, y: 3.3, w: 8.8, h: 1.8 });

// 10. Contributions ----------------------------------------------------------
s = content("Team contributions and Q&A");
s.addText(bullets([
  `${TEAM.split("·")[0].trim()}: [data pipeline, SVM baseline, ...]`,
  `${(TEAM.split("·")[1] || "[Name 2]").trim()}: [CNN training, demo, write-up, ...]`,
  "Repository commits show individual contribution",
]), { ...body, x: 0.6, y: 1.3, w: 8.8, h: 2.2 });
s.addText("Thank you. Questions?", { x: 0.6, y: 3.9, w: 8.8, h: 0.8, fontFace: "Calibri", fontSize: 28, bold: true, color: NAVY, isTextBox: true, margin: 0 });

pres.writeFile({ fileName: path.join(__dirname, "slides.pptx") }).then(f => console.log("wrote", f));
