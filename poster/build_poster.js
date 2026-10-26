// Builds the CIKM'26 A0 portrait poster for RAD.
// Usage: node poster/build_poster.js   (requires pptxgenjs)
const path = require("path");
const pptxgen = require("pptxgenjs");

const SKILL_THEME = process.env.APPLY_THEME_JS;
const FIG = (f) => path.join(__dirname, "figures", f);
const OUT = path.join(__dirname, "RAD_CIKM26_poster_A0.pptx");

const THEME = {
  name: "RAD Poster",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1A1A1A", lt1: "FFFFFF", dk2: "1C1C1C", lt2: "ECEEF1",
    accent1: "CFAE70", // Vanderbilt gold
    accent2: "8A6A1E", // dark gold (text on white)
    accent3: "D62728", // RAD red (matches paper plots)
    accent4: "1F77B4", // RDL blue (matches paper plots)
    accent5: "A3A8AF", // baseline gray
    accent6: "5B6168", // muted text
    hlink: "1F77B4", folHlink: "8A6A1E",
  },
};

// A0 portrait: 841 x 1189 mm
const W = 33.11, H = 46.81;
const M = 0.8, G = 0.5;
const COLW = (W - 2 * M - 2 * G) / 3;
const X1 = M, X2 = M + COLW + G, X3 = M + 2 * (COLW + G);
const SPAN2 = 2 * COLW + G;
const PAD = 0.4;

const pres = new pptxgen();
pres.defineLayout({ name: "A0_PORTRAIT", width: W, height: H });
pres.layout = "A0_PORTRAIT";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "RAD: Rule-Augmented Relational Anomaly Detection";
pres.author = "Noah Dahle, Anne Tumlin, Ngoc Tran, Xenofon Koutsoukos, Tyler Derr";
pres.subject = "CIKM 2026 poster";
const C = pres.SchemeColor;

const HEADER_H = 5.8;
const TITLE_W = 17.6;
pres.defineSlideMaster({
  title: "Poster",
  background: { color: C.background2 },
  objects: [
    { rect: { x: 0, y: 0, w: W, h: HEADER_H, fill: { color: C.text2 } } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.45, w: TITLE_W, h: 2.8,
        fontFace: THEME.headFontFace, fontSize: 84, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 },
      text: "Poster title" } },
    { placeholder: { options: { name: "authors", type: "body", x: M, y: 3.5, w: TITLE_W, h: 1.9,
        fontFace: THEME.bodyFontFace, fontSize: 36, color: C.background1, valign: "top", margin: 0 },
      text: "Authors" } },
  ],
});

const slide = pres.addSlide({ masterName: "Poster" });

// ---------- header ----------
slide.addText(
  [
    { text: "RAD", options: { color: C.accent1, breakLine: false } },
    { text: ": Rule-Augmented", options: { breakLine: true } },
    { text: "Relational Anomaly Detection" },
  ],
  { placeholder: "title" },
);
slide.addText(
  [
    { text: "Noah Dahle, Anne Tumlin, Ngoc Tran, Xenofon Koutsoukos, Tyler Derr", options: { bold: true, breakLine: true } },
    { text: "Vanderbilt University, Nashville, TN, USA", options: { fontSize: 30, color: C.accent1, breakLine: true } },
    { text: "CIKM ’26 · Rome, Italy", options: { fontSize: 30, color: C.accent5 } },
  ],
  { placeholder: "authors" },
);
const QR = 3.7;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: W - M - QR, y: 0.55, w: QR, h: QR, rectRadius: 0.15, fill: { color: C.background1 }, objectName: "QR frame",
});
slide.addImage({ path: FIG("qr.png"), x: W - M - QR + 0.2, y: 0.75, w: QR - 0.4, h: QR - 0.4,
  altText: "QR code linking to github.com/noahd15/RAD_RelationalAnomalyDetection", objectName: "QR code" });
slide.addText("Code & data", {
  x: W - M - QR, y: 0.55 + QR + 0.05, w: QR, h: 0.5, align: "center", fontSize: 24, color: C.background1, margin: 0, isTextBox: true,
});
// conference logo has black lettering, so it sits on a white tile like the QR code
const CIKM_H = QR - 0.5, CIKM_W = CIKM_H * 952 / 728, CIKM_TILE = CIKM_W + 0.5;
const cikmX = W - M - QR - 0.4 - CIKM_TILE;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: cikmX, y: 0.55, w: CIKM_TILE, h: QR, rectRadius: 0.15, fill: { color: C.background1 }, objectName: "CIKM logo frame",
});
slide.addImage({ path: FIG("cikm2026.png"), x: cikmX + 0.25, y: 0.8, w: CIKM_W, h: CIKM_H,
  altText: "CIKM 2026 logo", objectName: "CIKM logo" });
const V_H = 3.0, V_W = V_H * 1733 / 1217;
slide.addImage({ path: FIG("vanderbilt_v.png"), x: cikmX - 0.6 - V_W, y: 0.55 + (QR - V_H) / 2, w: V_W, h: V_H,
  altText: "Vanderbilt University logo", objectName: "Vanderbilt logo" });

// ---------- helpers ----------
function shadow() {
  return { type: "outer", color: "000000", opacity: 0.12, blur: 8, offset: 2, angle: 90 };
}
function card(x, y, w, h, num, title) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.25, fill: { color: C.background1 }, line: { type: "none" }, shadow: shadow(),
    objectName: `Card ${num}`,
  });
  const d = 0.85;
  slide.addShape(pres.shapes.OVAL, { x: x + PAD, y: y + 0.32, w: d, h: d, fill: { color: C.accent1 }, objectName: `Badge ${num}` });
  slide.addText(String(num), {
    x: x + PAD, y: y + 0.32, w: d, h: d, align: "center", valign: "middle", fontFace: THEME.headFontFace,
    fontSize: 36, bold: true, color: C.text1, margin: 0, isTextBox: true,
  });
  slide.addText(title, {
    x: x + PAD + d + 0.3, y: y + 0.32, w: w - 2 * PAD - d - 0.3, h: d, valign: "middle", fontFace: THEME.headFontFace,
    fontSize: 44, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: `Heading ${num}`,
  });
  return { cx: x + PAD, cy: y + 1.45, cw: w - 2 * PAD };
}
const BODY = 26;
function bullets(items, opts) {
  slide.addText(
    items.map((t, i) => ({
      text: t,
      options: { bullet: { indent: 24 }, breakLine: i < items.length - 1, paraSpaceAfter: 10 },
    })),
    { fontSize: BODY, color: C.text1, valign: "top", margin: 0, isTextBox: true, ...opts },
  );
}

// ---------- key message band ----------
{
  const y = HEADER_H + 0.5, h = 3.6;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: M, y, w: W - 2 * M, h, rectRadius: 0.25, fill: { color: C.background1 }, shadow: shadow(), objectName: "Key message card",
  });
  slide.addText(
    [
      { text: "Key idea  ", options: { bold: true, color: C.accent2, fontFace: THEME.headFontFace } },
      { text: "Mine symbolic rules from the data, inject them into target nodes ", options: {} },
      { text: "before", options: { italic: true, bold: true } },
      { text: " graph message passing, and train a heterogeneous graph autoencoder with a ranking loss — better anomaly retrieval in multi-table databases.", options: {} },
    ],
    { x: M + PAD, y: y + 0.3, w: 12.6, h: h - 0.6, fontSize: 31, color: C.text1, valign: "middle", margin: 0, isTextBox: true },
  );
  const stats = [
    ["1.2", "best average AUPRC rank of 10 methods across 3 benchmarks"],
    ["0.659", "AUPRC on LANL cyber events vs 0.525 for the best baseline"],
    ["0.601", "LANL precision@k: 60% of top alerts are real red-team logins (best baseline 0.540)"],
  ];
  const sx0 = M + PAD + 12.6 + 0.6, sw = (W - M - PAD - sx0 - 2 * 0.4) / 3;
  stats.forEach(([n, label], i) => {
    const sx = sx0 + i * (sw + 0.4);
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: sx, y: y + 0.3, w: sw, h: h - 0.6, rectRadius: 0.2, fill: { color: C.background2 }, objectName: `Stat ${i + 1}`,
    });
    slide.addText(n, {
      x: sx + 0.3, y: y + 0.45, w: sw - 0.6, h: 1.45, fontFace: THEME.headFontFace, fontSize: 80, bold: true,
      color: C.accent2, margin: 0, valign: "middle", isTextBox: true,
    });
    slide.addText(label, {
      x: sx + 0.3, y: y + 1.95, w: sw - 0.6, h: 1.25, fontSize: 24, color: C.text1, margin: 0, valign: "top", isTextBox: true,
    });
  });
}

// ---------- row 1 ----------
const R1 = HEADER_H + 0.5 + 3.6 + 0.5;
const R1H = 15.4;
// 1. Problem
const P_H = 6.7;
{
  const { cx, cy, cw } = card(X1, R1, COLW, P_H, 1, "The Problem");
  slide.addText(
    "A login can look normal as a single row yet be suspicious given who logged in, where, and what happened just before. Such anomalies live across tables.",
    { x: cx, y: cy, w: cw, h: 1.85, fontSize: BODY, color: C.text1, valign: "top", margin: 0, isTextBox: true },
  );
  bullets(
    [
      "Flattening into one feature matrix loses entity identity, schema roles and multi-hop context",
      "Relational deep learning keeps structure but has no anomaly objective",
      "Symbolic rules capture behavior, but usually sit outside representation learning",
    ],
    { x: cx, y: cy + 1.95, w: cw, h: 3.75 },
  );
}
// 2. Benchmark
{
  const y = R1 + P_H + G, h = R1H - P_H - G;
  const { cx, cy, cw } = card(X1, y, COLW, h, 2, "A New Benchmark");
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 } } });
  const rows = [
    [hdr("Task"), hdr("Target"), hdr("Records"), hdr("Anomaly")],
    ["LANL", "Auth event", "1.65B", "Red-team login"],
    ["Amazon", "Customer", "15.0M", "Unexpected review churn"],
    ["H&M", "Customer", "16.7M", "Unexpected purchase churn"],
  ];
  slide.addTable(rows, {
    x: cx, y: cy, w: cw, colW: [1.75, 2.0, 1.75, cw - 5.5], fontSize: 21, color: C.text1, valign: "middle",
    border: { type: "solid", pt: 1, color: "D5D8DC" }, rowH: 0.5, margin: [0.04, 0.1, 0.04, 0.1], objectName: "Benchmark table",
  });
  const fw = cw, fh = fw * 813 / 1934;
  slide.addImage({ path: FIG("churn.png"), x: cx, y: cy + 2.35, w: fw, h: fh,
    altText: "Unexpected-churn label construction: churned users with an expected prior activity decline are removed; the rest are anomalies.",
    objectName: "Churn label figure" });
}
// 3. Method
{
  const { cx, cy, cw } = card(X2, R1, SPAN2, R1H, 3, "RAD: Rule-Augmented Relational Anomaly Detection");
  const fs = R1H - 1.45 - PAD;
  slide.addImage({ path: FIG("overview.png"), x: cx, y: cy, w: fs, h: fs * 3943 / 3939,
    altText: "Overview of RAD: rule mining with random forests and LLM refinement, rule-feature augmentation, relational graph construction, heterogeneous graph autoencoder and losses.",
    objectName: "RAD overview figure" });
  const tx = cx + fs + 0.45, tw = cw - fs - 0.45;
  const steps = [
    ["Relational graph", "Rows become typed nodes, primary–foreign keys become typed edges; only data available at time t is used."],
    ["Rule mining", "Random-forest paths become conjunctive predicates, ranked by support, confidence and lift. An LLM prunes, tightens and merges them."],
    ["Rule injection", "Each target node gets a binary rule vector r(v), concatenated to its features before message passing."],
    ["Rule-aware scoring", "2-layer heterogeneous GraphSAGE autoencoder trained with reconstruction + pairwise ranking loss."],
  ];
  let sy = cy;
  const d = 0.62, stepH = 2.72;
  steps.forEach(([t, b], i) => {
    slide.addShape(pres.shapes.OVAL, { x: tx, y: sy + 0.02, w: d, h: d, fill: { color: C.text2 }, objectName: `Step badge ${i + 1}` });
    slide.addText(String(i + 1), { x: tx, y: sy + 0.02, w: d, h: d, align: "center", valign: "middle", fontSize: 26,
      bold: true, color: C.accent1, margin: 0, isTextBox: true });
    slide.addText(t, { x: tx + d + 0.2, y: sy, w: tw - d - 0.2, h: 0.66, fontFace: THEME.headFontFace, fontSize: 30,
      bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
    slide.addText(b, { x: tx + d + 0.2, y: sy + 0.72, w: tw - d - 0.2, h: stepH - 0.85, fontSize: 24, color: C.text1,
      valign: "top", margin: 0, isTextBox: true });
    sy += stepH;
  });
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: tx, y: sy + 0.05, w: tw, h: cy + fs - sy - 0.05, rectRadius: 0.15,
    fill: { color: C.background2 }, objectName: "LLM note" });
  slide.addText(
    [
      { text: "No label leakage. ", options: { bold: true } },
      { text: "The LLM only proposes predicates, which are re-validated on train/validation data. It never assigns labels or scores." },
    ],
    { x: tx + 0.3, y: sy + 0.2, w: tw - 0.6, h: cy + fs - sy - 0.35, fontSize: 22, color: C.text1, valign: "middle", margin: 0, isTextBox: true },
  );
}

// ---------- row 2: results ----------
const R2 = R1 + R1H + G;
const R2H = 9.2;
const MODELS = ["RAD (rules, no BCE)", "RAD (rules, BCE)", "RDL weighted", "RDL unweighted", "AE+Rank", "DRL", "AE",
  "Deep SVDD", "Isolation Forest", "Supervised MLP"];
// Table 4 of the paper: means over 10 seeds. Columns: LANL, Amazon, H&M (AUROC, AUPRC, P@k), then avg rank (AUROC, AUPRC, P@k).
const T4 = [
  [0.996, 0.659, 0.601, 0.693, 0.043, 0.050, 0.828, 0.143, 0.129, 2.0, 1.2, 2.0],
  [0.860, 0.557, 0.572, 0.695, 0.043, 0.051, 0.787, 0.110, 0.125, 3.3, 1.8, 2.3],
  [0.553, 0.007, 0.021, 0.720, 0.037, 0.062, 0.700, 0.035, 0.064, 4.7, 4.8, 3.7],
  [0.685, 0.015, 0.026, 0.713, 0.037, 0.062, 0.693, 0.039, 0.064, 4.3, 3.8, 3.3],
  [0.943, 0.525, 0.540, 0.518, 0.016, 0.015, 0.456, 0.013, 0.005, 5.2, 5.2, 5.7],
  [0.673, 0.004, 0.005, 0.463, 0.013, 0.003, 0.456, 0.013, 0.009, 7.5, 8.2, 7.2],
  [0.888, 0.009, 0.000, 0.474, 0.014, 0.014, 0.408, 0.012, 0.007, 6.7, 7.3, 7.8],
  [0.736, 0.005, 0.001, 0.458, 0.013, 0.003, 0.412, 0.012, 0.002, 7.7, 8.7, 8.8],
  [0.644, 0.003, 0.000, 0.450, 0.013, 0.001, 0.402, 0.012, 0.004, 9.7, 9.3, 9.5],
  [0.965, 0.050, 0.093, 0.659, 0.027, 0.041, 0.642, 0.027, 0.039, 4.0, 4.7, 4.7],
];
{
  const { cx, cy, cw } = card(X1, R2, SPAN2, R2H, 4, "Main Results (10 seeds)");
  const best = T4[0].map((_, j) => {
    const col = T4.map((r) => r[j]);
    return j >= 9 ? Math.min(...col) : Math.max(...col);
  });
  const head = (t, span) => ({ text: t, options: { bold: true, align: "center", color: C.background1, fill: { color: C.text2 }, colspan: span } });
  const sub = (t) => ({ text: t, options: { bold: true, align: "center", fill: { color: C.background2 } } });
  const metrics = ["AUROC", "AUPRC", "P@k"];
  const rows = [
    [{ text: "Model", options: { bold: true, color: C.background1, fill: { color: C.text2 }, rowspan: 2, valign: "middle" } },
      head("LANL", 3), head("Amazon", 3), head("H&M", 3), head("Avg. rank ↓", 3)],
    [...metrics, ...metrics, ...metrics, ...metrics].map(sub),
    ...T4.map((r, i) => {
      const isRad = i < 2;
      const fill = isRad ? { fill: { color: "F6EEDC" } } : {};
      return [
        { text: MODELS[i], options: { bold: isRad, ...fill } },
        ...r.map((v, j) => ({
          text: j >= 9 ? v.toFixed(1) : v.toFixed(3),
          options: { align: "center", bold: v === best[j], color: v === best[j] ? C.accent2 : C.text1, ...fill },
        })),
      ];
    }),
  ];
  const mw = 3.7;
  slide.addTable(rows, {
    x: cx, y: cy, w: cw, colW: [mw, ...Array(12).fill((cw - mw) / 12)], rowH: 0.52, fontSize: 21, color: C.text1,
    valign: "middle", border: { type: "solid", pt: 1, color: "D5D8DC" }, margin: [0.03, 0.08, 0.03, 0.08], objectName: "Main results table",
  });
  slide.addText(
    "Means over 10 seeds (std. dev. in the paper). Bold = best per column. AUPRC and P@k matter most under heavy class imbalance.",
    { x: cx, y: cy + 12 * 0.52 + 0.2, w: cw, h: 0.55, fontSize: 21, italic: true, color: C.accent6, margin: 0, isTextBox: true },
  );
}
{
  const { cx, cy, cw } = card(X3, R2, COLW, R2H, 5, "Ranking at a Glance");
  const order = MODELS.map((m, i) => [m.replace("rules, ", ""), T4[i][10]]).sort((a, b) => b[1] - a[1]); // top of chart = best
  const colors = order.map(([m]) => (m.startsWith("RAD") ? THEME.colors.accent2 : THEME.colors.accent5));
  slide.addChart(pres.charts.BAR, [{ name: "Avg. AUPRC rank", labels: order.map((o) => o[0]), values: order.map((o) => o[1]) }], {
    x: cx, y: cy, w: cw, h: R2H - 1.45 - 1.25, barDir: "bar", chartColors: colors, barGapWidthPct: 45,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 20, dataLabelFormatCode: "0.0", dataLabelFontFace: "+mn-lt",
    dataLabelColor: THEME.colors.dk1, catAxisLabelFontSize: 20, catAxisLabelFontFace: "+mn-lt", catAxisLabelColor: THEME.colors.dk1,
    valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 11, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLineShow: false, showLegend: false, objectName: "Average AUPRC rank chart",
  });
  slide.addText("Average AUPRC rank across LANL, Amazon and H&M (lower is better). Both RAD variants rank ahead of every baseline, including the supervised MLP.",
    { x: cx, y: R2 + R2H - 1.35, w: cw, h: 1.05, fontSize: 21, italic: true, color: C.accent6, margin: 0, valign: "top", isTextBox: true });
}

// ---------- row 3 ----------
const R3 = R2 + R2H + G;
const FOOT = 1.05;
const R3H = H - M - FOOT - R3;
{
  const { cx, cy, cw } = card(X1, R3, COLW, R3H, 6, "What Matters? Ablations");
  const fh = cw * 1212 / 2814;
  slide.addImage({ path: FIG("ablation.png"), x: cx, y: cy, w: cw, h: fh,
    altText: "Ablation bar charts for LANL, Amazon and H&M comparing RAD with and without BCE, GAE without rules, GAE with RF sampled rules and GAE with RF features.",
    objectName: "Ablation figure" });
  bullets(
    [
      "Rules help: rule-augmented variants beat the no-rules GAE, most clearly on AUPRC",
      "Compact predicates beat raw random-forest decision features",
      "Edge reconstruction (BCE) is optional: the no-BCE variant ranks best overall",
    ],
    { x: cx, y: cy + fh + 0.3, w: cw, h: R3H - 1.45 - fh - 0.6, fontSize: 24 },
  );
}
{
  const { cx, cy, cw } = card(X2, R3, SPAN2, R3H, 7, "Robust to the Anomaly Definition");
  const textH = 1.3;
  const ph = R3H - 1.45 - textH - PAD;
  const aw = ph * 2823 / 2083, hw = ph * 2823 / 2083;
  const gap = (cw - aw - hw) / 3;
  slide.addImage({ path: FIG("sens_amazon.png"), x: cx + gap, y: cy, w: aw, h: ph,
    altText: "Amazon: AUROC and AUPRC of RAD vs supervised RDL at 0.5%, 1.5% and 3.0% anomaly prevalence.", objectName: "Sensitivity Amazon" });
  slide.addImage({ path: FIG("sens_hm.png"), x: cx + 2 * gap + aw, y: cy, w: hw, h: ph,
    altText: "H&M: AUROC and AUPRC of RAD vs supervised RDL at 0.5%, 1.5% and 3.0% anomaly prevalence.", objectName: "Sensitivity H&M" });
  slide.addText(
    "Varying the unexpected-churn prevalence (0.5%, 1.5%, 3.0%) leaves the RAD-vs-RDL comparison stable: the 1.5% default is not a cherry-picked threshold. Solid lines = AUROC, dotted = AUPRC.",
    { x: cx, y: cy + ph + 0.2, w: cw, h: textH, fontSize: 24, color: C.text1, margin: 0, valign: "top", isTextBox: true },
  );
}

// ---------- footer ----------
slide.addText(
  [
    { text: "Paper: ", options: { bold: true } },
    { text: "doi.org/10.1145/3799682.3841086", options: { hyperlink: { url: "https://doi.org/10.1145/3799682.3841086" } } },
    { text: "    Code: ", options: { bold: true } },
    { text: "github.com/noahd15/RAD_RelationalAnomalyDetection", options: { hyperlink: { url: "https://github.com/noahd15/RAD_RelationalAnomalyDetection" } } },
    { text: "    Supported by NSA (H98230-23-C-0279) and NSF (IIS-2239881, ECCS-2325417, IIS-2524380, DGE-2622415).", options: {} },
  ],
  { x: M, y: H - M - FOOT + 0.3, w: W - 2 * M, h: FOOT - 0.3, fontSize: 20, color: C.accent6, valign: "middle", margin: 0, isTextBox: true },
);

slide.addNotes(
  "Two-minute pitch. " +
  "1) Problem: in relational databases, anomalies often depend on context across tables; flattening loses that, and relational deep learning has no anomaly objective. " +
  "2) Idea: RAD mines random-forest rules, refines them with an LLM into compact predicates, and injects them as binary features into target nodes before message passing. A heterogeneous graph autoencoder is trained with reconstruction plus a pairwise ranking loss. " +
  "3) Benchmark: LANL red-team logins plus unexpected churn on Amazon and H&M. " +
  "4) Results: best average AUPRC rank (1.2 of 10 methods); LANL AUPRC 0.659 vs 0.525 and P@k 0.601 vs 0.540. Amazon: RDL wins AUROC and P@k, RAD wins AUPRC. " +
  "5) Ablations: rules help, compact predicates beat raw RF features, edge reconstruction is optional. Results are stable across 0.5 to 3 percent prevalence.",
);

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (SKILL_THEME) {
    const { applyTheme } = require(SKILL_THEME);
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT);
})();
