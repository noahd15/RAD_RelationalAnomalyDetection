// RAD CIKM'26 poster — single A0 portrait slide, built with pptxgenjs.
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/isis/home/dahlenw/.claude/skills/synced/35df0b8e-ffab-4486-841e-a92661c858c4_c1ef5b37-0b79-4a5e-b1cc-390072bd4253/pptx/scripts/apply_theme.js");

const S = __dirname + "/";
const OUT = __dirname + "/../RAD_CIKM2026_poster.pptx";

// ---- theme ----------------------------------------------------------------
const THEME = {
  name: "RAD Poster",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2933", lt1: "FFFFFF", dk2: "1B2A41", lt2: "EEF2F7",
    accent1: "C86F1B", accent2: "2F6DB5", accent3: "9AA5B1", accent4: "2CA02C",
    accent5: "D62728", accent6: "9467BD", hlink: "2F6DB5", folHlink: "9467BD",
  },
};
const INK = "1F2933", NAVY = "1B2A41", PANEL = "FFFFFF", GROUND = "EEF2F7";
const AMBER = "C86F1B", BLUE = "2F6DB5", MUTED = "5B6670", LINE = "D5DCE5", PALE = "C9D4E3";

const pres = new pptxgen();
pres.defineLayout({ name: "A0_PORTRAIT", width: 33.11, height: 46.81 });
pres.layout = "A0_PORTRAIT";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Noah Dahle"; pres.title = "RAD: Rule-Augmented Relational Anomaly Detection (CIKM 2026 poster)";

const slide = pres.addSlide();
slide.background = { color: GROUND };

// ---- type scale (pt) ------------------------------------------------------
const T = { title: 58, authors: 34, affil: 28, head: 36, body: 24, small: 21, table: 16, callout: 54 };

// ---- geometry -------------------------------------------------------------
const M = 0.8, W = 33.11, GAP = 0.5;
const COLW = (W - 2 * M - 2 * GAP) / 3;           // 10.17
const COLX = [M, M + COLW + GAP, M + 2 * (COLW + GAP)];
const PAD = 0.4;
const HEADER_H = 5.3, COL_TOP = M + HEADER_H + 0.5, COL_H = 39.0, COL_BOTTOM = COL_TOP + COL_H;
const cw = COLW - 2 * PAD;                          // 9.37 panel content width

// ---- crude text-fit estimator (Calibri ~0.5em avg glyph width) ------------
const warnings = [];
function estH(runs, w, pt, paraAfterPt = 8) {
  const text = Array.isArray(runs) ? runs.map(r => r.text).join("") : runs;
  const paras = text.split("\n");
  const cpl = Math.floor(w / (pt / 72 * 0.5));
  let lines = 0;
  for (const p of paras) lines += Math.max(1, Math.ceil(p.length / cpl));
  return lines * (pt / 72) * 1.2 + (paras.length - 1) * paraAfterPt / 72;
}
function check(label, runs, w, h, pt, paraAfterPt) {
  const need = estH(runs, w, pt, paraAfterPt);
  if (need > h) warnings.push(`${label}: est ${need.toFixed(2)}" > box ${h.toFixed(2)}"`);
}

// helpers -------------------------------------------------------------------
function panel(x, y, w, h, title, opts = {}) {
  const dark = !!opts.dark;
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.18, fill: { color: dark ? NAVY : PANEL }, line: { color: dark ? NAVY : LINE, width: 1.5 },
    objectName: "panel: " + title,
  });
  check("heading " + title, title, w - 2 * PAD, 0.75, T.head);
  slide.addText(title, {
    x: x + PAD, y: y + PAD - 0.05, w: w - 2 * PAD, h: 0.75, fontFace: THEME.headFontFace, fontSize: T.head, bold: true,
    color: dark ? "FFFFFF" : NAVY, margin: 0, isTextBox: true, valign: "top", objectName: "heading: " + title,
  });
  return y + PAD + 0.95;
}
function body(label, runs, x, y, w, h, extra = {}) {
  const base = { fontSize: T.body, color: INK, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 8, objectName: label };
  const o = Object.assign({ x, y, w, h }, base, extra);
  const txt = runs.map(r => r.text + (r.options && r.options.breakLine ? "\n" : "")).join("");
  check(label, txt, w, h, o.fontSize, o.paraSpaceAfter);
  slide.addText(runs, o);
}
function bullets(label, items, x, y, w, h, extra = {}) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: { indent: 20 }, breakLine: i < items.length - 1 } }));
  body(label, runs, x, y, w - 0.3, h, Object.assign({ paraSpaceAfter: 10 }, extra));
}
const b = (text) => ({ text, options: { bold: true } });
const r = (text) => ({ text });
const sub = (text) => ({ text, options: { subscript: true } });
const sup = (text) => ({ text, options: { superscript: true } });
const it = (text) => ({ text, options: { italic: true } });
const br = () => ({ text: "", options: { breakLine: true } });
function formulaBox(label, runs, x, y, w, h, pt) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.12, fill: { color: GROUND }, line: { color: GROUND }, objectName: "formula box: " + label });
  slide.addText(runs, { x: x + 0.15, y, w: w - 0.3, h, fontFace: THEME.headFontFace, fontSize: pt, color: NAVY, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "formula: " + label });
}

// =============================== HEADER ====================================
slide.addShape(pres.ShapeType.roundRect, { x: M, y: M, w: W - 2 * M, h: HEADER_H, rectRadius: 0.18, fill: { color: NAVY }, line: { color: NAVY }, objectName: "header" });
const QR = 3.3, QRY = M + 0.55;
const qrX2 = W - M - 0.5 - QR, qrX1 = qrX2 - 0.5 - QR;
const titleW = qrX1 - 0.15 - 0.5 - (M + 0.6);
slide.addText("RAD: Rule-Augmented Relational Anomaly Detection", {
  x: M + 0.6, y: M + 0.45, w: titleW, h: 1.5, fontFace: THEME.headFontFace, fontSize: T.title, bold: true,
  color: "FFFFFF", margin: 0, isTextBox: true, valign: "middle", objectName: "title",
});
slide.addText("Noah Dahle · Anne Tumlin · Ngoc Tran · Xenofon Koutsoukos · Tyler Derr",
  { x: M + 0.6, y: M + 2.2, w: titleW, h: 0.7, fontSize: T.authors, color: "FFFFFF", margin: 0, isTextBox: true, objectName: "authors" });
slide.addText([r("Vanderbilt University, Nashville, Tennessee, USA   ·   "), it("CIKM ’26, Rome, Italy, November 7–11, 2026")],
  { x: M + 0.6, y: M + 3.0, w: titleW, h: 0.6, fontSize: T.affil, color: PALE, margin: 0, isTextBox: true, objectName: "affiliation" });
slide.addText("Paper: arxiv.org/abs/2608.23468   ·   Code: github.com/noahd15/RAD_RelationalAnomalyDetection   ·   noah.dahle@vanderbilt.edu",
  { x: M + 0.6, y: M + 3.85, w: titleW, h: 0.6, fontSize: T.small, color: PALE, margin: 0, isTextBox: true, objectName: "links" });
for (const [x, img, label] of [[qrX1, "qr_paper.png", "Paper (arXiv)"], [qrX2, "qr_code.png", "Code & data"]]) {
  slide.addShape(pres.ShapeType.roundRect, { x: x - 0.15, y: QRY - 0.15, w: QR + 0.3, h: QR + 0.3, rectRadius: 0.1, fill: { color: "FFFFFF" }, line: { color: "FFFFFF" }, objectName: "qr frame " + label });
  slide.addImage({ path: S + img, x, y: QRY, w: QR, h: QR, objectName: "QR " + label });
  slide.addText(label, { x: x - 0.15, y: QRY + QR + 0.25, w: QR + 0.3, h: 0.5, fontSize: T.small, bold: true, color: "FFFFFF", align: "center", margin: 0, isTextBox: true, objectName: "qr label " + label });
}

// =============================== COLUMN 1 ==================================
let x = COLX[0], y = COL_TOP, h, cy;

// -- Why relational detection?
h = 9.9;
cy = panel(x, y, COLW, h, "Why relational detection?");
body("c1 intro", [
  r("Anomaly detection is central to cybersecurity, fraud, and healthcare, yet the data usually lives in "),
  b("multi-table relational databases"), r(", not one feature matrix."),
], x + PAD, cy, cw, 1.4);
body("c1 example", [
  it("A login can look normal on its own, yet be suspicious because the user–machine pair never co-occurred, the machine is otherwise used only by service accounts, and the user hit four other machines in the last ten minutes."),
], x + PAD, cy + 1.5, cw, 2.6, { color: MUTED });
body("c1 gaps label", [b("Two gaps in current practice")], x + PAD, cy + 4.2, cw, 0.5);
bullets("c1 gaps", [
  "Flattening tables into one matrix (joins, aggregates, hand-crafted features) discards entity identity, schema roles, multi-hop dependencies, and temporal context.",
  "Relational deep learning (RDL) keeps the structure but has no anomaly-oriented objective and no way to use symbolic evidence such as rules or analyst patterns.",
], x + PAD, cy + 4.8, cw, h - (cy - y) - 4.8 - 0.3);
y += h + GAP;

// -- Problem definition
h = 6.3;
cy = panel(x, y, COLW, h, "Problem definition");
body("c1 problem", [
  r("Given a database "), it("D"), r(" with schema "), it("S"), r(", a target table "), it("T"), sub("target"),
  r(", and its heterogeneous graph "), it("G"), r(" = ("), it("V, E, τ"), r("), learn a scoring function "),
  b("s : V"), { text: "target", options: { bold: true, subscript: true } }, b(" → ℝ"),
  r(" so that anomalous target nodes rank above normal ones: "), it("s(v"), sub("i"), it(") > s(v"), sub("j"), it(")"), r(" whenever "), it("y"), sub("i"), r(" = 1 and "), it("y"), sub("j"), r(" = 0."),
], x + PAD, cy, cw, 2.6);
bullets("c1 problem bullets", [
  "Evaluated on held-out target nodes as a ranking, not a calibrated per-instance label.",
  "Temporal consistency: s(v) may use only structure and attributes observed at or before tᵥ.",
], x + PAD, cy + 2.7, cw, h - (cy - y) - 2.7 - 0.3);
y += h + GAP;

// -- Contributions
h = 6.9;
cy = panel(x, y, COLW, h, "Contributions");
bullets("c1 contributions", [
  "We formalize relational anomaly detection as distinct from tabular and graph anomaly detection: anomalousness can depend on cross-table, multi-hop, temporal, and schema-dependent interactions.",
  "RAD: mine rules from a flattened view, inject them into target nodes, and learn anomaly scores with heterogeneous message passing, reconstruction, and pairwise ranking.",
  "A benchmark joining LANL cybersecurity event detection with two unexpected user-churn tasks built from Amazon and H&M (RelBench), under one ranking protocol.",
], x + PAD, cy, cw, h - (cy - y) - 0.3);
y += h + GAP;

// -- Benchmark
h = COL_BOTTOM - y;
cy = panel(x, y, COLW, h, "Benchmark: three tasks");
const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center", valign: "middle", fontSize: 17 } });
const cell = (t, o = {}) => ({ text: t, options: Object.assign({ align: "center", valign: "middle" }, o) });
const bcol = [1.3, 1.65, 1.0, 1.35, 1.35];
slide.addTable([
  [hdr("Task"), hdr("Target"), hdr("Tables"), hdr("Records"), hdr("Targets"), hdr("Anomaly")],
  [cell("LANL", { bold: true }), cell("Auth. event"), cell("7"), cell("1.65B"), cell("1.64M"), cell("Red-team authentication")],
  [cell("Amazon", { bold: true }), cell("Customer"), cell("3"), cell("15.0M"), cell("1.85M"), cell("Unexpected review churn")],
  [cell("H&M", { bold: true }), cell("Customer"), cell("3"), cell("16.7M"), cell("1.37M"), cell("Unexpected purchase churn")],
], {
  x: x + PAD, y: cy, w: cw, colW: bcol.concat([cw - bcol.reduce((a, c) => a + c, 0)]),
  fontSize: 18, color: INK, border: { type: "solid", color: LINE, pt: 1 }, rowH: 0.6, margin: 0.06, objectName: "benchmark table",
});
let ty = cy + 4 * 0.6 + 0.65;
body("c1 churn", [
  b("Unexpected churn. "), r("Ordinary churn is often predictable, so a churned user is anomalous only if prior activity did "), b("not"), r(" already signal disengagement: "),
  it("D(u)"), r(" = 1["), it("g"), sub("u"), r(" ≤ −θ(μ"), sub("u"), r(")],  "), it("y"), { text: "anom", options: { italic: true, superscript: true } }, r(" = "), it("y"), { text: "churn", options: { italic: true, superscript: true } }, r(" · (1 − "), it("D(u)"), r("). Default prevalence 1.5%; 0.5% and 3.0% in the sweep."),
], x + PAD, ty, cw, 2.7);
const f2w = cw, f2h = f2w * 813 / 1934;
let fy = ty + 2.8;
slide.addImage({ path: S + "fig-000.png", x: x + PAD, y: fy, w: f2w, h: f2h, objectName: "Figure: unexpected-churn filtering" });
body("c1 fig2 caption", [r("Churned users whose activity already declined are removed; the remaining churners are the unexpected-churn anomalies.")],
  x + PAD, fy + f2h + 0.15, cw, 0.95, { fontSize: T.small, color: MUTED });
body("c1 lanl", [
  b("LANL. "), r("Authentication events are targets; red-team authentications are anomalies. We score sampled relational subgraphs around anchor events with their linked users, computers, processes, DNS, and flows."),
], x + PAD, fy + f2h + 1.2, cw, COL_BOTTOM - (fy + f2h + 1.2) - 0.3, { fontSize: T.small });

// =============================== COLUMN 2 ==================================
x = COLX[1]; y = COL_TOP;

// -- RAD overview
const f1w = 8.1, f1h = f1w;
h = 15.7;
cy = panel(x, y, COLW, h, "RAD in four stages");
slide.addImage({ path: S + "fig1_hi-04.png", x: x + (COLW - f1w) / 2, y: cy - 0.1, w: f1w, h: f1h, objectName: "Figure: RAD overview" });
let sy = cy + f1h + 0.15;
const stage = (n, title, text) => [{ text: n + "   ", options: { bold: true, color: AMBER } }, b(title + ". "), r(text), br()];
body("c2 stages", [].concat(
  stage("1", "Relational graph construction", "Rows become typed nodes, primary–foreign keys become typed edges; type-specific input projections; temporal constraint respected."),
  stage("2", "Rule mining and validation", "A random forest on an auxiliary flattened view of the targets; each root-to-leaf path is a conjunctive predicate q(v) = c₁(v) ∧ … ∧ cₘ(v), ranked by support, confidence, and lift, then frozen."),
  stage("3", "Rule-augmented node features", "Each retained rule becomes a binary feature rₖ(v), concatenated to the target node before message passing."),
  stage("4", "Heterogeneous graph autoencoder", "Two-layer relation-aware GraphSAGE; the anomaly score is the reconstruction error, trained with pairwise ranking supervision."),
), x + PAD, sy, cw, y + h - sy - 0.3, { paraSpaceAfter: 7, fontSize: 23 });
y += h + GAP;

// -- Key design choice
h = 7.2;
cy = panel(x, y, COLW, h, "Design choice: direct injection");
formulaBox("augmentation", [
  it("x̃"), sub("v"), r(" = [ "), it("x"), sub("v"), r(" ‖ "), it("r"), r("("), it("v"), r(") ]"), br(),
  it("r"), r("("), it("v"), r(") = [ "), it("r"), sub("1"), r("("), it("v"), r("), …, "), it("r"), sub("K"), r("("), it("v"), r(") ] ∈ {0,1}"), sup("K"), r(",   "), it("r"), sub("k"), r("("), it("v"), r(") = 1 iff "), it("v"), r(" satisfies rule "), it("k"),
], x + PAD, cy, cw, 1.6, 28);
bullets("c2 injection", [
  "Rules enter the target node before graph encoding, so symbolic evidence shapes the learned representation and interacts with relational context.",
  "Post-hoc fusion (adding rule scores after the graph model has scored) can nudge scores but cannot change what the encoder learned; the ablation favors direct injection.",
  "Mining uses only training/validation labels; at test time the frozen predicates are simply evaluated.",
], x + PAD, cy + 1.8, cw, y + h - (cy + 1.8) - 0.3);
y += h + GAP;

// -- Rule mining and LLM refinement
h = 8.5;
cy = panel(x, y, COLW, h, "Rule mining and refinement");
formulaBox("rule stats", [
  r("support"), sub("S"), r("(q) = Σ"), sub("v∈S"), r(" 1[q(v) = 1]"), br(),
  r("confidence"), sub("S"), r("(q) = Σ 1[q(v) = 1 ∧ y"), sub("v"), r(" = 1]  /  Σ 1[q(v) = 1]"), br(),
  r("lift"), sub("S"), r("(q) = confidence"), sub("S"), r("(q) / π"), sub("S"), br(),
  r("π"), sub("S"), r(" = anomaly prevalence on split S"),
], x + PAD, cy, cw, 2.2, 22);
bullets("c2 mining", [
  "Predicates touching identifiers, labels, unavailable columns, or future information are dropped, as are those below minimum coverage or duplicated.",
  "Optional LLM step: given the schema, feature statistics, compact positives, and sampled seed rules, it proposes SQL WHERE predicates that prune, tighten, combine, or generalize them.",
  "The LLM never labels, scores, or sees test data. Each proposal is re-executed and re-scored with the same support / confidence / lift filter: compaction, not a new signal.",
], x + PAD, cy + 2.4, cw, y + h - (cy + 2.4) - 0.3);
y += h + GAP;

// -- Training objective
h = COL_BOTTOM - y;
cy = panel(x, y, COLW, h, "Training objective");
formulaBox("loss", [it("ℒ"), r(" = "), it("ℒ"), sub("attr"), r(" + "), it("α ℒ"), sub("struct"), r(" + "), it("β ℒ"), sub("rank")], x + PAD, cy, cw, 1.0, 32);
body("c2 loss", [
  b("ℒ"), { text: "attr", options: { bold: true, subscript: true } }, r(" = ‖x̃"), sub("v"), r(" − x̂"), sub("v"), r("‖²  reconstructs rule-augmented target features; the error is the anomaly signal."), br(),
  b("ℒ"), { text: "struct", options: { bold: true, subscript: true } }, r("  BCE on observed vs. sampled typed edges, â"), sub("uv"), sup("(r)"), r(" = σ(z"), sub("u"), r("ᵀ W"), sub("r"), r(" z"), sub("v"), r("). Optional: α = 0 in the “no BCE” variant."), br(),
  b("ℒ"), { text: "rank", options: { bold: true, subscript: true } }, r(" = 𝔼"), sub("(i,j)∼P×N"), r("[softplus(s"), sub("j"), r(" − s"), sub("i"), r(")]  ranks labeled anomalies above presumed normals: a ranking under heavy imbalance, not a classifier."),
], x + PAD, cy + 1.2, cw, y + h - (cy + 1.2) - 0.3, { paraSpaceAfter: 6 });

// =============================== COLUMN 3 ==================================
x = COLX[2]; y = COL_TOP;

// -- Main results
h = 11.9;
cy = panel(x, y, COLW, h, "Main results (10 seeds)");
const c3 = (cw - 0.6) / 3;
const call = (cx, num, label) => {
  slide.addText(num, { x: cx, y: cy - 0.1, w: c3, h: 1.0, fontFace: THEME.headFontFace, fontSize: T.callout, bold: true, color: AMBER, align: "center", margin: 0, isTextBox: true, valign: "bottom", objectName: "callout " + label });
  body("callout label " + label, [r(label)], cx, cy + 0.95, c3, 0.8, { fontSize: 20, color: MUTED, align: "center" });
};
call(x + PAD, "1.2", "best avg. AUPRC rank (next best 1.8)");
call(x + PAD + c3 + 0.3, "0.659", "LANL AUPRC (next best 0.525)");
call(x + PAD + 2 * (c3 + 0.3), "3 / 3", "H&M metrics where RAD is best");

const rh = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center", valign: "middle", fontSize: 12 } });
const grp = (t, span) => ({ text: t, options: { bold: true, color: NAVY, fill: { color: GROUND }, align: "center", valign: "middle", colspan: span, fontSize: 15 } });
const num = (t, best) => ({ text: t, options: { align: "center", valign: "middle", bold: !!best, color: best ? AMBER : INK } });
const nm = (t, rad) => ({ text: t, options: { align: "left", valign: "middle", bold: !!rad, color: rad ? AMBER : INK } });
const rows = [
  ["RAD (no BCE)", ".996", ".659", ".601", ".693", ".043", ".050", ".828", ".143", ".129", "1.2", true, [0,1,2,4,6,7,8]],
  ["RAD (BCE)",           ".860", ".557", ".572", ".695", ".043", ".051", ".787", ".110", ".125", "1.8", true, [4]],
  ["RDL weighted",        ".553", ".007", ".021", ".720", ".037", ".062", ".700", ".035", ".064", "4.8", false, [3,5]],
  ["RDL unweighted",      ".685", ".015", ".026", ".713", ".037", ".062", ".693", ".039", ".064", "3.8", false, [5]],
  ["AE + Rank",           ".943", ".525", ".540", ".518", ".016", ".015", ".456", ".013", ".005", "5.2", false, []],
  ["DRL",                 ".673", ".004", ".005", ".463", ".013", ".003", ".456", ".013", ".009", "8.2", false, []],
  ["AE",                  ".888", ".009", ".000", ".474", ".014", ".014", ".408", ".012", ".007", "7.3", false, []],
  ["Deep SVDD",           ".736", ".005", ".001", ".458", ".013", ".003", ".412", ".012", ".002", "8.7", false, []],
  ["Isolation Forest",    ".644", ".003", ".000", ".450", ".013", ".001", ".402", ".012", ".004", "9.3", false, []],
  ["Supervised MLP",      ".965", ".050", ".093", ".659", ".027", ".041", ".642", ".027", ".039", "4.7", false, []],
];
const tableRows = [
  [grp("", 1), grp("LANL", 3), grp("Amazon", 3), grp("H&M", 3), grp("Rank", 1)],
  [rh("Model"), rh("AUROC"), rh("AUPRC"), rh("P@k"), rh("AUROC"), rh("AUPRC"), rh("P@k"), rh("AUROC"), rh("AUPRC"), rh("P@k"), rh("AUPRC")],
].concat(rows.map(([name, ...vals]) => {
  const metrics = vals.slice(0, 9), rank = vals[9], rad = vals[10], best = vals[11];
  return [nm(name, rad)].concat(metrics.map((v, i) => num(v, best.includes(i)))).concat([num(rank, rad)]);
}));
const nameW = 2.1, restW = (cw - nameW) / 10, ROWH = 0.5;
slide.addTable(tableRows, {
  x: x + PAD, y: cy + 1.95, w: cw, colW: [nameW].concat(Array(10).fill(restW)),
  fontSize: T.table, color: INK, border: { type: "solid", color: LINE, pt: 0.75 }, rowH: ROWH, margin: 0.04, objectName: "main results table",
});
body("c3 results note", [
  b("Amber"), r(" = best per column; Rank = average AUPRC rank over the three datasets. AUPRC and P@k matter most under severe imbalance: analysts inspect only the top of the list. On Amazon, supervised RDL wins AUROC and P@k while RAD wins AUPRC; on LANL and H&M, RAD leads clearly. Avg. AUROC rank: RAD 2.0, next best 3.3."),
], x + PAD, cy + 1.95 + 12 * ROWH + 0.25, cw, y + h - (cy + 1.95 + 12 * ROWH + 0.25) - 0.3, { fontSize: T.small });
y += h + GAP;

// -- Ablations
const f3w = cw, f3h = f3w * 2498 / 5664;
h = 9.6;
cy = panel(x, y, COLW, h, "Ablations (RQ2–RQ4)");
slide.addImage({ path: S + "fig3_vec-1.png", x: x + PAD, y: cy, w: f3w, h: f3h, objectName: "Figure: ablations" });
bullets("c3 ablation bullets", [
  "Rules (RQ2): compact predicate rules beat a rule-free GAE and raw random-forest feature injection, most clearly on AUPRC and P@k.",
  "LLM refinement (RQ4) helps most on LANL; elsewhere RF rules are competitive, so read it as rule compaction.",
  "Edge reconstruction (RQ3): the no-BCE variant is strongest overall; BCE hurts on LANL (noisy sampled topology), is neutral on Amazon, and loses on H&M.",
], x + PAD, cy + f3h + 0.2, cw, y + h - (cy + f3h + 0.2) - 0.3, { fontSize: 22 });
y += h + GAP;

// -- Prevalence sensitivity (native charts)
h = 8.1;
cy = panel(x, y, COLW, h, "Label-prevalence sweep (RQ5)");
const chartW = (cw - 0.3) / 2, chartH = 4.5;
const labels = ["0.5%", "1.5%", "3.0%"];
const chartOpts = (title) => ({
  showTitle: true, title, titleFontSize: 20, titleColor: NAVY, titleFontFace: "+mj-lt",
  chartColors: [AMBER, BLUE], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 11,
  showLegend: true, legendPos: "b", legendFontSize: 16, legendColor: INK, legendFontFace: "+mn-lt",
  catAxisLabelFontSize: 16, catAxisLabelColor: MUTED, catAxisLabelFontFace: "+mn-lt", catAxisTitle: "target anomaly rate", showCatAxisTitle: true, catAxisTitleFontSize: 15, catAxisTitleColor: MUTED, catAxisTitleFontFace: "+mn-lt",
  valAxisLabelFontSize: 16, valAxisLabelColor: MUTED, valAxisLabelFontFace: "+mn-lt", valAxisLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 0.06, valAxisMajorUnit: 0.02,
  valAxisTitle: "AUPRC", showValAxisTitle: true, valAxisTitleFontSize: 15, valAxisTitleColor: MUTED, valAxisTitleFontFace: "+mn-lt",
  valGridLine: { color: LINE, size: 0.75 }, catGridLine: { style: "none" },
  showValue: false, plotArea: { fill: { color: "FFFFFF" } }, chartArea: { fill: { color: "FFFFFF" } },
});
slide.addChart(pres.ChartType.line, [
  { name: "RAD", labels, values: [0.0073, 0.0272, 0.0541] },
  { name: "Supervised RDL", labels, values: [0.0061, 0.0198, 0.0408] },
], Object.assign({ x: x + PAD, y: cy, w: chartW, h: chartH, objectName: "chart: Amazon prevalence sweep" }, chartOpts("Amazon")));
slide.addChart(pres.ChartType.line, [
  { name: "RAD", labels, values: [0.0127, 0.0199, 0.0410] },
  { name: "Supervised RDL", labels, values: [0.0047, 0.0161, 0.0382] },
], Object.assign({ x: x + PAD + chartW + 0.3, y: cy, w: chartW, h: chartH, objectName: "chart: H&M prevalence sweep" }, chartOpts("H&M")));
body("c3 sweep note", [
  r("Only the label-filtering rate changes (retained positives: Amazon 520 / 1575 / 3198; H&M 71 / 214 / 434); graph, features, model, and protocol are fixed. RAD stays above supervised RDL on AUPRC at every rate, so the 1.5% default is not a cherry-picked threshold."),
], x + PAD, cy + chartH + 0.1, cw, y + h - (cy + chartH + 0.1) - 0.3, { fontSize: T.small });
y += h + GAP;

// -- Takeaways (dark)
h = COL_BOTTOM - y;
cy = panel(x, y, COLW, h, "Takeaways", { dark: true });
bullets("c3 takeaways", [
  "Relational anomaly detection is a joint problem of representation, scoring, and behavioral evidence; flattening alone or structure alone is not enough.",
  "Where symbolic evidence enters matters: rules injected before message passing beat post-hoc fusion.",
  "Ranking supervision drives retrieval under severe imbalance; edge reconstruction is dataset-dependent and should stay optional.",
  "Limits: rule mining needs labels and is bounded by the flattened view; Amazon/H&M are evaluated on a validation partition (RelBench withholds test labels); LANL uses sampled subgraphs.",
], x + PAD, cy, cw, y + h - cy - 0.3, { color: "FFFFFF", paraSpaceAfter: 9 });

// =============================== FOOTER ====================================
slide.addText("Supported by the National Security Agency (H98230-23-C-0279) and the National Science Foundation (IIS-2239881, ECCS-2325417, IIS-2524380, DGE-2622415).",
  { x: M, y: COL_BOTTOM + 0.2, w: W - 2 * M, h: 0.5, fontSize: 18, color: MUTED, align: "center", margin: 0, isTextBox: true, objectName: "footer" });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
  if (warnings.length) { console.log("FIT WARNINGS:"); warnings.forEach(w => console.log("  " + w)); } else console.log("no fit warnings");
})();
