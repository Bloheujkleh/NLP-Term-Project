import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const HOME = process.env.HOME || process.env.USERPROFILE || "";
const ARTIFACT_UTILS = path.join(
  HOME,
  ".codex",
  "plugins",
  "cache",
  "openai-primary-runtime",
  "presentations",
  "26.513.11550",
  "skills",
  "presentations",
  "scripts",
  "artifact_tool_utils.mjs",
);

const { ensureArtifactToolWorkspace, importArtifactTool, saveBlobToFile } = await import(pathToFileURL(ARTIFACT_UTILS).href);

const WORKSPACE = path.join(ROOT, "outputs", "final-presentation");
const PREVIEW_DIR = path.join(WORKSPACE, "preview");
const OUT = path.join(ROOT, "deliverables", "Turkish_Legal_RAG_Presentation.pptx");

await ensureArtifactToolWorkspace(WORKSPACE);
const artifact = await importArtifactTool(WORKSPACE);
const { Presentation, PresentationFile, paint, stroke } = artifact;

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });

const C = {
  ink: "#0B2545",
  blue: "#2E74B5",
  pale: "#F4F7FB",
  line: "#D6DEE9",
  text: "#1D2939",
  muted: "#667085",
  gold: "#B7791F",
  green: "#1B7F5A",
  red: "#B42318",
  white: "#FFFFFF",
};

function addBox(slide, x, y, w, h, fill = C.white, line = C.line) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: w, height: h },
    fill: paint(fill),
    line: stroke(line),
  });
}

function addText(slide, value, x, y, w, h, opts = {}) {
  const shape = addBox(slide, x, y, w, h, opts.fill || C.white, opts.line || (opts.fill || C.white));
  shape.text.set(value);
  shape.text.fontSize = opts.size || 24;
  shape.text.typeface = opts.font || "Aptos";
  shape.text.color = opts.color || C.text;
  shape.text.bold = Boolean(opts.bold);
  shape.text.alignment = opts.align || "left";
  shape.text.verticalAlignment = opts.valign || "mid";
  shape.text.insets = opts.insets || { left: 10, right: 10, top: 6, bottom: 6 };
  return shape;
}

function title(slide, kicker, claim) {
  addText(slide, kicker.toUpperCase(), 64, 38, 360, 28, { size: 13, bold: true, color: C.blue, fill: C.pale, line: C.pale });
  addText(slide, claim, 64, 72, 1010, 72, { size: 30, bold: true, color: C.ink });
  addBox(slide, 64, 150, 1110, 2, C.blue, C.blue);
}

function footer(slide, n) {
  addText(slide, `Turkish Legal RAG | CENG493 | ${n}`, 1010, 675, 190, 24, { size: 10, color: C.muted, align: "right" });
}

function metricCard(slide, x, y, w, h, value, label, color = C.ink) {
  addBox(slide, x, y, w, h, C.pale, C.line);
  addText(slide, value, x + 14, y + 14, w - 28, 44, { size: 30, bold: true, color, fill: C.pale, line: C.pale });
  addText(slide, label, x + 14, y + 60, w - 28, 42, { size: 14, color: C.muted, fill: C.pale, line: C.pale });
}

function bar(slide, x, y, maxW, label, value, color) {
  addText(slide, label, x, y - 2, 230, 26, { size: 14, color: C.text });
  addBox(slide, x + 240, y, maxW, 20, "#E9EEF5", "#E9EEF5");
  addBox(slide, x + 240, y, maxW * value, 20, color, color);
  addText(slide, value.toFixed(3), x + 250 + maxW, y - 3, 90, 26, { size: 14, bold: true, color: color });
}

function slide01() {
  const slide = deck.slides.add();
  addBox(slide, 0, 0, 1280, 720, C.ink, C.ink);
  addText(slide, "Improving Turkish Legal Question Answering", 70, 120, 980, 64, { size: 40, bold: true, color: C.white, fill: C.ink, line: C.ink });
  addText(slide, "with an Optimized RAG Pipeline", 70, 188, 850, 58, { size: 34, bold: true, color: "#D7E3F4", fill: C.ink, line: C.ink });
  addText(slide, "CENG493 Term Project", 74, 275, 420, 34, { size: 20, color: "#D7E3F4", fill: C.ink, line: C.ink });
  metricCard(slide, 74, 410, 220, 120, "7,579", "legal corpus chunks", C.blue);
  metricCard(slide, 322, 410, 220, 120, "1,000", "retrieval eval queries", C.blue);
  metricCard(slide, 570, 410, 220, 120, "240", "gold QA questions", C.blue);
  metricCard(slide, 818, 410, 220, 120, "0.975", "BM25 Recall@10", C.green);
  footer(slide, 1);
}

function slide02() {
  const slide = deck.slides.add();
  title(slide, "Problem", "Legal QA must be evaluated for grounding, not only fluent answers.");
  addText(slide, "Input", 90, 220, 210, 40, { size: 20, bold: true, color: C.blue });
  addText(slide, "A Turkish legal question", 90, 265, 380, 54, { size: 24, bold: true, fill: C.pale, line: C.line });
  addText(slide, "Output", 760, 220, 210, 40, { size: 20, bold: true, color: C.blue });
  addText(slide, "A source-supported answer with citation", 760, 265, 380, 54, { size: 24, bold: true, fill: C.pale, line: C.line });
  addBox(slide, 505, 284, 190, 4, C.gold, C.gold);
  addText(slide, "Risk: fluent legal hallucinations can look credible without being grounded in any source.", 180, 440, 860, 76, { size: 24, bold: true, color: C.ink, fill: "#FFF7E6", line: "#F0C36A", align: "center" });
  footer(slide, 2);
}

function slide03() {
  const slide = deck.slides.add();
  title(slide, "Dataset", "The provided data supports full Scenario 1 evaluation.");
  const cards = [
    ["corpus.jsonl", "7,579", "retrieval corpus"],
    ["rag_eval.json", "1,000", "gold chunk retrieval"],
    ["gold_benchmark.json", "240", "verified QA + sources"],
    ["embedding.jsonl", "2,059", "triplet tuning data"],
    ["reranker.jsonl", "6,752", "cross-encoder pairs"],
    ["llm.jsonl", "13,758", "SFT examples"],
  ];
  cards.forEach((c, i) => {
    const x = 80 + (i % 3) * 370;
    const y = 210 + Math.floor(i / 3) * 170;
    addBox(slide, x, y, 320, 120, C.pale, C.line);
    addText(slide, c[0], x + 18, y + 16, 280, 26, { size: 17, bold: true, color: C.ink, fill: C.pale, line: C.pale });
    addText(slide, c[1], x + 18, y + 46, 120, 44, { size: 30, bold: true, color: C.blue, fill: C.pale, line: C.pale });
    addText(slide, c[2], x + 130, y + 54, 160, 32, { size: 15, color: C.muted, fill: C.pale, line: C.pale });
  });
  footer(slide, 3);
}

function slide04() {
  const slide = deck.slides.add();
  title(slide, "Architecture", "The pipeline separates retrieval, ranking, and grounded answer generation.");
  const steps = ["Question", "Retriever", "Top-k chunks", "Answer generator", "Cited answer"];
  steps.forEach((s, i) => {
    const x = 70 + i * 235;
    addBox(slide, x, 290, 180, 80, i === 1 ? "#E8F1FB" : C.pale, C.blue);
    addText(slide, s, x + 12, 312, 156, 34, { size: 18, bold: true, color: C.ink, fill: i === 1 ? "#E8F1FB" : C.pale, line: i === 1 ? "#E8F1FB" : C.pale, align: "center" });
    if (i < steps.length - 1) addBox(slide, x + 184, 327, 46, 4, C.gold, C.gold);
  });
  addText(slide, "Implemented: BM25, dense MiniLM, hybrid retrieval, embedding fine-tuning, reranker fine-tuning, LLM/SFT smoke training, QA metrics, and judge-based faithfulness.", 120, 470, 1040, 70, { size: 20, color: C.text, fill: C.white, line: C.white, align: "center" });
  footer(slide, 4);
}

function slide05() {
  const slide = deck.slides.add();
  title(slide, "Retrieval", "BM25 is the strongest baseline on Turkish legal text.");
  bar(slide, 120, 245, 520, "BM25 Recall@10", 0.975, C.green);
  bar(slide, 120, 305, 520, "Dense Recall@10", 0.676, C.red);
  bar(slide, 120, 365, 520, "Hybrid Recall@10", 0.969, C.blue);
  metricCard(slide, 820, 230, 250, 115, "0.863", "BM25 MRR", C.green);
  metricCard(slide, 820, 370, 250, 115, "0.890", "BM25 nDCG@10", C.green);
  addText(slide, "Interpretation: exact legal terms, article numbers and court references make lexical retrieval very competitive.", 130, 555, 930, 54, { size: 20, color: C.ink, fill: "#F6F8FA", line: C.line });
  footer(slide, 5);
}

function slide06() {
  const slide = deck.slides.add();
  title(slide, "Reranker", "Legal-domain fine-tuning fixes much of the pretrained mismatch.");
  metricCard(slide, 90, 225, 240, 120, "0.810", "Pretrained Recall@10, 100q", C.red);
  metricCard(slide, 365, 225, 240, 120, "0.970", "Fine-tuned Recall@10, 100q", C.green);
  metricCard(slide, 640, 225, 240, 120, "0.915", "Fine-tuned Recall@10, 1000q", C.blue);
  metricCard(slide, 915, 225, 240, 120, "0.975", "BM25 Recall@10, 1000q", C.green);
  addText(slide, "Conclusion: reranker fine-tuning works, but raw BM25 remains strongest on the full benchmark, so the live demo keeps BM25 ranking.", 150, 440, 900, 92, { size: 22, bold: true, color: C.ink, fill: "#EAF7F1", line: "#A6D8BF", align: "center" });
  footer(slide, 6);
}

function slide07() {
  const slide = deck.slides.add();
  title(slide, "Embedding Tuning", "Dense fine-tuning must be validated, not assumed to help.");
  metricCard(slide, 130, 235, 250, 120, "0.676", "Base dense Recall@10", C.blue);
  metricCard(slide, 445, 235, 250, 120, "0.591", "CPU triplet tuned Recall@10", C.red);
  metricCard(slide, 760, 235, 250, 120, "41 min", "CPU training runtime", C.gold);
  addText(slide, "Result: the naive triplet setup degraded dense retrieval. This is a useful ablation and justifies keeping BM25 in the final demo.", 145, 455, 900, 84, { size: 23, bold: true, color: C.ink, fill: "#FFF1F0", line: "#FDA29B", align: "center" });
  footer(slide, 7);
}

function slide08() {
  const slide = deck.slides.add();
  title(slide, "QA Evaluation", "Grounded extractive answers are faithful but bounded by source ranking.");
  metricCard(slide, 90, 230, 210, 120, "0.799", "Token F1", C.blue);
  metricCard(slide, 330, 230, 210, 120, "0.908", "Top-5 source hit", C.green);
  metricCard(slide, 570, 230, 210, 120, "0.813", "Citation accuracy", C.gold);
  metricCard(slide, 810, 230, 210, 120, "0.961", "Faithfulness proxy", C.green);
  addText(slide, "Judge-based faithfulness adds a stricter semantic check: 206 of 240 answers were source-supported, score 0.858.", 160, 455, 860, 64, { size: 22, bold: true, color: C.ink, fill: "#F6F8FA", line: C.line, align: "center" });
  footer(slide, 8);
}

function slide09() {
  const slide = deck.slides.add();
  title(slide, "Error Analysis", "Failures split cleanly into retrieval and ranking problems.");
  metricCard(slide, 210, 230, 280, 130, "22", "gold source missing from top-5", C.red);
  metricCard(slide, 650, 230, 280, 130, "23", "gold source in top-5 but not top-1", C.gold);
  addText(slide, "Optimization map: first-stage retrieval should address the 22 misses; a domain-tuned reranker should address the 23 ranking failures.", 150, 455, 930, 78, { size: 24, bold: true, color: C.ink, fill: C.white, line: C.white, align: "center" });
  footer(slide, 9);
}

function slide10() {
  const slide = deck.slides.add();
  title(slide, "Conclusion", "The project is reproducible, measured, and demo-ready.");
  const items = [
    "BM25 is the current strongest baseline.",
    "Embedding and reranker fine-tuning were run and evaluated on CPU.",
    "LLM/SFT smoke training works but is not reliable enough for live demo.",
    "Judge faithfulness and error analysis make grounding auditable.",
  ];
  items.forEach((item, i) => {
    addBox(slide, 110, 215 + i * 85, 920, 56, i === 0 ? "#EAF7F1" : C.pale, C.line);
    addText(slide, item, 140, 226 + i * 85, 860, 34, { size: 22, bold: i === 0, color: C.ink, fill: i === 0 ? "#EAF7F1" : C.pale, line: i === 0 ? "#EAF7F1" : C.pale });
  });
  footer(slide, 10);
}

[slide01, slide02, slide03, slide04, slide05, slide06, slide07, slide08, slide09, slide10].forEach(fn => fn());

await fs.mkdir(path.dirname(OUT), { recursive: true });
await fs.mkdir(PREVIEW_DIR, { recursive: true });
for (let i = 0; i < deck.slides.count; i += 1) {
  const slide = deck.slides.getItem(i);
  const png = await deck.export({ slide, format: "png", scale: 1 });
  await saveBlobToFile(png, path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`));
}
const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(OUT);
console.log(`Wrote ${OUT}`);
