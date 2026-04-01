// Wolfson Architecture Slides Generator v2
// Fixes: arrow directions, layout overflow, routing

const pptxgen = require("/opt/homebrew/lib/node_modules/pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10" × 5.625"
pres.title = "Wolfson Architecture";

// ─── Palette ──────────────────────────────────────────────────────────────────
const C = {
  bg:         "1E2761",
  boxA:       "2A3570",
  boxB:       "162050",
  accent:     "253880",
  border:     "CADCFC",
  titleText:  "FFFFFF",
  labelText:  "FFFFFF",
  detailText: "CADCFC",
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function box(slide, x, y, w, h, label, details, opts = {}) {
  const fill = opts.accent ? C.accent : (opts.dark ? C.boxB : C.boxA);
  const borderWidth = opts.accent ? 2 : 1.5;

  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color: fill },
    line: { color: C.border, width: borderWidth },
    rectRadius: 0.08,
  });

  if (!details || details.length === 0) {
    slide.addText(label, {
      x, y, w, h,
      fontSize: 12,
      bold: true,
      color: C.labelText,
      fontFace: "Calibri",
      align: "center",
      valign: "middle",
      margin: 4,
    });
  } else {
    const labelH = 0.26;
    const detailY = y + labelH + 0.04;
    const detailH = h - labelH - 0.04;

    slide.addText(label, {
      x: x + 0.05, y, w: w - 0.1, h: labelH,
      fontSize: 11,
      bold: true,
      color: C.labelText,
      fontFace: "Calibri",
      align: "center",
      valign: "middle",
      margin: 0,
    });

    const richDetails = details.map((d, i) => ({
      text: d,
      options: {
        fontSize: 8.5,
        color: C.detailText,
        fontFace: "Calibri",
        breakLine: i < details.length - 1,
      },
    }));
    slide.addText(richDetails, {
      x: x + 0.08, y: detailY, w: w - 0.16, h: detailH,
      align: "left",
      valign: "top",
      margin: 2,
    });
  }
}

// Arrow: left-to-right horizontal. x1 < x2.
function arrowH(slide, x1, x2, y) {
  slide.addShape(pres.shapes.LINE, {
    x: x1, y, w: x2 - x1, h: 0,
    line: { color: C.border, width: 1.5, endArrowType: "arrow" },
  });
}

// Arrow: top-to-bottom vertical. y1 < y2.
function arrowV(slide, x, y1, y2) {
  slide.addShape(pres.shapes.LINE, {
    x, y: y1, w: 0, h: y2 - y1,
    line: { color: C.border, width: 1.5, endArrowType: "arrow" },
  });
}

// Line segment (no arrow)
function line(slide, x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  slide.addShape(pres.shapes.LINE, {
    x: x1, y: y1, w: dx, h: dy,
    line: { color: C.border, width: 1.5 },
  });
}

// Elbow connector: goes down from (x1,y1) to yMid, across to x2, then down to y2 (with arrow)
function elbowDown(slide, x1, y1, x2, y2) {
  const yMid = (y1 + y2) / 2;
  line(slide, x1, y1, x1, yMid);
  line(slide, x1, yMid, x2, yMid);
  arrowV(slide, x2, yMid, y2);
}

function accentBar(slide) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.055,
    fill: { color: C.border },
    line: { color: C.border, width: 0 },
  });
}

function slideTitle(slide, title) {
  slide.addText(title, {
    x: 0.35, y: 0.07, w: 9.3, h: 0.35,
    fontSize: 13,
    bold: true,
    color: C.titleText,
    fontFace: "Calibri",
    align: "left",
    valign: "middle",
    margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SLIDE 1 — Input Pipeline
// ═══════════════════════════════════════════════════════════════════════════════

const s1 = pres.addSlide();
s1.background = { color: C.bg };
accentBar(s1);
slideTitle(s1, "Wolfson — Input Pipeline");

// ── Layout constants ──────────────────────────────────────────────────────────
// Slide: 10" × 5.625"
// Top row: Bass → MidiListener → PhraseDetector → PhraseAnalyzer
// Side branch from MidiListener downward: BeatEstimator
// Middle: PhraseMemory (centred)
// Bottom: ArcController (wide, centred)

const g = 0.14;   // gap for arrow clearance

// Standard box dimensions
const bw = 1.55;
const bh = 0.48;

// Row Y values
const r1y = 0.60;   // main top row
const r2y = 1.65;   // BeatEstimator
const r3y = 2.75;   // PhraseMemory
const r4y = 4.10;   // ArcController

// Col X (left edges) for top row: 4 boxes
// Need to fit in ~0.3 to 9.7 with arrows between
// PhraseAnalyzer is at far right and wider, so:
const c0x = 0.30;           // Bass
const c1x = 2.15;           // MidiListener
const c2x = 4.00;           // PhraseDetector

// PhraseAnalyzer: wide, taller, at right
const paW = 2.80;
const paH = 1.50;
const paX = 6.95;
const paY = r1y - 0.08;

// Draw top-row boxes
box(s1, c0x, r1y, bw, bh, "Bass (pitch-to-MIDI)");
box(s1, c1x, r1y, bw, bh, "MidiListener");
box(s1, c2x, r1y, bw, bh, "PhraseDetector");
box(s1, paX, paY, paW, paH, "PhraseAnalyzer", [
  "contour, density",
  "Q&A type, swing",
  "dynamics, energy profile",
  "pitch classes",
  "interval motifs, lyrical motifs",
], { accent: true });

// Top-row arrows (left-to-right): Bass → MidiListener → PhraseDetector → PhraseAnalyzer
arrowH(s1, c0x + bw + g, c1x - g, r1y + bh / 2);
arrowH(s1, c1x + bw + g, c2x - g, r1y + bh / 2);
arrowH(s1, c2x + bw + g, paX - g, r1y + bh / 2);

// BeatEstimator (branch down from MidiListener)
const beX = c1x;
const beY = r2y;
box(s1, beX, beY, bw, bh, "BeatEstimator", ["live tempo"]);
arrowV(s1, c1x + bw / 2, r1y + bh + g, beY - g);

// PhraseMemory (centred)
const pmW = 2.70;
const pmH = 0.95;
const pmX = (10 - pmW) / 2;
const pmY = r3y;
box(s1, pmX, pmY, pmW, pmH, "PhraseMemory", [
  "stores phrases + motifs",
  "stores lyrical motifs",
  "both voices (bass + sax)",
], { dark: true });

// PhraseAnalyzer → PhraseMemory
// Route: from PA left side, horizontally left, with arrow pointing left into PM right side.
// PA left side x = paX. PM right side x = pmX + pmW.
// Both are at roughly the same vertical region (paY + paH/2 vs pmY + pmH/2).
// We'll route: horizontal from PA left side to a junction at PA-left, then down to PM mid-y, then left to PM right.
// Simplest: PA bottom-left corner → down to pmMidY → left to PM right edge.
{
  const junctionX = paX + 0.30;         // slightly inside PA left edge for clean exit
  const startY    = paY + paH + g;      // just below PA bottom
  const pmRightX  = pmX + pmW;
  const pmMidY    = pmY + pmH / 2;
  // Vertical segment down from PA bottom
  line(s1, junctionX, startY, junctionX, pmMidY);
  // Horizontal segment left to PM right side: start at junctionX, end at pmRightX+g
  // Arrow head should be at pmRightX (going left). We must draw rightward (pmRightX→junctionX)
  // with endArrowType so head is at junctionX... no.
  // Use a downward-then-right L: start the horiz from pmRightX+g going right to junctionX, beginArrow=arrow
  // Visually: the line goes from junctionX leftward to pmRightX, arrow points left (into PM).
  s1.addShape(pres.shapes.LINE, {
    x: pmRightX + g, y: pmMidY,
    w: junctionX - pmRightX - g, h: 0,
    line: { color: C.border, width: 1.5, beginArrowType: "arrow" },
  });
}

// BeatEstimator → PhraseMemory
// BE bottom-centre → elbow → PM left side mid
{
  const srcX = beX + bw / 2;
  const srcBotY = beY + bh + g;
  const dstLeftX = pmX;
  const dstMidY = pmY + pmH / 2;
  line(s1, srcX, srcBotY, srcX, dstMidY);
  arrowH(s1, srcX, dstLeftX, dstMidY);
}

// ArcController
const acW = 4.50;
const acH = 1.00;
const acX = (10 - acW) / 2;
const acY = r4y;
box(s1, acX, acY, acW, acH, "ArcController", [
  "5-min performance arc  ·  leadership & proactive mode",
  "bass pitch-class tracking  ·  energy arc selection",
  "motif + lyrical motif selection  ·  register contrast scheduling",
  "rhythmic density + complementarity  ·  stage swing baseline",
], { accent: true });

// PhraseMemory → ArcController
arrowV(s1, pmX + pmW / 2, pmY + pmH + g, acY - g);

// Annotation
s1.addText("continues on Slide 2 →", {
  x: acX + acW + 0.15, y: acY + acH / 2 - 0.13, w: 2.2, h: 0.26,
  fontSize: 8.5,
  color: C.detailText,
  fontFace: "Calibri",
  italic: true,
  align: "left",
  valign: "middle",
  margin: 0,
});


// ═══════════════════════════════════════════════════════════════════════════════
//  SLIDE 2 — Generation & Output
// ═══════════════════════════════════════════════════════════════════════════════

const s2 = pres.addSlide();
s2.background = { color: C.bg };
accentBar(s2);
slideTitle(s2, "Wolfson — Generation & Output");

// ── Layout ────────────────────────────────────────────────────────────────────
// Slide: 10" × 5.625"
// All rows measured from top; chain (thinning→MIDI→Synth) must fit within 5.55"
//
// Planned layout (recalc to give visible arrows ~0.22" each):
//   Row1: ArcController     y=0.52  h=0.44  bot=0.96
//   arrow 0.22
//   Row2: HarmonyController y=1.18  h=0.52  bot=1.70
//   arrow 0.22
//   Row3: PhraseGenerator   y=1.92  h=1.18  bot=3.10
//   split stem 0.28
//   Row4: Output boxes      y=3.48  h=0.65  bot=4.13
//   MidiOutput gap 0.12     y=4.25  h=0.46  bot=4.71
//   Synth gap 0.11          y=4.82  h=0.36  bot=5.18  ✓ fits

const s2g = 0.08;  // arrow clearance from box edge

const s2r1y = 0.52;
const s2r2y = 1.18;
const s2r3y = 1.92;
const outY  = 3.48;
const outH  = 0.65;

// ArcController (carried over) ─────────────────────────────────────────────────
const a2W = 3.60;
const a2H = 0.44;
const a2X = (10 - a2W) / 2;
box(s2, a2X, s2r1y, a2W, a2H, "ArcController (carried over)", [], { accent: true });

// HarmonyController ────────────────────────────────────────────────────────────
const hcW = 3.00;
const hcH = 0.52;
const hcX = (10 - hcW) / 2;
box(s2, hcX, s2r2y, hcW, hcH, "HarmonyController", [
  "mode, progression, pedal  ·  tritone substitution",
]);

// Arrow: ArcController → HarmonyController
arrowV(s2, 5.0, s2r1y + a2H + s2g, s2r2y - s2g);

// PhraseGenerator (LSTM) ───────────────────────────────────────────────────────
const pgW = 6.20;
const pgH = 1.18;
const pgX = (10 - pgW) / 2;
box(s2, pgX, s2r3y, pgW, pgH, "PhraseGenerator (LSTM)", [
  "LSTM + chord conditioning  ·  pitch range limits + register gravity",
  "register contrast  ·  scale pitch bias  ·  contour steering",
  "swing/triplet bias  ·  energy arc shaping",
  "long-note penalty + singable duration bias  ·  motivic development",
  "voice leading  ·  modal leap bonus (P4/P5)  ·  repetition penalty",
  "rest injection  ·  beat accumulator",
], { accent: true });

// Arrow: HarmonyController → PhraseGenerator
arrowV(s2, 5.0, s2r2y + hcH + s2g, s2r3y - s2g);

// ── Three output branches ──────────────────────────────────────────────────────

// Branch 1: PhraseMemory (left)
const pm2W = 2.20;
const pm2X = 0.30;
box(s2, pm2X, outY, pm2W, outH, "PhraseMemory", [
  "motifs, recall,",
  "self-play seed",
]);

// Branch 2: WebAudienceDisplay / OscOutput (centre)
const disp2W = 2.70;
const disp2X = (10 - disp2W) / 2;
box(s2, disp2X, outY, disp2W, outH, "WebAudienceDisplay / OscOutput", [
  "always sees full",
  "intended phrase",
]);

// Branch 3: Performance thinning (right) → chain vertically to MidiOutput → Synth
const thinW = 1.90;
const thinX = 6.85;
box(s2, thinX, outY, thinW, outH, "Performance thinning", [
  "short notes dropped",
  "stochastically",
]);

// MidiOutput (below thinning)
const thinBotY = outY + outH;
const midi2Y   = thinBotY + 0.12;
const midi2H   = 0.46;
box(s2, thinX, midi2Y, thinW, midi2H, "MidiOutput", [
  "energy arc × peak × end taper",
]);
arrowV(s2, thinX + thinW / 2, thinBotY + s2g, midi2Y - s2g);

// Synth (below MidiOutput)
const midi2Bot  = midi2Y + midi2H;
const synth2Y   = midi2Bot + 0.11;
const synthH2   = 0.36;
const synthW2   = 1.55;
const synth2X   = thinX + (thinW - synthW2) / 2;
box(s2, synth2X, synth2Y, synthW2, synthH2, "Synth (sax voice)");
arrowV(s2, thinX + thinW / 2, midi2Bot + s2g, synth2Y - s2g);

// ── PhraseGenerator T-split ────────────────────────────────────────────────────
const pgBotX  = pgX + pgW / 2;
const pgBotY  = s2r3y + pgH;
const splitY  = outY - 0.25;

// Vertical stem from PG bottom
line(s2, pgBotX, pgBotY + s2g, pgBotX, splitY);

// Horizontal bar of T
const leftBranchX  = pm2X + pm2W / 2;
const midBranchX   = disp2X + disp2W / 2;
const rightBranchX = thinX + thinW / 2;
line(s2, leftBranchX, splitY, rightBranchX, splitY);

// Arrows down to each branch box
arrowV(s2, leftBranchX,  splitY, outY - s2g);
arrowV(s2, midBranchX,   splitY, outY - s2g);
arrowV(s2, rightBranchX, splitY, outY - s2g);

// ── Annotation ────────────────────────────────────────────────────────────────
s2.addText("← carried over from Slide 1", {
  x: a2X + a2W + 0.15, y: s2r1y + a2H / 2 - 0.12, w: 2.5, h: 0.25,
  fontSize: 8.5,
  color: C.detailText,
  fontFace: "Calibri",
  italic: true,
  align: "left",
  valign: "middle",
  margin: 0,
});

// ═══════════════════════════════════════════════════════════════════════════════
//  Write file
// ═══════════════════════════════════════════════════════════════════════════════

pres.writeFile({ fileName: "/Users/davidderoure/wolfson/docs/architecture.pptx" })
  .then(() => console.log("Saved: /Users/davidderoure/wolfson/docs/architecture.pptx"))
  .catch(err => { console.error("Error:", err); process.exit(1); });
