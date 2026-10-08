// Builds MuleRadar_Submission.docx from the measured results. Run: node build_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, ImageRun, LevelFormat, Footer, Header, PageNumber, ExternalHyperlink, TabStopType,
} = require("docx");

const R = path.join(__dirname, "..", "results");
const tr = JSON.parse(fs.readFileSync(path.join(R, "train_results.json")));
const ev = JSON.parse(fs.readFileSync(path.join(R, "eval_rings.json")));
const bench = JSON.parse(fs.readFileSync(path.join(R, "bench.json")));
const pct = (x, d = 1) => (x * 100).toFixed(d);
const L = Object.values(tr.ladder), full = L[L.length - 1];
const b2048 = bench.rows.find(r => r.batch === 2048), bMax = bench.rows[bench.rows.length - 1];
const acc = ev.accounts, rings = ev.rings;
const k = n => n.toLocaleString("en-US");

const INK = "0F1B26", MUTED = "56687A", ACCENT = "0B6E63", SOFT = "E3F2EF", LINE = "D7DEE4", SUNK = "F1F4F6";
const FONT = "Arial";
const W = 9638; // A4 text width with 2 cm margins (DXA)

// ---------- helpers ----------
const t = (text, o = {}) => new TextRun({ text, font: FONT, ...o });
const p = (runs, o = {}) => new Paragraph({ children: Array.isArray(runs) ? runs : [t(runs)], spacing: { after: 120, line: 288 }, ...o });
const rich = (s, o = {}) => {             // **bold** inline markup
  const parts = s.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map(x => x.startsWith("**") ? t(x.slice(2, -2), { bold: true, ...o }) : t(x, o));
};
const para = (s, o = {}) => p(rich(s), o);
const bullet = (s, level = 0) => new Paragraph({ children: rich(s), numbering: { reference: "bullets", level }, spacing: { after: 80, line: 276 } });
const num = (s, ref = "steps") => new Paragraph({ children: rich(s), numbering: { reference: ref, level: 0 }, spacing: { after: 80, line: 276 } });
const h1 = s => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [t(s)], spacing: { before: 360, after: 160 } });
const h2 = s => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t(s)], spacing: { before: 240, after: 100 } });
const img = (file, w, h, caption) => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(__dirname, "fig", file)), transformation: { width: w, height: h },
      altText: { title: caption, description: caption, name: file } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [t(caption, { italics: true, size: 18, color: MUTED })] }),
];
const pngSize = f => { const b = fs.readFileSync(path.join(__dirname, "fig", f)); return [b.readUInt32BE(16), b.readUInt32BE(20)]; };
const fit = (f, widthPx) => { const [w, h] = pngSize(f); return [widthPx, Math.round(widthPx * h / w)]; };

const border = { style: BorderStyle.SINGLE, size: 4, color: LINE };
const borders = { top: border, bottom: border, left: border, right: border };
function table(widths, rows, { header = true, shadeFirstCol = false } = {}) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: header && i === 0,
      children: r.map((c, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA }, borders,
        margins: { top: 70, bottom: 70, left: 110, right: 110 },
        shading: header && i === 0 ? { fill: INK, type: ShadingType.CLEAR, color: "auto" }
          : shadeFirstCol && j === 0 ? { fill: SUNK, type: ShadingType.CLEAR, color: "auto" } : undefined,
        children: [new Paragraph({ spacing: { after: 0, line: 264 },
          children: header && i === 0 ? [t(c, { bold: true, color: "FFFFFF", size: 19 })] : rich(c, { size: 19 }) })],
      })),
    })),
  });
}

// ---------- at a glance band ----------
const kpis = [
  [`${pct(full.f1[0])}%`, "Laundering-transaction F1", "best published: 64.8%"],
  [`${rings.detected}/${rings.known_attempts_in_test}`, "Known laundering rings caught", "test period"],
  [`${pct(acc.wrongful_flags_reduction)}%`, "Fewer innocent accounts flagged", "vs threshold rule, equal recall"],
  [`${(bMax.gpu_txn_s / 1e6).toFixed(2)}M/s`, "Transfers scored per second", "one A100 GPU"],
  [`${b2048.gpu_p50_ms.toFixed(0)} ms`, "Per batch of 2,048 transfers", "IBM's CPU system: 148 ms"],
];
const kw = Math.floor(W / 5);
const glance = new Table({
  width: { size: kw * 5, type: WidthType.DXA }, columnWidths: Array(5).fill(kw),
  rows: [new TableRow({ children: kpis.map(([v, l, s]) => new TableCell({
    width: { size: kw, type: WidthType.DXA }, borders: { top: { style: BorderStyle.SINGLE, size: 18, color: ACCENT }, bottom: border, left: border, right: border },
    shading: { fill: SOFT, type: ShadingType.CLEAR, color: "auto" }, margins: { top: 120, bottom: 120, left: 120, right: 100 },
    children: [
      new Paragraph({ spacing: { after: 40 }, children: [t(v, { bold: true, size: 34, color: ACCENT })] }),
      new Paragraph({ spacing: { after: 20, line: 252 }, children: [t(l, { bold: true, size: 17 })] }),
      new Paragraph({ spacing: { after: 0, line: 252 }, children: [t(s, { size: 15, color: MUTED })] }),
    ] })) })],
});

// ---------- document ----------
const children = [
  new Paragraph({ spacing: { after: 40 }, children: [t("MuleRadar", { bold: true, size: 56, color: INK })] }),
  new Paragraph({ spacing: { after: 120 }, children: [t("Catching money-mule rings while the money is still in the system", { size: 28, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 240 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: LINE, space: 8 } },
    children: [t("Challenge #5: Combating Digital Fraud and Financial Crime", { bold: true, size: 20 }),
      t("   ·   Team: Abdussamad Ali, Abubakar Abdullahi, Umar Mohammed Bashir, Abdulbasit Idris Ali", { size: 20, color: MUTED })] }),
  para("MuleRadar watches the live transfer graph on a single GPU, recognises the **shapes that laundering leaves behind** (fan-in, fan-out, pass-through, scatter-gather and round-trip cycles), and raises a ring-level alert with plain-English reasons and a draft Suspicious Transaction Report. It holds the ring, not every unusual new customer."),
  new Paragraph({ spacing: { before: 120, after: 80 }, children: [t("AT A GLANCE (measured on the public IBM AML benchmark)", { bold: true, size: 17, color: MUTED })] }),
  glance,

  // 1
  h1("1. Problem & Opportunity"),
  para("**Fraud proceeds leave Nigerian banks through chains of mule accounts faster than today's controls can react.**"),
  bullet("Nigerian financial institutions lost **₦52.26 billion to fraud in 2024**, three times the ₦17.67 billion of 2023 and 4.5× the 2020 level. Attempted fraud rose 338% in a year (NIBSS)."),
  bullet("The rails are fast and huge: NIBSS processed **11.2 billion e-payments worth ₦1.07 quadrillion in 2024**, about 355 transfers every second on average. Stolen money is split, passed across accounts and cashed out within hours."),
  bullet("Most transaction monitoring still checks **one account at a time against fixed thresholds**, often in overnight batches. A mule ring is a network pattern: each account looks ordinary on its own, and the crime only shows in how money moves between them."),
  bullet("Regulators are raising the cost of weak controls: seven of the largest Nigerian banks were fined a combined **US$10.7 million in 2024**, more than 60× the year before, including for AML failures, and CBN has introduced a fixed penalty regime for AML breaches."),
  bullet("Blunt rules also **hurt financial inclusion**. Threshold rules put large numbers of legitimate customers under suspicion, and the newly banked (students, small traders, first-time digital users) are the ones whose accounts get frozen, while mule recruiters target exactly these groups."),
  para("**The opportunity:** detect whole rings, not single accounts, inside the window in which a hold can still stop the money, and do it precisely enough that banks freeze far fewer innocent customers."),

  // 2
  h1("2. Innovative Solution"),
  para("MuleRadar is a streaming detection engine with an analyst copilot. Every incoming transfer passes through five stages:"),
  ...img("pipeline.png", ...fit("pipeline.png", 620), "Figure 1. MuleRadar pipeline. Everything up to the risk model runs on one GPU."),
  num("**GPU sliding-window transaction graph.** Transfers arrive in time order. For each micro-batch the engine rebuilds the last 96 hours of the graph on the GPU (about 2 million transfers in roughly 1 ms) as time-sorted lists of each account's payments."),
  num("**Four custom CUDA detectors** compute, for every incoming transfer and using only the past, the laundering signals that published research identifies as decisive:"),
  bullet("**Account velocity:** counts, value and number of distinct counterparties sent to and received from in 6-hour and 72-hour windows.", 1),
  bullet("**Pass-through:** how much the sender received in the previous 48 hours, how much of it is being forwarded, and how soon after the last credit. Mules keep little and forward fast.", 1),
  bullet("**Scatter-gather:** how many middlemen sit on time-ordered two-hop paths between the same sender and receiver.", 1),
  bullet("**Temporal cycles:** whether this transfer closes a time-ordered round trip of 2 to 6 accounts, with money returning to where it started.", 1),
  num("**AI risk scoring.** Gradient-boosted trees (XGBoost on the GPU) score each transfer. Alerted transfers are joined into rings, and every alert carries reason codes derived from the model's per-feature contributions (SHAP), for example \"receiver collected from 10 different accounts in 72h\"."),
  num("**GenAI analyst copilot.** For each ring, a large language model drafts the Suspicious Transaction Report narrative for the NFIU from the ring's evidence only: accounts, transfers with IDs, timeline and reason codes. A validator rejects any draft that cites a transaction or account that is not in the evidence, and the LLM never makes the hold-or-clear decision. In production it calls an open-weight model on the bank's own GPU, so no customer data leaves the bank."),
  num("**Proportionate actions instead of blanket freezes:** monitor, then step-up verification (BVN selfie or in-app confirmation), then hold outbound transfers on ring members, then escalate with an STR."),
  h2("Where AI, GenAI and emerging technology are used"),
  table([2000, 3500, 4138], [
    ["Technology", "Role in MuleRadar", "Why it matters"],
    ["GPU computing (CUDA)", "Four purpose-built kernels for temporal laundering patterns; the transfer graph lives in GPU memory", "Real-time network analysis at national payment volumes on one server"],
    ["Machine learning", "Gradient-boosted trees on 48 behavioural and network features, trained on the GPU", "Precise scoring: 84.8% of alerted transfers are true laundering"],
    ["Explainable AI (SHAP)", "Per-alert reason codes in plain English", "Analysts and regulators can see why an account was flagged"],
    ["Generative AI (LLM)", "Drafts the NFIU Suspicious Transaction Report from the ring's evidence, with a grounding check", "Cuts analyst writing time; the human approves every report"],
    ["Graph analytics", "Alerts are linked into rings of connected accounts", "Act on the whole network, not one account at a time"],
  ], { shadeFirstCol: true }),
  para(""),
  para("**What is new:** the detectors are not generic graph libraries. They are purpose-built CUDA kernels for time-ordered laundering patterns, verified to give exactly the same output as a CPU reference implementation, running in a streaming loop on one GPU."),

  // 3
  h1("3. Impact & Value"),
  para(`All numbers below were measured by us on the public IBM "Transactions for Anti-Money Laundering" benchmark (HI-Small: 5.08 million transfers, 515,000 accounts, 18 days), replayed as a stream, with **the same time-ordered 60/20/20 train, validation and test split used by the published papers**. The test set is the last 20% of the stream and was never seen during training.`),
  table([3300, 2700, 3638], [
    ["Result", "MuleRadar", "Reference"],
    ["Laundering-transaction F1", `**${pct(full.f1[0])}% ± ${pct(full.f1[1])}** (3 seeds)`, "IBM Graph Feature Preprocessor + XGBoost (2024): 63.2–64.8%; PNA graph neural network: 56.8%; plain XGBoost: 19.8%"],
    ["Precision of alerts", `**${pct(full.precision[0])}%** (recall ${pct(full.recall[0])}%)`, ""],
    ["Known laundering rings flagged", `**${rings.detected} of ${rings.known_attempts_in_test}** (${pct(rings.ring_recall, 0)}%)`, "All cycle, scatter-gather, gather-scatter and stack rings caught"],
    ["Innocent accounts flagged per 10,000", `**${acc.muleradar.innocent_flagged_per_10k.toFixed(1)}**`, `${k(Math.round(acc.threshold_rule_same_recall.innocent_flagged_per_10k))} for a 24-hour outbound-amount rule at the same recall: **${pct(acc.wrongful_flags_reduction)}% fewer** wrongful flags`],
    ["Throughput on one A100 GPU", `**${(bMax.gpu_txn_s / 1e6).toFixed(2)} million** transfers/s`, `${bMax.speedup.toFixed(0)}× the same algorithms on ${bench.cpu_threads} CPU cores (${b2048.speedup.toFixed(0)}× at batch 2,048); IBM's system: about 14,000/s on 64 CPU threads`],
    ["Latency per batch of 2,048 transfers", `**${b2048.gpu_p50_ms.toFixed(1)} ms**`, "IBM's system: 148 ms"],
  ], { shadeFirstCol: true }),
  ...img("ablation.png", ...fit("ablation.png", 560), "Figure 2. Each detector family adds accuracy (mean ± std over 3 seeds)."),
  ...img("throughput.png", ...fit("throughput.png", 560), "Figure 3. Same algorithm on GPU and CPU, end to end per micro-batch."),
  h2("What this means for a Nigerian bank"),
  bullet(`**Fewer wrongful freezes, more inclusion.** To catch the same half of the mule accounts, a typical amount rule would put ${k(acc.threshold_rule_same_recall.flagged)} accounts under suspicion. MuleRadar flags ${k(acc.muleradar.flagged)}, of which ${pct(acc.muleradar.precision, 0)}% are truly involved. For students and small traders, that is the difference between a working account and a frozen one.`),
  bullet("**Capacity to spare.** Nigeria's 2024 average is about 355 e-payments per second. One GPU sustains over a million per second, so a single server covers a large bank's peak load with wide headroom, and the same software runs on a cheaper GPU for microfinance banks and fintechs."),
  bullet("**Analyst time.** Each alert arrives as a ring with reasons and a drafted STR narrative, instead of a list of isolated threshold hits."),
  h2("Limits we state openly"),
  para(`The benchmark is synthetic, because real bank labels are confidential. In it, laundering is mostly generated as one payment format, which lets every model, ours included, alert on a ring's very first transfer; our measured "${pct(rings.stoppable_value_share, 0)}% of ring value still stoppable at the first alert" is therefore an upper bound. On real data the first alert will come a few hops in. The network detectors are what carry detection once that shortcut is gone, and the ablation shows they lift F1 from ${pct(L[0].f1[0])}% to ${pct(full.f1[0])}%.`),

  // 4
  h1("4. Target Users & Implementation"),
  para("**Who benefits.** Fraud and AML operations teams and compliance officers at commercial banks first; then microfinance banks, payment service banks and fintechs, which carry the same CBN and NFIU obligations with much smaller teams. Customers benefit directly through fewer wrongful freezes and faster recovery of stolen funds."),
  h2("How it deploys"),
  bullet("Reads a copy of the transfer feed (NIP / ISO 20022 messages or the core-banking transaction log). No change to core systems."),
  bullet("Runs on **one on-premises GPU server**. Customer data never leaves the bank, and the copilot LLM is an open-weight model on the same hardware."),
  bullet("Sends alerts to the bank's existing case-management tool through an API. Analysts approve every hold and every STR."),
  h2("Rollout"),
  num("**Back-test** on 3 to 6 months of the bank's historical transfers and confirmed fraud cases.", "rollout"),
  num("**Shadow mode:** live alerts go to analysts only, and precision and wrongful-flag rates are measured against the current system.", "rollout"),
  num("**Analyst-in-the-loop holds** on tier-3 and tier-4 rings.", "rollout"),
  num("**Automated step-up verification** for tier-2 accounts.", "rollout"),
  h2("Feasibility today"),
  para("Every result in Section 3 was produced by our working prototype: data pipeline, CUDA engine, CPU reference, model training, ring evaluation, STR drafting and the analyst console (Figure 4). A live mode replays the held-out transfer stream through the engine on the A100 and streams alerts and rings to the console in the browser, with a switch to run the same detectors on the CPU for comparison. It is built in Python with CuPy (custom CUDA C++ kernels), Numba, XGBoost, Polars and FastAPI, and each step is reproducible with one script. The STR module is wired for an on-premises LLM endpoint; in the prototype the drafts come from the evidence-grounded template, because our GPU cluster could not download model weights during the build."),
  ...img("console.png", ...fit("console.png", 600), "Figure 4. MuleRadar analyst console: ring queue, money-flow graph, reason codes and transfers."),
  para("**Business model.** An annual licence per institution, tiered by transaction volume, with a managed-appliance option for microfinance banks that have no GPU infrastructure."),

  // 5
  h1("5. Team & Expertise"),
  table([2300, 1700, 5638], [
    ["Member", "Role", "Contribution to MuleRadar"],
    ["**Abdussamad Ali**", "CEO", "Leads the venture and the GPU and AI build: problem framing, the CUDA detection engine and model evaluation. Owns bank partnerships and go-to-market."],
    ["**Abubakar Abdullahi**", "CTO", "Owns the technical architecture and deployment path: streaming ingestion from NIP and core-banking feeds, the on-premises GPU server, security, and integration with case-management tools."],
    ["**Umar Mohammed Bashir**", "Product Manager", "Turns detections into analyst workflow: tiered actions, the STR drafting flow for the NFIU, CBN and NFIU compliance requirements, and the pilot rollout plan."],
    ["**Abdulbasit Idris Ali**", "UI Designer", "Designs the analyst console: alert queue, ring visualisation, plain-English reason codes and the review-and-approve STR screen."],
  ], { shadeFirstCol: true }),
  para(""),
  para("Together the team covers what a bank needs in order to adopt MuleRadar: a detection engine that works at national scale (CEO and CTO), a workflow that fits compliance obligations (Product), and an interface analysts can act on in seconds (Design)."),

  h1("Sources"),
  ...[
    ["NIBSS 2024 fraud report (via 21st Century Chronicle)", "https://21stcenturychronicle.com/nigerian-banks-lost-n52-26-billion-to-fraud-in-2024-report/"],
    ["NIBSS 2024 e-payment volumes (TechCabal Insights)", "https://insights.techcabal.com/nigerian-payments-report-2024-online-transfers-dominate-atm-transactions-decline/"],
    ["Regulatory fines on Nigerian banks, 2024 (BusinessFront)", "https://businessfront.com/finance/insights/regulatory-fines-nigerias-top-lenders/"],
    ["Blanuša et al., Graph Feature Preprocessor, ICAIF 2024", "https://arxiv.org/abs/2402.08593"],
    ["Egressy et al., Provably Powerful GNNs for Directed Multigraphs, AAAI 2024", "https://arxiv.org/abs/2306.11586"],
  ].map(([label, url]) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 },
    children: [new ExternalHyperlink({ link: url, children: [t(label, { color: ACCENT, underline: {}, size: 19 })] })] })),
  new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 },
    children: [t("Altman et al., Realistic Synthetic Financial Transactions for Anti-Money Laundering Models, NeurIPS 2023 (IBM AML dataset)", { size: 19 })] }),
];

const doc = new Document({
  creator: "MuleRadar team", title: "MuleRadar: hackathon submission", description: "Challenge #5 submission",
  styles: {
    default: { document: { run: { font: FONT, size: 21, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: INK }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 23, bold: true, color: ACCENT }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 270 } } } } ] },
    { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
    { reference: "rollout", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [t("MuleRadar · Challenge #5 submission", { size: 16, color: MUTED })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [t("Page ", { size: 16, color: MUTED }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: MUTED }),
        t(" of ", { size: 16, color: MUTED }), new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 16, color: MUTED })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(__dirname, "MuleRadar_Submission.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
