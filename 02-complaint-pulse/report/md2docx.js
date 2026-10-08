// Minimal Markdown -> DOCX for the submission (headings, **bold**, bullets, numbered lists, pipe tables,
// images with {width=NN%}, block quote, title line "% Title"). Styled like the MuleRadar report.
// Usage: node md2docx.js report.md out.docx "Header text"
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, ImageRun, LevelFormat, Footer, Header, PageNumber, ExternalHyperlink,
} = require("docx");

const [, , SRC, OUT, HEADER] = process.argv;
const base = path.dirname(SRC);
const INK = "121827", MUTED = "58637A", ACCENT = "3D4FB8", SOFT = "E3E7FA", LINE = "D9DDE8", SUNK = "F1F2F7";
const FONT = "Arial", W = 9638;
const t = (text, o = {}) => new TextRun({ text, font: FONT, ...o });

function inline(s, o = {}) {           // **bold**, *italic*, [text](url)
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)]+\))/g;
  let last = 0, m;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push(t(s.slice(last, m.index), o));
    const x = m[0];
    if (x.startsWith("**")) out.push(t(x.slice(2, -2), { ...o, bold: true }));
    else if (x.startsWith("[")) {
      const mm = x.match(/\[([^\]]+)\]\(([^)]+)\)/);
      out.push(new ExternalHyperlink({ link: mm[2], children: [t(mm[1], { ...o, color: ACCENT, underline: {} })] }));
    } else out.push(t(x.slice(1, -1), { ...o, italics: true }));
    last = m.index + x.length;
  }
  if (last < s.length) out.push(t(s.slice(last), o));
  return out;
}
const border = { style: BorderStyle.SINGLE, size: 4, color: LINE };
const borders = { top: border, bottom: border, left: border, right: border };

function table(rows) {
  const cols = rows[0].length;
  const weights = rows[0].map((_, j) => Math.max(...rows.map(r => (r[j] || "").length)));
  const tot = weights.reduce((a, b) => a + Math.min(b, 60) + 8, 0);
  const widths = weights.map(w => Math.floor(W * (Math.min(w, 60) + 8) / tot));
  widths[cols - 1] += W - widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA }, borders, margins: { top: 60, bottom: 60, left: 100, right: 100 },
        shading: i === 0 ? { fill: INK, type: ShadingType.CLEAR, color: "auto" } : j === 0 ? { fill: SUNK, type: ShadingType.CLEAR, color: "auto" } : undefined,
        children: [new Paragraph({ spacing: { after: 0, line: 252 }, children: inline(c.trim(), i === 0 ? { bold: true, color: "FFFFFF", size: 18 } : { size: 18 }) })],
      })),
    })),
  });
}

function image(alt, file, widthPct) {
  const buf = fs.readFileSync(path.join(base, file));
  const w0 = buf.readUInt32BE(16), h0 = buf.readUInt32BE(20);
  const wpx = Math.round(640 * widthPct / 100);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true, keepLines: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: Math.round(wpx * h0 / w0) },
        altText: { title: alt, description: alt, name: path.basename(file) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [t(alt, { italics: true, size: 18, color: MUTED })] }),
  ];
}

const lines = fs.readFileSync(SRC, "utf8").split("\n");
const children = [];
let i = 0, title = "";
while (i < lines.length) {
  const l = lines[i];
  if (l.startsWith("% ")) { title = l.slice(2); children.push(new Paragraph({ spacing: { after: 160 }, children: [t(title, { bold: true, size: 40, color: INK })] })); i++; continue; }
  if (/^---\s*$/.test(l) || !l.trim()) { i++; continue; }
  if (l.startsWith("### ")) { children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [t(l.slice(4))] })); i++; continue; }
  if (l.startsWith("## ")) { children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [t(l.slice(3))] })); i++; continue; }
  if (l.startsWith("> ")) {
    const q = []; while (i < lines.length && lines[i].startsWith(">")) q.push(lines[i++].replace(/^>\s?/, ""));
    children.push(new Paragraph({ spacing: { before: 60, after: 200, line: 288 }, shading: { fill: SOFT, type: ShadingType.CLEAR, color: "auto" },
      border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT, space: 8 } }, children: inline(q.join(" ")) }));
    continue;
  }
  const im = l.match(/^!\[([^\]]*)\]\(([^)]+)\)(\{width=(\d+)%\})?/);
  if (im) { children.push(...image(im[1], im[2], +(im[4] || 100))); i++; continue; }
  if (l.startsWith("|")) {
    const rows = [];
    while (i < lines.length && lines[i].startsWith("|")) {
      const cells = lines[i].trim().replace(/^\||\|$/g, "").split("|");
      if (!cells.every(c => /^\s*:?-+:?\s*$/.test(c))) rows.push(cells);
      i++;
    }
    children.push(table(rows), new Paragraph({ spacing: { after: 120 }, children: [] }));
    continue;
  }
  const b = l.match(/^(\s*)- (.*)$/);
  if (b) {
    let text = b[2]; i++;
    while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*- /.test(lines[i])) text += " " + lines[i++].trim();
    children.push(new Paragraph({ children: inline(text), numbering: { reference: "bullets", level: b[1].length >= 2 ? 1 : 0 }, spacing: { after: 70, line: 276 } }));
    continue;
  }
  const n = l.match(/^\d+\. (.*)$/);
  if (n) {
    let text = n[1]; i++;
    while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*- /.test(lines[i])) text += " " + lines[i++].trim();
    children.push(new Paragraph({ children: inline(text), numbering: { reference: "steps", level: 0 }, spacing: { after: 70, line: 276 } }));
    continue;
  }
  // paragraph: join until blank / block start
  let text = l.trim(); i++;
  while (i < lines.length && lines[i].trim() && !/^(#|>|\||- |\d+\. |!\[)/.test(lines[i].trim())) text += " " + lines[i++].trim();
  children.push(new Paragraph({ children: inline(text), spacing: { after: 120, line: 288 } }));
}

const doc = new Document({
  creator: "ComplaintPulse team", title,
  styles: {
    default: { document: { run: { font: FONT, size: 21, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 30, bold: true, color: INK },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 4 } } } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 23, bold: true, color: ACCENT },
        paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 270 } } } }] },
    { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [t(HEADER || "", { size: 16, color: MUTED })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [t("Page ", { size: 16, color: MUTED }),
      new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: MUTED }), t(" of ", { size: 16, color: MUTED }),
      new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 16, color: MUTED })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
