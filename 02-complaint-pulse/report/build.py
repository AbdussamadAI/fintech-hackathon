"""Build ComplaintPulse_Submission.docx/.pdf from SUBMISSION.md (single source of truth).

  * fills the N-ATLaS placeholders from results/normalize.json, llm_eval.json, topics_summary.json when present
  * replaces the ASCII pipeline with the figure and inserts the timeline, kernel and console figures
  * pandoc -> docx using the MuleRadar report as the style reference, then LibreOffice -> PDF

Usage: python report/build.py
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
R = ROOT / "results"
REF = ROOT.parent / "01-mule-radar" / "report" / "MuleRadar_Submission.docx"
SOFFICE = shutil.which("soffice") or "/Applications/LibreOffice.app/Contents/MacOS/soffice"


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def llm_table():
    n, ev = load("normalize.json"), load("llm_eval.json")
    if not n and not ev:
        return "(N-ATLaS evaluation pending)"
    langs = ["English", "Pidgin", "Hausa", "Yoruba", "Igbo"]
    lines = ["", "", "| Language | Issue accuracy, all complaints | Issue macro-F1 | Language identified | Issue accuracy on N-ATLaS-rewritten complaints |",
             "|---|---|---|---|---|"]
    for L in langs:
        a = n["by_language"][L] if n else {}
        rw = ev["rewritten"][L] if ev else {}
        f = lambda x: f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "–"  # noqa: E731
        lines.append(f"| {L} | {f(a.get('issue_accuracy'))} | {f(a.get('issue_macro_f1'))} | {f(a.get('language_id_accuracy'))} | {f(rw.get('issue_accuracy'))} |")
    extra = []
    if n:
        extra.append(f"N-ATLaS read {n['distinct']:,} distinct redacted complaints (covering all 249,553) at "
                     f"{n['per_second']:.1f} per second on one A100, with {n['parse_errors']} unparseable outputs.")
    if ev:
        extra.append(f"The rewritten set ({ev['n']} complaints, {ev['n'] // 5} per language) was paraphrased freely by N-ATLaS "
                     "to test robustness beyond our templates.")
    return "\n".join(lines) + "\n\n" + " ".join(extra)


def topic_result():
    t = load("topics_summary.json")
    if not t:
        return "(pending)"
    m = t["metrics"]["by_language"]
    worst = min(m.values(), key=lambda v: v["same_topic_as_majority"])["same_topic_as_majority"]
    return (f"with topics built from N-ATLaS English summaries, {worst * 100:.0f}–"
            f"{max(v['same_topic_as_majority'] for v in m.values()) * 100:.0f}% of complaints in every language land in the same "
            f"topic as complaints about the same problem in other languages (purity {t['metrics']['purity']:.2f}, NMI {t['metrics']['nmi']:.2f})")


def main():
    md = (ROOT / "SUBMISSION.md").read_text()
    md = md.replace("__LLM_TABLE__", llm_table()).replace("__LLM_TOPIC_RESULT__", topic_result())
    # ASCII pipeline -> figure
    md = re.sub(r"```\nall channels.*?```", "![Figure 1. ComplaintPulse pipeline: everything after the channels runs on one GPU.](fig/pipeline.png){width=100%}",
                md, flags=re.S)
    md = md.replace("- **Repost storm:**", "![Figure 2. Time to alert on the held-out stream: ComplaintPulse vs the daily-count rule.](fig/timeline.png){width=92%}\n\n- **Repost storm:**")
    md = md.replace("K2 is what makes it possible", "![Figure 3. Same algorithms on CPU and A100 GPU; outputs identical.](fig/kernels.png){width=100%}\n\nK2 is what makes it possible")
    if (HERE / "fig" / "console.png").exists():
        md = md.replace("**Business model.**", "![Figure 4. ComplaintPulse live console on the A100: issue radar, volume, multilingual feed.](fig/console.png){width=100%}\n\n**Business model.**")
    md = re.sub(r"^# (.+)$", r"% \1", md, count=1, flags=re.M)        # title block
    src = HERE / "report.md"
    src.write_text(md)
    out = HERE / "ComplaintPulse_Submission.docx"
    env = {**__import__("os").environ, "NODE_PATH": str(REF.parent / "node_modules")}
    subprocess.run(["node", str(HERE / "md2docx.js"), str(src), str(out), "ComplaintPulse · Challenge #2 submission"],
                   check=True, cwd=HERE, env=env, stderr=subprocess.DEVNULL)
    subprocess.run([SOFFICE, "--headless", "--convert-to", "pdf", "--outdir", str(HERE), str(out)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", out, "and PDF")


if __name__ == "__main__":
    main()
