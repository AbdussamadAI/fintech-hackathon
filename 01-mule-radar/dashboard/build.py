"""Inject demo.json into the dashboard template -> dashboard/index.html (single self-contained file).

Usage: python dashboard/build.py path/to/demo.json
"""
import json
import sys
from pathlib import Path

here = Path(__file__).parent
demo = json.loads(Path(sys.argv[1]).read_text())
for r in demo["rings"]:                       # keep the page light
    r["timeline_full"] = r["timeline_full"][:80]
data = json.dumps(demo, separators=(",", ":"), default=str).replace("</", "<\\/")
html = (here / "template.html").read_text().replace("/*DATA*/", data)
(here / "index.html").write_text(html)
print(f"wrote {here / 'index.html'} ({len(html) / 1e3:.0f} kB)")
