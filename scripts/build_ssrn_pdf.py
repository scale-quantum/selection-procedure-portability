"""Build an SSRN-ready PDF from the English paper. Run from the repository root."""

from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Selection Procedure Portability.md"
OUT = ROOT / "Selection Procedure Portability.pdf"

CSS = """
@page { size: A4; margin: 1.8cm 1.6cm 1.8cm 1.6cm; }
body { font-family: Times New Roman, Times, serif; font-size: 10pt; line-height: 1.35; color: #111; }
h1 { font-size: 16pt; line-height: 1.25; margin: 0 0 10pt 0; }
h2 { font-size: 12.5pt; margin: 16pt 0 6pt 0; }
h3 { font-size: 11pt; margin: 12pt 0 5pt 0; }
p, li { margin: 0 0 7pt 0; }
table { width: 100%; border-collapse: collapse; font-size: 7pt; margin: 0 0 10pt 0; }
th, td { border: 0.4pt solid #444; padding: 2pt 3pt; vertical-align: top; }
th { background: #eee; }
pre, code { font-family: Courier New, monospace; font-size: 8pt; }
pre { border: 0.4pt solid #ccc; padding: 6pt; margin: 0 0 10pt 0; }
img { width: 16cm; }
.meta { font-size: 11pt; margin: 0 0 12pt 0; }
"""


def link_callback(uri, rel):
    if uri.startswith("figures/"):
        return str(ROOT / uri.replace("/", "\\"))
    return uri


def main() -> None:
    md = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(
        md,
        extensions=["extra", "sane_lists", "fenced_code", "tables"],
    )
    html = (
        "<html><head><meta charset='utf-8'/><style>"
        + CSS
        + "</style></head><body>"
        + body
        + "</body></html>"
    )
    with OUT.open("wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, link_callback=link_callback, encoding="utf-8")
    if result.err:
        raise SystemExit(f"PDF failed with {result.err} error(s)")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
