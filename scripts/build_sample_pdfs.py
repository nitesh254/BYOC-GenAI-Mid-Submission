"""Convert CC BY 4.0 raw markdown policy documents into clean multi-page sample PDFs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "basecamp_policies"
SAMPLE_DIR = ROOT_DIR / "data" / "sample"

POLICY_BUNDLES: dict[str, tuple[str, list[str]]] = {
    "01_refund_and_cancellation_policy.pdf": (
        "37signals / Basecamp Refund & Cancellation Policies",
        ["refund.md", "cancellation.md"],
    ),
    "02_security_and_sla_policy.pdf": (
        "37signals / Basecamp Security Overview & Service Level Agreement",
        ["security.md", "sla.md"],
    ),
    "03_privacy_policy.pdf": (
        "37signals / Basecamp Privacy Policy",
        ["privacy.md"],
    ),
    "04_taxes_and_acceptable_use_policy.pdf": (
        "37signals / Basecamp Taxes & Use Restrictions Policy",
        ["taxes.md", "abuse.md"],
    ),
}


def strip_frontmatter(text: str) -> str:
    """Remove leading YAML frontmatter delimited by --- lines."""
    match = re.match(r"^---\s*\n.*?\n---\s*\n", text, flags=re.DOTALL)
    return text[match.end() :] if match else text


def clean_markdown_line(line: str) -> str:
    """Convert inline markdown links, HTML spans, and Liquid tags into plain readable text."""
    line = re.sub(r"\{\{\s*site\.email_support\s*\}\}", "support@37signals.com", line)
    line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", line)
    line = re.sub(r"\[([^\]]+)\]\[[^\]]+\]", r"\1", line)
    line = re.sub(r"<[^>]+>", "", line)
    line = line.replace("**", "").replace("__", "").replace("*", "")
    return line.strip()


def escape_xml(text: str) -> str:
    """Escape XML special characters for ReportLab Paragraph markup."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_pdf(output_path: Path, doc_title: str, source_files: Iterable[Path]) -> None:
    """Render one or more markdown policy files into a clean multi-page PDF."""
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=16,
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "DocH1",
        parent=styles["Heading1"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=body_style,
        leftIndent=16,
        bulletIndent=6,
    )

    story = [
        Paragraph(escape_xml(doc_title), title_style),
        Paragraph(
            "Source: basecamp/policies (37signals LLC) | License: CC BY 4.0",
            body_style,
        ),
        Spacer(1, 0.15 * inch),
    ]

    files = list(source_files)
    for idx, md_path in enumerate(files):
        raw_text = strip_frontmatter(md_path.read_text(encoding="utf-8"))
        for raw_line in raw_text.splitlines():
            stripped = raw_line.strip()
            if not stripped or re.match(r"^\[[^\]]+\]:\s*https?://", stripped):
                continue

            if stripped.startswith("### "):
                cleaned = escape_xml(clean_markdown_line(stripped[4:]))
                story.append(Paragraph(cleaned, h2_style))
            elif stripped.startswith("## "):
                cleaned = escape_xml(clean_markdown_line(stripped[3:]))
                story.append(Paragraph(cleaned, h1_style))
            elif stripped.startswith("# "):
                cleaned = escape_xml(clean_markdown_line(stripped[2:]))
                story.append(Paragraph(cleaned, h1_style))
            elif stripped.startswith("* ") or stripped.startswith("- "):
                cleaned = escape_xml(clean_markdown_line(stripped[2:]))
                story.append(Paragraph(f"&bull; {cleaned}", bullet_style))
            else:
                cleaned = escape_xml(clean_markdown_line(stripped))
                if cleaned:
                    story.append(Paragraph(cleaned, body_style))

        if idx < len(files) - 1:
            story.append(PageBreak())

    def add_page_number(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(LETTER[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
        canvas.drawString(0.75 * inch, 0.5 * inch, output_path.name)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def main() -> None:
    """Build all sample PDFs from the raw policy markdown files."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for pdf_name, (title, md_names) in POLICY_BUNDLES.items():
        source_paths = [RAW_DIR / name for name in md_names]
        missing = [str(p) for p in source_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing raw policy files: {missing}")
        output_pdf = SAMPLE_DIR / pdf_name
        build_pdf(output_pdf, title, source_paths)
        print(f"Built {output_pdf}")


if __name__ == "__main__":
    main()
