#!/usr/bin/env python3
"""
Column-aware text extraction for 2-column PDFs with broken font encoding.

Why this exists:
- pypdf/pymupdf often fail on PDFs with no ToUnicode CMap (garbage chars).
- `pdftotext` (poppler) decodes correctly but `-layout` interleaves columns
  because it reconstructs lines across the full page width.

This script uses `pdftotext -bbox` to get correctly-decoded words with
positions, then reconstructs reading order itself:
  1. Group words into lines by y-position.
  2. Per page, detect the column gutter (a vertical whitespace gap near
     the horizontal center).
  3. For each line, check whether it actually straddles the gutter.
       - If yes with a real gap at the gutter -> split into left/right,
         accumulate into column buffers.
       - If no (a header, title, or full-width table row) -> treat as a
         "spanning" line: flush accumulated L/R column buffers first
         (left column top-to-bottom, then right column top-to-bottom),
         emit the spanning line, then keep going.

Usage:
    python3 extract_columns.py input.pdf > output.txt
    python3 extract_columns.py input.pdf --debug   # print column/gutter info
"""
import subprocess
import sys
import xml.etree.ElementTree as ET
import tempfile
import os
import argparse

LINE_Y_TOL = 2.0          # points; words within this y-range are one line
MIN_COLUMN_GAP = 10.0     # points; a gap this big (near center) signals a column break
GUTTER_SEARCH_LO = 0.35   # only treat a gap as the gutter within this fraction of page width
GUTTER_SEARCH_HI = 0.65


def run_pdftotext_bbox(pdf_path):
    with tempfile.TemporaryDirectory() as td:
        out_xml = os.path.join(td, "out.xml")
        subprocess.run(
            ["pdftotext", "-bbox", pdf_path, out_xml],
            check=True, capture_output=True
        )
        with open(out_xml, "r", encoding="utf-8") as f:
            return f.read()


def parse_pages(xml_text):
    ns = ""
    root = ET.fromstring(xml_text)
    pages = []
    for page_el in root.iter():
        if page_el.tag.endswith("page"):
            width = float(page_el.attrib["width"])
            height = float(page_el.attrib["height"])
            words = []
            for w in page_el:
                if w.tag.endswith("word"):
                    text = (w.text or "").strip()
                    if not text:
                        continue
                    words.append({
                        "x0": float(w.attrib["xMin"]),
                        "x1": float(w.attrib["xMax"]),
                        "y0": float(w.attrib["yMin"]),
                        "y1": float(w.attrib["yMax"]),
                        "text": text,
                    })
            pages.append({"width": width, "height": height, "words": words})
    return pages


def group_lines(words):
    """Group words into lines by y-position; return list of lines,
    each a list of words sorted left-to-right, sorted top-to-bottom."""
    if not words:
        return []
    ws = sorted(words, key=lambda w: (w["y0"], w["x0"]))
    lines = []
    cur = [ws[0]]
    cur_y = ws[0]["y0"]
    for w in ws[1:]:
        if abs(w["y0"] - cur_y) <= LINE_Y_TOL:
            cur.append(w)
        else:
            lines.append(cur)
            cur = [w]
            cur_y = w["y0"]
    lines.append(cur)
    for line in lines:
        line.sort(key=lambda w: w["x0"])
    return lines


def find_line_gap(line, width):
    """Return (gap_lo, gap_hi) for the largest inter-word gap in this line
    if it's wide enough and near the horizontal center, else None.
    Used only to *sample* candidate gutter positions across the page."""
    lo = width * GUTTER_SEARCH_LO
    hi = width * GUTTER_SEARCH_HI
    best_gap, best = 0.0, None
    for i in range(len(line) - 1):
        gap = line[i + 1]["x0"] - line[i]["x1"]
        mid = (line[i + 1]["x0"] + line[i]["x1"]) / 2.0
        if gap > best_gap and lo <= mid <= hi:
            best_gap = gap
            best = (line[i]["x1"], line[i + 1]["x0"])
    if best is not None and best_gap >= MIN_COLUMN_GAP:
        return best
    return None


def estimate_page_gutter(lines, width):
    """Sample the gutter position from lines that clearly show it, then
    return the median (gutter_lo, gutter_hi). None if not enough evidence
    (page is probably single-column)."""
    samples = [find_line_gap(line, width) for line in lines]
    samples = [s for s in samples if s is not None]
    if len(samples) < 3:
        return None
    los = sorted(s[0] for s in samples)
    his = sorted(s[1] for s in samples)
    return (los[len(los) // 2], his[len(his) // 2])


CROSS_MARGIN = 2.0  # points; ignore near-boundary overlap (float/kerning noise)


def classify_line(line, gutter):
    """Classify a line against the page's estimated gutter.
    Returns ('left', words) | ('right', words) | ('split', left_words, right_words)
    | ('span', words)."""
    g_lo, g_hi = gutter

    def overlap(w):
        return min(w["x1"], g_hi) - max(w["x0"], g_lo)

    crossing = [w for w in line if overlap(w) > CROSS_MARGIN]
    if crossing:
        return ("span", line)
    left = [w for w in line if w["x1"] <= g_lo + CROSS_MARGIN]
    right = [w for w in line if w["x0"] >= g_hi - CROSS_MARGIN]
    if left and right:
        return ("split", left, right)
    if left:
        return ("left", left)
    if right:
        return ("right", right)
    return ("span", line)


def line_text(line):
    return " ".join(w["text"] for w in line)


def render_page(page, debug=False):
    lines = group_lines(page["words"])
    if not lines:
        return ""

    gutter = estimate_page_gutter(lines, page["width"])
    if debug:
        print(f"  [gutter estimate: {gutter}]", file=sys.stderr)

    out_chunks = []

    if gutter is None:
        # Not enough evidence of a 2-column layout on this page -> just
        # read top to bottom, left to right (already correct order).
        for line in lines:
            out_chunks.append(line_text(line))
        return "\n".join(out_chunks)

    left_buf, right_buf = [], []

    def flush():
        nonlocal left_buf, right_buf
        for l in left_buf:
            out_chunks.append(line_text(l))
        if left_buf and right_buf:
            out_chunks.append("")  # blank line between columns
        for r in right_buf:
            out_chunks.append(line_text(r))
        left_buf, right_buf = [], []

    for line in lines:
        kind, *parts = classify_line(line, gutter)
        if kind == "left":
            left_buf.append(parts[0])
        elif kind == "right":
            right_buf.append(parts[0])
        elif kind == "split":
            left_buf.append(parts[0])
            right_buf.append(parts[1])
        else:  # span: title, header, footer, wide table row
            flush()
            out_chunks.append(line_text(parts[0]))
    flush()
    return "\n".join(out_chunks)


def extract_pdf_text(pdf_path, debug=False):
    xml_text = run_pdftotext_bbox(pdf_path)
    pages = parse_pages(xml_text)
    all_text = []
    for i, page in enumerate(pages):
        if debug:
            print(f"  page {i+1}", file=sys.stderr)
        all_text.append(f"--- page {i+1} ---\n" + render_page(page, debug))
    return "\n\n".join(all_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="a PDF file, or a directory of PDFs for batch mode")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("-o", "--output", help="output txt path (single-file mode; default stdout)")
    ap.add_argument("--outdir", default="extracted_text",
                     help="output directory for batch mode (default: ./extracted_text)")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        os.makedirs(args.outdir, exist_ok=True)
        pdfs = sorted(f for f in os.listdir(args.input) if f.lower().endswith(".pdf"))
        print(f"Found {len(pdfs)} PDFs in {args.input}", file=sys.stderr)
        failures = []
        for fname in pdfs:
            in_path = os.path.join(args.input, fname)
            out_path = os.path.join(args.outdir, os.path.splitext(fname)[0] + ".txt")
            print(f"-> {fname}", file=sys.stderr)
            try:
                text = extract_pdf_text(in_path, args.debug)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception as e:
                print(f"   FAILED: {e}", file=sys.stderr)
                failures.append(fname)
        print(f"\nDone. {len(pdfs) - len(failures)}/{len(pdfs)} succeeded.", file=sys.stderr)
        if failures:
            print("Failed files:\n  " + "\n  ".join(failures), file=sys.stderr)
    else:
        result = extract_pdf_text(args.input, args.debug)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result)
        else:
            print(result)


if __name__ == "__main__":
    main()
