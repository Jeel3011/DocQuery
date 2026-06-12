"""Extraction FIDELITY check — ground-truth-free, runs on ANY document.

The generic answer to "what about new docs": no rule set reads every PDF
perfectly, so every ingested doc SELF-REPORTS extraction fidelity. The check
cross-references the extracted grids against the PDF's OWN text layer
(pdfplumber's independent line assembly): every text line whose trailing run is
>=2 value tokens (a data row — cells are right-aligned trailing) must have all
its numbers present in that page's grids, as cell values or as label text. A
page where numbers vanished is flagged ON ARRIVAL — a silent row drop becomes a
visible per-page stat, not a wrong/missing answer three questions later.

Needs no labels, no gold, no API call. The eval gate
(eval/test_extraction_completeness.py) asserts 0 uncovered lines on the
hand-verified corpus; ingestion logs the same stat for every new doc.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from src.components.table_extraction import _FOOTNOTE_TOK, _VALUE_TOK


def norm_val(tok: str) -> str:
    """Canonical numeric-token form across text/grid layers: strip currency,
    percent, negative-parens and commas; keep digits, dot and sign."""
    s = (tok or "").strip()
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[($),%]", "", s).strip()
    if not s:
        return ""
    return ("-" + s) if neg and not s.startswith("-") else s


def is_year_or_day(norm: str) -> bool:
    """Bare year (1900-2100) / day-of-month (<=31) → period-header furniture."""
    try:
        v = float(norm)
    except ValueError:
        return False
    if v != int(v):
        return False
    return 1900 <= v <= 2100 or 0 <= v <= 31


def text_data_lines(page) -> List[Any]:
    """(line_text, [all normalized numeric tokens]) per data-looking text line."""
    try:
        text = page.extract_text() or ""
    except Exception:  # noqa: BLE001
        return []
    out = []
    for line in text.split("\n"):
        if "http" in line.lower():
            continue  # sec.gov footer chrome, never a statement row
        toks = [t for t in (tok.strip() for tok in line.split())
                if t and t != "$" and not _FOOTNOTE_TOK.match(t)]
        is_val = [bool(_VALUE_TOK.match(t)) for t in toks]
        split = len(toks)
        while split > 0 and is_val[split - 1]:
            split -= 1
        trailing = [v for v in (norm_val(t) for t in toks[split:]) if v]
        if len(trailing) < 2:
            continue  # a lone trailing number (page footer) is not a data row
        if all(is_year_or_day(v) for v in trailing):
            continue  # a period/date header line, not data
        all_nums = [n for t, v in zip(toks, is_val) if v for n in [norm_val(t)] if n]
        out.append((line.strip(), all_nums))
    return out


def grid_value_pool(tables: List[Any], page_no: int) -> set:
    """Every normalized number the extractor kept for a page — cell values AND
    numbers embedded in label/section text (folded parentheticals are preserved
    data, not drops)."""
    pool = set()
    for tb in tables:
        tb_page = getattr(tb, "page_number", None)
        if tb_page is None:
            md = tb.to_metadata() if hasattr(tb, "to_metadata") else {}
            tb_page = md.get("page")
        if tb_page != page_no:
            continue
        rows = getattr(tb, "rows", None)
        if rows is None:
            rows = (tb.to_metadata() if hasattr(tb, "to_metadata") else {}).get("rows", [])
        for r in rows:
            for k, v in r.items():
                if k in ("section", "label"):
                    for tok in str(v).split():
                        if _VALUE_TOK.match(tok):
                            n = norm_val(tok)
                            if n:
                                pool.add(n)
                    continue
                n = norm_val(str(v))
                if n:
                    pool.add(n)
    return pool


def fidelity_report(pdf_path: str, tables: List[Any]) -> Dict[str, Any]:
    """Per-doc fidelity stat: which table-bearing pages have text-layer data
    lines whose numbers never made it into any extracted grid. Never raises."""
    report: Dict[str, Any] = {"doc": pdf_path, "pages_checked": 0,
                              "data_lines": 0, "uncovered": []}
    try:
        import pdfplumber
        table_pages = sorted({getattr(t, "page_number", None) for t in tables
                              if getattr(t, "page_number", None)})
        with pdfplumber.open(pdf_path) as pdf:
            for pno in table_pages:
                if pno < 1 or pno > len(pdf.pages):
                    continue
                pool = grid_value_pool(tables, pno)
                lines = text_data_lines(pdf.pages[pno - 1])
                report["pages_checked"] += 1
                report["data_lines"] += len(lines)
                for line, nums in lines:
                    missing = [v for v in nums if v not in pool]
                    if missing:
                        report["uncovered"].append(
                            {"page": pno, "line": line[:120], "missing": missing})
    except Exception as e:  # noqa: BLE001 — fidelity must never break ingestion
        report["error"] = str(e)
    return report
