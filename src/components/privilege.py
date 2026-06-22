"""Privilege firewall (F1e — plans/F1_VAULT_PLAN.md §1).

A document marked `privileged` is attorney-client / work-product material. The firewall rule
(the plan's exact words): a privileged document is **excluded from any shared / cross-vault
surface** and **watermarked in exports**. Critically — it is NOT hidden from its OWN vault: you
privileged it, you can still use it in its own matter. The exclusion is a CROSS-VAULT boundary.

This module is the small, pure, testable core of that rule:

  - `exclude_privileged(docs)` — given doc rows headed for a SHARED / cross-vault surface (e.g.
    F6 Firm Brain reusing work across matters), drop the privileged ones. F6 is the main
    consumer; it doesn't exist yet, so this helper is the contract F6 will call — gate-proven
    now so the firewall is correct the day the surface lands. A single-vault agent run does NOT
    call this (that's the doc's own vault — privilege doesn't exclude there).
  - `export_watermark(...)` / `needs_watermark(...)` — the notice line stamped on any export
    that contains privileged material, so a privileged doc never leaves the system unmarked.

Pure functions over plain dicts/bools — no I/O, never raise on bad input. $0, no LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List


def is_privileged(doc: Dict[str, Any]) -> bool:
    """True iff a document row is flagged privileged. Tolerant of a missing/None key (legacy
    rows pre-migration-011 read as not-privileged) — never raises."""
    try:
        return bool(doc.get("privileged", False))
    except Exception:  # noqa: BLE001
        return False


def exclude_privileged(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop privileged docs from a set headed for a SHARED / cross-vault surface.

    Returns a NEW list (does not mutate the input). The caller is any surface that crosses a
    vault boundary — never a single-vault run (privilege does not exclude a doc from its own
    matter). Non-privileged and legacy/untagged docs pass through unchanged."""
    if not docs:
        return []
    return [d for d in docs if not is_privileged(d)]


def privileged_doc_ids(docs: List[Dict[str, Any]]) -> List[str]:
    """The ids of the privileged docs in a set — for an export-watermark check or an audit
    trail of what was withheld from a shared surface. Never raises."""
    out: List[str] = []
    for d in (docs or []):
        if is_privileged(d) and d.get("id"):
            out.append(d["id"])
    return out


# ── Export watermark ──────────────────────────────────────────────────────────────────────

# The notice stamped on an export that contains privileged material. Kept as a single constant
# so the wording is consistent across the .docx (export.py) and redline (redline.py) paths.
WATERMARK_NOTICE = (
    "PRIVILEGED & CONFIDENTIAL — ATTORNEY-CLIENT / WORK PRODUCT. "
    "This document contains privileged material and is not for distribution."
)


def needs_watermark(docs: List[Dict[str, Any]]) -> bool:
    """True iff any doc in the export set is privileged → the export must carry the watermark."""
    return any(is_privileged(d) for d in (docs or []))


def export_watermark(docs: List[Dict[str, Any]]) -> str:
    """The watermark notice to stamp on an export, or '' when nothing in the set is privileged
    (no watermark on a clean export). A pure string builder — the export/redline path decides
    WHERE to place it (header line, footer, first page)."""
    return WATERMARK_NOTICE if needs_watermark(docs) else ""
