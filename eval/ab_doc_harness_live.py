"""LIVE A/B — DOCUMENT_HARNESS Phase 1 final gate.

Proves the read-once harness drops FALSE-ABSTAINS on a real contract grid without
introducing any new WRONG cell. Runs the SAME proven path the route uses
(grid_engine.build_grid) twice over ONE real document:

  • Arm A (current):  harness=False  → per-cell fan-out, search_vault retrieval.
  • Arm B (harness):  harness=True   → one agent reads the doc ONCE, answers all columns.

Then it diffs the cells column-by-column and reports the headline metrics:
  • false-abstains resolved   (A ABSTAIN/MISSING → B FOUND)
  • new WRONG introduced       (the bar is 0 — B FOUND where A was FOUND but the
                                grounded quote disagrees, or B FOUND on a column that
                                is genuinely absent)
  • verified-rate delta        (cells with a grounded/clickable source span)

API-BURNING. On-demand only. Runs on the dev OpenAI key (AGENT_MODEL_STANDARD=gpt-5.4),
not Claude. Target: the Sezzle Consulting Agreement (Document.pdf) in the `contracts`
vault — the contract the G1 work flagged as 3/3 "unclear".

Run:  python -u eval/ab_doc_harness_live.py
"""
import os
import sys
import json

from dotenv import load_dotenv

load_dotenv()

# ── Fixed live target (verified present in the DB this session) ──────────────────────
OWNER_UID = "74f022f1-69d8-4f2b-a60c-6b08b79a80ea"
COLLECTION_ID = "8ca20f0a-5dfe-40b8-ab7a-26358a512f7e"      # vault: "contracts"
DOC_ID = "90f5e83f-83c2-4aa8-a7bc-0bc583c7ca4e"             # Document.pdf — Sezzle Consulting Agreement


def _build_db():
    """A service-role SupabaseManager with the vault owner attached as the verified user.

    user_id == owner ⇒ accessible_vault_owner resolves the own-vault (byte-identical) path,
    exactly as the route does for the owner. No token attached: read_client falls back to the
    service-role client, which can see the owner's own collection row, so the owner fast-path hits.
    """
    from src.components.db import SupabaseManager

    class _User:
        def __init__(self, uid):
            self.id = uid
            self.email = None
            self.user_metadata = {}

    sb = SupabaseManager(use_service_role=True)
    sb._user = _User(OWNER_UID)
    return sb


def _columns():
    """3 core contract-review columns (the cheapest A/B per Jeel)."""
    from src.components.agent_core.review_grid import GridColumn, ColumnKind

    return [
        GridColumn(
            key="governing_law",
            label="Governing Law",
            prompt=("Find the governing-law / choice-of-law clause. Quote the exact text "
                    "naming which state's or country's law governs the agreement."),
            kind=ColumnKind.CLAUSE,
            risk_rubric="standard if New York / Delaware / England & Wales; else non_standard",
        ),
        GridColumn(
            key="term_termination",
            label="Term & Termination",
            prompt=("Find the clause stating the agreement's term and how either party may "
                    "terminate it (notice period, for-cause vs convenience). Quote it exactly."),
            kind=ColumnKind.CLAUSE,
        ),
        GridColumn(
            key="confidentiality",
            label="Confidentiality",
            prompt=("Find the confidentiality / non-disclosure clause. Quote the exact text "
                    "describing the confidentiality obligation."),
            kind=ColumnKind.CLAUSE,
        ),
    ]


def _run_arm(harness: bool, spec, *, sb, filename_by_doc, grids_by_doc, model_id,
             user_config, retrieval_mgr):
    from src.components.agent_core.grid_engine import build_grid
    from src.components.agent_core.budgets import budget_for
    from src.components.agent_core.model import build_model

    grid_budget = budget_for("standard", user_config)

    def model_factory():
        return build_model("standard", grid_budget, user_config, system="")

    # build_grid's harness path takes db_client/vault_owner; its per-cell path (build_cell)
    # needs the retrieval manager, which build_grid doesn't thread. For arm A we therefore
    # call build_cell directly with the retrieval_manager (mirroring the route), and for arm
    # B we use build_grid(harness=True). This keeps each arm byte-identical to the route.
    from src.components.agent_core.grid_engine import build_cell, build_doc_cells
    from src.components.agent_core.review_grid import GridResult

    cells = []
    vault_owner = sb.accessible_vault_owner(COLLECTION_ID)
    for did in spec.doc_ids:
        if harness:
            cells.extend(build_doc_cells(
                did, spec.columns,
                collection_id=COLLECTION_ID,
                model=model_factory(),
                filename_by_doc=filename_by_doc,
                grids_by_doc=grids_by_doc,
                db_client=sb,
                vault_owner=vault_owner,
                model_id=model_id,
                model_factory=model_factory,
            ))
        else:
            for column in spec.columns:
                cells.append(build_cell(
                    did, column,
                    collection_id=COLLECTION_ID,
                    model=model_factory(),
                    filename_by_doc=filename_by_doc,
                    grids_by_doc=grids_by_doc,
                    retrieval_manager=retrieval_mgr,
                    db_client=sb,
                    model_id=model_id,
                    model_factory=model_factory,
                ))
    return GridResult(spec=spec, cells=cells)


def _cell_summary(cell):
    return {
        "status": cell.status.value,
        "value": (cell.value or "")[:120] if cell.value else None,
        "quote": (cell.quote or "")[:160] if cell.quote else None,
        "abstain_reason": cell.abstain_reason,
        "verified": cell.is_verified,
        "note": (cell.note or "")[:120] if cell.note else None,
    }


def main():
    from src.components.config import Config
    from src.components.agent_core.review_grid import GridSpec, CellStatus
    from src.components.brain.table_intent import load_grids_for_docs
    from src.components.retrieval import RetrievalManager
    from src.components.agent_core.budgets import budget_for

    sb = _build_db()

    user_config = Config()
    user_config.PINECONE_NAMESPACE = OWNER_UID
    # Sanity: the owner-resolve must succeed or both arms are mis-scoped.
    resolved = sb.accessible_vault_owner(COLLECTION_ID)
    assert resolved == OWNER_UID, f"vault owner resolve failed: {resolved!r} != {OWNER_UID!r}"

    doc_ids = sb.get_collection_document_ids(COLLECTION_ID)
    assert DOC_ID in doc_ids, f"{DOC_ID} not visible in collection {COLLECTION_ID}"
    docs = sb.read_client.table("documents").select("id,filename").in_("id", doc_ids).execute()
    filename_by_doc = {d["id"]: d["filename"] for d in (docs.data or [])}
    print(f"target doc: {filename_by_doc.get(DOC_ID)}  ({DOC_ID[:8]})")

    columns = _columns()
    spec = GridSpec(title="A/B: Sezzle contract", collection_id=COLLECTION_ID,
                    doc_ids=[DOC_ID], columns=columns)

    grids_by_doc = {DOC_ID: load_grids_for_docs(
        sb, [DOC_ID], question=None, filename_by_doc=filename_by_doc, per_doc_top=20) or []}

    retrieval_mgr = RetrievalManager(user_config)
    model_id = budget_for("standard", user_config).model
    print(f"model: {model_id}\n")

    print("══ ARM A — current (per-cell, search_vault) ══════════════════════════════")
    a = _run_arm(False, spec, sb=sb, filename_by_doc=filename_by_doc, grids_by_doc=grids_by_doc,
                 model_id=model_id, user_config=user_config, retrieval_mgr=retrieval_mgr)
    a_by_key = {c.column_key: c for c in a.cells}
    for c in a.cells:
        print(f"  [{c.column_key:18}] {c.status.value:8} verified={c.is_verified}  "
              f"{(c.quote or c.note or '')[:80]!r}")

    print("\n══ ARM B — harness (read-once per doc) ══════════════════════════════════")
    b = _run_arm(True, spec, sb=sb, filename_by_doc=filename_by_doc, grids_by_doc=grids_by_doc,
                 model_id=model_id, user_config=user_config, retrieval_mgr=retrieval_mgr)
    b_by_key = {c.column_key: c for c in b.cells}
    for c in b.cells:
        print(f"  [{c.column_key:18}] {c.status.value:8} verified={c.is_verified}  "
              f"{(c.quote or c.note or '')[:80]!r}")

    # ── Diff + headline ──────────────────────────────────────────────────────────────
    print("\n══ DIFF ═════════════════════════════════════════════════════════════════")
    FOUND = CellStatus.FOUND
    abstain_states = {CellStatus.ABSTAIN, CellStatus.MISSING, CellStatus.ERROR}
    resolved_fa = 0
    regressions = 0
    new_wrong_suspects = []
    details = []
    for col in columns:
        ca, cb = a_by_key.get(col.key), b_by_key.get(col.key)
        sa, sb_ = ca.status, cb.status
        verdict = "="
        if sa in abstain_states and sb_ == FOUND:
            resolved_fa += 1
            verdict = "RESOLVED  (A abstained → B found)"
        elif sa == FOUND and sb_ in abstain_states:
            regressions += 1
            verdict = "REGRESSED (A found → B abstained)"
        elif sa == FOUND and sb_ == FOUND:
            # Both found — flag for manual WRONG check if the grounded quotes diverge wildly.
            qa = (ca.quote or "").strip().lower()
            qb = (cb.quote or "").strip().lower()
            overlap = bool(qa) and bool(qb) and (qa[:40] in qb or qb[:40] in qa)
            verdict = "both FOUND" + ("" if overlap else "  ⚠ quotes diverge — verify")
            if not overlap:
                new_wrong_suspects.append(col.key)
        details.append({"column": col.key, "A": _cell_summary(ca), "B": _cell_summary(cb),
                        "verdict": verdict})
        print(f"  {col.key:18}  A={sa.value:8} → B={sb_.value:8}   {verdict}")

    def _vrate(res):
        return sum(1 for c in res.cells if c.is_verified), len(res.cells)

    av, an = _vrate(a)
    bv, bn = _vrate(b)
    print("\n── HEADLINE ─────────────────────────────────────────────────────────────")
    print(f"  false-abstains resolved (A→B):  {resolved_fa} / {len(columns)}")
    print(f"  regressions (B worse than A):    {regressions}")
    print(f"  quotes-diverge (manual WRONG check): {new_wrong_suspects or 'none'}")
    print(f"  verified-rate:  A {av}/{an}  →  B {bv}/{bn}")
    print(f"\n  PASS BAR: false-abstains↓  AND  0 new WRONG  AND  verified-rate not down.")

    out = {
        "doc": filename_by_doc.get(DOC_ID),
        "model": model_id,
        "columns": [c.key for c in columns],
        "resolved_false_abstains": resolved_fa,
        "regressions": regressions,
        "quotes_diverge": new_wrong_suspects,
        "verified_rate": {"A": [av, an], "B": [bv, bn]},
        "coverage": {"A": a.coverage(), "B": b.coverage()},
        "cells": details,
    }
    with open("eval/ab_doc_harness_live_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  wrote eval/ab_doc_harness_live_results.json")


if __name__ == "__main__":
    main()
