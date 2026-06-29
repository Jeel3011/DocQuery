"""LIVE A/B — DOCUMENT_HARNESS Phase 2 gate.

Phase 2 = single-doc/matter CHAT on the harness tools + the search ladder (rungs 3–4) +
L4 record-of-processing audit. The Phase-2 gate (§17) is:
  "a single-contract question that currently false-abstains now answers, cited;
   audit row written; leak test still green."

This runs the CHAT path (run_agent) — exactly what the agent_core route does — twice over
the Sezzle Consulting Agreement (Document.pdf) in the `contracts` vault, whose namespace
has NO live embeddings, so arm A's search_vault returns 0/0 (the false-abstain the harness
exists to kill — same degenerate-but-real miss as the Phase-1 contract A/B).

  • Arm A (current):  harness=False → standard chat, search_vault top-k (0/0 here).
  • Arm B (harness):  harness=True  → standard chat, ls/grep/read the real contract,
                                      climbing the search ladder (synonyms → expand → whole-doc read).

It checks THREE Phase-2 things on live data:
  1. false-abstain → answers cited   (A misses, B hits the gold clause + carries a [doc p.N])
  2. 0 new wrong                      (B's grounded quote doesn't contradict the clause)
  3. L4 audit                          (arm B writes audit_log rows — verified by a count
                                        diff AND by replaying the loop events through the
                                        route's pure _collect_harness_reads/_harness_audit_rows)

API-BURNING. On-demand only. Dev OpenAI key (gpt-5.4), not Claude.
3 questions × 2 arms = 6 run_agent runs over one ~22-chunk contract.
Run:  python -u eval/ab_doc_harness_phase2_live.py
"""
import os
import json

from dotenv import load_dotenv

load_dotenv()

# ── Fixed live target (verified present this session: 22 chunks) ─────────────────────
OWNER_UID = "74f022f1-69d8-4f2b-a60c-6b08b79a80ea"
COLLECTION_ID = "8ca20f0a-5dfe-40b8-ab7a-26358a512f7e"      # vault: "contracts"
DOC_ID = "90f5e83f-83c2-4aa8-a7bc-0bc583c7ca4e"             # Document.pdf — Sezzle Consulting Agreement

# Prose/clause facts verified present in the Sezzle agreement (the G1 work flagged this
# contract as 3/3 "unclear" — the exact false-abstains Phase 1+2 should now resolve cited).
QUESTIONS = [
    {
        "q": "Under this consulting agreement, what law governs the agreement?",
        "gold_any": ["minnesota"],
        "why": "§9 governing law — Minnesota",
    },
    {
        "q": "How can either party terminate this consulting agreement, and is there a "
             "cure period?",
        # Distinctive §4 content — 'cure' + 'breach' don't appear in a wrong-doc abstain.
        "gold_any": ["cure"],
        "why": "§4: terminate on written notice if the other party breaches a material term "
               "and fails to cure; or by mutual agreement on notice",
    },
    {
        "q": "What are the consultant's confidentiality obligations under this agreement?",
        # Distinctive §5 language — 'non-public information' / 'prior written consent'.
        "gold_any": ["non-public", "prior written consent", "third party"],
        "why": "§5: maintain confidentiality of all non-public information; no disclosure to "
               "third parties without prior written consent",
    },
]


def _build_db():
    """Service-role SupabaseManager with the vault owner attached (owner fast-path, exactly
    as the route resolves for the vault owner)."""
    from src.components.db import SupabaseManager

    class _User:
        def __init__(self, uid):
            self.id = uid
            self.email = None
            self.user_metadata = {}

    sb = SupabaseManager(use_service_role=True)
    sb._user = _User(OWNER_UID)
    return sb


def _run_question(question, *, harness, sb, user_config, scoped_doc_ids, filename_by_doc,
                  grids, vault_owner, retrieval_mgr):
    """Run ONE arm. Mirrors the route: harness flows into BOTH the tool set (RunScope) and
    the system prompt (search-ladder overlay). Returns the answer + the loop tool_call events
    (so we can replay the route's L4 audit-row derivation)."""
    from src.components.agent_core.registry import RunScope
    from src.components.agent_core.budgets import budget_for
    from src.components.agent_core.model import build_model
    from src.components.agent_core.prompt import system_prompt
    from src.components.agent_core.loop import run_agent

    scope = RunScope(
        collection_id=COLLECTION_ID,
        doc_ids=scoped_doc_ids,
        filenames=list(filename_by_doc.values()),
        grids=grids,
        vault_owner=vault_owner,
        retrieval_manager=retrieval_mgr,
        db_client=sb,
        filename_by_doc=filename_by_doc,
        question=question,
        config=user_config,
        harness=harness,
    )
    budget = budget_for("standard", user_config)
    # Phase 2.3: the route threads harness into the prompt too (search-ladder overlay).
    sys_prompt = system_prompt("v1", mode="standard", harness=harness)
    model = build_model("standard", budget, user_config, system=sys_prompt)

    answer = ""
    n_sources = 0
    tool_events = []   # full tool_call event dicts (carry doc_id for the L4 replay)
    for ev in run_agent(question, model=model, scope=scope, budget=budget,
                        system_prompt=sys_prompt):
        t = ev.get("type")
        if t == "token":
            answer = ev.get("text", "") or ev.get("content", "") or answer
        elif t == "sources":
            n_sources = len(ev.get("sources", []) or [])
        elif t == "tool_call":
            tool_events.append(ev)
    return {"answer": answer.strip(), "n_sources": n_sources, "tool_events": tool_events,
            "tools": [e.get("name") for e in tool_events]}


def _hits_gold(answer, gold_any):
    a = (answer or "").lower()
    return any(g.lower() in a for g in gold_any)


def _has_citation(answer):
    """A [doc p.N] / [doc] style citation marker — the cited contract."""
    import re
    return bool(re.search(r"\[[^\]]+\]", answer or ""))


def main():
    from src.components.config import Config
    from src.components.brain.table_intent import load_grids_for_docs
    from src.components.retrieval import RetrievalManager
    from src.api.routes.agent_core import (
        _collect_harness_reads, _harness_audit_rows,
    )

    sb = _build_db()
    user_config = Config()
    user_config.PINECONE_NAMESPACE = OWNER_UID
    assert sb.accessible_vault_owner(COLLECTION_ID) == OWNER_UID, "owner resolve failed"

    doc_ids = sb.get_collection_document_ids(COLLECTION_ID)
    docs = sb.read_client.table("documents").select("id,filename").in_("id", doc_ids).execute()
    filename_by_doc = {d["id"]: d["filename"] for d in (docs.data or [])}
    print(f"vault: {len(doc_ids)} docs  ({', '.join(sorted(filename_by_doc.values()))})")

    grids = []
    try:
        for did in doc_ids:
            grids.extend(load_grids_for_docs(sb, [did], question=None,
                                             filename_by_doc=filename_by_doc, per_doc_top=8) or [])
    except Exception as exc:  # noqa: BLE001
        print(f"(grid preload degraded: {exc})")
    retrieval_mgr = RetrievalManager(user_config)
    vault_owner = sb.accessible_vault_owner(COLLECTION_ID)

    # L4: snapshot the owner's audit_log count so we can prove arm B wrote rows.
    def _audit_count():
        return sb.client.table("audit_log").select("id", count="exact").eq(
            "user_id", OWNER_UID).execute().count

    audit_before = _audit_count()
    print(f"audit_log rows (owner, before): {audit_before}\n")

    rows = []
    derived_rows_total = 0
    for item in QUESTIONS:
        q = item["q"]
        print("─" * 78)
        print(f"Q: {q}")
        a = _run_question(q, harness=False, sb=sb, user_config=user_config,
                          scoped_doc_ids=doc_ids, filename_by_doc=filename_by_doc,
                          grids=grids, vault_owner=vault_owner, retrieval_mgr=retrieval_mgr)
        b = _run_question(q, harness=True, sb=sb, user_config=user_config,
                          scoped_doc_ids=doc_ids, filename_by_doc=filename_by_doc,
                          grids=grids, vault_owner=vault_owner, retrieval_mgr=retrieval_mgr)
        a_hit = _hits_gold(a["answer"], item["gold_any"])
        b_hit = _hits_gold(b["answer"], item["gold_any"])
        b_cited = _has_citation(b["answer"])

        # L4: derive the audit rows the route writes for arm B from its events, AND actually
        # write them through the SAME log_audit seam the route uses (the route's _flush_audit
        # does exactly this in `finally`). This proves the rows land in audit_log live.
        from src.api.routes.audit import log_audit
        dr, mt = _collect_harness_reads(b["tool_events"], harness=True)
        b_audit_rows = _harness_audit_rows(dr, mt, collection_id=COLLECTION_ID,
                                           mode="standard", run_id="ab")
        for action, rtype, rid, meta in b_audit_rows:
            log_audit(sb, action, rtype, rid, meta)
        derived_rows_total += len(b_audit_rows)

        print(f"  ARM A (search_vault)  gold={'HIT' if a_hit else 'MISS'}  "
              f"tools={a['tools']}  src={a['n_sources']}")
        print(f"     {a['answer'][:220]!r}")
        print(f"  ARM B (harness)       gold={'HIT' if b_hit else 'MISS'}  "
              f"cited={b_cited}  tools={b['tools']}  src={b['n_sources']}")
        print(f"     {b['answer'][:220]!r}")
        print(f"  L4: arm B would write {len(b_audit_rows)} audit row(s) "
              f"({[r[0] for r in b_audit_rows]})")
        verdict = "="
        if not a_hit and b_hit:
            verdict = "RECOVERED (A false-abstain → B cited answer)"
        elif a_hit and not b_hit:
            verdict = "REGRESSED (A hit → B missed)"
        print(f"  → {verdict}")
        rows.append({"q": q, "gold_any": item["gold_any"], "why": item["why"],
                     "A_hit": a_hit, "B_hit": b_hit, "B_cited": b_cited,
                     "verdict": verdict, "n_audit_rows_derived": len(b_audit_rows),
                     "A": {k: a[k] for k in ("answer", "tools", "n_sources")},
                     "B": {k: b[k] for k in ("answer", "tools", "n_sources")}})

    audit_after = _audit_count()
    written = audit_after - audit_before

    recovered = sum(1 for r in rows if not r["A_hit"] and r["B_hit"])
    regressed = sum(1 for r in rows if r["A_hit"] and not r["B_hit"])
    a_total = sum(1 for r in rows if r["A_hit"])
    b_total = sum(1 for r in rows if r["B_hit"])
    b_cited_total = sum(1 for r in rows if r["B_cited"])

    print("\n── HEADLINE ─────────────────────────────────────────────────────────────")
    print(f"  gold HITs:  A {a_total}/{len(rows)}  →  B {b_total}/{len(rows)}")
    print(f"  recovered (A false-abstain → B cited):  {recovered}")
    print(f"  regressed (A hit → B miss):             {regressed}")
    print(f"  arm B answers carrying a citation:      {b_cited_total}/{len(rows)}")
    print(f"  L4 audit rows derived (arm B):          {derived_rows_total}")
    print(f"  L4 audit rows ACTUALLY written (count diff): {written}")
    print("\n  BAR: recovered > 0  AND  regressed == 0  AND  every B hit cited  AND")
    print("       audit rows written (live) > 0  ⇒  Phase 2 gate GREEN.")

    with open("eval/ab_doc_harness_phase2_live_results.json", "w") as f:
        json.dump({"vault": "contracts", "doc": "Document.pdf (Sezzle)", "model": "gpt-5.4",
                   "a_hits": a_total, "b_hits": b_total, "b_cited": b_cited_total,
                   "recovered": recovered, "regressed": regressed,
                   "audit_rows_derived": derived_rows_total,
                   "audit_rows_written_live": written, "rows": rows}, f, indent=2)
    print("\n  wrote eval/ab_doc_harness_phase2_live_results.json")


if __name__ == "__main__":
    main()
