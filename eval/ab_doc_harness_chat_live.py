"""LIVE A/B (HARDER) — harness vs current on a FINANCE vault WITH real embeddings.

The first A/B (eval/ab_doc_harness_live.py) ran on a prose-contract vault whose namespace
had no embeddings, so arm A's search_vault returned 0/0 — a strong but degenerate miss.
This one runs the CHAT path (run_agent over a whole collection) on the `very big` US 10-K
vault, whose namespace DOES have live embeddings (probed: search_vault returns 20/20 above
threshold). So arm A is the retriever doing its genuine best on big 360-chunk docs — the
honest "does the harness still win when RAG actually works?" test.

Questions are PROSE facts verifiable directly in goog-20231231's own chunks (non-circular),
the kind top-k retrieval fumbles on a 360-chunk doc (buried MD&A, near-identical year
phrasing). The numeric moat (table_lookup/compute) is unchanged by the harness, so a fair
text-retrieval A/B targets PROSE, not numbers.

  • Arm A (current):  harness=False → standard mode, search_vault top-k.
  • Arm B (harness):  harness=True  → standard mode, ls/grep/read the real docs.

For each question we run both arms and check whether the gold fact appears, grounded, in
the answer. Headline: how many golds arm A misses that arm B recovers, and 0 new wrong.

API-BURNING. On-demand only. Dev OpenAI key (gpt-5.4), not Claude.
Run:  python -u eval/ab_doc_harness_chat_live.py
"""
import os
import json

from dotenv import load_dotenv

load_dotenv()

OWNER_UID = "74f022f1-69d8-4f2b-a60c-6b08b79a80ea"
COLLECTION_ID = "79ae7a9c-7f27-4bdf-be11-0c5b640105de"   # vault: "very big" (US 10-Ks)

# Prose facts verified present in goog-20231231.pdf's own chunks this session.
QUESTIONS = [
    {
        "q": "In Alphabet's 2023 10-K, what percentage of consolidated revenues were "
             "international in 2023?",
        "gold_any": ["53%", "approximately 53", "53 percent"],
        "why": "verbatim: 'International revenues accounted for approximately 53% of our "
               "consolidated revenues in 2023'",
    },
    {
        "q": "In Alphabet's 2023 10-K, roughly what share of total revenues came from "
             "online advertising in 2023?",
        "gold_any": ["75%", "more than 75", "greater than 75", "75 percent"],
        "why": "verbatim: 'more than 75% of total revenues from online advertising in 2023'",
    },
    {
        "q": "In Alphabet's 2023 10-K, how do the rights of the different classes of "
             "common/capital stock differ?",
        "gold_any": ["voting", "identical", "except with respect to voting"],
        "why": "verbatim: 'rights ... are identical, except with respect to voting'",
    },
]


def _build_db():
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
    sys_prompt = system_prompt("v1", mode="standard")
    model = build_model("standard", budget, user_config, system=sys_prompt)

    answer = ""
    n_sources = 0
    tool_calls = []
    for ev in run_agent(question, model=model, scope=scope, budget=budget,
                        system_prompt=sys_prompt):
        t = ev.get("type")
        if t == "token":
            # The loop emits the WHOLE final answer as one 'token' event (text=final_text).
            answer = ev.get("text", "") or ev.get("content", "") or answer
        elif t == "sources":
            n_sources = len(ev.get("sources", []) or [])
        elif t == "tool_call":
            tool_calls.append(ev.get("name"))
    return {"answer": answer.strip(), "n_sources": n_sources, "tools": tool_calls}


def _hits_gold(answer, gold_any):
    a = (answer or "").lower()
    return any(g.lower() in a for g in gold_any)


def main():
    from src.components.config import Config
    from src.components.brain.table_intent import load_grids_for_docs
    from src.components.retrieval import RetrievalManager

    sb = _build_db()
    user_config = Config()
    user_config.PINECONE_NAMESPACE = OWNER_UID
    assert sb.accessible_vault_owner(COLLECTION_ID) == OWNER_UID, "owner resolve failed"

    doc_ids = sb.get_collection_document_ids(COLLECTION_ID)
    docs = sb.read_client.table("documents").select("id,filename").in_("id", doc_ids).execute()
    filename_by_doc = {d["id"]: d["filename"] for d in (docs.data or [])}
    print(f"vault: {len(doc_ids)} docs  ({', '.join(sorted(filename_by_doc.values()))})\n")

    # Grids preloaded once (the numeric moat is shared by both arms — unchanged by harness).
    grids = []
    try:
        for did in doc_ids:
            grids.extend(load_grids_for_docs(sb, [did], question=None,
                                             filename_by_doc=filename_by_doc, per_doc_top=8) or [])
    except Exception as exc:  # noqa: BLE001
        print(f"(grid preload degraded: {exc})")
    retrieval_mgr = RetrievalManager(user_config)
    vault_owner = sb.accessible_vault_owner(COLLECTION_ID)

    rows = []
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
        print(f"  ARM A (search_vault)  gold={'HIT' if a_hit else 'MISS'}  "
              f"tools={a['tools']}  src={a['n_sources']}")
        print(f"     {a['answer'][:200]!r}")
        print(f"  ARM B (harness)       gold={'HIT' if b_hit else 'MISS'}  "
              f"tools={b['tools']}  src={b['n_sources']}")
        print(f"     {b['answer'][:200]!r}")
        verdict = "="
        if not a_hit and b_hit:
            verdict = "RECOVERED (A missed → B hit)"
        elif a_hit and not b_hit:
            verdict = "REGRESSED (A hit → B missed)"
        print(f"  → {verdict}")
        rows.append({"q": q, "gold_any": item["gold_any"], "A_hit": a_hit, "B_hit": b_hit,
                     "verdict": verdict, "A": a, "B": b})

    recovered = sum(1 for r in rows if not r["A_hit"] and r["B_hit"])
    regressed = sum(1 for r in rows if r["A_hit"] and not r["B_hit"])
    a_total = sum(1 for r in rows if r["A_hit"])
    b_total = sum(1 for r in rows if r["B_hit"])
    print("\n── HEADLINE ─────────────────────────────────────────────────────────────")
    print(f"  gold HITs:  A {a_total}/{len(rows)}  →  B {b_total}/{len(rows)}")
    print(f"  recovered (A miss → B hit):  {recovered}")
    print(f"  regressed (A hit → B miss):  {regressed}")
    print(f"\n  BAR: B hits ≥ A hits  AND  0 regressions (harness no worse with real embeddings).")

    with open("eval/ab_doc_harness_chat_live_results.json", "w") as f:
        json.dump({"vault": "very big", "model": "gpt-5.4",
                   "a_hits": a_total, "b_hits": b_total, "recovered": recovered,
                   "regressed": regressed, "rows": rows}, f, indent=2)
    print("\n  wrote eval/ab_doc_harness_chat_live_results.json")


if __name__ == "__main__":
    main()
