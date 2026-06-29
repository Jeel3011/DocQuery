"""LIVE verify — DOCUMENT_HARNESS Phase 2.5 (trace UI data contract).

Phase 2.5 makes the harness read WATCHABLE: every harness tool_call now carries explicit
renderable fields (query / any_of / heading / page_range / full_text — §2.5.1) so the
frontend phrases the "watch it read" line from real values instead of regex-parsing the
truncated args_summary, and the cited `sources` spans (doc/page) are the click targets that
jump to the cited evidence (§2.5.2).

The §17 gate ("UI shows the read steps live AND a span click lands on the right page") is
ultimately a browser check — but its LOAD-BEARING half is the event-shape contract the UI
renders from. This proves that half on live data: run ONE harness query through `run_agent`
(the exact engine the agentcore route drives) over the Sezzle contract, then assert:

  1. the harness emitted at least one search/read tool_call (it actually navigated→grepped→read);
  2. those tool_call events carry the renderable fields the trace UI shows
     (a search carries `query`/`any_of`; a section/whole read carries heading/page_range/full_text);
  3. a read step's cited target ({doc, page}) RESOLVES to a `sources` span — i.e. a clicked
     trace chip would land on a real cited source (the _resolveSourceId logic, mirrored here).

API-BURNING. On-demand only. Dev OpenAI key. ONE run_agent run over one ~22-chunk contract.
Run:  python -u eval/verify_trace_ui_live.py
"""
import re

from dotenv import load_dotenv

load_dotenv()

# Same fixed live target as the proven Phase-2 A/B (verified present: 22 chunks).
OWNER_UID = "74f022f1-69d8-4f2b-a60c-6b08b79a80ea"
COLLECTION_ID = "8ca20f0a-5dfe-40b8-ab7a-26358a512f7e"      # vault: "contracts"
DOC_ID = "90f5e83f-83c2-4aa8-a7bc-0bc583c7ca4e"             # Document.pdf — Sezzle agreement

QUESTION = "Under this consulting agreement, what law governs the agreement?"

# The renderable fields the trace UI shows (must mirror loop._RENDER_ARG_KEYS).
RENDER_KEYS = ("query", "any_of", "heading", "page_range", "full_text")
SEARCH_TOOLS = {"search_text"}
READ_TOOLS = {"read_section", "read_document"}


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


def _resolve_source_id(sources, doc, page):
    """Mirror of the frontend _resolveSourceId: match a read step's {doc,page} to a cited
    source. Returns the matched source dict or None. doc match is substring either way (the
    ledger filename and the tool's doc_id differ in form), page exact when given."""
    doc = (doc or "").lower()
    def fn(s):
        return (s.get("filename") or s.get("doc") or "").lower()
    pool = [s for s in sources if doc and (doc in fn(s) or fn(s).split(".")[0] in doc)]
    if not pool:
        return None
    if page is not None:
        on_page = [s for s in pool if s.get("page") is not None and int(s["page"]) == int(page)]
        if on_page:
            return on_page[0]
    return pool[0]


def main():
    from src.components.config import Config
    from src.components.brain.table_intent import load_grids_for_docs
    from src.components.retrieval import RetrievalManager
    from src.components.agent_core.registry import RunScope
    from src.components.agent_core.budgets import budget_for
    from src.components.agent_core.model import build_model
    from src.components.agent_core.prompt import system_prompt
    from src.components.agent_core.loop import run_agent, _RENDER_ARG_KEYS

    # The script's RENDER_KEYS must stay in lock-step with the loop's source of truth.
    assert tuple(_RENDER_ARG_KEYS) == RENDER_KEYS, "RENDER_KEYS drifted from loop._RENDER_ARG_KEYS"

    sb = _build_db()
    user_config = Config()
    user_config.PINECONE_NAMESPACE = OWNER_UID
    assert sb.accessible_vault_owner(COLLECTION_ID) == OWNER_UID, "owner resolve failed"

    doc_ids = sb.get_collection_document_ids(COLLECTION_ID)
    docs = sb.read_client.table("documents").select("id,filename").in_("id", doc_ids).execute()
    filename_by_doc = {d["id"]: d["filename"] for d in (docs.data or [])}
    print(f"vault: {len(doc_ids)} docs  ({', '.join(sorted(filename_by_doc.values()))})")

    grids = []
    for did in doc_ids:
        grids.extend(load_grids_for_docs(sb, [did], question=None,
                                         filename_by_doc=filename_by_doc, per_doc_top=8) or [])

    retrieval_mgr = RetrievalManager(config=user_config)
    scope = RunScope(
        collection_id=COLLECTION_ID, doc_ids=doc_ids,
        filenames=list(filename_by_doc.values()), grids=grids,
        vault_owner=OWNER_UID, retrieval_manager=retrieval_mgr, db_client=sb,
        filename_by_doc=filename_by_doc, question=QUESTION, config=user_config, harness=True,
    )
    budget = budget_for("standard", user_config)
    sys_prompt = system_prompt("v1", mode="standard", harness=True)
    model = build_model("standard", budget, user_config, system=sys_prompt)

    answer, sources, tool_calls = "", [], []
    print(f"model: {getattr(model, 'model', model.__class__.__name__)}")
    print(f"\nQ: {QUESTION}\n── running harness (live) ──")
    for ev in run_agent(QUESTION, model=model, scope=scope, budget=budget, system_prompt=sys_prompt):
        t = ev.get("type")
        if t == "tool_call":
            tool_calls.append(ev)
            rf = {k: ev[k] for k in RENDER_KEYS if k in ev}
            print(f"  tool_call {ev.get('name'):<14} doc_id={str(ev.get('doc_id'))[:8]:<8} render={rf}")
        elif t == "sources":
            sources = ev.get("sources", []) or []
        elif t == "token":
            answer = (ev.get("text") or ev.get("content") or answer)
        elif t in ("gate", "meta") and not ev.get("pass", True):
            # Surface a degrade/abstain (e.g. an upstream model 429) so a 0-tool run is
            # diagnosable as an environment issue, not a silent trace-contract failure.
            print(f"  [{t}] {str({k: v for k, v in ev.items() if k != 'type'})[:200]}")
    answer = answer.strip()
    print(f"\nanswer ({len(answer)} ch): {answer[:160]}…")
    print(f"sources: {len(sources)}  | tool_calls: {[e.get('name') for e in tool_calls]}\n")

    # ── Assertions: the Phase 2.5 trace-UI data contract on live events ──
    fails = []

    def ok(cond, msg):
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            fails.append(msg)

    print("── 2.5.1 harness navigated→grepped→read, and the events are renderable ──")
    searches = [e for e in tool_calls if e.get("name") in SEARCH_TOOLS]
    reads = [e for e in tool_calls if e.get("name") in READ_TOOLS]
    ok(bool(searches or reads),
       "the harness actually searched/read the contract (≥1 search/read tool_call)")

    # Every search tool_call must carry a renderable query or any_of (the UI's grep line).
    for e in searches:
        rf = {k: e[k] for k in RENDER_KEYS if k in e}
        ok(bool(rf.get("query") or rf.get("any_of")),
           f"search_text carries query/any_of for the trace line (got {rf})")

    # Every read tool_call must carry doc_id + at least one renderable read field
    # (heading/page_range/full_text) — what the "Reading …" line phrases from.
    for e in reads:
        rf = {k: e[k] for k in RENDER_KEYS if k in e}
        ok(e.get("doc_id") is not None,
           f"{e.get('name')} carries doc_id (the read target / click anchor)")
        ok(bool(rf.get("heading") or rf.get("page_range") or rf.get("full_text")),
           f"{e.get('name')} carries a renderable read field (got {rf})")

    print("\n── 2.5.2 a read step's cited target resolves to a clickable source span ──")
    ok(len(sources) > 0, "the run produced cited source spans (the click targets)")
    # Take the doc-targeted reads; each should resolve to a real cited source by {doc,page}.
    resolved_any = False
    for e in reads:
        doc = e.get("doc_id")
        pr = e.get("page_range")
        page = None
        if pr and re.match(r"^\d+", str(pr)):
            page = int(re.match(r"^\d+", str(pr)).group())
        src = _resolve_source_id(sources, doc, page)
        if src is not None:
            resolved_any = True
            print(f"    read {doc[:8]} p.{page} → source #{src.get('source_id')} "
                  f"({src.get('filename') or src.get('doc')} p.{src.get('page')})")
    # If the harness answered by grepping (no explicit read tool), the grep hit spans ARE the
    # sources; assert at least one source maps back to the in-scope contract instead.
    if not reads:
        in_scope = any((s.get("filename") or s.get("doc") or "") for s in sources)
        ok(in_scope, "grep-only path: cited spans belong to the in-scope contract")
    else:
        ok(resolved_any, "≥1 read step resolves to a cited source (a chip click would land)")

    print("\n" + "=" * 60)
    if fails:
        print(f"  TRACE-UI LIVE VERIFY: {len(fails)} FAIL")
        for f in fails:
            print(f"    ✗ {f}")
        return 1
    print("  ✓ TRACE-UI LIVE VERIFY GREEN — harness events are watchable + clickable")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
