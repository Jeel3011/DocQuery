"""T0 — Abstain autopsy harness (tool_hard.md §T0; MEASUREMENT FIRST).

A CLASSIFICATION instrument, not a fixer. It runs a fixed question set through the REAL
agent loop (`run_agent` — the same code the live `/query/agentcore/stream` route drives,
with the SAME scope assembly) and **labels every non-answer by its CAUSE**, mapping each
run to a link in the §0.1 abstain pipeline:

  retrieval_empty           (A) — search_vault returned 0 / all-low / failed spans.
  tool_abstain_unrecovered  (B) — a table_lookup/compute abstain the agent never resolved.
  loop_death                (C) — budget exhausted (empty vs non-empty ledger distinguished).
  gate_redaction            (D) — the model produced a final answer the gate REDACTED; we
                                  additionally flag whether a redacted figure ACTUALLY traced
                                  to a ledger cell (the false-positive abstain → the T8 corpus).
  true_abstain                  — the documents genuinely don't answer it (the moat working).
  answered                      — a verified answer shipped (not a non-answer; for the denominator).

WHY this is a thin AGGREGATOR, not a new harness (tool_hard.md §0.3): `tracer.py` ALREADY
journals every event (repeated calls, dead/zero-provenance tools, gate fails, abstain,
budget death). T0 reuses that event stream verbatim and adds the ONE label the tracer
lacks — *did a gate-redacted figure actually trace*. It changes NO live code (it drives
`run_agent` directly, exactly like `eval/agentcore_eval.py`), so the §"flag-off = byte
identical" prime rule is honoured trivially: this file is never on the request path.

Output: a per-cause histogram + the transcript of every `gate_redaction` (candidate T8
bugs), written to `eval/abstain_autopsy_results.json`.

The acceptance test for the WHOLE plan (tool_hard.md): T0's `true_abstain` share rises (we
abstain only when we should) while every other CAUSE falls toward 0, with the WRONG-rate
corpus still 0.

⚠️ API-BURNING — runs the live model over multi-step agent loops. ON-DEMAND ONLY, when
Jeel asks. Smoke-test the WIRING with no spend first:  python -u eval/abstain_autopsy.py --dry-run
Then a single live question:                          python -u eval/abstain_autopsy.py --limit 1
Full run:                                             python -u eval/abstain_autopsy.py
Point at another collection's question set:           python -u eval/abstain_autopsy.py --questions eval/<file>.json
"""
import sys, os, json, time, uuid, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.components.config import Config

# Default to the multihop set — the one question file with a wired LIVE collection_id
# (finance + multi-hop). Legal / "what's in this vault" sets can be pointed at via
# --questions once their collections exist; the harness is corpus-agnostic.
QUESTIONS_DEFAULT = "eval/eval_questions_multihop.json"
OUT_DEFAULT = "eval/abstain_autopsy_results.json"


# ── Scope assembly — MIRRORS src/api/routes/agent_core.py (the live path) ────────────
# Self-contained (the old eval/brain_solo_eval.py helpers were deleted in the law-first
# sweep). We rebuild the route's RunScope: owner-resolved namespace + filename routing +
# per-doc grid preload (per_doc_top=20, the value the route actually uses).

def _scope_to_collection(config, collection_id):
    """Namespace = the collection owner; return (filenames, service-role db_client).
    Mirrors the route's owner resolution so retrieval hits the right Pinecone namespace."""
    from src.components.db import SupabaseManager
    svc = SupabaseManager(use_service_role=True)
    owner_row = (
        svc.client.table("collections").select("user_id")
        .eq("id", collection_id).single().execute().data
    )
    if owner_row and owner_row.get("user_id"):
        config.PINECONE_NAMESPACE = owner_row["user_id"]
        svc._user = type("User", (), {"id": owner_row["user_id"]})()
    filenames = [d["filename"] for d in svc.get_collection_documents(collection_id)]
    return filenames, svc


def _build_scope(config, sb, collection_id, filenames, question):
    """Recreate the route's RunScope exactly: doc_id↔filename map + per-doc grid preload."""
    from src.components.agent_core.registry import RunScope
    from src.components.brain.table_intent import load_grids_for_docs
    from src.components.retrieval import RetrievalManager

    doc_ids = sb.get_collection_document_ids(collection_id) or []
    owner_id = getattr(sb, "user_id", None) or getattr(getattr(sb, "_user", None), "id", None)
    docs = sb.client.table("documents").select("id,filename").in_(
        "id", doc_ids).eq("user_id", owner_id).execute()
    filename_by_doc = {d["id"]: d["filename"] for d in (docs.data or [])}
    routed = set(filenames)
    scoped_doc_ids = [did for did, fn in filename_by_doc.items() if fn in routed] or doc_ids

    grids = load_grids_for_docs(
        sb, scoped_doc_ids, question=question,
        filename_by_doc=filename_by_doc,
        per_doc_top=20,  # the value the live route uses (not the eval's old 8)
    )
    return RunScope(
        collection_id=collection_id, doc_ids=scoped_doc_ids,
        filenames=list(filenames), grids=grids,
        retrieval_manager=RetrievalManager(config), db_client=sb,
        filename_by_doc=filename_by_doc, question=question, config=config,
    )


# ── The per-cause classifier (the heart of T0) ───────────────────────────────────────
# Pure event-stream analysis + ONE recomputed label (figure-traced-but-redacted). No
# live-code hook — every signal below is already on the §3.6 event stream the tracer sees.

# These tools answer numeric questions; a failed/zero call from them is the (B) cause.
_NUMERIC_TOOLS = {"table_lookup", "compute", "list_metrics"}
_RETRIEVAL_TOOLS = {"search_vault"}


def _figure_traced_but_redacted(draft, ledger_payloads):
    """The ONE label the tracer lacks (tool_hard.md §T0/§0.3): of the figures in the model's
    DRAFT, did at least one ACTUALLY trace to a ledger cell — yet the answer was redacted?
    That is a false-positive abstain (the gate over-redacted a correct answer = a T8 bug).

    Reuses the SAME tracer used by the gate (`figure_traces_to_cells`) so this label is
    exactly the gate's own notion of 'traces', not a re-implementation that could disagree."""
    try:
        from src.components.brain.monitoring.invariants import figure_traces_to_cells
    except Exception:
        return None  # invariants unavailable → can't label; leave it unflagged

    # figure_traces_to_cells only reads `c.value`, so a minimal shim suffices — the SAME
    # trick gates.py uses (`_CellLike`), because the real CellRef has 8 required fields.
    class _CellLike:
        __slots__ = ("value",)
        def __init__(self, value): self.value = value

    cells = []
    for p in ledger_payloads or []:
        if not isinstance(p, dict):
            continue
        v = p.get("value")
        if isinstance(v, (int, float)):
            cells.append(_CellLike(float(v)))
    if not cells:
        return False  # nothing to trace against → not a false positive (genuinely ungrounded)
    chk = figure_traces_to_cells(draft or "", cells)
    # decided + ok ⇒ every stated figure traced; the gate redacting it anyway is the bug.
    # decided + not ok with SOME traced is also suspicious, but we report the clean signal.
    return bool(chk.decided and chk.ok and _has_a_figure(draft))


_NUM = None
def _has_a_figure(text):
    import re
    global _NUM
    if _NUM is None:
        _NUM = re.compile(r"\$?\s*\d[\d,]*\.?\d*")
    for tok in _NUM.findall(text or ""):
        t = tok.strip()
        if "," in t or "." in t or "$" in t:
            return True
        try:
            if abs(float(t.replace(",", ""))) >= 10000:
                return True
        except ValueError:
            continue
    return False


def classify_cause(events):
    """Map ONE run's §3.6 event stream to a single CAUSE label + supporting detail.

    Precedence is deliberate (most-specific failure wins): a run that both looped AND
    hit budget is `loop_death` only if it never grounded; a redaction is reported even
    when retrieval was weak, because the redaction is the link that lost the answer."""
    meta = next((e for e in reversed(events) if e.get("type") == "meta"), {})
    abstained = bool(meta.get("abstained"))
    degraded = bool(meta.get("degrade"))

    # Reconstruct the signals from the stream.
    tool_results = [e for e in events if e.get("type") == "tool_result"]
    gate_events = [e for e in events if e.get("type") == "gate"]
    sources_ev = next((e for e in reversed(events) if e.get("type") == "sources"), {})
    ledger_payloads = sources_ev.get("sources", []) or []
    # The model's last draft before the gate ran (loop.py yields resp.text[:500] as
    # agent_thought; the final `token` carries the possibly-REDACTED answer).
    thoughts = [e.get("text", "") for e in events if e.get("type") == "agent_thought"]
    last_draft = thoughts[-1] if thoughts else ""
    final_token = next((e.get("text", "") for e in reversed(events)
                        if e.get("type") == "token"), "")

    def _failed_or_zero(ev):
        return (not ev.get("ok")) or (ev.get("n_provenance") or 0) == 0

    retrieval_calls = [e for e in tool_results if e.get("name") in _RETRIEVAL_TOOLS]
    numeric_calls = [e for e in tool_results if e.get("name") in _NUMERIC_TOOLS]
    retrieval_all_dead = bool(retrieval_calls) and all(_failed_or_zero(e) for e in retrieval_calls)
    numeric_any_dead = any(_failed_or_zero(e) for e in numeric_calls)

    budget_hit = any(e.get("name") == "budget" and e.get("pass") is False for e in gate_events)
    output_redacted = any(
        e.get("name") == "output" and e.get("pass") is False for e in gate_events
    )
    # Distinguish a shipped answer from an empty one (the ledger may still hold figures).
    ledger_nonempty = len(ledger_payloads) > 0

    # ── If a verified answer shipped, it's not a non-answer (the denominator). ──────────
    if not abstained and not output_redacted and final_token.strip():
        return {"cause": "answered", "detail": "verified answer shipped",
                "false_positive": False}

    if degraded:
        return {"cause": "model_degrade", "detail": meta.get("error", "model error/degrade"),
                "false_positive": False}

    # ── (D) gate_redaction — the answer existed but the gate removed it. ────────────────
    if output_redacted:
        fp = _figure_traced_but_redacted(last_draft or final_token, ledger_payloads)
        gate_fails = [e for e in gate_events
                      if e.get("pass") is False and e.get("name") not in ("budget", "output")]
        why = "; ".join(f"{e.get('name')}: {e.get('detail','')}" for e in gate_fails)[:300]
        return {
            "cause": "gate_redaction",
            "detail": why or "output gate redacted ungated claims",
            "false_positive": bool(fp),  # the T8-corpus flag: a traced figure was redacted
            "draft_head": (last_draft or "")[:300],
            "final_head": (final_token or "")[:300],
        }

    # ── (C) loop_death — budget exhausted before finishing. ────────────────────────────
    if budget_hit:
        return {
            "cause": "loop_death",
            "detail": ("budget exhausted, ledger NON-EMPTY (partial figures rendered)"
                       if ledger_nonempty else
                       "budget exhausted, ledger EMPTY (nothing grounded)"),
            "false_positive": False,
            "ledger_nonempty": ledger_nonempty,
        }

    # ── (A) retrieval_empty — search_vault never produced usable spans. ─────────────────
    if retrieval_all_dead and not ledger_nonempty:
        return {"cause": "retrieval_empty",
                "detail": f"search_vault dead on all {len(retrieval_calls)} call(s)",
                "false_positive": False}

    # ── (B) tool_abstain_unrecovered — a numeric tool abstained, never resolved. ────────
    if numeric_any_dead and abstained:
        return {"cause": "tool_abstain_unrecovered",
                "detail": "table_lookup/compute abstained and the agent never recovered",
                "false_positive": False}

    # ── true_abstain — abstained with none of the above mechanical causes (the moat). ───
    if abstained or not final_token.strip():
        return {"cause": "true_abstain",
                "detail": "abstained with no mechanical cause — likely genuinely unanswerable",
                "false_positive": False}

    # Shouldn't reach here, but never crash a measurement run.
    return {"cause": "answered", "detail": "fell through (treated as answered)",
            "false_positive": False}


# ── Drive ONE run through the real loop, capturing every event ───────────────────────

def _run_one(config, sb, collection_id, filenames, question, mode):
    from src.components.agent_core.budgets import budget_for
    from src.components.agent_core.model import build_model
    from src.components.agent_core.prompt import system_prompt
    from src.components.agent_core.loop import run_agent
    from src.components.agent_core.registry import REGISTRY
    from src.components.agent_core.tracer import RunTracer

    scope = _build_scope(config, sb, collection_id, filenames, question)
    budget = budget_for(mode, config)
    sys_prompt = system_prompt("v1", mode=mode)
    model = build_model(mode, budget, config, system=sys_prompt)
    tracer = RunTracer(run_id=uuid.uuid4().hex[:12], question=question, mode=mode)

    events = []
    answer_parts = []
    t0 = time.perf_counter()
    for ev in run_agent(question, model=model, scope=scope, budget=budget,
                        system_prompt=sys_prompt, registry=REGISTRY):
        tracer.record(ev)
        if ev.get("type") != "token_delta":  # token_delta floods; the tracer keeps it
            events.append(ev)
        if ev.get("type") == "token":
            answer_parts.append(ev.get("text", ""))
    health = tracer.finish()
    return ("".join(answer_parts).strip(), events, health,
            time.perf_counter() - t0)


def main():
    args = sys.argv[1:]
    limit = None
    mode = "standard"
    out = OUT_DEFAULT
    questions_file = QUESTIONS_DEFAULT
    dry = "--dry-run" in args
    for i, a in enumerate(args):
        if a.startswith("--limit"):
            limit = int(a.split("=", 1)[1]) if "=" in a else int(args[i + 1])
        elif a.startswith("--mode"):
            mode = (a.split("=", 1)[1] if "=" in a else args[i + 1]).lower()
        elif a.startswith("--out"):
            out = a.split("=", 1)[1] if "=" in a else args[i + 1]
        elif a.startswith("--questions"):
            questions_file = a.split("=", 1)[1] if "=" in a else args[i + 1]

    raw = json.load(open(questions_file))
    meta = next((q for q in raw if "_collection_id" in q), {})
    questions = [q for q in raw if "question" in q]
    if limit:
        questions = questions[:limit]
    collection_id = meta.get("_collection_id")
    if not collection_id:
        sys.exit(f"No _collection_id in {questions_file} — point --questions at a set with one.")

    if dry:
        # Offline: validate wiring + the classifier on SYNTHETIC event streams, no API.
        from collections import Counter
        by_type = Counter(q.get("query_type", "untyped") for q in questions)
        print(f"[dry-run] {len(questions)} questions · collection {collection_id} · mode={mode}")
        print(f"[dry-run] query_types: {dict(by_type)}")
        # Prove the loop/scope/budget/model builders import + construct (no model call).
        from src.components.agent_core.budgets import budget_for
        from src.components.agent_core.prompt import system_prompt
        b = budget_for(mode, Config())
        assert system_prompt("v1", mode=mode) and b.max_steps > 0
        print(f"[dry-run] budget(mode={mode}): steps={b.max_steps} tokens={b.token_budget} "
              f"wall={b.wall_clock_s}s")
        # Smoke-test the classifier on hand-built event streams (the real $0 unit check).
        _self_test_classifier()
        print("[dry-run] ✓ wiring + classifier valid — ready for the paid run when Jeel says go.")
        return 0

    config = Config()
    if not getattr(config, "USE_AGENT_CORE", False):
        print("⚠ USE_AGENT_CORE is off in config — T0 drives run_agent directly so it still "
              "runs, but set it true in .env to match the live route.")
    filenames, sb = _scope_to_collection(config, collection_id)

    print(f"ABSTAIN AUTOPSY (T0) · mode={mode} · {len(questions)} questions · "
          f"collection {collection_id} → {len(filenames)} docs\n")

    from collections import Counter, defaultdict
    causes = Counter()
    by_type = defaultdict(Counter)
    rows = []
    redaction_corpus = []  # every gate_redaction (the T8 candidates)
    false_positive_redactions = 0

    for qi, item in enumerate(questions, 1):
        q = item["question"]
        qtype = item.get("query_type", "untyped")
        try:
            answer, events, health, lat = _run_one(
                config, sb, collection_id, filenames, q, mode)
        except Exception as e:
            answer, events, health, lat = "", [{"type": "meta", "abstained": True,
                                                "degrade": True, "error": str(e)}], \
                {"flags": [f"run error: {e}"]}, 0.0
            print(f"[{qi}] run ERROR: {e}")

        verdict = classify_cause(events)
        cause = verdict["cause"]
        causes[cause] += 1
        by_type[qtype][cause] += 1
        if cause == "gate_redaction":
            if verdict.get("false_positive"):
                false_positive_redactions += 1
            redaction_corpus.append({"q": q, "query_type": qtype, **verdict})

        rows.append({
            "q": q, "query_type": qtype, "cause": cause,
            "false_positive": verdict.get("false_positive", False),
            "detail": verdict.get("detail", ""),
            "answer_head": answer[:200], "latency_s": round(lat, 1),
            "health_flags": health.get("flags", []),
        })
        fp = " ⚠FALSE-POSITIVE(traced-but-redacted)" if verdict.get("false_positive") else ""
        print(f"[{qi}/{len(questions)}] {cause:<24}{fp} [{qtype:<14}] {lat:.0f}s | {q[:46]}")
        print(f"       → {verdict.get('detail','')[:120]}")

    n = len(questions)
    print("\n" + "=" * 70)
    print("  ABSTAIN-CAUSE HISTOGRAM (the T0 headline — what must fall toward 0):")
    print("=" * 70)
    # The non-answer causes (everything but 'answered') are what the plan drives down,
    # except 'true_abstain' which is the moat working (we WANT that to be the residue).
    order = ["answered", "true_abstain", "retrieval_empty", "tool_abstain_unrecovered",
             "loop_death", "gate_redaction", "model_degrade"]
    for c in order:
        if causes.get(c):
            tag = ("  ← the moat working (target: the ONLY large non-answer bucket)"
                   if c == "true_abstain" else
                   "  ← shipped (denominator)" if c == "answered" else
                   "  ← drive toward 0")
            print(f"    {c:<26} {causes[c]:>3}/{n}{tag}")
    if false_positive_redactions:
        print(f"\n  🔴 gate_redaction FALSE-POSITIVES (traced figure redacted) : "
              f"{false_positive_redactions}  → these ARE the T8 corpus.")
    else:
        print("\n  ✓ no false-positive redactions detected (a traced figure was never redacted).")

    print("\n  per query_type:")
    for qt, c in sorted(by_type.items()):
        parts = " ".join(f"{k}={v}" for k, v in c.most_common())
        print(f"    {qt:<18} {parts}")

    with open(out, "w") as f:
        json.dump({
            "mode": mode, "n": n, "questions_file": questions_file,
            "collection_id": collection_id,
            "causes": dict(causes),
            "false_positive_redactions": false_positive_redactions,
            "by_type": {k: dict(v) for k, v in by_type.items()},
            "redaction_corpus": redaction_corpus,  # the T8 candidate fixtures
            "rows": rows,
        }, f, indent=2, default=str)
    print(f"\n  per-question rows + the gate_redaction corpus written to {out}")
    print("  (feed redaction_corpus into eval/test_output_gates.py as T8 fixtures.)")
    return 0


# ── $0 self-test of the classifier on synthetic event streams (runs in --dry-run) ────

def _self_test_classifier():
    """Hand-built event streams exercise each CAUSE branch with no API. This is the
    regression guard for the classifier logic itself — the one piece of T0 that has
    real logic (the rest is plumbing the proven eval harness already exercises)."""
    cases = [
        # answered: a verified answer shipped.
        ([{"type": "tool_result", "name": "compute", "ok": True, "n_provenance": 2},
          {"type": "sources", "sources": [{"value": 513983.0}]},
          {"type": "token", "text": "Net sales were 513,983 [amzn p.41]."},
          {"type": "meta", "abstained": False}], "answered", False),
        # gate_redaction FALSE-POSITIVE: the draft's figure traces, yet output failed.
        ([{"type": "agent_thought", "text": "Alphabet FY21 R&D was 31,562 [goog p.62]."},
          {"type": "gate", "name": "verify_citations", "pass": False, "detail": "uncited"},
          {"type": "gate", "name": "output", "pass": False, "detail": "redacted"},
          {"type": "sources", "sources": [{"value": 31562.0}]},
          {"type": "token", "text": "I could not verify the requested figures."},
          {"type": "meta", "abstained": True}], "gate_redaction", True),
        # gate_redaction TRUE positive: ungrounded figure, no cell to trace.
        ([{"type": "agent_thought", "text": "Revenue was 999,999."},
          {"type": "gate", "name": "verify_numbers", "pass": False, "detail": "no cells"},
          {"type": "gate", "name": "output", "pass": False, "detail": "redacted"},
          {"type": "sources", "sources": []},
          {"type": "token", "text": "I could not verify the figures."},
          {"type": "meta", "abstained": True}], "gate_redaction", False),
        # loop_death (empty ledger).
        ([{"type": "gate", "name": "budget", "pass": False, "detail": "step budget"},
          {"type": "sources", "sources": []},
          {"type": "token", "text": "I ran out of my analysis budget."},
          {"type": "meta", "abstained": True}], "loop_death", False),
        # retrieval_empty.
        ([{"type": "tool_result", "name": "search_vault", "ok": True, "n_provenance": 0},
          {"type": "tool_result", "name": "search_vault", "ok": True, "n_provenance": 0},
          {"type": "sources", "sources": []},
          {"type": "token", "text": "I couldn't find anything relevant."},
          {"type": "meta", "abstained": True}], "retrieval_empty", False),
        # tool_abstain_unrecovered.
        ([{"type": "tool_result", "name": "search_vault", "ok": True, "n_provenance": 3},
          {"type": "tool_result", "name": "compute", "ok": False, "n_provenance": 0},
          {"type": "sources", "sources": [{"value": 1.0}]},
          {"type": "token", "text": "I could not compute that."},
          {"type": "meta", "abstained": True}], "tool_abstain_unrecovered", False),
        # true_abstain.
        ([{"type": "tool_result", "name": "search_vault", "ok": True, "n_provenance": 4},
          {"type": "sources", "sources": [{"value": 5.0}]},
          {"type": "token", "text": "The documents do not state a 2019 figure."},
          {"type": "meta", "abstained": True}], "true_abstain", False),
    ]
    for i, (events, want_cause, want_fp) in enumerate(cases, 1):
        v = classify_cause(events)
        assert v["cause"] == want_cause, \
            f"[classifier case {i}] cause: got {v['cause']!r}, want {want_cause!r}"
        assert bool(v.get("false_positive")) == want_fp, \
            f"[classifier case {i}] false_positive: got {v.get('false_positive')}, want {want_fp}"
    print(f"[dry-run] ✓ classifier self-test: {len(cases)}/{len(cases)} cause+fp labels correct")


if __name__ == "__main__":
    sys.exit(main())
