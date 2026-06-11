"""A5 — end-to-end agent-core eval (AGENT_CORE_PLAN §5 A5, GATE A).

Runs the n=27 multihop measurement set through the REAL agent loop (`run_agent`,
the same code the live route drives), with the SAME scope assembly as
`src/api/routes/agent_core.py`, and classifies each answer 3-way
(CORRECT / ABSTAINED / WRONG) with the same judge as `brain_solo_eval`. WRONG-rate is
the headline; per-question it records cost/latency/steps and the full §3.6 event
trace, so a failure says EXACTLY what to fix (tool recall? gate too strict? prompt?)
— no architecture relitigating (§5 A5).

This is the instrument that ends the per-question whack-a-mole: one run surfaces every
structural failure at once, with traces, instead of one painful UI session each.

GATE A (the phase bar): WRONG ≤ 1/27 AND CORRECT ≥ 20/27 AND p50 standard latency
≤ 90s AND mean cost ≤ $0.40/question.

⚠️ API-BURNING — runs the live model (gpt-5.4 / Opus per env) over ~27 multi-step agent
loops. ON-DEMAND ONLY, when Jeel asks. Smoke-test with `--limit 1` first.

Run:  python -u eval/agentcore_eval.py [--limit N] [--mode standard|deep] [--out FILE]
Offline structure check (NO API): python -u eval/agentcore_eval.py --dry-run
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.components.config import Config

QUESTIONS = "eval/eval_questions_multihop.json"
OUT_DEFAULT = "eval/agentcore_results.json"

# Reuse brain_solo's proven collection scope + 3-way judge verbatim (one source of truth).
from eval.brain_solo_eval import _scope_to_collection, _classify
from langchain_openai import ChatOpenAI


# ── Scope assembly — MIRRORS src/api/routes/agent_core.py exactly ────────────────
def _build_scope(config, sb, collection_id, filenames, question):
    """Recreate the route's RunScope: doc_id↔filename map + per-doc grid preload."""
    from src.components.agent_core.registry import RunScope
    from src.components.brain.table_intent import load_grids_for_docs
    from src.components.retrieval import RetrievalManager

    doc_ids = sb.get_collection_document_ids(collection_id) or []
    docs = sb.client.table("documents").select("id,filename").in_(
        "id", doc_ids).eq("user_id", sb._user.id).execute()
    filename_by_doc = {d["id"]: d["filename"] for d in (docs.data or [])}
    routed = set(filenames)
    scoped_doc_ids = [did for did, fn in filename_by_doc.items() if fn in routed] or doc_ids

    grids = load_grids_for_docs(
        sb, scoped_doc_ids, question=question,
        filename_by_doc=filename_by_doc, per_doc_top=8,  # the BUG-C fix the route uses
    )
    return RunScope(
        collection_id=collection_id, doc_ids=scoped_doc_ids,
        filenames=list(filenames), grids=grids,
        retrieval_manager=RetrievalManager(config), db_client=sb,
        filename_by_doc=filename_by_doc, question=question,
    )


def _run_one(config, sb, collection_id, filenames, question, mode):
    """One full agent run. Returns (answer, trace, steps, tokens, abstained, latency_s)."""
    from src.components.agent_core.budgets import budget_for
    from src.components.agent_core.model import build_model
    from src.components.agent_core.prompt import system_prompt
    from src.components.agent_core.loop import run_agent
    from src.components.agent_core.registry import REGISTRY
    from src.components.agent_core.tracer import RunTracer
    import uuid

    scope = _build_scope(config, sb, collection_id, filenames, question)
    budget = budget_for(mode, config)
    sys_prompt = system_prompt("v1")
    model = build_model(mode, budget, config, system=sys_prompt)
    tracer = RunTracer(run_id=uuid.uuid4().hex[:12], question=question, mode=mode)

    trace, answer_parts = [], []
    steps = tokens = 0
    abstained = False
    t0 = time.perf_counter()
    for ev in run_agent(question, model=model, scope=scope, budget=budget,
                        system_prompt=sys_prompt, registry=REGISTRY):
        tracer.record(ev)
        t = ev.get("type")
        if t == "agent_step":
            steps = ev.get("n", steps)
        elif t == "tool_call":
            trace.append(f"call {ev.get('name')} {str(ev.get('args_summary'))[:120]}")
        elif t == "tool_result":
            trace.append(f"  → {ev.get('name')} ok={ev.get('ok')} {str(ev.get('summary'))[:120]}")
        elif t == "gate":
            trace.append(f"  GATE {ev.get('name')} pass={ev.get('pass')} {str(ev.get('detail'))[:100]}")
        elif t == "token":
            answer_parts.append(ev.get("text", ""))
        elif t == "meta":
            steps = ev.get("steps", steps)
            tokens = ev.get("tokens", tokens)
            abstained = bool(ev.get("abstained", False))
    health = tracer.finish()
    return ("".join(answer_parts).strip(), trace, steps, tokens, abstained,
            time.perf_counter() - t0, health)


# rough cost estimate (output-token-dominant; refine per vendor pricing if needed)
def _est_cost(tokens, mode):
    # gpt-5.4 ballpark blended $/1k tokens; standard vs deep differ by model tier.
    per_1k = 0.010 if mode == "standard" else 0.020
    return (tokens / 1000.0) * per_1k


def main():
    args = sys.argv[1:]
    limit = None
    mode = "standard"
    out = OUT_DEFAULT
    dry = "--dry-run" in args
    for i, a in enumerate(args):
        if a.startswith("--limit"):
            limit = int(a.split("=", 1)[1]) if "=" in a else int(args[i + 1])
        elif a.startswith("--mode"):
            mode = (a.split("=", 1)[1] if "=" in a else args[i + 1]).lower()
        elif a.startswith("--out"):
            out = a.split("=", 1)[1] if "=" in a else args[i + 1]

    raw = json.load(open(QUESTIONS))
    meta = next((q for q in raw if "_collection_id" in q), {})
    questions = [q for q in raw if "question" in q]
    if limit:
        questions = questions[:limit]
    collection_id = meta.get("_collection_id")
    if not collection_id:
        sys.exit("No _collection_id in questions file.")

    if dry:
        # Offline: just validate the harness wiring + the question set shape, no API.
        from collections import Counter
        by_type = Counter(q.get("query_type", "untyped") for q in questions)
        print(f"[dry-run] {len(questions)} questions, collection {collection_id}")
        print(f"[dry-run] query_types: {dict(by_type)}")
        print(f"[dry-run] mode={mode}, out={out}. Harness imports OK. No API spend.")
        # prove the scope/loop/budget/model builders import + construct (no model call)
        from src.components.agent_core.budgets import budget_for
        from src.components.agent_core.prompt import system_prompt
        b = budget_for(mode, Config())
        assert system_prompt("v1") and b.max_steps > 0
        print(f"[dry-run] budget(mode={mode}): steps={b.max_steps} "
              f"tokens={b.token_budget} wall={b.wall_clock_s}s")
        print("[dry-run] ✓ wiring valid — ready for the paid run when Jeel says go.")
        return 0

    config = Config()
    if not getattr(config, "USE_AGENT_CORE", False):
        print("⚠ USE_AGENT_CORE is off in config — the eval drives run_agent directly so it "
              "still runs, but set it true in .env to match the live route.")
    filenames, sb = _scope_to_collection(config, collection_id)
    judge_llm = ChatOpenAI(model="gpt-4o", temperature=0.0,
                           api_key=config.OPENAI_API_KEY, request_timeout=30)

    print(f"AGENT-CORE eval · mode={mode} · {len(questions)} questions · "
          f"collection {collection_id} → {len(filenames)} docs\n")

    from collections import defaultdict
    buckets = {"CORRECT": 0, "ABSTAINED": 0, "WRONG": 0}
    by_type = defaultdict(lambda: {"CORRECT": 0, "ABSTAINED": 0, "WRONG": 0})
    rows = []
    lat_list, cost_list = [], []

    for qi, item in enumerate(questions, 1):
        q = item["question"]
        required = item.get("answer_must_include", [])
        qtype = item.get("query_type", "untyped")
        try:
            answer, trace, steps, tokens, abstained, lat, health = _run_one(
                config, sb, collection_id, filenames, q, mode)
        except Exception as e:
            answer, trace, steps, tokens, abstained, lat, health = \
                "", [f"RUN ERROR: {e}"], 0, 0, True, 0.0, {"flags": [f"run error: {e}"]}
            print(f"[{qi}] run ERROR: {e}")
        bucket = _classify(q, required, answer, judge_llm)
        buckets[bucket] += 1
        by_type[qtype][bucket] += 1
        cost = _est_cost(tokens, mode)
        lat_list.append(lat); cost_list.append(cost)
        flags = health.get("flags", [])
        rows.append({
            "q": q, "query_type": qtype, "bucket": bucket,
            "answer_head": answer[:200], "steps": steps, "tokens": tokens,
            "latency_s": round(lat, 1), "est_cost_usd": round(cost, 3),
            "abstained": abstained, "health_flags": flags, "trace": trace,
        })
        print(f"[{qi}/{len(questions)}] {bucket:<9} [{qtype:<16}] {steps}st {lat:.0f}s "
              f"${cost:.2f} | {q[:50]}")
        print(f"       → {(answer[:150].strip() or '(empty)')}")
        # Surface loose-end flags inline — the whole point: see them across all 27 at once.
        for fl in flags:
            print(f"       ⚠ {fl}")

    n = len(questions)
    lat_list.sort()
    p50 = lat_list[len(lat_list) // 2] if lat_list else 0
    p95 = lat_list[int(len(lat_list) * 0.95)] if lat_list else 0
    mean_cost = sum(cost_list) / n if n else 0

    print("\n" + "=" * 68)
    print(f"  CORRECT   : {buckets['CORRECT']}/{n}")
    print(f"  ABSTAINED : {buckets['ABSTAINED']}/{n}   (safe — coverage gap, not wrong)")
    print(f"  WRONG     : {buckets['WRONG']}/{n}   (DANGEROUS — the headline)")
    print("=" * 68)
    print(f"  WRONG-rate: {buckets['WRONG']/n:.0%}   p50={p50:.0f}s  p95={p95:.0f}s  "
          f"mean ${mean_cost:.2f}/q")
    print("\n  per query_type:")
    for qt, b in sorted(by_type.items()):
        tot = sum(b.values())
        print(f"    {qt:<18} C={b['CORRECT']} A={b['ABSTAINED']} W={b['WRONG']}  (n={tot})")

    # GATE A
    gate_a = (buckets["WRONG"] <= 1 and buckets["CORRECT"] >= 20
              and p50 <= 90 and mean_cost <= 0.40) if n >= 27 else None
    if gate_a is None:
        print("\n  (partial run — GATE A needs the full n=27)")
    else:
        print(f"\n  GATE A: {'✓ PASS' if gate_a else '✗ FAIL'} "
              f"(WRONG≤1 CORRECT≥20 p50≤90s cost≤$0.40)")

    with open(out, "w") as f:
        json.dump({"mode": mode, "n": n, "buckets": buckets,
                   "p50_s": p50, "p95_s": p95, "mean_cost_usd": round(mean_cost, 3),
                   "by_type": {k: dict(v) for k, v in by_type.items()},
                   "gate_a": gate_a, "rows": rows}, f, indent=2, default=str)
    print(f"\n  per-question traces written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
