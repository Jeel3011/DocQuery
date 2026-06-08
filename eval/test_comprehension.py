"""Comprehension gate — BRAIN_REASONING_PLAN §5.2 / Stage C2.

Measures QueryIR accuracy on the §7 question set, INDEPENDENT of downstream (no
retrieval, no compute — just question → QueryIR). The headline bar is **type-accuracy
≥ 0.9** (§5.2). We also report entity / period / constraint extraction as secondary
signals (these feed the planner; the planner gate measures their end use).

Ground truth = each eval question's `query_type`. The eval set predates the 7-type
QueryIR basis and uses a COARSER 4-type vocabulary, so each eval type maps to the SET of
QueryIR types it legitimately covers (an answer is correct if it lands any of them):
    single_hop_control → {lookup}
    extremum_pivot     → {extremum_pivot}
    lookup_pivot       → {lookup_pivot}
    qualitative        → {qualitative, exists}   ← the eval's "qualitative" bucket is
        actually mostly yes/no predicates ("profit or a loss?", "did X reach
        profitability?", "increase every year?") — which the finer basis classes as
        `exists`. Accepting either is HONEST, not metric-gaming: a profit/loss question
        genuinely is `exists`. (`compare`/`compound` arrive when the set broadens.)

Only the comprehension LLM is called (one cheap call per question). Requires OPENAI_API_KEY.

Run: python eval/test_comprehension.py
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.components.config import Config
from src.components.brain.comprehension import comprehend

QUESTIONS = "eval/eval_questions_multihop.json"

# eval query_type → the SET of acceptable QueryIR question_types (the eval is coarser)
TYPE_MAP = {
    "single_hop_control": {"lookup"},
    "extremum_pivot": {"extremum_pivot"},
    "lookup_pivot": {"lookup_pivot"},
    "qualitative": {"qualitative", "exists"},
}


def main():
    raw = json.load(open(QUESTIONS))
    questions = [q for q in raw if "question" in q]

    cfg = Config()
    from langchain_openai import ChatOpenAI
    # request_timeout is load-bearing: without it a single hung connection blocks the
    # whole 27-call run indefinitely (observed). Fail fast → comprehend() degrades that
    # one question to a minimal IR rather than stalling the gate.
    llm = ChatOpenAI(model=cfg.LLM_MODEL_NAME, temperature=0.0, api_key=cfg.OPENAI_API_KEY,
                     request_timeout=20, max_retries=1)

    print(f"Comprehension gate (§5.2) — QueryIR type-accuracy on n={len(questions)}\n")
    from collections import defaultdict
    type_hits = 0
    degraded = 0
    by_type = defaultdict(lambda: {"hit": 0, "n": 0})
    # secondary: did we recover a named entity / a constraint where one is expected?
    entity_hits = entity_n = 0
    for qi, item in enumerate(questions, 1):
        q = item["question"]
        gold_eval = item.get("query_type", "")
        accept = TYPE_MAP.get(gold_eval, {gold_eval})
        ir = comprehend(q, llm)
        ok = (ir.question_type in accept)
        type_hits += int(ok)
        degraded += int(ir.degraded)
        label = "/".join(sorted(accept))
        by_type[label]["n"] += 1
        by_type[label]["hit"] += int(ok)

        # secondary entity check: pivot/qualitative Qs name at least one entity
        if accept & {"extremum_pivot", "lookup_pivot", "qualitative", "exists"}:
            entity_n += 1
            entity_hits += int(len(ir.entities) >= 1)

        mark = "✓" if ok else "✗"
        print(f"  [{mark}] {label:<22} got={ir.question_type:<15} "
              f"ent={ir.entities} per={ir.periods} con={ir.constraints or '{}'}")
        if not ok:
            print(f"        Q: {q[:80]}")

    n = len(questions)
    acc = type_hits / n if n else 0.0
    print("\n" + "=" * 68)
    print(f"  TYPE-ACCURACY (headline): {type_hits}/{n} = {acc:.0%}   bar ≥ 90%")
    print(f"  degraded (parse-failed → minimal IR): {degraded}/{n}")
    if entity_n:
        print(f"  entity recovery (pivot/qualitative Qs name ≥1 entity): "
              f"{entity_hits}/{entity_n} = {entity_hits/entity_n:.0%}")
    print("  By type (hit / n):")
    for t in sorted(by_type):
        b = by_type[t]
        print(f"    {t:<16} {b['hit']:>2} / {b['n']:>2}")
    print("=" * 68)
    ok = acc >= 0.9
    print("  ✓ §5.2 bar MET (type-accuracy ≥ 90%)" if ok
          else f"  ✗ BAR FAILED (type-accuracy {acc:.0%} < 90%)")

    json.dump({"n": n, "type_accuracy": acc, "type_hits": type_hits,
               "degraded": degraded, "by_type": dict(by_type)},
              open("eval/comprehension_results.json", "w"), indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
