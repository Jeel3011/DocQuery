"""Executor gate — BRAIN_REASONING_PLAN §5.3 / Stage C4 (the spine, end to end).

Runs the FULL deterministic spine — `plan(ir) → execute(plan, grids)` — over REAL grids
extracted from the test PDFs, with NO LLM and NO retrieval. It proves the executive layer
SEQUENCES a bridge correctly: select the pivot over a complete series, BIND it in the
workspace, then read the follow-on metric FOR that bound pivot — or abstain cleanly when a
dependency can't be resolved. This is the behavior the §2 single-pass path gets wrong
(confident wrong year); here the pivot is a provenance-carrying object, not a word in one
LLM call.

Why offline + grid-fixture: the live end-to-end gate (`brain_solo_eval.py`) hits
Pinecone/Supabase/OpenAI and is run on demand. This gate isolates the EXECUTOR the same
way test_selection/test_grounding isolate the kernel/grounding — feeding it the directed-
retrieval grid set directly (the complete series the §5.3 executor arm would fetch) and a
KNOWN-GOOD QueryIR, so a failure indicts the executor, not retrieval or comprehension.

The bar is 100% on these checks (a wrong pivot binding is the §4a-forbidden confident-wrong
error this layer exists to eliminate; a missing dependency must ABSTAIN, never guess).

Run: python eval/test_executor.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.table_extraction import extract_tables_from_pdf
from src.components.brain.analyst import Grid
from src.components.brain.comprehension import QueryIR
from src.components.brain.executive import plan, execute

AMZN22 = "test docs/amzn-20221231.pdf"
AMZN23 = "test docs/amzn-20231231.pdf"

_grids_cache: dict = {}


def grids_for(*docs) -> list:
    """The directed-retrieval grid set for these docs (all grids in each PDF). Cached."""
    out = []
    for d in docs:
        if d not in _grids_cache:
            _grids_cache[d] = [
                Grid(t.to_metadata(), doc=d.split("/")[-1], page=t.page_number)
                for t in extract_tables_from_pdf(d)
            ]
        out.extend(_grids_cache[d])
    return out


def _ir(qt, metrics=(), entities=(), periods=(), constraints=None) -> QueryIR:
    return QueryIR(raw="", question_type=qt, metrics=list(metrics),
                   entities=list(entities), periods=list(periods),
                   constraints=dict(constraints or {}))


PASS, FAIL = "PASS", "FAIL"
results = []


def check(desc, cond, detail=""):
    results.append((PASS if cond else FAIL, desc, detail))
    print(f"  [{PASS if cond else FAIL}] {desc}" + (f": {detail}" if detail else ""))


def run_plan(ir, grids):
    qp = plan(ir)
    return qp, execute(qp, grids)


def main():
    print("Executor gate (§5.3 / C4) — full plan→execute spine over real grids\n")

    # ── Bridge 1: extremum_pivot — AWS op income first exceeds $20B → 2022, then read
    #    total net sales for 2022 → $513,983M. The canonical Q2 (the 2021-below trap).
    g = grids_for(AMZN22)
    ir = _ir("extremum_pivot", ["operating income", "total net sales"], ["AWS", "Amazon"],
             constraints={"predicate": "first_exceeds", "threshold": 20000})
    qp, res = run_plan(ir, g)
    pivot = res.workspace.value_of("pivot_year")
    check("bridge plan is a 2-goal DAG (select→lookup)",
          len(qp.goals) == 2 and qp.is_bridge, f"goals={[gg.type for gg in qp.goals]}")
    check("pivot_year BOUND to 2022 (first_exceeds, not the 2021-below trap)",
          str(pivot) == "2022", f"pivot_year={pivot}")
    check("follow-on lookup reads total net sales FOR the bound year → 513,983",
          res.ok and res.answer_binding and abs(res.answer_binding.value - 513983) < 1,
          f"answer={res.answer_binding.value if res.answer_binding else None}")
    check("answer carries provenance (≥1 source cell)",
          res.ok and len(res.cells) >= 1, f"cells={len(res.cells)}")

    # ── Bridge 2: extremum_pivot, pivot IS the answer — AWS net sales first exceeds $80B
    #    → 2022 (one metric, no follow-on read; the select binding is the final answer).
    ir2 = _ir("extremum_pivot", ["net sales"], ["AWS", "Amazon"],
              constraints={"predicate": "first_exceeds", "threshold": 80000})
    qp2, res2 = run_plan(ir2, grids_for(AMZN22, AMZN23))
    check("single-metric extremum → 1-goal plan, no bridge",
          len(qp2.goals) == 1 and not qp2.is_bridge, f"goals={[gg.type for gg in qp2.goals]}")
    check("AWS net sales first exceeds $80B → pivot binding 2022",
          res2.ok and str(res2.answer_binding.value) == "2022",
          f"answer={res2.answer_binding.value if res2.answer_binding else None}")

    # ── Control: a single_hop lookup must resolve directly (no pivot, no over-think).
    ir3 = _ir("lookup", ["total net sales"], ["Amazon"], ["2022"])
    qp3, res3 = run_plan(ir3, grids_for(AMZN22))
    check("single_hop lookup is a 1-goal plan (no select, no bridge)",
          len(qp3.goals) == 1 and qp3.goals[0].type == "lookup" and not qp3.is_bridge)
    check("single_hop total net sales 2022 → 513,983 (direct read)",
          res3.ok and abs(res3.answer_binding.value - 513983) < 1,
          f"answer={res3.answer_binding.value if res3.answer_binding else None}")

    # ── Abstain: a bridge whose pivot can't resolve must ABSTAIN, never guess the
    #    follow-on. Threshold no AWS op-income year crosses ($999B) → no pivot → abstain.
    ir4 = _ir("extremum_pivot", ["operating income", "total net sales"], ["AWS", "Amazon"],
              constraints={"predicate": "first_exceeds", "threshold": 999000})
    qp4, res4 = run_plan(ir4, grids_for(AMZN22))
    check("unresolvable pivot → whole bridge ABSTAINS (no follow-on guess)",
          (not res4.ok) and res4.answer_binding is not None
          and res4.answer_binding.abstained,
          f"ok={res4.ok}, reason={res4.reason[:60]}")

    # ── Routing: an exists/qualitative goal is DEFERRED to claims+verifier, not the
    #    kernel, and is NOT a confident-wrong (it's "the spine has nothing to compute").
    ir5 = _ir("exists", ["operating income"], ["International", "Amazon"], ["2022"])
    qp5, res5 = run_plan(ir5, grids_for(AMZN22))
    check("exists goal routed to claims+verifier (deferred, not kernel)",
          len(res5.needs_claims) == 1 and res5.needs_claims[0].type == "exists",
          f"needs_claims={[s.type for s in res5.needs_claims]}")

    # ── summary ────────────────────────────────────────────────────────────────
    n = len(results)
    npass = sum(1 for r in results if r[0] == PASS)
    print("\n" + "=" * 64)
    print(f"  {npass}/{n} checks passed", "✓ §5.3 executor spine OK" if npass == n
          else "✗ EXECUTOR GATE FAILED")
    print("=" * 64)
    return 0 if npass == n else 1


if __name__ == "__main__":
    sys.exit(main())
