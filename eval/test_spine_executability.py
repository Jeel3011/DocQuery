"""Spine executability gate — SPINE_TRUST_BREAKS H1.

The planner gate (test_planner) scores plan STRUCTURE — right types, deps, pivot bindings.
It does NOT check that a plan is actually RUNNABLE. That blind spot let `lookup_pivot` and
`compare` sit in the coordinator's `_SPINE_TYPES` while their planner builders emitted a
`select` goal with NO kernel `op` — so the kernel rejected them ("operation not allowed:
None"), they always abstained, and the spine silently fell through to the old path on 2 of
its 3 declared types. A capability the spine pretended to have.

This gate asserts the invariant that closes H1: EVERY question type the coordinator
dispatches to the spine (`_SPINE_TYPES`) must produce a plan whose every `select`/`compute`
goal carries a whitelisted kernel op. If someone re-adds a type before building its op, this
fails loudly here instead of degrading to a silent fall-through in production.

No LLM, no retrieval — pure structural check over plan() output. Run:
    python eval/test_spine_executability.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.brain.comprehension import QueryIR
from src.components.brain.executive import plan
from src.components.brain.analyst import WHITELISTED_OPS
from src.components.brain.meta_reasoner import _SPINE_TYPES

PASS, FAIL = "PASS", "FAIL"
results = []


def check(desc, cond, detail=""):
    results.append((PASS if cond else FAIL, desc, detail))
    print(f"  [{PASS if cond else FAIL}] {desc}" + (f": {detail}" if detail else ""))


# A representative IR per spine type — enough structure that the builder emits its real
# selectors (so a missing op is exposed, not hidden by an early degrade).
def _representative_ir(qt):
    if qt == "extremum_pivot":
        return QueryIR(raw="which year did AWS operating income first exceed $20B",
                       question_type=qt, metrics=["operating income"], entities=["AWS"],
                       periods=[], constraints={"predicate": "first_exceeds", "threshold": 20000})
    if qt == "lookup_pivot":
        return QueryIR(raw="the year MSFT revenue was closest to $200B, what was net income",
                       question_type=qt, metrics=["total revenue", "net income"],
                       entities=["Microsoft"], periods=[], constraints={})
    if qt == "compare":
        return QueryIR(raw="was Amazon 2023 revenue higher than Microsoft's",
                       question_type=qt, metrics=["total revenue"],
                       entities=["Amazon", "Microsoft"], periods=["2023"], constraints={})
    # fallback generic
    return QueryIR(raw="x", question_type=qt, metrics=["net sales"], entities=["AWS"])


def main():
    print("Spine executability gate (SPINE_TRUST_BREAKS H1) — every dispatched type must "
          "plan an executable spine\n")
    print(f"  _SPINE_TYPES = {_SPINE_TYPES}\n")

    for qt in _SPINE_TYPES:
        qp = plan(_representative_ir(qt))
        bad = [g for g in qp.goals
               if g.type in ("select", "compute")
               and (g.selector or {}).get("op") not in WHITELISTED_OPS]
        check(f"{qt}: every select/compute goal carries a whitelisted kernel op",
              not bad,
              "OK" if not bad else f"op-less/invalid goals: "
              f"{[(g.id, (g.selector or {}).get('op')) for g in bad]} — this type would always "
              f"ABSTAIN and silently fall through (H1)")

    # Regression sentinel: the two types we PULLED must stay out until their op exists.
    for gone in ("lookup_pivot", "compare"):
        check(f"{gone} stays OUT of _SPINE_TYPES until its kernel op is built",
              gone not in _SPINE_TYPES,
              "absent ✓" if gone not in _SPINE_TYPES else
              "PRESENT — re-added without an executable op; build the op first")

    _summary()


def _summary():
    n = sum(1 for s, _d, _x in results if s == PASS)
    print("\n" + "=" * 70)
    ok = n == len(results)
    print(f"  {n}/{len(results)} checks passed " + ("✓ H1 closed — no dead spine type"
          if ok else "✗ a dispatched spine type is NOT executable (would silently fall through)"))
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
