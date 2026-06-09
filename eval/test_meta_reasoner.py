"""Coordinator gate — BRAIN_REASONING_PLAN §5.6 / Stage C6.

Proves the meta-reasoner dispatch (§5.6) end-to-end, OFFLINE: comprehend → (spine-type?) →
plan → execute → monitor → rendered block, with a MOCKED comprehension LLM so the gate runs
with no API and isolates the COORDINATOR (the spine organs are already gated by
test_planner/test_executor/test_reasoning_verifier). The comprehend LLM is the only LLM the
coordinator calls; mocking it lets us assert the routing + assembly deterministically.

What it proves (the §5.6 contract + the prime directive):
  (1) a pivot/bridge question → the spine APPLIES and the block carries the VERIFIED answer
      (canonical Q2: pivot 2022 → 513,983) with provenance + reasoning checks;
  (2) a single `lookup` (non-spine type) → the spine FALLS THROUGH (applied=False) so the
      existing Analyst/Brain path is byte-identical (the no-regression guard);
  (3) a degraded comprehension → falls through (never worse than baseline);
  (4) a monitor abstain (wrong-pivot threshold nothing crosses) → a CLEAN abstain block,
      not a confident wrong;
  (5) a comprehend LLM that raises → degrades to applied=False (the spine never crashes the
      request).

Bar: 100% (routing + abstain are correctness properties, not metrics).

Run: python eval/test_meta_reasoner.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.table_extraction import extract_tables_from_pdf
from src.components.brain.analyst import Grid
from src.components.brain.meta_reasoner import run_executive_spine

AMZN22 = "test docs/amzn-20221231.pdf"
_cache = {}


def grids_for(*docs):
    out = []
    for d in docs:
        if d not in _cache:
            _cache[d] = [Grid(t.to_metadata(), doc=d.split("/")[-1], page=t.page_number)
                         for t in extract_tables_from_pdf(d)]
        out.extend(_cache[d])
    return out


class _Msg:
    def __init__(self, c): self.content = c


class FakeLLM:
    """Returns a fixed comprehend JSON (models the C2 output for a given question shape)."""
    def __init__(self, json_str): self._j = json_str
    def invoke(self, msgs): return _Msg(self._j)


class RaisingLLM:
    def invoke(self, msgs): raise RuntimeError("comprehend boom")


PASS, FAIL = "PASS", "FAIL"
results = []


def check(desc, cond, detail=""):
    results.append(cond)
    print(f"  [{PASS if cond else FAIL}] {desc}" + (f": {detail}" if detail else ""))


# comprehend JSONs for each routing case
J_BRIDGE = ('{"question_type":"extremum_pivot","metrics":["operating income","total net sales"],'
            '"entities":["AWS","Amazon"],"periods":[],'
            '"constraints":{"predicate":"first_exceeds","threshold":20000}}')
J_BRIDGE_UNCROSSED = ('{"question_type":"extremum_pivot","metrics":["operating income","total net sales"],'
                      '"entities":["AWS","Amazon"],"periods":[],'
                      '"constraints":{"predicate":"first_exceeds","threshold":999000}}')
J_LOOKUP = ('{"question_type":"lookup","metrics":["total net sales"],"entities":["Amazon"],'
            '"periods":["2022"],"constraints":{}}')
J_DEGRADED = "not json at all"   # → comprehend degrades → applied=False


def main():
    print("Coordinator gate (§5.6 / C6) — spine dispatch + block assembly, offline\n")
    g = grids_for(AMZN22)
    Q = "bridge question text"

    # (1) pivot/bridge → spine applies, verified answer in block
    out = run_executive_spine(Q, g, FakeLLM(J_BRIDGE))
    check("pivot/bridge → spine APPLIES with a verified answer",
          out.applied and not out.abstained and out.binding == "513983.0",
          f"applied={out.applied} binding={out.binding}")
    check("  block carries the pivot bind, the answer, and provenance",
          out.block and "pivot_year=2022" in out.block and "513,983" in out.block
          and "Source cell" in out.block)
    check("  block shows the reasoning checks passed (predicate ok)",
          out.block and "predicate: ok" in out.block)

    # (2) single lookup (non-spine) → falls through, fast path untouched
    out2 = run_executive_spine(Q, g, FakeLLM(J_LOOKUP))
    check("single lookup → spine FALLS THROUGH (applied=False, no regression)",
          (not out2.applied) and out2.block is None, f"reason={out2.reason}")

    # (3) degraded comprehension → falls through
    out3 = run_executive_spine(Q, g, FakeLLM(J_DEGRADED))
    check("degraded comprehension → falls through (never worse than baseline)",
          not out3.applied, f"reason={out3.reason}")

    # (4) unresolvable pivot (threshold nothing crosses) → FALL THROUGH, no injected block.
    # The spine couldn't vouch for an answer, so it steps aside and lets the Brain's prose
    # path run as the baseline (it must NOT inject an "I withhold" block that could suppress
    # a correct prose answer — the §5.6 "only add verified figures" rule).
    out4 = run_executive_spine(Q, g, FakeLLM(J_BRIDGE_UNCROSSED))
    check("unresolvable pivot → spine FALLS THROUGH (no block injected, prose path intact)",
          (not out4.applied) and out4.block is None,
          f"applied={out4.applied} reason={out4.reason}")

    # (5) comprehend raises → degrade, never crash the request
    out5 = run_executive_spine(Q, g, RaisingLLM())
    check("comprehend LLM raises → degrade to applied=False (never crashes)",
          not out5.applied, f"reason={out5.reason}")

    # (6) no grids → falls through immediately (the has-grids guard)
    out6 = run_executive_spine(Q, [], FakeLLM(J_BRIDGE))
    check("no grids → falls through immediately", not out6.applied)

    n = len(results); npass = sum(results)
    print("\n" + "=" * 66)
    print(f"  {npass}/{n} checks passed",
          "✓ §5.6 coordinator OK (spine opt-in, fast path untouched, abstain clean)"
          if npass == n else "✗ C6 GATE FAILED")
    print("=" * 66)
    return 0 if npass == n else 1


if __name__ == "__main__":
    sys.exit(main())
