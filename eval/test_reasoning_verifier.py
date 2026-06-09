"""Reasoning-verifier gate — BRAIN_REASONING_PLAN §5.5 / Stage C5.

Proves the self-monitoring organ converts a confident WRONG → ABSTAIN while passing
genuine answers, DETERMINISTICALLY and OFFLINE (no LLM, no retrieval). The headline §5.5
gate is the live `brain_solo_eval.py` (WRONG-rate → ~0), which is run on demand; this gate
isolates the verifier the same way test_executor isolates the spine — feeding it real
plan→execute output plus deliberately-corrupted bindings that model the exact failure modes
§5.5 targets, and asserting the verdict.

What it proves:
  (1) a CORRECT spine result (real Q2 bridge: pivot 2022 → 513,983) PASSES — the monitor
      must not over-abstain on a right answer (the §7 abstention-usefulness guard);
  (2) a wrong-PIVOT first_exceeds (pivot tampered to 2021, which is BELOW the threshold)
      → ABSTAIN via the predicate check — the canonical confident-wrong the atomic verifier
      can't catch;
  (3) a "first" that isn't first (a prior period already exceeded) → ABSTAIN;
  (4) an INCOMPLETE argmax (saw 2 of 3 entities) → ABSTAIN via completeness;
  (5) an unbound pivot ($var never resolved) → ABSTAIN via the binding check;
  (6) a figure stated in the answer that traces to NO source cell → ABSTAIN (figure→cell);
  (7) an upstream spine abstain stays abstained (not resurrected).

Bar: 100% (a missed wrong-pivot is the §4a-forbidden confident-wrong this organ exists to
eliminate; an over-abstain on the correct case is the opposite failure §7 flags).

Run: python eval/test_reasoning_verifier.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.table_extraction import extract_tables_from_pdf
from src.components.brain.analyst import Grid, CellRef
from src.components.brain.comprehension import QueryIR
from src.components.brain.executive import plan, execute
from src.components.brain.executive.workspace import Binding, Workspace
from src.components.brain.executive.executor import ExecResult
from src.components.brain.monitoring import monitor

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


def _ir(qt, metrics=(), entities=(), periods=(), constraints=None):
    return QueryIR(raw="", question_type=qt, metrics=list(metrics), entities=list(entities),
                   periods=list(periods), constraints=dict(constraints or {}))


def _cell(period, value):
    return CellRef(doc="x", page=1, table_id="t", section="AWS",
                   label="Operating income", period=period, raw=str(value), value=float(value))


PASS, FAIL = "PASS", "FAIL"
results = []


def check(desc, cond, detail=""):
    results.append(cond)
    print(f"  [{PASS if cond else FAIL}] {desc}" + (f": {detail}" if detail else ""))


def main():
    print("Reasoning-verifier gate (§5.5 / C5) — WRONG→ABSTAIN over real spine output\n")
    g = grids_for(AMZN22)

    # ── (1) CORRECT bridge passes ────────────────────────────────────────────────
    ir = _ir("extremum_pivot", ["operating income", "total net sales"], ["AWS", "Amazon"],
             constraints={"predicate": "first_exceeds", "threshold": 20000})
    qp = plan(ir)
    res = execute(qp, g)
    v = monitor(qp, res, answer_text="In fiscal 2022 Amazon's total net sales were $513,983 million.")
    check("CORRECT bridge (pivot 2022 → 513,983) PASSES (no over-abstain)",
          v.ok, f"abstain={v.abstain} reason={v.reason[:50]}")
    check("  predicate check recorded as ok in the trail",
          any("predicate: ok" in c for c in v.checks), f"checks={v.checks[:2]}")

    # ── (2) wrong-PIVOT first_exceeds → predicate fails ──────────────────────────
    # tamper the resolved pivot to 2021 ($18,532M, BELOW the $20B threshold) — exactly the
    # §2 confident-wrong (a true sentence about the wrong year). The atomic verifier would
    # pass it; the reasoning verifier must NOT.
    bad = Binding(name="pivot_year", value="2021", op="first_exceeds", threshold=20000.0,
                  candidates=[_cell("2020", 13531), _cell("2021", 18532), _cell("2022", 22841)],
                  cells=[_cell("2021", 18532)], confidence=1.0)
    res.workspace.write(bad)
    res.workspace.write_alias("A", bad)
    v2 = monitor(qp, res)
    check("wrong-pivot first_exceeds (2021 is BELOW $20B) → ABSTAIN",
          v2.abstain and "predicate" in v2.failures, f"failures={v2.failures}")

    # ── (3) a 'first' that isn't first → predicate fails ─────────────────────────
    not_first = Binding(name="pivot_year", value="2022", op="first_exceeds", threshold=10000.0,
                        candidates=[_cell("2020", 13531), _cell("2021", 18532), _cell("2022", 22841)],
                        cells=[_cell("2022", 22841)], confidence=1.0)
    # threshold 10000: 2020 (13531) ALREADY exceeds it → 2022 is not the FIRST
    res3 = execute(qp, g)
    res3.workspace.write(not_first); res3.workspace.write_alias("A", not_first)
    v3 = monitor(qp, res3)
    check("'first' that isn't first (a prior period already exceeded) → ABSTAIN",
          v3.abstain and "predicate" in v3.failures, f"failures={v3.failures}")

    # ── (4) incomplete argmax → completeness fails ───────────────────────────────
    # Model the exact failure mode: the spine RESOLVED an argmax winner (ok=True) but the
    # op only saw 2 of the plan's 3 entity rows (one entity row didn't resolve). The
    # binding is non-abstained, so the completeness check must run and fire. We build the
    # ExecResult directly so res.ok=True (a real grid that drops an entity would do the
    # same, but constructing it keeps the test deterministic and on-point).
    ir4 = _ir("extremum_pivot", ["net income"], ["Google", "Microsoft", "Amazon"], ["2022"],
              constraints={"predicate": "argmax"})
    qp4 = plan(ir4)                       # plan lists 3 entity rows
    short = Binding(name="pivot_entity", value="AWS", op="argmax",
                    candidates=[_cell("2022", 100), _cell("2022", 200)],   # saw only 2 of 3
                    cells=[_cell("2022", 200)], confidence=1.0)
    ws4 = Workspace(); ws4.write(short); ws4.write_alias(qp4.final, short)
    res4 = ExecResult(ok=True, answer_binding=short, workspace=ws4)
    v4 = monitor(qp4, res4)
    check("incomplete argmax (saw 2 of 3 entities) → ABSTAIN via completeness",
          v4.abstain and "completeness" in v4.failures, f"failures={v4.failures}")

    # ── (5) unbound pivot → binding check fails ──────────────────────────────────
    # a bridge whose pivot abstained: the executor already abstains, so monitor() should
    # propagate the upstream abstain (spine failure), which is itself the §4a-correct path.
    ir5 = _ir("extremum_pivot", ["operating income", "total net sales"], ["AWS", "Amazon"],
              constraints={"predicate": "first_exceeds", "threshold": 999000})  # nothing crosses
    qp5 = plan(ir5)
    res5 = execute(qp5, g)
    v5 = monitor(qp5, res5)
    check("unresolvable pivot bridge → ABSTAIN (spine/binding)",
          v5.abstain, f"reason={v5.reason[:55]}")

    # ── (6) untraceable figure → figure→cell fails ───────────────────────────────
    v6 = monitor(qp, execute(qp, g),
                 answer_text="Amazon's total net sales were $999,999 million in fiscal 2022.")
    check("answer states a figure ($999,999) that traces to NO cell → ABSTAIN",
          v6.abstain and "figure→cell" in v6.failures, f"failures={v6.failures}")

    # ── (7) genuine figure (scaled) traces fine → passes ─────────────────────────
    v7 = monitor(qp, execute(qp, g),
                 answer_text="Amazon reported about $514 billion in total net sales for fiscal 2022.")
    check("scaled-but-real figure ($514B ≈ 513,983M) traces to a cell → PASS",
          v7.ok, f"abstain={v7.abstain} reason={v7.reason[:50]}")

    n = len(results); npass = sum(results)
    print("\n" + "=" * 66)
    print(f"  {npass}/{n} checks passed",
          "✓ §5.5 reasoning verifier OK (WRONG→ABSTAIN, no over-abstain)"
          if npass == n else "✗ C5 GATE FAILED")
    print("=" * 66)
    return 0 if npass == n else 1


if __name__ == "__main__":
    sys.exit(main())
