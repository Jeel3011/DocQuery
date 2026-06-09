"""Planner gate — BRAIN_REASONING_PLAN §5.3 / Stage C3.

Measures PLAN-ACCURACY on the §7 question set, **independent of execution** (§5.3): we
score the SHAPE of the QueryPlan `plan()` produces — the right sub-goal types, the right
dependency edges, and a named pivot binding where the question needs one — and run NO
retrieval and NO compute. The headline bar is **plan-accuracy ≥ 0.9**.

Two reasons the gate is structural-only:
  1. §5.3 demands independence from execution — a plan is correct if its DECOMPOSITION is
     right, regardless of the number the kernel would later read.
  2. The planner is deterministic (no LLM of its own), so it is fed a QueryIR. To isolate
     the PLANNER from comprehension drift (and to not burn the API on every run) the
     default mode builds the QueryIR fixtures DIRECTLY from each eval question's known
     structure — i.e. "given a correct IR, does the planner decompose it correctly?".
     `--llm` runs the full stack (real comprehend() → plan()) when you want the
     end-to-end signal; that exercises C2+C3 together and needs OPENAI_API_KEY.

What "correct plan" means per eval query_type (the EXPECTED structure):
  single_hop_control → exactly one `lookup` goal, no deps, no pivot binding.
  extremum_pivot     → a `select` goal that BINDS a pivot (pivot_year|pivot_entity);
                       if a second metric is asked, a `lookup` depends_on it. (= a bridge.)
  lookup_pivot       → a `select` goal binding pivot_year + a `lookup` depends_on it. (bridge)
  qualitative        → a single `exists` or `qualitative` goal, no kernel select. (The eval's
                       coarse 'qualitative' bucket is mostly yes/no predicates the finer IR
                       basis types `exists`; accepting either is honest, per test_comprehension.)

Run: python eval/test_planner.py            # default: IR-fixture mode (no LLM, offline)
     python eval/test_planner.py --llm       # full stack via comprehend() (needs API key)
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.brain.comprehension import QueryIR
from src.components.brain.executive import plan

QUESTIONS = "eval/eval_questions_multihop.json"


# ── IR fixtures: a correct QueryIR per eval question (mirrors what C2 comprehend()
#    produces — verified at 93% type-accuracy). The planner gate measures the planner,
#    so we hand it a KNOWN-GOOD IR and check the decomposition. Indexed by question text
#    prefix (stable across edits). Only the fields the planner reads are populated.
def _ir(qt, metrics=(), entities=(), periods=(), constraints=None) -> QueryIR:
    return QueryIR(raw="", question_type=qt, metrics=list(metrics),
                   entities=list(entities), periods=list(periods),
                   constraints=dict(constraints or {}))


# question-prefix → (QueryIR fixture, expected-plan spec). The expected spec is:
#   types:  multiset of sub-goal types the plan must contain
#   bridge: True ⇒ at least one goal has a dependency (a pivot feeds a downstream read)
#   binds:  a pivot-binding name that must appear on some goal (None ⇒ none required)
FIXTURES = {
    # ── lookup_pivot (stated condition picks a pivot, then read a value) ──
    "In the fiscal year that Microsoft's total revenue was closest": (
        _ir("lookup_pivot", ["total revenue"], ["Microsoft", "Alphabet"]),
        {"types": ["select", "lookup"], "bridge": True, "binds": "pivot_year"}),
    "Amazon's AWS segment operating income was $24,631": (
        _ir("lookup_pivot", ["operating income", "total net sales"], ["AWS", "Amazon"]),
        {"types": ["select", "lookup"], "bridge": True, "binds": "pivot_year"}),
    "In the fiscal year that Alphabet reported total revenue of about": (
        _ir("lookup_pivot", ["total revenue", "research and development"], ["Alphabet"]),
        {"types": ["select", "lookup"], "bridge": True, "binds": "pivot_year"}),
    "Microsoft's fiscal 2023 operating income was $88,523": (
        _ir("lookup_pivot", ["operating income", "total net sales"], ["Microsoft", "Amazon"]),
        {"types": ["select", "lookup"], "bridge": True, "binds": "pivot_year"}),
    "In the fiscal year Amazon posted a net loss": (
        _ir("lookup_pivot", ["net income", "total net sales"], ["Amazon"]),
        {"types": ["select", "lookup"], "bridge": True, "binds": "pivot_year"}),
    "Google Cloud first became profitable on an operating basis": (
        _ir("lookup_pivot", ["operating income", "research and development"], ["Google Cloud", "Alphabet"]),
        {"types": ["select", "lookup"], "bridge": True, "binds": "pivot_year"}),
    "In the fiscal year Microsoft completed its largest disclosed": (
        _ir("lookup_pivot", ["acquisition", "total net sales"], ["Microsoft", "Amazon"]),
        {"types": ["select", "lookup"], "bridge": True, "binds": "pivot_year"}),
    "AWS net sales were $90,757 million": (
        _ir("lookup_pivot", ["net sales", "operating income"], ["AWS", "Amazon"]),
        {"types": ["select", "lookup"], "bridge": True, "binds": "pivot_year"}),

    # ── extremum_pivot (an extremum binds a pivot, then maybe read a value) ──
    "In the fiscal year Amazon's AWS segment operating income first exceeded": (
        _ir("extremum_pivot", ["operating income", "total net sales"], ["AWS", "Amazon"],
            constraints={"predicate": "first_exceeds", "threshold": 20000}),
        {"types": ["select", "lookup"], "bridge": True, "binds": "pivot_year"}),
    "For the company among Google, Microsoft, and Amazon with the highest total": (
        _ir("extremum_pivot", ["total revenue", "research and development"],
            ["Google", "Microsoft", "Amazon"], ["2023"],
            constraints={"predicate": "argmax"}),
        {"types": ["select", "lookup"], "bridge": True, "binds": "pivot_entity"}),
    "Across these filings, in the fiscal year Microsoft acquired the largest": (
        _ir("extremum_pivot", ["acquisition", "total revenue"], ["Microsoft"],
            constraints={"predicate": "argmax"}),
        {"types": ["select", "lookup"], "bridge": True, "binds": "pivot_year"}),
    "Among Google, Microsoft, and Amazon, which company reported the lowest net income": (
        _ir("extremum_pivot", ["net income"], ["Google", "Microsoft", "Amazon"], ["2022"],
            constraints={"predicate": "argmin"}),
        {"types": ["select"], "bridge": False, "binds": "pivot_entity"}),
    "Looking at Amazon's AWS segment net sales over 2020-2023": (
        _ir("extremum_pivot", ["net sales"], ["AWS", "Amazon"],
            constraints={"predicate": "first_exceeds", "threshold": 80000}),
        {"types": ["select"], "bridge": False, "binds": "pivot_year"}),
    "For the company among Google, Microsoft, and Amazon with the highest net income": (
        _ir("extremum_pivot", ["net income", "net income"], ["Google", "Microsoft", "Amazon"], ["2021"],
            constraints={"predicate": "argmax"}),
        {"types": ["select", "lookup"], "bridge": True, "binds": "pivot_entity"}),
    "In the fiscal year Alphabet's R&D expense first exceeded $40": (
        _ir("extremum_pivot", ["research and development", "total revenue"], ["Alphabet"],
            constraints={"predicate": "first_exceeds", "threshold": 40000}),
        {"types": ["select", "lookup"], "bridge": True, "binds": "pivot_year"}),
    "Among Amazon's three reportable segments": (
        _ir("extremum_pivot", ["operating income"],
            ["North America", "International", "AWS"], ["2022"],
            constraints={"predicate": "argmax"}),
        {"types": ["select"], "bridge": False, "binds": "pivot_entity"}),
    "Over fiscal 2021 to 2023, in which year did Amazon's total net sales grow": (
        _ir("extremum_pivot", ["total net sales"], ["Amazon"],
            constraints={"predicate": "argmax"}),
        {"types": ["select"], "bridge": False, "binds": "pivot_year"}),

    # ── qualitative / exists (yes-no predicate; never the kernel-select path) ──
    "Alphabet reports a 'Google Cloud' operating segment": (
        _ir("exists", ["operating income"], ["Google Cloud", "Alphabet"], ["2023"]),
        {"types_any": ["exists", "qualitative"], "bridge": False, "binds": None}),
    "Did Amazon's International segment operate at a profit or a loss": (
        _ir("exists", ["operating income"], ["International", "Amazon"], ["2022"]),
        {"types_any": ["exists", "qualitative"], "bridge": False, "binds": None}),
    "Across fiscal 2021 to 2023, was Google Cloud's operating result": (
        _ir("exists", ["operating income"], ["Google Cloud"], ["2022"]),
        {"types_any": ["exists", "qualitative"], "bridge": False, "binds": None}),
    "Based on these filings, did Amazon's total net sales increase every": (
        _ir("exists", ["total net sales"], ["Amazon"]),
        {"types_any": ["exists", "qualitative"], "bridge": False, "binds": None}),
    "According to these filings, which had a larger fiscal 2023 operating loss": (
        _ir("exists", ["operating income"], ["Google Cloud", "Alphabet"], ["2023"]),
        {"types_any": ["exists", "qualitative"], "bridge": False, "binds": None}),

    # ── single_hop_control (one value, no pivot — must NOT become a bridge) ──
    "What was Amazon's total net sales in fiscal 2023?": (
        _ir("lookup", ["total net sales"], ["Amazon"], ["2023"]),
        {"types": ["lookup"], "bridge": False, "binds": None}),
    "What was Alphabet's total revenue in fiscal 2022?": (
        _ir("lookup", ["total revenue"], ["Alphabet"], ["2022"]),
        {"types": ["lookup"], "bridge": False, "binds": None}),
    "What was Microsoft's operating income in fiscal 2023?": (
        _ir("lookup", ["operating income"], ["Microsoft"], ["2023"]),
        {"types": ["lookup"], "bridge": False, "binds": None}),
    "What was Amazon's AWS segment operating income in fiscal 2023?": (
        _ir("lookup", ["operating income"], ["AWS", "Amazon"], ["2023"]),
        {"types": ["lookup"], "bridge": False, "binds": None}),
    "What was Alphabet's research and development expense in fiscal 2021?": (
        _ir("lookup", ["research and development"], ["Alphabet"], ["2021"]),
        {"types": ["lookup"], "bridge": False, "binds": None}),
}


def _match_fixture(question: str):
    """Find the FIXTURES entry whose key is a prefix of the question (stable lookup)."""
    for prefix, val in FIXTURES.items():
        if question.startswith(prefix):
            return val
    return None


def _score_plan(qp, spec) -> tuple[bool, str]:
    """Score a QueryPlan against the expected structure. Correct iff:
      • the multiset of goal types matches `types` (or — for qualitative — every goal's
        type is in `types_any`), AND
      • bridge presence matches `bridge` (a dep edge exists iff a pivot feeds a read), AND
      • a goal binds the required pivot variable `binds` (when one is required).
    Returns (ok, reason-on-fail). Pure structure — no execution.
    """
    got_types = sorted(g.type for g in qp.goals)

    if "types_any" in spec:
        allowed = set(spec["types_any"])
        if not all(g.type in allowed for g in qp.goals):
            return False, f"types {got_types} ⊄ {sorted(allowed)}"
    else:
        if got_types != sorted(spec["types"]):
            return False, f"types {got_types} ≠ {sorted(spec['types'])}"

    if qp.is_bridge != spec["bridge"]:
        return False, f"bridge={qp.is_bridge} ≠ expected {spec['bridge']}"

    want_bind = spec.get("binds")
    if want_bind is not None:
        if not any(g.binds == want_bind for g in qp.goals):
            bound = [g.binds for g in qp.goals]
            return False, f"missing pivot binding {want_bind!r} (got {bound})"

    # a bridge must actually wire the pivot var into a downstream input ($var reference)
    if spec["bridge"]:
        refs = " ".join(str(g.inputs) for g in qp.goals)
        if "$" not in refs:
            return False, "bridge has no $var reference wiring the pivot downstream"

    return True, ""


def main(use_llm: bool = False):
    raw = json.load(open(QUESTIONS))
    questions = [q for q in raw if "question" in q]

    llm = None
    if use_llm:
        from dotenv import load_dotenv
        load_dotenv()
        from src.components.config import Config
        from langchain_openai import ChatOpenAI
        from src.components.brain.comprehension import comprehend
        cfg = Config()
        llm = ChatOpenAI(model=cfg.LLM_MODEL_NAME, temperature=0.0,
                         api_key=cfg.OPENAI_API_KEY, request_timeout=20, max_retries=1)

    mode = "full stack (comprehend→plan)" if use_llm else "IR-fixture (planner-isolated)"
    print(f"Planner gate (§5.3) — plan-accuracy on n={len(questions)}  [{mode}]\n")

    from collections import defaultdict
    hits = 0
    skipped = 0
    by_type = defaultdict(lambda: {"hit": 0, "n": 0})

    for item in questions:
        q = item["question"]
        gold = item.get("query_type", "")
        fx = _match_fixture(q)
        if fx is None:
            print(f"  [?] no fixture for: {q[:70]}")
            skipped += 1
            continue
        ir_fixture, spec = fx

        if use_llm:
            from src.components.brain.comprehension import comprehend
            ir = comprehend(q, llm)
        else:
            ir = ir_fixture

        qp = plan(ir)
        ok, why = _score_plan(qp, spec)
        hits += int(ok)
        by_type[gold]["n"] += 1
        by_type[gold]["hit"] += int(ok)

        mark = "✓" if ok else "✗"
        shape = "→".join(f"{g.type}{'('+g.binds+')' if g.binds and g.binds!='answer' else ''}"
                         for g in qp.goals)
        print(f"  [{mark}] {gold:<19} {shape:<34} {'' if ok else '  ✗ ' + why}")
        if not ok:
            print(f"        Q: {q[:78]}")

    n = len(questions) - skipped
    acc = hits / n if n else 0.0
    print("\n" + "=" * 70)
    print(f"  PLAN-ACCURACY (headline): {hits}/{n} = {acc:.0%}   bar ≥ 90%")
    if skipped:
        print(f"  skipped (no fixture): {skipped}")
    print("  By eval type (hit / n):")
    for t in sorted(by_type):
        b = by_type[t]
        print(f"    {t:<20} {b['hit']:>2} / {b['n']:>2}")
    print("=" * 70)
    passed = acc >= 0.9 and skipped == 0
    print("  ✓ §5.3 bar MET (plan-accuracy ≥ 90%, all questions covered)" if passed
          else f"  ✗ BAR FAILED (plan-accuracy {acc:.0%}{', '+str(skipped)+' uncovered' if skipped else ''})")

    json.dump({"n": n, "plan_accuracy": acc, "hits": hits, "skipped": skipped,
               "mode": mode, "by_type": dict(by_type)},
              open("eval/planner_results.json", "w"), indent=2)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(use_llm="--llm" in sys.argv))
