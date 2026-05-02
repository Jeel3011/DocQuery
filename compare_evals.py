"""
DocQuery — Side-by-Side RAG Evaluation Comparator

Compares two eval_results JSON files (e.g. baseline vs hybrid) and
prints a formatted table with delta values and a final verdict.

Usage:
    python compare_evals.py                                          # default paths
    python compare_evals.py eval_results_baseline.json eval_results_hybrid.json
    python compare_evals.py --a eval_results_baseline.json --b eval_results_hybrid.json
"""

import sys
import json
import argparse
import math
from pathlib import Path

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

BAR_WIDTH = 20


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def bar(score: float, width: int = BAR_WIDTH) -> str:
    filled = int(score * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def delta_marker(d: float) -> str:
    if d > 0.005:
        return f"  {d:+.4f}  [UP]"
    elif d < -0.005:
        return f"  {d:+.4f}  [DOWN]"
    else:
        return f"  {d:+.4f}  [~]"


def compare(path_a: str, path_b: str):
    data_a = load(path_a)
    data_b = load(path_b)

    mode_a = data_a.get("mode", Path(path_a).stem).upper()
    mode_b = data_b.get("mode", Path(path_b).stem).upper()

    agg_a = data_a["aggregate_scores"]
    agg_b = data_b["aggregate_scores"]

    cfg_a = data_a.get("pipeline_config", {})
    cfg_b = data_b.get("pipeline_config", {})

    # ── Header ──────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  DocQuery RAG Evaluation — Comparison Report")
    print("=" * 72)
    print(f"  A (left)  : {mode_a}  [{Path(path_a).name}]")
    print(f"  B (right) : {mode_b}  [{Path(path_b).name}]")
    print()

    # ── Pipeline config diff ────────────────────────────────────────────────
    print("-" * 72)
    print(f"  {'Config Key':<30} {'A':>12}   {'B':>12}")
    print("-" * 72)
    all_keys = sorted(set(list(cfg_a.keys()) + list(cfg_b.keys())))
    for k in all_keys:
        va = str(cfg_a.get(k, "N/A"))
        vb = str(cfg_b.get(k, "N/A"))
        marker = "  <--" if va != vb else ""
        print(f"  {k:<30} {va:>12}   {vb:>12}{marker}")

    # ── Metric comparison table ─────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  {'Metric':<22} {'A Score':>8}  {'A Bar':<{BAR_WIDTH+2}}  {'B Score':>8}  {'B Bar':<{BAR_WIDTH+2}}  Delta")
    print("=" * 72)

    deltas = []
    for metric in METRICS:
        sa = agg_a.get(metric, 0.0)
        sb = agg_b.get(metric, 0.0)
        d = sb - sa
        deltas.append(d)
        print(
            f"  {metric:<22} {sa:>8.4f}  {bar(sa):<{BAR_WIDTH+2}}  "
            f"{sb:>8.4f}  {bar(sb):<{BAR_WIDTH+2}}{delta_marker(d)}"
        )

    # ── Averages ────────────────────────────────────────────────────────────
    avg_a = sum(agg_a.values()) / len(agg_a)
    avg_b = sum(agg_b.values()) / len(agg_b)
    avg_delta = avg_b - avg_a

    print("-" * 72)
    print(
        f"  {'AVERAGE':<22} {avg_a:>8.4f}  {bar(avg_a):<{BAR_WIDTH+2}}  "
        f"{avg_b:>8.4f}  {bar(avg_b):<{BAR_WIDTH+2}}{delta_marker(avg_delta)}"
    )

    # ── Timing ──────────────────────────────────────────────────────────────
    print()
    print("-" * 72)
    pt_a = data_a.get("pipeline_time_s", "?")
    pt_b = data_b.get("pipeline_time_s", "?")
    et_a = data_a.get("eval_time_s", "?")
    et_b = data_b.get("eval_time_s", "?")
    nq_a = data_a.get("num_questions", "?")
    nq_b = data_b.get("num_questions", "?")
    print(f"  Questions evaluated  : A={nq_a}   B={nq_b}")
    print(f"  Pipeline time        : A={pt_a}s  B={pt_b}s")
    print(f"  RAGAS eval time      : A={et_a}s  B={et_b}s")

    # ── Per-question breakdown ───────────────────────────────────────────────
    pq_a = {q["question"]: q for q in data_a.get("per_question", [])}
    pq_b = {q["question"]: q for q in data_b.get("per_question", [])}
    common_qs = [q for q in pq_a if q in pq_b]

    if common_qs:
        print()
        print("=" * 72)
        print("  Per-Question Delta  (B minus A)")
        print("=" * 72)
        for i, q in enumerate(common_qs, 1):
            qa = pq_a[q]
            qb = pq_b[q]
            print(f"\n  Q{i}: {q[:60]}...")
            for metric in METRICS:
                d = qb.get(metric, 0.0) - qa.get(metric, 0.0)
                if d > 0.01:
                    marker = "[UP]"
                elif d < -0.01:
                    marker = "[DOWN]"
                else:
                    marker = "[~]"
                print(f"       {metric:<22} A={qa.get(metric,0):.3f}  B={qb.get(metric,0):.3f}  delta={d:+.3f}  {marker}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    improvements = sum(1 for d in deltas if d > 0.005)
    regressions  = sum(1 for d in deltas if d < -0.005)
    if avg_delta > 0.01:
        verdict = f"[WINNER: {mode_b}]  +{avg_delta:.4f} avg improvement"
    elif avg_delta < -0.01:
        verdict = f"[WINNER: {mode_a}]  {mode_b} regressed by {abs(avg_delta):.4f}"
    else:
        verdict = "[TIE]  No meaningful difference between modes (<0.01 avg delta)"
    print(f"  {verdict}")
    print(f"  Metrics improved : {improvements}/4   Regressions: {regressions}/4")
    print("=" * 72)
    print()


def main():
    default_a = str(Path(__file__).parent / "eval_results_baseline.json")
    default_b = str(Path(__file__).parent / "eval_results_hybrid.json")

    parser = argparse.ArgumentParser(description="Compare two DocQuery RAG eval results")
    parser.add_argument("--a", type=str, default=default_a, help="Path to result A (baseline)")
    parser.add_argument("--b", type=str, default=default_b, help="Path to result B (hybrid)")
    # Positional fallback: compare_evals.py file_a file_b
    parser.add_argument("pos_a", nargs="?", default=None)
    parser.add_argument("pos_b", nargs="?", default=None)
    args = parser.parse_args()

    path_a = args.pos_a or args.a
    path_b = args.pos_b or args.b

    for p in [path_a, path_b]:
        if not Path(p).exists():
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)

    compare(path_a, path_b)


if __name__ == "__main__":
    main()
