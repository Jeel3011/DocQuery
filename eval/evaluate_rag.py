"""
RAGAS Evaluation Runner for DocQuery.

Usage:
    # Baseline single-doc (dense + reranker + multi-query):
    python evaluate_rag.py --mode baseline --questions eval_questions_v2.json

    # Multi-doc with routing-recall (Phase 2+):
    python evaluate_rag.py --mode multidoc --questions eval_questions_multidoc.json

    # CI gate — exits non-zero on regression:
    python evaluate_rag.py --mode baseline --ci --baseline eval_results_baseline.json

    # Hybrid (BM25 + dense RRF + reranker + multi-query):
    python evaluate_rag.py --mode hybrid --questions eval_questions_v2.json
"""

import sys
import json
import argparse
from pathlib import Path

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.components.config import Config
from src.components.evaluation import RAGASEvaluator
from eval.routing_recall import compute_routing_recall


# ── Regression thresholds ─────────────────────────────────────────────────────
# A metric is "regressed" if it drops by more than this fraction vs. the saved
# baseline. Fail CI; block the merge.
_REGRESSION_TOLERANCE = 0.05        # 5 pp drop = fail
_ROUTING_RECALL_FLOOR = 0.80        # Stage-1 router must surface ≥80% of gold docs


def _load_baseline(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _check_regression(current: dict, baseline: dict) -> list[str]:
    """Return a list of failure messages (empty = no regression)."""
    failures = []
    cur_scores = current.get("aggregate_scores", {})
    base_scores = baseline.get("aggregate_scores", {})
    for metric, base_val in base_scores.items():
        if base_val is None:
            continue
        cur_val = cur_scores.get(metric)
        if cur_val is None:
            failures.append(f"  REGRESSION  {metric}: was {base_val:.4f}, now NaN (evaluation failed)")
            continue
        drop = base_val - cur_val
        if drop > _REGRESSION_TOLERANCE:
            failures.append(
                f"  REGRESSION  {metric}: was {base_val:.4f}, now {cur_val:.4f}  "
                f"(dropped {drop:.4f} > tolerance {_REGRESSION_TOLERANCE})"
            )
    # Routing recall
    cur_rr = current.get("routing_recall")
    if cur_rr is not None and cur_rr < _ROUTING_RECALL_FLOOR:
        failures.append(
            f"  REGRESSION  routing_recall: {cur_rr:.4f} < floor {_ROUTING_RECALL_FLOOR}"
        )
    return failures


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on DocQuery")
    parser.add_argument(
        "--mode",
        choices=["baseline", "hybrid", "multidoc"],
        default="baseline",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--questions",
        default=None,
        help="Path to questions JSON (default: auto-selected by mode)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Pinecone namespace override",
    )
    parser.add_argument(
        "--collection-id",
        default=None,
        dest="collection_id",
        help="Collection ID to scope multi-doc retrieval (Phase 1+)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: compare against --baseline and exit non-zero on regression",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to saved baseline results JSON for --ci comparison",
    )
    args = parser.parse_args()

    # Default question file per mode
    eval_dir = Path(__file__).parent
    default_questions = {
        "baseline": str(eval_dir / "eval_questions_v2.json"),
        "hybrid":   str(eval_dir / "eval_questions_v2.json"),
        "multidoc": str(eval_dir / "eval_questions_multidoc.json"),
    }
    questions_path = args.questions or default_questions[args.mode]
    output_path = args.output or str(eval_dir / f"eval_results_{args.mode}.json")

    print(f"\n{'='*62}")
    print("  DocQuery RAG Evaluation")
    print(f"  Mode       : {args.mode.upper()}")
    print(f"  Questions  : {questions_path}")
    print(f"  Output     : {output_path}")
    if args.namespace:
        print(f"  Namespace  : {args.namespace}")
    if args.collection_id:
        print(f"  Collection : {args.collection_id}")
    if args.ci:
        print(f"  CI mode    : ON  (baseline: {args.baseline or 'none'})")
    print(f"{'='*62}\n")

    config = Config()
    if args.namespace:
        config.PINECONE_NAMESPACE = args.namespace
    if args.mode == "hybrid":
        config.USE_HYBRID_SEARCH = True
        print("  [+] USE_HYBRID_SEARCH = True\n")
    else:
        config.USE_HYBRID_SEARCH = False

    evaluator = RAGASEvaluator(config)

    # ── Run standard RAGAS eval ───────────────────────────────────────────────
    scores = evaluator.evaluate(
        eval_dataset_path=questions_path,
        output_path=None,   # we'll save after augmenting
        mode=args.mode,
        collection_id=args.collection_id,
    )

    # ── Routing-recall (multi-doc mode only) ─────────────────────────────────
    if args.mode == "multidoc":
        print("\nComputing routing recall@N...")
        rr = compute_routing_recall(
            questions_path=questions_path,
            config=config,
            collection_id=args.collection_id,
        )
        scores["routing_recall"] = rr
        scores["routing_recall_floor"] = _ROUTING_RECALL_FLOOR
        print(f"  routing_recall@N = {rr:.4f}  (floor = {_ROUTING_RECALL_FLOOR})")
        if rr < _ROUTING_RECALL_FLOOR:
            print(f"  [WARN] routing_recall below floor — Stage-1 router missing docs")
        else:
            print(f"  [OK]   routing_recall above floor")

    # ── Single-doc non-regression run (always, as part of CI) ────────────────
    if args.ci:
        print("\nRunning single-doc non-regression suite...")
        single_doc_scores = evaluator.evaluate(
            eval_dataset_path=str(eval_dir / "eval_questions_v2.json"),
            output_path=None,
            mode="baseline",
            collection_id=None,
        )
        scores["single_doc_non_regression"] = single_doc_scores.get("aggregate_scores", {})

    # ── Save results ──────────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\n  Results saved -> {output_path}")

    # ── Summary banner ────────────────────────────────────────────────────────
    agg = scores.get("aggregate_scores", {})
    valid = {k: v for k, v in agg.items() if v is not None}
    avg = sum(valid.values()) / len(valid) if valid else 0.0
    print(f"\n{'='*62}")
    print(f"  OVERALL AVERAGE [{args.mode.upper()}]: {avg:.4f}")
    if avg >= 0.8:
        print("  [OK]   Pipeline quality GOOD")
    elif avg >= 0.5:
        print("  [WARN] Pipeline quality MODERATE")
    else:
        print("  [FAIL] Pipeline quality LOW")
    print(f"{'='*62}\n")

    # ── CI regression check ───────────────────────────────────────────────────
    if args.ci:
        if not args.baseline or not Path(args.baseline).exists():
            print("  [CI] No baseline file provided — saving current as new baseline")
            import shutil
            shutil.copy(output_path, args.baseline or output_path)
            sys.exit(0)

        baseline = _load_baseline(args.baseline)
        failures = _check_regression(scores, baseline)
        if failures:
            print("\n  [CI FAIL] Regressions detected:")
            for f in failures:
                print(f)
            print("\n  Merge blocked. Fix regressions before merging.\n")
            sys.exit(1)
        else:
            print("  [CI PASS] No regressions detected.\n")
            sys.exit(0)


if __name__ == "__main__":
    main()
