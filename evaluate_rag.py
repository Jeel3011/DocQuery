"""
RAGAS Evaluation Runner for DocQuery.

Usage:
    # Baseline (dense + reranker + multi-query):
    python evaluate_rag.py --mode baseline --questions eval_questions_v2.json --output eval_results_baseline.json

    # Hybrid (BM25 + dense RRF + reranker + multi-query):
    python evaluate_rag.py --mode hybrid --questions eval_questions_v2.json --output eval_results_hybrid.json

    # Quick defaults (baseline, v2 question set):
    python evaluate_rag.py
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.components.config import Config
from src.components.evaluation import RAGASEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on DocQuery")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "hybrid"],
        default="baseline",
        help="Retrieval mode: 'baseline' (dense+reranker) or 'hybrid' (BM25+dense RRF+reranker)",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=str(Path(__file__).parent / "eval_questions_v2.json"),
        help="Path to evaluation questions JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results JSON (default: eval_results_<mode>.json)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Optional Pinecone namespace override (default: use Config().PINECONE_NAMESPACE)",
    )
    args = parser.parse_args()

    # Default output path is named after the mode
    output_path = args.output or str(
        Path(__file__).parent / f"eval_results_{args.mode}.json"
    )

    print(f"\n{'='*60}")
    print("  DocQuery RAG Evaluation")
    print(f"  Mode       : {args.mode.upper()}")
    print(f"  Questions  : {args.questions}")
    print(f"  Output     : {output_path}")
    if args.namespace:
        print(f"  Namespace  : {args.namespace}")
    print(f"{'='*60}\n")

    # Build config — patch hybrid flag at runtime so no .env change needed
    config = Config()
    if args.namespace:
        config.PINECONE_NAMESPACE = args.namespace

    if args.mode == "hybrid":
        config.USE_HYBRID_SEARCH = True
        print("  [+] USE_HYBRID_SEARCH = True  (BM25 + Dense RRF activated)\n")
    else:
        config.USE_HYBRID_SEARCH = False
        print("  [i] USE_HYBRID_SEARCH = False (dense + reranker only)\n")

    evaluator = RAGASEvaluator(config)

    # Run evaluation
    scores = evaluator.evaluate(
        eval_dataset_path=args.questions,
        output_path=output_path,
        mode=args.mode,
    )

    # Summary banner
    agg = scores["aggregate_scores"]
    avg_score = sum(agg.values()) / len(agg)
    print(f"\n{'='*60}")
    print(f"  OVERALL AVERAGE SCORE  [{args.mode.upper()}]: {avg_score:.4f}")
    if avg_score >= 0.8:
        print("  [OK] RAG pipeline quality is GOOD")
    elif avg_score >= 0.5:
        print("  [WARN] RAG pipeline quality is MODERATE -- review weak metrics")
    else:
        print("  [FAIL] RAG pipeline quality is LOW -- needs improvement")
    print(f"  Results saved -> {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
