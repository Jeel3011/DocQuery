"""
RAGAS Evaluation Runner for DocQuery.

Usage:
    python evaluate_rag.py
    python evaluate_rag.py --questions path/to/questions.json
    python evaluate_rag.py --output path/to/results.json
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
        "--questions",
        type=str,
        default=str(Path(__file__).parent / "eval_questions.json"),
        help="Path to evaluation questions JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "eval_results.json"),
        help="Path to save evaluation results",
    )
    args = parser.parse_args()

    # Initialize
    config = Config()
    evaluator = RAGASEvaluator(config)

    # Run evaluation
    scores = evaluator.evaluate(
        eval_dataset_path=args.questions,
        output_path=args.output,
    )

    # Summary
    agg = scores["aggregate_scores"]
    avg_score = sum(agg.values()) / len(agg)
    print(f"\n{'='*60}")
    print(f"  OVERALL AVERAGE SCORE: {avg_score:.4f}")
    if avg_score >= 0.8:
        print("  ✅ RAG pipeline quality is GOOD")
    elif avg_score >= 0.5:
        print("  ⚠️  RAG pipeline quality is MODERATE — review weak metrics")
    else:
        print("  ❌ RAG pipeline quality is LOW — needs improvement")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
