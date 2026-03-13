"""
RAGAS Evaluation Module for DocQuery RAG Pipeline.

Uses RAGAS metrics to evaluate retrieval and generation quality:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are retrieved contexts relevant to the question?
- Context Recall: Was all needed information retrieved?
"""

import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.components.config import Config
from src.components.retrieval import RetrievalManager
from src.components.genration import AnswerGenration
from src.logger import get_logger

logger = get_logger(__name__)


class RAGASEvaluator:
    """Evaluates the DocQuery RAG pipeline using RAGAS metrics."""

    METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

    def __init__(self, config: Config):
        self.config = config
        self.retrieval_mgr = RetrievalManager(config)
        self.generator = AnswerGenration(config)

    def load_eval_dataset(self, path: str) -> List[Dict[str, Any]]:
        """Load evaluation questions from a JSON file.
        
        Expected format:
        [
            {
                "question": "...",
                "ground_truth": "..."
            },
            ...
        ]
        """
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} evaluation questions from {path}")
        return data

    def run_pipeline_on_questions(
        self, questions: List[Dict[str, Any]]
    ) -> Dict[str, List]:
        """Run the RAG pipeline on each question and collect results.
        
        Returns a dict with keys: question, answer, contexts, ground_truth
        suitable for RAGAS evaluation.
        """
        results = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for i, item in enumerate(questions, 1):
            question = item["question"]
            ground_truth = item["ground_truth"]

            print(f"  [{i}/{len(questions)}] Querying: {question[:60]}...")

            # Retrieve
            docs = self.retrieval_mgr.retrieve(question)
            contexts = [doc.page_content for doc in docs]

            # Generate
            if docs:
                gen_result = self.generator.generate(question, docs)
                answer = gen_result["answer"]
            else:
                answer = "No relevant sources found."

            results["question"].append(question)
            results["answer"].append(answer)
            results["contexts"].append(contexts)
            results["ground_truth"].append(ground_truth)

            logger.info(
                f"Q{i}: retrieved {len(docs)} docs, answer length={len(answer)}"
            )

        return results

    def evaluate(
        self,
        eval_dataset_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run full RAGAS evaluation.
        
        Args:
            eval_dataset_path: Path to JSON file with test questions.
            output_path: Optional path to save results JSON.
            
        Returns:
            Dict with per-metric scores and per-question breakdown.
        """
        print("\n" + "=" * 60)
        print("RAGAS EVALUATION")
        print("=" * 60)

        # 1. Load questions
        questions = self.load_eval_dataset(eval_dataset_path)
        print(f"\nLoaded {len(questions)} test questions.\n")

        # 2. Run pipeline
        print("Running pipeline on questions...")
        t1 = time.time()
        pipeline_results = self.run_pipeline_on_questions(questions)
        t2 = time.time()
        print(f"\nPipeline complete in {t2 - t1:.2f}s\n")

        # 3. Create HuggingFace dataset for RAGAS
        dataset = Dataset.from_dict(pipeline_results)

        # 4. Evaluate with RAGAS
        print("Running RAGAS evaluation (this calls the LLM for scoring)...")
        t3 = time.time()
        ragas_result = evaluate(dataset=dataset, metrics=self.METRICS)
        t4 = time.time()
        print(f"RAGAS evaluation complete in {t4 - t3:.2f}s\n")

        # 5. Format results
        scores = {
            "aggregate_scores": {
                metric: float(ragas_result[metric])
                for metric in [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                ]
            },
            "num_questions": len(questions),
            "pipeline_time_s": round(t2 - t1, 2),
            "eval_time_s": round(t4 - t3, 2),
        }

        # Per-question breakdown from the ragas dataframe
        try:
            df = ragas_result.to_pandas()
            per_question = []
            for _, row in df.iterrows():
                per_question.append({
                    "question": row.get("question", ""),
                    "answer": str(row.get("answer", ""))[:200],
                    "faithfulness": float(row.get("faithfulness", 0)),
                    "answer_relevancy": float(row.get("answer_relevancy", 0)),
                    "context_precision": float(row.get("context_precision", 0)),
                    "context_recall": float(row.get("context_recall", 0)),
                })
            scores["per_question"] = per_question
        except Exception as e:
            logger.warning(f"Could not extract per-question scores: {e}")

        # 6. Print table
        self._print_results(scores)

        # 7. Save results
        if output_path:
            with open(output_path, "w") as f:
                json.dump(scores, f, indent=2)
            print(f"\nResults saved to: {output_path}")

        return scores

    @staticmethod
    def _print_results(scores: Dict[str, Any]):
        """Pretty-print the evaluation results."""
        print("=" * 60)
        print("AGGREGATE SCORES")
        print("=" * 60)
        print(f"{'Metric':<25} {'Score':>10}")
        print("-" * 37)
        for metric, score in scores["aggregate_scores"].items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {metric:<23} {score:>6.4f}  {bar}")
        print("-" * 37)

        if "per_question" in scores:
            print(f"\n{'='*60}")
            print("PER-QUESTION BREAKDOWN")
            print(f"{'='*60}")
            for i, q in enumerate(scores["per_question"], 1):
                print(f"\n  Q{i}: {q['question'][:55]}...")
                print(f"      Faith: {q['faithfulness']:.3f} | "
                      f"Relevancy: {q['answer_relevancy']:.3f} | "
                      f"Ctx Prec: {q['context_precision']:.3f} | "
                      f"Ctx Recall: {q['context_recall']:.3f}")

        print(f"\n  Pipeline time: {scores['pipeline_time_s']}s")
        print(f"  RAGAS eval time: {scores['eval_time_s']}s")
        print(f"  Questions evaluated: {scores['num_questions']}")
        print("=" * 60)
