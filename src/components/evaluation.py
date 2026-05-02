"""
RAGAS Evaluation Module for DocQuery RAG Pipeline.

Uses RAGAS metrics to evaluate retrieval and generation quality:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are retrieved contexts relevant to the question?
- Context Recall: Was all needed information retrieved?
"""

import json
import math
import time
import traceback
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
        self, questions: List[Dict[str, Any]], mode: str = "baseline"
    ) -> Dict[str, List]:
        """Run the RAG pipeline on each question and collect results.

        Args:
            questions: List of {question, ground_truth} dicts.
            mode:      'baseline' (dense+reranker) or 'hybrid' (BM25+dense+reranker).

        Returns:
            Dict with keys: question, answer, contexts, ground_truth.
        """
        results = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        use_multi_query = self.config.USE_MULTI_QUERY

        for i, item in enumerate(questions, 1):
            question = item["question"]
            ground_truth = item["ground_truth"]

            print(f"  [{i}/{len(questions)}] [{mode.upper()}] {question[:65]}...")

            # Multi-query: generate variants then retrieve
            if use_multi_query:
                variants = self.generator.generate_query_variants(
                    question, n=self.config.MULTI_QUERY_COUNT
                )
                all_queries = [question] + variants
                docs = self.retrieval_mgr.retrieve_multi_query(all_queries)
            else:
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
                "Q%d: retrieved %d docs, answer length=%d", i, len(docs), len(answer)
            )

        return results

    def evaluate(
        self,
        eval_dataset_path: str,
        output_path: Optional[str] = None,
        mode: str = "baseline",
    ) -> Dict[str, Any]:
        """Run full RAGAS evaluation.

        Args:
            eval_dataset_path: Path to JSON file with test questions.
            output_path:       Optional path to save results JSON.
            mode:              'baseline' or 'hybrid' — recorded in output for comparison.

        Returns:
            Dict with aggregate scores, per-question breakdown, and pipeline config.
        """
        print("\n" + "=" * 60)
        print(f"RAGAS EVALUATION  [mode: {mode.upper()}]")
        print("=" * 60)

        # 1. Load questions
        questions = self.load_eval_dataset(eval_dataset_path)
        print(f"\nLoaded {len(questions)} test questions.\n")

        # 2. Run pipeline
        print("Running pipeline on questions...")
        t1 = time.time()
        pipeline_results = self.run_pipeline_on_questions(questions, mode=mode)
        t2 = time.time()
        print(f"\nPipeline complete in {t2 - t1:.2f}s\n")

        # 3. Create HuggingFace dataset for RAGAS
        dataset = Dataset.from_dict(pipeline_results)

        # 4. Evaluate with RAGAS
        print("Running RAGAS evaluation (this calls the LLM for scoring)...")
        t3 = time.time()
        ragas_result = None
        eval_error = None
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            llm = LangchainLLMWrapper(ChatOpenAI(model=self.config.LLM_MODEL_NAME))
            emb = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(model=self.config.EMBEDDING_MODEL_NAME)
            )
            ragas_result = evaluate(
                dataset=dataset,
                metrics=self.METRICS,
                llm=llm,
                embeddings=emb,
            )
        except Exception as e:
            eval_error = str(e)
            logger.error("RAGAS evaluation failed: %s", e)
            logger.debug("RAGAS traceback:\n%s", traceback.format_exc())
        t4 = time.time()

        # 5. Format results — B1: guard NaN values so JSON stays valid
        def _safe_float(v) -> float:
            """Return 0.0 for NaN/None so the score is always a valid JSON number."""
            try:
                f = float(v)
                return 0.0 if math.isnan(f) else f
            except (TypeError, ValueError):
                return 0.0

        scores = {
            "mode": mode,
            "pipeline_config": {
                "use_reranker": self.config.USE_RERANKER,
                "reranker_model": self.config.RERANKER_MODEL if self.config.USE_RERANKER else None,
                "use_multi_query": self.config.USE_MULTI_QUERY,
                "multi_query_count": self.config.MULTI_QUERY_COUNT if self.config.USE_MULTI_QUERY else None,
                "use_hybrid_search": self.config.USE_HYBRID_SEARCH,
                "hybrid_fetch_k": self.config.HYBRID_FETCH_K if self.config.USE_HYBRID_SEARCH else None,
                "top_k": self.config.TOP_K,
                "rerank_top_k": self.config.RERANK_TOP_K,
                "embedding_model": self.config.EMBEDDING_MODEL_NAME,
                "llm_model": self.config.LLM_MODEL_NAME,
            },
            "aggregate_scores": {
                metric: _safe_float(ragas_result[metric]) if ragas_result else 0.0
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
            "eval_status": "ok" if ragas_result else "failed",
            "eval_error": eval_error,
        }

        # Re-compute overall average ignoring any remaining NaN slots
        valid_scores = [
            v for v in scores["aggregate_scores"].values()
            if not math.isnan(v)
        ]
        scores["average_score"] = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0

        # Per-question breakdown from the ragas dataframe
        try:
            if ragas_result is None:
                raise RuntimeError("No RAGAS result available due to evaluation failure")
            df = ragas_result.to_pandas()
            per_question = []
            for _, row in df.iterrows():
                per_question.append({
                    "question": row.get("question", ""),
                    "answer": str(row.get("answer", ""))[:200],
                    "faithfulness": _safe_float(row.get("faithfulness")),
                    "answer_relevancy": _safe_float(row.get("answer_relevancy")),
                    "context_precision": _safe_float(row.get("context_precision")),
                    "context_recall": _safe_float(row.get("context_recall")),
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
