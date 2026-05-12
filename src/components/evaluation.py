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
from src.components.generation import AnswerGenration
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

        # 5. Format results
        # B1 FIX: Return None for NaN/None instead of silently replacing with 0.0.
        # Replacing NaN with 0.0 inflates the average and misrepresents quality.
        # Instead we surface which metrics produced NaN so the caller can report
        # honestly (e.g. "4/6 valid, 2 NaN on math-heavy questions").
        def _safe_float(v) -> Optional[float]:
            """Return None for NaN/None, float otherwise."""
            try:
                f = float(v)
                return None if math.isnan(f) else f
            except (TypeError, ValueError):
                return None

        _metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        raw_scores = {
            k: _safe_float(ragas_result[k]) if ragas_result else None
            for k in _metric_keys
        }
        nan_metrics = [k for k, v in raw_scores.items() if v is None]
        valid_scores_map = {k: v for k, v in raw_scores.items() if v is not None}

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
            # raw_scores: None means NaN (LLM couldn't score it), float means valid
            "aggregate_scores": raw_scores,
            "nan_metrics": nan_metrics,
            "num_questions": len(questions),
            "pipeline_time_s": round(t2 - t1, 2),
            "eval_time_s": round(t4 - t3, 2),
            "eval_status": "ok" if ragas_result else "failed",
            "eval_error": eval_error,
        }

        # Average excludes NaN slots — never divide by zero
        if valid_scores_map:
            scores["average_score"] = round(sum(valid_scores_map.values()) / len(valid_scores_map), 4)
            if nan_metrics:
                scores["average_note"] = (
                    f"Excludes {len(nan_metrics)} NaN metric(s): {nan_metrics}. "
                    "NaN typically occurs when RAGAS cannot verify claims in formula-heavy text."
                )
            else:
                scores["average_note"] = "All metrics valid."
        else:
            scores["average_score"] = 0.0
            scores["average_note"] = "No valid metric scores — RAGAS evaluation may have failed entirely."

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
                    # Use None for NaN here too — honest per-question scores
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
                def _fmt(v): return f"{v:.3f}" if v is not None else " NaN"
                print(f"\n  Q{i}: {q['question'][:55]}...")
                print(f"      Faith: {_fmt(q['faithfulness'])} | "
                      f"Relevancy: {_fmt(q['answer_relevancy'])} | "
                      f"Ctx Prec: {_fmt(q['context_precision'])} | "
                      f"Ctx Recall: {_fmt(q['context_recall'])}")

        print(f"\n  Pipeline time: {scores['pipeline_time_s']}s")
        print(f"  RAGAS eval time: {scores['eval_time_s']}s")
        print(f"  Questions evaluated: {scores['num_questions']}")
        print("=" * 60)
