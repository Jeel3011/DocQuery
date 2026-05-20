"""
DocQuery — Admin API Routes (Phase 5)

Provides internal endpoints for triggering and reading RAGAS evaluation runs.
These endpoints require authentication (same Bearer token as all other routes)
but are intended for admin/owner use only.

Why this is differentiating:
- Very few portfolio RAG projects have a live evaluation API endpoint.
- Demonstrates eval-driven development: you don't ship model changes without
  running evals, and you can trigger evals programmatically via API.
- Harvey runs "BigLaw Bench" before every model deployment — this is the
  student-scale equivalent.
"""

import os
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends

from src.api.dependencies import get_current_user

router = APIRouter(prefix="/admin", tags=["Admin"])

# Path to the default eval questions file (relative to project root)
_DEFAULT_EVAL_PATH = os.getenv("EVAL_QUESTIONS_PATH", "eval_questions_v2.json")
_DEFAULT_OUTPUT_PATH = os.getenv("EVAL_OUTPUT_PATH", "eval_results.json")


@router.post("/eval/run", status_code=202)
async def run_evaluation(
    questions_path: str = _DEFAULT_EVAL_PATH,
    sb=Depends(get_current_user),
):
    """
    Trigger a RAGAS evaluation run asynchronously via Celery.

    The evaluation runs in the background so it does not block the API.
    Use GET /admin/eval/results to poll for the latest results.

    Returns the Celery task_id so the caller can track progress.
    """
    try:
        from src.worker.tasks import run_evaluation_task  # noqa: F401
        task = run_evaluation_task.delay(
            questions_path=questions_path,
            output_path=_DEFAULT_OUTPUT_PATH,
            user_id=sb.user_id,
        )
        return {
            "task_id": task.id,
            "status": "started",
            "questions_path": questions_path,
            "message": "Evaluation started. Poll GET /api/v1/admin/eval/results for output.",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start evaluation task: {exc}",
        )


@router.get("/eval/results")
async def get_eval_results(sb=Depends(get_current_user)):
    """
    Return the latest RAGAS evaluation results from eval_results.json.

    Honest NaN handling: metrics that returned NaN (e.g., faithfulness on
    formula-heavy text) are surfaced as null rather than silently zeroed.
    The response includes an 'average_note' field explaining any exclusions.
    """
    output_path = Path(_DEFAULT_OUTPUT_PATH)
    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No evaluation results found. "
                "Run POST /api/v1/admin/eval/run first."
            ),
        )
    try:
        with open(output_path, "r") as f:
            results = json.load(f)
        return results
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read evaluation results: {exc}",
        )
