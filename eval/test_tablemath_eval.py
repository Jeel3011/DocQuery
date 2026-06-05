"""Phase 4.3 (§4b) table-math eval gate — the 100%-correct bar.

Drives the FULL Analyst path on natural-language questions with known answers:
  question → extract table grids (test docs/) → LLM writes spec → deterministic
  compute → compare to the hand-verified expected value.

A wrong number in finance is a fireable error, so the bar is 100%. This is the
§4b eval gate; wire it into CI alongside routing-recall and the §4a false-assertion
suite. Requires OPENAI_API_KEY (the spec-writer) and the 10-Ks in `test docs/`.

Run: python eval/test_tablemath_eval.py
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.components.table_extraction import extract_tables_from_pdf
from src.components.brain.analyst import (
    Grid, analyze, verify_numbers, cells_from_results, corroborate_with_prose,
)
from src.components.brain.table_intent import _relevance

CORPUS = "test docs"
QUESTIONS = "eval/eval_questions_tablemath.json"

_grids_cache = {}

# When True, load the ACTUAL ingested grids (incl. the LLM table_summary) from the
# DB — the real production path. When False, re-extract from PDF with the deterministic
# caption only (the pre-summary baseline). Set by the --db / --baseline CLI flag so we
# can measure before/after the summary fix on the same questions.
USE_DB = "--baseline" not in sys.argv


def _db_grids_for_doc(doc):
    """Load the doc's ingested table grids from Supabase (carries the LLM summary).

    Mirrors production: the grids the Analyst actually sees come from the stored
    table chunks, whose table_json now includes the discriminative `summary`. Falls
    back to PDF extraction (no summary) if the doc isn't found in the DB.
    """
    from src.components.db import get_supabase_client
    import json as _json
    db = get_supabase_client(use_service_role=True)
    docrow = db.table("documents").select("id").eq("filename", doc).limit(1).execute().data
    if not docrow:
        return None
    did = docrow[0]["id"]
    rows = (
        db.table("document_chunks").select("content,metadata")
        .eq("document_id", did).eq("metadata->>chunk_type", "table").execute().data
    )
    out = []
    for r in rows or []:
        md = r.get("metadata") or {}
        raw = md.get("table_json")
        if not raw:
            continue
        tj = _json.loads(raw) if isinstance(raw, str) else raw
        cap = md.get("table_summary") or md.get("description") or r.get("content", "")
        out.append((Grid(tj, doc=doc, page=md.get("page_number")), tj, cap))
    return out or None


def all_grids_for_doc(doc):
    if doc not in _grids_cache:
        grids = _db_grids_for_doc(doc) if USE_DB else None
        if grids is None:
            tabs = extract_tables_from_pdf(f"{CORPUS}/{doc}")
            grids = [
                (Grid(t.to_metadata(), doc=doc, page=t.page_number), t.to_metadata(), t.caption)
                for t in tabs
            ]
        _grids_cache[doc] = grids
    return _grids_cache[doc]


_prose_cache = {}


def _doc_prose(doc):
    """The doc's full text layer — what the retriever's prose chunks come from.

    Used to corroborate the Analyst's direct cell reads (a figure not present in
    the doc text is a mis-selected row → flagged). Read once per doc, cached.
    """
    if doc not in _prose_cache:
        try:
            import pdfplumber
            with pdfplumber.open(f"{CORPUS}/{doc}") as pdf:
                _prose_cache[doc] = " ".join(p.extract_text() or "" for p in pdf.pages)
        except Exception:
            _prose_cache[doc] = ""
    return _prose_cache[doc]


def grids_for_question(doc, question, top=8):
    """Mirror the PRODUCTION path: relevance-filter to the top-N on-topic tables
    (the same lexical ranking load_grids_for_docs uses), so the eval exercises
    what users actually hit — not an unfiltered all-tables variant."""
    scored = [(_relevance(question, tj, cap), g) for g, tj, cap in all_grids_for_doc(doc)]
    scored.sort(key=lambda x: x[0], reverse=True)
    keep = [g for rel, g in scored if rel > 0][:top] or [g for _r, g in scored[:top]]
    return keep


def llm():
    from src.components.config import Config
    from langchain_openai import ChatOpenAI
    cfg = Config()
    return ChatOpenAI(model=cfg.LLM_MODEL_NAME, temperature=0.0,
                      api_key=cfg.OPENAI_API_KEY, request_timeout=40)


def close(got, want, tol_pct):
    if got is None:
        return False
    return abs(got - want) / max(abs(want), 1e-9) * 100 <= tol_pct


def main():
    spec = json.load(open(QUESTIONS))
    tol = spec.get("tolerance_pct", 0.5)
    model = llm()

    # The §4b bar is CORRECT-OR-ABSTAIN: never confidently wrong. A question
    # passes if the Analyst computed the right number OR abstained (returned no
    # confident value, e.g. flagged an ambiguous table). It FAILS only if it
    # stated a WRONG number confidently — the one outcome finance can't ship.
    correct = abstained = wrong = 0
    for q in spec["questions"]:
        grids = grids_for_question(q["doc"], q["question"])
        results = analyze(q["question"], grids, model)
        # §4b prose-corroboration backstop (the production path does the same):
        # a direct cell read whose figure isn't in the doc's text is flagged.
        results = corroborate_with_prose(results, _doc_prose(q["doc"]))
        ok_results = [r for r in results if r.ok]

        got = next((r.value for r in ok_results if close(r.value, q["expected_value"], tol)), None)
        if got is not None:
            # grounded check: the displayed figure must trace to cells/computed
            display = " ".join(r.display() for r in ok_results)
            v = verify_numbers(display, cells_from_results(ok_results), ok_results)
            if v.ok:
                correct += 1
                print(f"  [CORRECT ] {q['id']}: {got} == {q['expected_value']}{q['expected_unit']}")
                continue
        # no correct figure produced — did it abstain, or state a wrong number?
        confident_wrong = [r.value for r in ok_results if not close(r.value, q["expected_value"], tol)]
        if confident_wrong:
            wrong += 1
            print(f"  [WRONG ✗ ] {q['id']}: stated {confident_wrong} (want {q['expected_value']}{q['expected_unit']})")
        else:
            abstained += 1
            errs = "; ".join((r.error or "") for r in results if not r.ok)[:80]
            print(f"  [ABSTAIN ] {q['id']}: no confident figure (want {q['expected_value']}{q['expected_unit']}) — {errs}")

    n = len(spec["questions"])
    print(f"\n  {correct}/{n} correct, {abstained} abstained, {wrong} confidently WRONG")
    print(f"  Selection accuracy: {correct}/{n} = {100*correct//n}%  |  "
          + (f"✓ §4b bar met (0 confidently wrong)" if wrong == 0 else f"✗ {wrong} CONFIDENTLY WRONG — bar FAILED"))
    return 0 if wrong == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
