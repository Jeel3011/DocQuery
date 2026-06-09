"""Brain-ALONE bucketed eval — answers "does the Brain give wrong or no answers?"

Runs the PLAIN Brain path (no multi-hop, no parallel decompose) on tough bridge
questions and classifies each answer into ONE of three buckets:

  CORRECT   — the answer states the required facts (it's right)
  ABSTAINED — the Brain says it can't verify / doesn't have the answer (SAFE failure)
  WRONG     — the Brain confidently states a fact that's missing or contradicts (DANGEROUS)

This is the distinction a pass/fail metric hides: an honest "I can't verify this" is
fundamentally different from a confident wrong number. For finance/legal, WRONG is the
only failure that actually matters; ABSTAINED is the Brain working as designed.

§7 (Stage B measurement overhaul) adds two diagnostics on top of the 3-way buckets:
  COVERAGE             — did retrieval surface the gold docs (≥ min_docs_required) that
                         hold the answer? Cross-tabbed with WRONG, this splits a
                         RETRIEVAL failure (answer never in front of the Brain → fix
                         directed retrieval) from a REASONING failure (Brain had the
                         evidence and still went wrong → fix grounding/executive/monitor).
  ABSTENTION-USEFULNESS — abstentions should cluster on genuinely-hard pivot/compound
                         questions, not on trivial single-hop lookups. Over-abstaining
                         on single_hop_control is flagged.
WRONG-rate (per query_type) stays the headline; coverage + abstention-usefulness are
the secondary instruments §7 requires so "it generalizes" is a measured claim.

Replicates chat.py's brain path exactly: per-file retrieve (top_k=BRAIN_CHUNKS_PER_DOC,
no threshold, no reranker) → group by doc → Brain.run().

Run:  python eval/brain_solo_eval.py
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.components.config import Config

QUESTIONS = "eval/eval_questions_multihop.json"


def _scope_to_collection(config, collection_id):
    """Namespace = collection owner; return (filenames, db_client). The db_client (svc) is
    returned so the spine step can load table grids exactly as chat.py does."""
    from src.components.db import SupabaseManager
    svc = SupabaseManager(use_service_role=True)
    owner = (
        svc.client.table("collections").select("user_id")
        .eq("id", collection_id).single().execute().data
    )
    if owner and owner.get("user_id"):
        config.PINECONE_NAMESPACE = owner["user_id"]
        svc._user = type("User", (), {"id": owner["user_id"]})()
    return [d["filename"] for d in svc.get_collection_documents(collection_id)], svc


_LOCAL_GRID_CACHE: dict = {}


def _local_grids_for_docs(filenames):
    """Re-extract grids from the local PDFs (in 'test docs/') with the CURRENT extractor —
    the corrected period-recovery. Cached per file. Used by the SPINE_LOCAL_GRIDS test path
    to validate the spine on clean grids without re-ingesting Supabase."""
    from src.components.table_extraction import extract_tables_from_pdf
    from src.components.brain.analyst import Grid
    out = []
    for fn in set(filenames):
        if fn not in _LOCAL_GRID_CACHE:
            path = os.path.join("test docs", fn)
            try:
                _LOCAL_GRID_CACHE[fn] = [
                    Grid(t.to_metadata(), doc=fn, page=t.page_number)
                    for t in extract_tables_from_pdf(path)
                ] if os.path.exists(path) else []
            except Exception:
                _LOCAL_GRID_CACHE[fn] = []
        out.extend(_LOCAL_GRID_CACHE[fn])
    return out


def _spine_block_for(config, sb, question, doc_chunks, spec_llm):
    """Build the executive-spine block for a question, EXACTLY as chat.py's Brain endpoint
    does — gated by USE_EXEC_SPINE. Returns (block_or_None, tag). This is what makes the
    eval actually test C2→C6: brain.run() consumes the block in REDUCE. When the flag is
    off, returns (None, 'off') and brain.run is byte-identical to the pre-spine path.

    Mirrors chat.py: has_numeric_intent gate → load_grids_for_docs → run_executive_spine.
    """
    if not getattr(config, "USE_EXEC_SPINE", False):
        return None, "off"
    try:
        from src.components.brain.table_intent import has_numeric_intent, load_grids_for_docs
        if not has_numeric_intent(question):
            return None, "non-numeric"
        filename_by_doc = {did: fn for did, (fn, _ch) in doc_chunks.items()}
        if os.getenv("SPINE_LOCAL_GRIDS") == "1":
            # TEST PATH: feed the spine grids freshly RE-EXTRACTED from the local PDFs (the
            # CORRECTED extraction), instead of the stale Supabase table_json. This isolates
            # "does the spine work on CLEAN grids?" without re-ingesting the collection.
            grids = _local_grids_for_docs(filename_by_doc.values())
        else:
            grids = load_grids_for_docs(
                sb, list(doc_chunks.keys()), question=question, filename_by_doc=filename_by_doc,
            )
        if not grids:
            return None, "no-grids"
        from src.components.brain.meta_reasoner import run_executive_spine
        outcome = run_executive_spine(question, grids, spec_llm)
        if outcome.applied and outcome.block:
            tag = f"ABSTAIN(pivot={outcome.binding})" if outcome.abstained else f"applied(pivot={outcome.binding})"
            return outcome.block, tag
        return None, f"fall-through({outcome.reason[:30]})"
    except Exception as exc:
        return None, f"error: {exc}"


def _brain_retrieve(retrieval_mgr, question, filenames, per_doc_k):
    """Per-file retrieval, exactly as chat.py's brain path does it.

    Returns (doc_chunks, retrieved_filenames): the second is the SET of filenames
    that actually produced chunks — the raw signal the §7 coverage metric uses
    (did we even consult the doc that holds the answer?). Tracked off the `fname`
    loop directly, so it doesn't depend on chunk-metadata shape.
    """
    doc_chunks = {}
    retrieved_filenames = set()
    for fname in filenames:
        try:
            chunks = retrieval_mgr.retrieve(
                question, fname, None, None, per_doc_k, False, False  # threshold off, reranker off
            )
        except Exception:
            chunks = []
        if chunks:
            doc_id = chunks[0].metadata.get("doc_id", fname)
            doc_chunks[doc_id] = (fname, chunks)
            retrieved_filenames.add(fname)
    return doc_chunks, retrieved_filenames


# ── 3-way classifier: CORRECT / ABSTAINED / WRONG ─────────────────────────────
_CLASSIFY_SYSTEM = (
    "You classify a document-Q&A answer into EXACTLY one bucket. You are given the "
    "question, the REQUIRED facts a correct answer must contain, and the ANSWER.\n\n"
    "Reply with ONE word:\n"
    "  CORRECT   — the answer states all required facts (different phrasing/rounding is fine).\n"
    "  ABSTAINED — the answer declines: it says it cannot verify, cannot find, does not have, "
    "or lacks the information. It does NOT assert a specific wrong value.\n"
    "  WRONG     — the answer confidently asserts a fact that is missing from, or contradicts, "
    "the required facts (e.g. states the wrong year or number as if it were correct).\n\n"
    "ABSTAINED vs WRONG is the key call: a hedge/refusal is ABSTAINED; a confident false "
    "statement is WRONG. One word only."
)


def _classify(question, required, answer, judge_llm):
    from langchain_core.messages import SystemMessage, HumanMessage
    if not (answer or "").strip():
        return "ABSTAINED"  # empty answer = no answer
    req = "; ".join(required)
    try:
        resp = judge_llm.invoke([
            SystemMessage(content=_CLASSIFY_SYSTEM),
            HumanMessage(content=f"Question: {question}\nRequired facts: {req}\n\nAnswer:\n{answer}"),
        ])
        v = (resp.content or "").strip().upper()
        for bucket in ("CORRECT", "ABSTAINED", "WRONG"):
            if bucket in v:
                return bucket
        return "WRONG"  # unparseable → treat conservatively as the dangerous bucket
    except Exception:
        return "WRONG"


def main():
    # --limit N runs only the first N questions (cheap smoke test before the full 27).
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit"):
            try:
                limit = int(a.split("=", 1)[1]) if "=" in a else int(sys.argv[sys.argv.index(a) + 1])
            except Exception:
                limit = None

    with open(QUESTIONS) as f:
        raw = json.load(f)
    meta = next((q for q in raw if "_collection_id" in q), {})
    questions = [q for q in raw if "question" in q]
    if limit:
        questions = questions[:limit]
    collection_id = meta.get("_collection_id")
    if not collection_id:
        sys.exit("No _collection_id in questions file.")

    config = Config()
    filenames, sb = _scope_to_collection(config, collection_id)
    spine_on = getattr(config, "USE_EXEC_SPINE", False)
    print(f"USE_EXEC_SPINE = {spine_on}" + ("  ← executive spine C2→C6 ACTIVE" if spine_on else "  (spine off — old single-pass path)"))
    per_doc_k = getattr(config, "BRAIN_CHUNKS_PER_DOC", 8)
    print(f"Collection {collection_id} → {len(filenames)} docs, k/doc={per_doc_k}")
    print(f"Brain-ALONE on {len(questions)} tough questions\n")

    from src.components.retrieval import RetrievalManager
    from src.components.brain.map_reduce import Brain
    from langchain_openai import ChatOpenAI

    retrieval_mgr = RetrievalManager(config)
    brain = Brain(config)
    judge_llm = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=config.OPENAI_API_KEY, request_timeout=30)
    # the spine's comprehend call uses the cheap model, same as chat.py's Analyst thread
    spec_llm = ChatOpenAI(model=config.LLM_MODEL_NAME, temperature=0.0,
                          api_key=config.OPENAI_API_KEY, request_timeout=20, max_retries=1)
    spine_tags = []   # per-question spine outcome, for the summary

    buckets = {"CORRECT": 0, "ABSTAINED": 0, "WRONG": 0}
    # Per-category breakdown — the §7 headline view. WRONG-rate is expected to
    # concentrate in extremum_pivot (the failure class Phase 4.6 Layer 1 targets);
    # single_hop_control WRONG must stay ~0 (the prime-directive regression guard).
    from collections import defaultdict
    by_type = defaultdict(lambda: {"CORRECT": 0, "ABSTAINED": 0, "WRONG": 0})
    # §7 coverage: per question, did retrieval surface the gold docs that hold the
    # answer? `covered` = (gold filenames that produced chunks) ≥ min_docs_required.
    # A question with NO declared gold docs is excluded from the coverage rate (can't
    # measure it) but still classified for WRONG-rate.
    covered_count = 0
    coverage_measurable = 0
    rows = []
    for qi, item in enumerate(questions, 1):
        q = item["question"]
        required = item.get("answer_must_include", [])
        qtype = item.get("query_type", "untyped")
        gold_docs = item.get("gold_doc_filenames", []) or []
        min_docs = item.get("min_docs_required", 1 if gold_docs else 0)
        doc_chunks, retrieved_fnames = _brain_retrieve(retrieval_mgr, q, filenames, per_doc_k)
        # coverage: how many of THIS question's gold docs actually produced chunks
        gold_hit = sorted(set(gold_docs) & retrieved_fnames)
        covered = bool(gold_docs) and len(gold_hit) >= min_docs
        if gold_docs:
            coverage_measurable += 1
            covered_count += int(covered)
        # Phase 4.6: build the executive-spine block (gated by USE_EXEC_SPINE), exactly as
        # chat.py does, and feed it to brain.run so REDUCE states the monitored figure.
        spine_block, spine_tag = _spine_block_for(config, sb, q, doc_chunks, spec_llm)
        spine_tags.append(spine_tag)
        try:
            result = brain.run(query=q, doc_chunks=doc_chunks, analyst_block=spine_block)
            answer = getattr(result, "answer", "") or ""
        except Exception as e:
            answer = ""
            print(f"[{qi}] Brain.run ERROR: {e}")
        bucket = _classify(q, required, answer, judge_llm)
        buckets[bucket] += 1
        by_type[qtype][bucket] += 1
        rows.append({
            "q": q, "query_type": qtype, "bucket": bucket,
            "answer_head": answer[:160],
            "gold_docs": gold_docs, "gold_hit": gold_hit,
            "min_docs": min_docs, "covered": covered if gold_docs else None,
        })
        cov_tag = ("" if not gold_docs
                   else f"  cov={len(gold_hit)}/{len(gold_docs)}{'' if covered else ' ✗MISS'}")
        spine_disp = "" if spine_tag in ("off",) else f"  spine={spine_tag}"
        print(f"[{qi}/{len(questions)}] {bucket:<9} [{qtype:<18}] {q[:55]}{cov_tag}{spine_disp}")
        print(f"       → {answer[:160].strip() or '(empty)'}\n")

    n = len(questions)
    print("=" * 68)
    print(f"  CORRECT   : {buckets['CORRECT']}/{n}")
    print(f"  ABSTAINED : {buckets['ABSTAINED']}/{n}   (safe — 'I can't verify this')")
    print(f"  WRONG     : {buckets['WRONG']}/{n}   (DANGEROUS — confident & false)")
    print("=" * 68)
    print(f"  Dangerous-wrong rate (headline): {buckets['WRONG']/n:.0%}\n")

    # ── spine activity: how often the executive spine actually ran (only meaningful with
    #    USE_EXEC_SPINE=true). 'applied' = the spine produced a verified figure block;
    #    'ABSTAIN' = the monitor withheld; the rest fell through to the old path.
    if spine_on:
        from collections import Counter
        sc = Counter("applied" if t.startswith("applied")
                     else "abstain" if t.startswith("ABSTAIN")
                     else "fallthrough" for t in spine_tags)
        print(f"  SPINE ACTIVITY: applied={sc['applied']}  abstain={sc['abstain']}  "
              f"fell-through={sc['fallthrough']}  (of {n})")
        print("=" * 68)

    # Per-category WRONG-rate — where the danger lives, and the regression guard.
    print("  By query_type (CORRECT / ABSTAINED / WRONG → WRONG-rate):")
    for qtype in sorted(by_type):
        b = by_type[qtype]
        tn = b["CORRECT"] + b["ABSTAINED"] + b["WRONG"]
        flag = "  ⚠️ REGRESSION" if (qtype == "single_hop_control" and b["WRONG"]) else ""
        print(f"    {qtype:<20} {b['CORRECT']:>2} / {b['ABSTAINED']:>2} / {b['WRONG']:>2}"
              f"   → {b['WRONG']/tn:.0%} wrong{flag}")
    print("=" * 68)

    # ── §7 COVERAGE: did we even consult the docs that hold the answer? ───────────
    # Separates a RETRIEVAL failure (answer not in front of the Brain) from a
    # REASONING failure (Brain had the evidence and still went wrong). A WRONG WITH
    # full coverage indicts the executive/grounding organs; a WRONG WITHOUT coverage
    # indicts directed-retrieval (the §5.3 executor arm).
    cov_rate = (covered_count / coverage_measurable) if coverage_measurable else None
    print(f"  COVERAGE (gold docs surfaced ≥ min_docs): "
          f"{covered_count}/{coverage_measurable}"
          + (f" = {cov_rate:.0%}" if cov_rate is not None else " (n/a)"))
    # cross-tab WRONG × coverage — the diagnostic that points at the right organ
    wrong_rows = [r for r in rows if r["bucket"] == "WRONG" and r["covered"] is not None]
    wrong_covered = sum(1 for r in wrong_rows if r["covered"])
    wrong_uncovered = len(wrong_rows) - wrong_covered
    print(f"    WRONG with FULL coverage   : {wrong_covered}   (reasoning/grounding failure → C1/C3/C5)")
    print(f"    WRONG with MISSING coverage: {wrong_uncovered}   (retrieval failure → C4 directed retrieval)")

    # ── §7 ABSTENTION-USEFULNESS: abstentions should cluster on genuinely-hard ───
    # questions, NOT trivial ones. We score an abstention as USEFUL if it lands on a
    # pivot/compound question and WASTEFUL if it lands on single_hop_control (a
    # trivial lookup the Brain should just answer). A high wasteful count = the
    # Brain is over-abstaining, which §7 flags as a (different) quality problem.
    HARD_TYPES = {"extremum_pivot", "lookup_pivot", "compare", "compound"}
    abst_rows = [r for r in rows if r["bucket"] == "ABSTAINED"]
    abst_useful = sum(1 for r in abst_rows if r["query_type"] in HARD_TYPES)
    abst_wasteful = sum(1 for r in abst_rows if r["query_type"] == "single_hop_control")
    abst_other = len(abst_rows) - abst_useful - abst_wasteful
    abst_score = (abst_useful / len(abst_rows)) if abst_rows else None
    print(f"  ABSTENTION-USEFULNESS: {len(abst_rows)} abstentions → "
          f"{abst_useful} useful (hard) / {abst_wasteful} wasteful (single-hop) / {abst_other} other"
          + (f"  | useful-rate {abst_score:.0%}" if abst_score is not None else ""))
    if abst_wasteful:
        print(f"    ⚠️ {abst_wasteful} abstention(s) on trivial single-hop lookups — over-abstaining.")
    print("=" * 68)

    with open("eval/brain_solo_results.json", "w") as f:
        json.dump({
            "collection_id": collection_id, "n": n,
            "buckets": buckets, "by_type": dict(by_type),
            "coverage": {
                "covered": covered_count, "measurable": coverage_measurable,
                "rate": cov_rate,
                "wrong_with_full_coverage": wrong_covered,
                "wrong_with_missing_coverage": wrong_uncovered,
            },
            "abstention_usefulness": {
                "total": len(abst_rows), "useful": abst_useful,
                "wasteful_single_hop": abst_wasteful, "other": abst_other,
                "useful_rate": abst_score,
            },
            "rows": rows,
        }, f, indent=2)
    print("Wrote eval/brain_solo_results.json")


if __name__ == "__main__":
    main()
