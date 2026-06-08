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
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.components.config import Config

QUESTIONS = "eval/eval_questions_multihop.json"


def _scope_to_collection(config, collection_id):
    """Namespace = collection owner; return the collection's filenames (mirrors routing_recall)."""
    from src.components.db import SupabaseManager
    svc = SupabaseManager(use_service_role=True)
    owner = (
        svc.client.table("collections").select("user_id")
        .eq("id", collection_id).single().execute().data
    )
    if owner and owner.get("user_id"):
        config.PINECONE_NAMESPACE = owner["user_id"]
        svc._user = type("User", (), {"id": owner["user_id"]})()
    return [d["filename"] for d in svc.get_collection_documents(collection_id)]


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
    with open(QUESTIONS) as f:
        raw = json.load(f)
    meta = next((q for q in raw if "_collection_id" in q), {})
    questions = [q for q in raw if "question" in q]
    collection_id = meta.get("_collection_id")
    if not collection_id:
        sys.exit("No _collection_id in questions file.")

    config = Config()
    filenames = _scope_to_collection(config, collection_id)
    per_doc_k = getattr(config, "BRAIN_CHUNKS_PER_DOC", 8)
    print(f"Collection {collection_id} → {len(filenames)} docs, k/doc={per_doc_k}")
    print(f"Brain-ALONE on {len(questions)} tough questions\n")

    from src.components.retrieval import RetrievalManager
    from src.components.brain.map_reduce import Brain
    from langchain_openai import ChatOpenAI

    retrieval_mgr = RetrievalManager(config)
    brain = Brain(config)
    judge_llm = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=config.OPENAI_API_KEY, request_timeout=30)

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
        try:
            result = brain.run(query=q, doc_chunks=doc_chunks)
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
        print(f"[{qi}/{len(questions)}] {bucket:<9} [{qtype:<18}] {q[:55]}{cov_tag}")
        print(f"       → {answer[:160].strip() or '(empty)'}\n")

    n = len(questions)
    print("=" * 68)
    print(f"  CORRECT   : {buckets['CORRECT']}/{n}")
    print(f"  ABSTAINED : {buckets['ABSTAINED']}/{n}   (safe — 'I can't verify this')")
    print(f"  WRONG     : {buckets['WRONG']}/{n}   (DANGEROUS — confident & false)")
    print("=" * 68)
    print(f"  Dangerous-wrong rate (headline): {buckets['WRONG']/n:.0%}\n")

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
