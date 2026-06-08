"""Brain-ALONE bucketed eval — answers "does the Brain give wrong or no answers?"

Runs the PLAIN Brain path (no multi-hop, no parallel decompose) on tough bridge
questions and classifies each answer into ONE of three buckets:

  CORRECT   — the answer states the required facts (it's right)
  ABSTAINED — the Brain says it can't verify / doesn't have the answer (SAFE failure)
  WRONG     — the Brain confidently states a fact that's missing or contradicts (DANGEROUS)

This is the distinction a pass/fail metric hides: an honest "I can't verify this" is
fundamentally different from a confident wrong number. For finance/legal, WRONG is the
only failure that actually matters; ABSTAINED is the Brain working as designed.

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
    """Per-file retrieval, exactly as chat.py's brain path does it."""
    doc_chunks = {}
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
    return doc_chunks


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
    rows = []
    for qi, item in enumerate(questions, 1):
        q = item["question"]
        required = item.get("answer_must_include", [])
        qtype = item.get("query_type", "untyped")
        doc_chunks = _brain_retrieve(retrieval_mgr, q, filenames, per_doc_k)
        try:
            result = brain.run(query=q, doc_chunks=doc_chunks)
            answer = getattr(result, "answer", "") or ""
        except Exception as e:
            answer = ""
            print(f"[{qi}] Brain.run ERROR: {e}")
        bucket = _classify(q, required, answer, judge_llm)
        buckets[bucket] += 1
        by_type[qtype][bucket] += 1
        rows.append({"q": q, "query_type": qtype, "bucket": bucket, "answer_head": answer[:160]})
        print(f"[{qi}/{len(questions)}] {bucket:<9} [{qtype:<18}] {q[:55]}")
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

    with open("eval/brain_solo_results.json", "w") as f:
        json.dump({
            "collection_id": collection_id, "n": n,
            "buckets": buckets, "by_type": dict(by_type), "rows": rows,
        }, f, indent=2)
    print("Wrote eval/brain_solo_results.json")


if __name__ == "__main__":
    main()
