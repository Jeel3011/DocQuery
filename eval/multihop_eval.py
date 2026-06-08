"""Multi-hop accuracy eval (Phase 4.5) — A/B the multi-hop loop vs parallel decompose.

Drives BOTH retrievers over the SAME live collection on bridge questions (answers that
require chaining a fact: find A, then use A to find B), generates an answer from each,
and scores two things per question:

  doc_recall      — fraction of gold_doc_filenames whose chunks were actually retrieved
  answer_correct  — does the generated answer contain the required facts? (LLM-judged,
                    phrasing-robust; falls back to substring match if the judge is down)

The hypothesis: on bridges the sequential MULTI-HOP loop should beat PARALLEL decompose,
because parallel fires all sub-queries at once and can't condition hop 2 on hop 1.

Run:
  python eval/multihop_eval.py                       # both retrievers, default collection
  python eval/multihop_eval.py --collection <id>     # override collection
  python eval/multihop_eval.py --retriever multihop  # run only one side

Requires OPENAI_API_KEY and the docs ingested into the collection's namespace.
"""
import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.components.config import Config

QUESTIONS = "eval/eval_questions_multihop.json"


# ── collection scoping (mirrors routing_recall.py: namespace = collection owner) ──
def _scope_to_collection(config, collection_id):
    """Point the Pinecone namespace at the collection owner and return its filenames."""
    from src.components.db import SupabaseManager
    svc = SupabaseManager(use_service_role=True)
    owner = (
        svc.client.table("collections").select("user_id")
        .eq("id", collection_id).single().execute().data
    )
    owner_user_id = owner["user_id"] if owner else None
    if owner_user_id:
        config.PINECONE_NAMESPACE = owner_user_id
        svc._user = type("User", (), {"id": owner_user_id})()
    filenames = [d["filename"] for d in svc.get_collection_documents(collection_id)]
    return filenames


# ── LLM judge for answer correctness ──────────────────────────────────────────
_JUDGE_SYSTEM = (
    "You grade whether a generated answer contains specific required facts. You are given "
    "the question, the required facts, and the answer. Reply with ONLY 'YES' if the answer "
    "states ALL required facts (allowing different phrasing, units, rounding, or formatting "
    "of the same value), or 'NO' if any required fact is missing or contradicted. One word."
)


def _judge(question, required, answer, judge_llm):
    """Return True if the answer covers every required fact. Substring fallback on failure."""
    from langchain_core.messages import SystemMessage, HumanMessage
    req = "; ".join(required)
    try:
        resp = judge_llm.invoke([
            SystemMessage(content=_JUDGE_SYSTEM),
            HumanMessage(content=f"Question: {question}\nRequired facts: {req}\n\nAnswer:\n{answer}"),
        ])
        verdict = (resp.content or "").strip().upper()
        return verdict.startswith("YES")
    except Exception:
        # Degrade to case-insensitive substring match (strict but deterministic).
        low = (answer or "").lower()
        return all(r.lower() in low for r in required)


# ── answer generation: single-call vs Brain map-reduce ────────────────────────
def _group_for_brain(docs):
    """Group a flat doc list into the Brain's {doc_id: (filename, [chunks])} shape.

    Mirrors chat.py's brain path: key by doc_id (fallback filename) so the Brain
    MAPs each document independently — which is why it doesn't overflow the way the
    single-call generate() does.
    """
    grouped = {}
    for d in docs:
        md = d.metadata or {}
        fname = md.get("filename", "?")
        doc_id = md.get("doc_id", fname)
        grouped.setdefault(doc_id, (fname, []))[1].append(d)
    return grouped


def _generate_answer(question, docs, generator, brain):
    """Produce an answer from retrieved docs. brain != None → map-reduce path."""
    if not docs:
        return ""
    if brain is not None:
        result = brain.run(query=question, doc_chunks=_group_for_brain(docs))
        return getattr(result, "answer", "") or ""
    out = generator.generate(query=question, retrieved_docs=docs)
    return out.get("answer", "")


# ── one retriever × one question ──────────────────────────────────────────────
def _run_one(retriever, generator, brain, question, filenames):
    """Retrieve → generate. Returns (answer_text, retrieved_filenames_set, hop_trail)."""
    res = retriever.retrieve_and_synthesize(question, filename_filters=filenames)
    docs = res["docs"]
    retrieved_files = {d.metadata.get("filename", "") for d in docs if d.metadata.get("filename")}
    answer = _generate_answer(question, docs, generator, brain)
    return answer, retrieved_files, res.get("sub_queries", [])


def _doc_recall(retrieved, gold):
    gold = set(gold or [])
    if not gold:
        return 1.0
    return len(retrieved & gold) / len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=None, help="collection id (default: from questions file)")
    ap.add_argument("--retriever", choices=["both", "multihop", "parallel"], default="both")
    ap.add_argument("--brain", action="store_true",
                    help="generate answers via the map-reduce Brain instead of the single-call "
                         "generate() — removes the 12k-token overflow/trim confound (Phase 4).")
    args = ap.parse_args()

    with open(QUESTIONS) as f:
        raw = json.load(f)
    meta = next((q for q in raw if "_collection_id" in q), {})
    questions = [q for q in raw if "question" in q]
    collection_id = args.collection or meta.get("_collection_id")
    if not collection_id:
        sys.exit("No collection id (pass --collection or set _collection_id in the questions file).")

    config = Config()
    filenames = _scope_to_collection(config, collection_id)
    print(f"Collection {collection_id} → {len(filenames)} docs, namespace={config.PINECONE_NAMESPACE[:8]}…")
    print(f"{len(questions)} bridge questions\n")

    from src.components.retrieval import RetrievalManager
    from src.components.generation import AnswerGenration  # (sic — class name in codebase)
    from src.components.agentic_retrieval import AgenticRetriever
    from src.components.multi_hop import MultiHopRetriever
    from langchain_openai import ChatOpenAI

    retrieval_mgr = RetrievalManager(config)
    generator = AnswerGenration(config)
    brain = None
    if args.brain:
        from src.components.brain.map_reduce import Brain
        brain = Brain(config)
        print("Answer generation: map-reduce BRAIN (overflow-safe)\n")
    else:
        print("Answer generation: single-call generate() (12k budget, may trim)\n")
    judge_llm = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=config.OPENAI_API_KEY, request_timeout=30)

    sides = []
    if args.retriever in ("both", "parallel"):
        sides.append(("parallel", AgenticRetriever(config=config, retrieval_mgr=retrieval_mgr)))
    if args.retriever in ("both", "multihop"):
        sides.append(("multihop", MultiHopRetriever(config=config, retrieval_mgr=retrieval_mgr)))

    agg = {name: {"recall": [], "correct": []} for name, _ in sides}

    for qi, item in enumerate(questions, 1):
        q = item["question"]
        gold = item.get("gold_doc_filenames", [])
        required = item.get("answer_must_include", [])
        print(f"[{qi}/{len(questions)}] {q[:78]}")
        for name, retriever in sides:
            try:
                answer, files, trail = _run_one(retriever, generator, brain, q, filenames)
            except Exception as e:
                print(f"    {name:<9} ERROR: {e}")
                agg[name]["recall"].append(0.0)
                agg[name]["correct"].append(0)
                continue
            recall = _doc_recall(files, gold)
            correct = _judge(q, required, answer, judge_llm) if answer else False
            agg[name]["recall"].append(recall)
            agg[name]["correct"].append(1 if correct else 0)
            hops = len(trail)
            print(f"    {name:<9} recall={recall:.2f}  answer={'✓' if correct else '✗'}  hops={hops}  trail={trail}")
        print()

    print("=" * 64)
    print(f"{'retriever':<12}{'doc_recall':>12}{'answer_acc':>12}{'n':>5}")
    for name, _ in sides:
        r = agg[name]["recall"]; c = agg[name]["correct"]
        rec = sum(r) / len(r) if r else 0.0
        acc = sum(c) / len(c) if c else 0.0
        print(f"{name:<12}{rec:>12.3f}{acc:>12.3f}{len(c):>5}")
    print("=" * 64)

    out = {
        "collection_id": collection_id,
        "n_questions": len(questions),
        "generation": "brain" if args.brain else "single_call",
        "results": {
            name: {
                "doc_recall": sum(agg[name]["recall"]) / len(agg[name]["recall"]) if agg[name]["recall"] else 0.0,
                "answer_accuracy": sum(agg[name]["correct"]) / len(agg[name]["correct"]) if agg[name]["correct"] else 0.0,
            }
            for name, _ in sides
        },
    }
    suffix = "_brain" if args.brain else ""
    path = f"eval/multihop_eval_results{suffix}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
