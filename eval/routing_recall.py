"""
Routing Recall Metric — Phase 2 eval gate.

Measures: of the gold_doc_filenames for each multi-doc question, how many
did Stage-1 routing (or, pre-router, retrieve_across_files) actually return
chunks from?

Formula:  routing_recall@N = |retrieved_filenames ∩ gold_filenames| / |gold_filenames|
Averaged across all questions that have gold_doc_filenames.

This is the PRIMARY accuracy gate for the Brain at scale.  Wrong answers at
scale almost always start with a routing miss — the source doc was never fetched.
"""

import json
from pathlib import Path
from typing import Optional


def compute_routing_recall(
    questions_path: str,
    config,
    collection_id: Optional[str] = None,
    routing_top_n: Optional[int] = None,
) -> float:
    """Return the average routing recall across all questions that have gold_doc_filenames.

    Args:
        questions_path:  Path to the multi-doc eval JSON.
        config:          Config instance (holds PINECONE_NAMESPACE, keys, etc.).
        collection_id:   Optional collection to scope retrieval.
        routing_top_n:   Number of docs the router is allowed to return.
                         Defaults to config.ROUTING_TOP_N (or 12).

    Returns:
        Float in [0, 1].  1.0 = perfect recall; 0.0 = routing missed everything.
        Returns 1.0 (vacuous pass) if no questions have gold_doc_filenames.
    """
    from src.components.retrieval import RetrievalManager
    from src.components.db import SupabaseManager

    with open(questions_path) as f:
        questions = json.load(f)

    scored = []

    top_n = routing_top_n or getattr(config, "ROUTING_TOP_N", 12)

    # If a collection_id is given, resolve (a) the owning user — whose user_id is
    # the Pinecone namespace the app scopes to (dependencies.get_user_config) — and
    # (b) the collection's filename list. Without scoping the namespace to the owner,
    # retrieval queries the default/empty namespace and returns nothing.
    filename_filters = None
    if collection_id:
        try:
            svc = SupabaseManager(use_service_role=True)
            owner = (
                svc.client.table("collections")
                .select("user_id")
                .eq("id", collection_id)
                .single()
                .execute()
                .data
            )
            owner_user_id = owner["user_id"] if owner else None
            if owner_user_id:
                # Scope the Pinecone namespace to the collection owner (app parity).
                config.PINECONE_NAMESPACE = owner_user_id
                # Set the user on the service-role client so RLS-style filters resolve.
                svc._user = type("User", (), {"id": owner_user_id})()
            filename_filters = [
                d["filename"] for d in svc.get_collection_documents(collection_id)
            ]
        except Exception as e:
            print(f"  [routing_recall] Could not resolve collection {collection_id}: {e}")

    retrieval_mgr = RetrievalManager(config)

    for item in questions:
        gold_filenames = item.get("gold_doc_filenames")
        if not gold_filenames:
            continue  # skip questions without gold labels

        question = item["question"]

        # Retrieve — using the current path (pre-router fan-out or router if Phase 3 active)
        try:
            docs = retrieval_mgr.retrieve(
                question,
                filename_filters=filename_filters or gold_filenames,
            )
        except Exception as e:
            print(f"  [routing_recall] Retrieval failed for '{question[:60]}': {e}")
            scored.append(0.0)
            continue

        retrieved_filenames = {
            d.metadata.get("filename", "")
            for d in docs
            if d.metadata.get("filename")
        }
        gold_set = set(gold_filenames)
        if not gold_set:
            continue

        recall = len(retrieved_filenames & gold_set) / len(gold_set)
        scored.append(recall)

        print(
            f"  Q: {question[:55]:<55} | "
            f"gold={len(gold_set)} retrieved={len(retrieved_filenames)} "
            f"hit={len(retrieved_filenames & gold_set)} recall={recall:.2f}"
        )

    if not scored:
        print("  [routing_recall] No questions with gold_doc_filenames — returning 1.0 (vacuous pass)")
        return 1.0

    avg = sum(scored) / len(scored)
    print(f"\n  Routing recall (avg over {len(scored)} questions): {avg:.4f}")
    return avg


def compute_router_recall(
    questions_path: str,
    config,
    collection_id: str,
    top_n: Optional[int] = None,
) -> float:
    """Recall of the Stage-1 DocumentRouter's RANKING (not just retrieval).

    Unlike compute_routing_recall (which feeds the whole filename list to the
    retriever and measures what comes back), this calls DocumentRouter.route()
    directly with a constrained ``top_n`` and measures whether the gold docs land
    in the router's top-N selection. This is the test that actually exercises
    ranking: with top_n < collection size, the router MUST choose and can miss.

    Args:
        questions_path: Path to the multi-doc eval JSON.
        config:         Config (PINECONE_NAMESPACE is set to the owner here).
        collection_id:  Collection to route within.
        top_n:          How many docs the router may return. Defaults to the size
                        of the largest gold set in the eval (so a perfect router
                        can still score 1.0), capped to force a real choice.

    Returns:
        Average router recall@top_n across questions with gold_doc_filenames.
    """
    import json as _json
    from src.components.document_router import DocumentRouter
    from src.components.db import SupabaseManager

    with open(questions_path) as f:
        questions = _json.load(f)

    # Resolve the owning user so the namespace + RLS scope match the app.
    svc = SupabaseManager(use_service_role=True)
    owner = (
        svc.client.table("collections").select("user_id")
        .eq("id", collection_id).single().execute().data
    )
    owner_user_id = owner["user_id"] if owner else None
    if owner_user_id:
        config.PINECONE_NAMESPACE = owner_user_id

    labelled = [q for q in questions if q.get("gold_doc_filenames")]
    if not labelled:
        print("  [router_recall] No labelled questions — returning 1.0 (vacuous pass)")
        return 1.0

    # Default top_n: the largest gold set, so a perfect router can score 1.0 while
    # still being forced to exclude the other docs (a real selection, not "return all").
    if top_n is None:
        top_n = max(len(q["gold_doc_filenames"]) for q in labelled)

    router = DocumentRouter(config)
    scored = []
    for item in labelled:
        gold = set(item["gold_doc_filenames"])
        try:
            selected = router.route(
                item["question"], collection_id, owner_user_id, top_n=top_n
            )
        except Exception as e:
            print(f"  [router_recall] route() failed for '{item['question'][:50]}': {e}")
            scored.append(0.0)
            continue
        sel = set(selected)
        recall = len(sel & gold) / len(gold)
        scored.append(recall)
        miss = gold - sel
        print(
            f"  Q: {item['question'][:52]:<52} | top_n={top_n} "
            f"gold={len(gold)} hit={len(sel & gold)} recall={recall:.2f}"
            + (f"  MISS={sorted(m[:18] for m in miss)}" if miss else "")
        )

    avg = sum(scored) / len(scored)
    print(f"\n  Router recall@{top_n} (avg over {len(scored)} questions): {avg:.4f}")
    return avg
