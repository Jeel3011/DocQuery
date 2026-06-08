"""Unit test for the multi-hop retrieval loop (Phase 4.5 / T1.5).

Run: python eval/test_multi_hop.py

NO network, NO API key, NO DB: the LLM gap-detector and the retrieval manager are
both faked, so this isolates the LOOP CONTROL LOGIC — the part that's hard to get
right (informed follow-ups, dedup across hops, hop budget, graceful degradation).
It builds the MultiHopRetriever WITHOUT calling __init__ (which would construct a
real ChatOpenAI), so it runs anywhere.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from langchain_core.documents import Document
from src.components.multi_hop import MultiHopRetriever


# ── fakes ────────────────────────────────────────────────────────────────────
class FakeResp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Returns a scripted gap-detector reply per call, then SUFFICIENT forever."""
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = 0
        self.seen_prompts = []

    def invoke(self, messages):
        self.calls += 1
        # messages = [SystemMessage, HumanMessage]; record the human content
        self.seen_prompts.append(messages[-1].content)
        reply = self.scripted.pop(0) if self.scripted else "SUFFICIENT"
        return FakeResp(reply)


class FakeRetriever:
    """Maps a query → list of docs by substring match against a script."""
    def __init__(self, by_query):
        self.by_query = by_query  # {substring: [Document, ...]}
        self.queries = []

    def retrieve(self, query, filename_filter=None, page_filter=None,
                 filename_filters=None, top_k=None):
        self.queries.append(query)
        for needle, docs in self.by_query.items():
            if needle.lower() in query.lower():
                return docs
        return []


def _doc(chunk_id, text, filename="f.pdf", page=1):
    return Document(page_content=text, metadata={
        "chunk_id": chunk_id, "filename": filename, "page_number": page,
    })


def _make(llm, retriever, max_hops=3, per_hop_k=5):
    """Construct without __init__ so no real ChatOpenAI is built."""
    r = MultiHopRetriever.__new__(MultiHopRetriever)
    r.config = type("C", (), {"USE_WEB_FALLBACK": False, "WEB_SEARCH_MAX_RESULTS": 3})()
    r.retrieval_mgr = retriever
    r.llm = llm
    r.max_hops = max_hops
    r.per_hop_k = per_hop_k
    return r


# ── tests ────────────────────────────────────────────────────────────────────
def test_informed_followup_and_dedup():
    """Hop 1 finds the acquirer; the gap detector asks an INFORMED follow-up that
    finds the spin-off. Both sets merge; a chunk seen in both hops is deduped once."""
    shared = _doc("c_shared", "appears in both hops")
    retriever = FakeRetriever({
        "acquirer of X":   [_doc("c1", "Acme acquired X"), shared],
        "Acme spin-off":   [_doc("c2", "Acme spun off Beta"), shared],
    })
    # After hop 1 → ask the informed query; after hop 2 → stop.
    llm = FakeLLM(["SEARCH: Acme spin-off subsidiaries", "SUFFICIENT"])
    r = _make(llm, retriever)

    out = r.retrieve_and_synthesize("acquirer of X")

    ids = sorted(d.metadata["chunk_id"] for d in out["docs"])
    assert ids == ["c1", "c2", "c_shared"], f"expected dedup merge, got {ids}"
    assert out["unique_docs"] == 3
    assert out["sub_queries"] == ["acquirer of X", "Acme spin-off subsidiaries"], out["sub_queries"]
    assert retriever.queries == ["acquirer of X", "Acme spin-off subsidiaries"]
    print("✓ informed follow-up issues the second query and dedups the shared chunk")


def test_stops_on_sufficient():
    """SUFFICIENT after hop 1 → exactly one retrieval, no follow-up."""
    retriever = FakeRetriever({"q": [_doc("c1", "answer is here")]})
    llm = FakeLLM(["SUFFICIENT"])
    r = _make(llm, retriever)

    out = r.retrieve_and_synthesize("q")

    assert out["sub_queries"] == ["q"], out["sub_queries"]
    assert len(retriever.queries) == 1, retriever.queries
    assert llm.calls == 1, llm.calls  # detector consulted once, said stop
    print("✓ SUFFICIENT ends the loop after one hop")


def test_hop_budget_caps_retrievals():
    """A detector that never says SUFFICIENT is bounded by max_hops; the LAST hop
    does NOT call the detector (no point asking for a query we can't run).

    Uses distinct multi-word follow-ups (each introduces a NEW fact) so they survive
    the reformulation guard — that's what a real informed hop looks like."""
    retriever = FakeRetriever({"": [_doc("c", "x")]})  # matches everything
    llm = FakeLLM([
        "FOUND: alpha\nNEXT: zeta subsidiary revenue breakdown",
        "FOUND: zeta\nNEXT: kappa division operating margin trend",
        "FOUND: kappa\nNEXT: omega goodwill impairment charge",
    ])
    r = _make(llm, retriever, max_hops=3)

    out = r.retrieve_and_synthesize("original distinct topic question")

    assert len(retriever.queries) == 3, retriever.queries          # exactly max_hops
    assert len(out["sub_queries"]) == 3, out["sub_queries"]
    assert llm.calls == 2, llm.calls  # detector consulted between hops only (3-1)
    print("✓ hop budget caps retrievals at max_hops; last hop skips the detector")


def test_found_next_format_informed_followup():
    """The new FOUND:/NEXT: format is parsed and the NEXT query (carrying a learned
    fact) drives hop 2."""
    retriever = FakeRetriever({
        "AWS operating income": [_doc("a", "AWS op income hit $22.6B in 2021")],
        "total net sales 2021":  [_doc("b", "Amazon total net sales 2021 = $469.8B")],
    })
    llm = FakeLLM([
        "FOUND: AWS operating income first exceeded $20B in fiscal 2021\n"
        "NEXT: Amazon total net sales 2021",
        "FOUND: total net sales 2021 = $469.8B\nNEXT: SUFFICIENT",
    ])
    r = _make(llm, retriever)

    out = r.retrieve_and_synthesize("AWS operating income $20B year total net sales")

    assert out["sub_queries"][1] == "Amazon total net sales 2021", out["sub_queries"]
    assert sorted(d.metadata["chunk_id"] for d in out["docs"]) == ["a", "b"]
    print("✓ FOUND/NEXT format parsed; learned fact ('2021') carried into hop 2")


def test_reformulation_is_rejected():
    """A follow-up that merely rephrases the original (high word overlap, no new fact)
    is rejected — the loop stops instead of wasting a hop. This is the exact failure
    seen in the first live run."""
    q = "fiscal year AWS operating income crossed twenty billion total net sales"
    retriever = FakeRetriever({"fiscal": [_doc("a", "x")]})
    # NEXT just reshuffles the same content words — no new fact.
    llm = FakeLLM(["FOUND: nothing concrete\n"
                   "NEXT: AWS operating income fiscal year total net sales crossed billion"])
    r = _make(llm, retriever, max_hops=3)

    out = r.retrieve_and_synthesize(q)

    assert out["sub_queries"] == [q], out["sub_queries"]   # no second hop
    assert len(retriever.queries) == 1, retriever.queries
    print("✓ rephrase-only follow-up rejected (reformulation guard)")


def test_bridge_query_with_learned_fact_survives_guard():
    """REGRESSION (2026-06-05 live run): a real bridge query reuses the question's nouns
    AND adds the learned fact (a year). The first guard (word-overlap) wrongly killed it;
    the fixed guard (new-token test) must let it through and run the second hop."""
    q = ("In the fiscal year Amazon AWS segment operating income first crossed $20B, "
         "what was Amazon total consolidated net sales?")
    retriever = FakeRetriever({
        "operating income": [_doc("a", "AWS op income $22.6B in 2021")],
        "2021":             [_doc("b", "Amazon total net sales 2021 = $469.8B")],
    })
    # The exact NEXT the live model produced — high word overlap but adds "2021".
    llm = FakeLLM([
        "FOUND: AWS operating income first crossed $20B in fiscal 2021\n"
        "NEXT: Amazon total consolidated net sales fiscal year 2021",
        "FOUND: net sales 2021 = $469.8B\nNEXT: SUFFICIENT",
    ])
    r = _make(llm, retriever)

    out = r.retrieve_and_synthesize(q)

    assert out["sub_queries"][1] == "Amazon total consolidated net sales fiscal year 2021", out["sub_queries"]
    assert sorted(d.metadata["chunk_id"] for d in out["docs"]) == ["a", "b"], out
    print("✓ bridge query carrying a learned fact (2021) survives the guard and runs hop 2")


def test_repeated_hop_is_rejected():
    """REGRESSION (2026-06-05 live run): the detector re-proposed the SAME follow-up on
    hops 2 and 3, wasting a round on identical docs. A follow-up that duplicates any
    earlier hop must stop the loop."""
    retriever = FakeRetriever({"": [_doc("c", "x")]})  # matches everything
    # hop1 = original; detector proposes "alpha beta gamma" for hop2, then the SAME
    # query again — the duplicate guard must fire before a 3rd retrieval.
    llm = FakeLLM([
        "FOUND: f1\nNEXT: alpha beta gamma report",
        "FOUND: f2\nNEXT: alpha beta gamma report",
    ])
    r = _make(llm, retriever, max_hops=3)

    out = r.retrieve_and_synthesize("distinct original question topic")

    assert out["sub_queries"] == ["distinct original question topic", "alpha beta gamma report"], out["sub_queries"]
    assert len(retriever.queries) == 2, retriever.queries  # original + one follow-up, no repeat
    print("✓ a follow-up duplicating an earlier hop stops the loop")


def test_degenerate_followup_stops():
    """A follow-up equal to the original question is a no-op → loop stops (no wasted hop)."""
    retriever = FakeRetriever({"start": [_doc("c1", "x")]})
    llm = FakeLLM(["SEARCH: start"])  # echoes the original
    r = _make(llm, retriever, max_hops=3)

    out = r.retrieve_and_synthesize("start")

    assert out["sub_queries"] == ["start"], out["sub_queries"]
    assert len(retriever.queries) == 1, retriever.queries
    print("✓ degenerate (echo) follow-up does not trigger another hop")


def test_llm_failure_degrades_to_single_pass():
    """Gap detector raising → loop ends with hop-1 evidence (never worse than single-pass)."""
    class BoomLLM:
        calls = 0
        def invoke(self, messages):
            BoomLLM.calls += 1
            raise RuntimeError("network down")
    retriever = FakeRetriever({"q": [_doc("c1", "found it")]})
    r = _make(BoomLLM(), retriever, max_hops=3)

    out = r.retrieve_and_synthesize("q")

    assert out["unique_docs"] == 1, out
    assert out["sub_queries"] == ["q"], out["sub_queries"]
    assert len(retriever.queries) == 1
    print("✓ gap-detector failure degrades to single-pass retrieval")


def test_empty_then_fallback_to_original():
    """Every hop returns nothing → the final fallback retries the original query once."""
    calls = {"n": 0}
    class OnceRetriever:
        def __init__(self): self.queries = []
        def retrieve(self, query, filename_filter=None, page_filter=None,
                     filename_filters=None, top_k=None):
            self.queries.append(query)
            calls["n"] += 1
            # first (hop) call empty; the fallback call returns a doc
            return [] if calls["n"] == 1 else [_doc("c1", "fallback hit")]
    retriever = OnceRetriever()
    llm = FakeLLM(["SUFFICIENT"])  # hop-1 empty evidence → detector says stop
    r = _make(llm, retriever, max_hops=3)

    out = r.retrieve_and_synthesize("nothing matches")

    assert out["unique_docs"] == 1, out
    # hop query + fallback query (both the original)
    assert retriever.queries == ["nothing matches", "nothing matches"], retriever.queries
    print("✓ all-empty hops trigger the original-query fallback")


def main():
    tests = [
        test_informed_followup_and_dedup,
        test_stops_on_sufficient,
        test_hop_budget_caps_retrievals,
        test_found_next_format_informed_followup,
        test_reformulation_is_rejected,
        test_bridge_query_with_learned_fact_survives_guard,
        test_repeated_hop_is_rejected,
        test_degenerate_followup_stops,
        test_llm_failure_degrades_to_single_pass,
        test_empty_then_fallback_to_original,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"✗ {t.__name__}: {e}")
        except Exception as e:  # noqa
            failed += 1
            print(f"✗ {t.__name__}: unexpected {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
