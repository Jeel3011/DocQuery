"""Tool-adapter gate — AGENT_CORE_PLAN §3.3 / A1.

Drives the four agent-core tool adapters on the REAL 8-doc grids and asserts the
three contracts the plan names for A1:

  1. ENVELOPE shape — every result has {ok, summary, data, provenance}; abstains
     carry `abstain_reason`, errors carry `error`.
  2. ABSTAIN semantics — table_lookup on the canonical MSFT total→component trap
     either resolves to the TRUE total (198,270) or ABSTAINS with candidates; it
     NEVER returns a forbidden component value as ok=True. compute returns the right
     value WITH cell provenance.
  3. NEVER RAISES — garbage args / missing deps / wrong types all return an envelope,
     never an exception (the §3.2 loop contract).

No DB, no API — grids are re-extracted deterministically from the PDFs (the same
offline pattern as test_grounding.py), so this runs in CI as a pure structural gate.
search_vault is exercised with a STUB retrieval manager (the live recall fixture set
is a Phase-B item per §3.3) — here we check its envelope/never-raise contract only.

Run: python -u eval/test_tools.py
"""

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.components.table_extraction import extract_tables_from_pdf
from src.components.brain.analyst import Grid
from src.components.agent_core.tools import (
    compute, read_document, search_vault, table_lookup, list_metrics,
    search_knowledge, SCHEMAS,
)

CORPUS = "test docs"
_grids_cache: dict = {}


def grids_for(doc):
    if doc not in _grids_cache:
        _grids_cache[doc] = [
            Grid(t.to_metadata(), doc=doc, page=t.page_number)
            for t in extract_tables_from_pdf(f"{CORPUS}/{doc}")
        ]
    return _grids_cache[doc]


# ── envelope shape assertions ───────────────────────────────────────────────────

def is_envelope(r) -> bool:
    if not isinstance(r, dict):
        return False
    if not all(key in r for key in ("ok", "summary", "data", "provenance")):
        return False
    if not isinstance(r["ok"], bool) or not isinstance(r["summary"], str):
        return False
    if not isinstance(r["provenance"], list):
        return False
    if r["ok"] is False and "abstain_reason" not in r and "error" not in r:
        return False
    return True


class Check:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def ok(self, cond, label):
        if cond:
            self.passed += 1
            print(f"  [PASS] {label}")
        else:
            self.failed += 1
            print(f"  [FAIL] {label}")


def main() -> int:
    c = Check()
    msft = grids_for("msft-10k_20220630.htm.pdf")
    amzn = grids_for("amzn-20221231.pdf")

    print("── compute (kernel adapter) ─────────────────────────────────────")
    # A real growth_pct on AMZN net sales should resolve with cell provenance.
    r = compute(
        {"op": "growth_pct", "row": {"section": "", "label": "Total net sales"},
         "from_period": "2021", "to_period": "2022"},
        amzn,
    )
    c.ok(is_envelope(r), "compute: returns a valid envelope")
    # Either it computed (ok + provenance) or it cleanly errored — never a raise.
    if r["ok"]:
        c.ok(len(r["provenance"]) >= 1, "compute: ok result carries cell provenance")
        c.ok("formula" in (r["data"] or {}), "compute: ok result carries a formula")
    else:
        c.ok("error" in r, "compute: non-ok result is a clean error (no crash)")
    # Bad op → error envelope, not a raise.
    r = compute({"op": "definitely_not_an_op", "row": {"label": "x"}}, amzn)
    c.ok(is_envelope(r) and not r["ok"] and "error" in r, "compute: bad op → error envelope")
    # Garbage spec → error envelope.
    r = compute("not a dict", amzn)
    c.ok(is_envelope(r) and not r["ok"], "compute: non-dict spec → error envelope (no raise)")
    # Missing grids → no crash.
    r = compute({"op": "value", "row": {"label": "x"}, "period": "2022"}, [])
    c.ok(is_envelope(r), "compute: empty grids → envelope (no raise)")

    print("\n── table_lookup (grounding adapter) ─────────────────────────────")
    # THE trap: 'total revenue' must resolve to the true total OR abstain with
    # candidates — NEVER return a forbidden component as ok=True.
    FORBIDDEN = {135620.0, 62650.0}
    r = table_lookup("total revenue", "2022", msft, aggregation="total")
    c.ok(is_envelope(r), "table_lookup: returns a valid envelope")
    if r["ok"]:
        val = (r["data"] or {}).get("value")
        c.ok(val not in FORBIDDEN, f"table_lookup: never binds a forbidden component (got {val})")
        c.ok(len(r["provenance"]) == 1, "table_lookup: ok result carries exactly one cell")
    else:
        c.ok("abstain_reason" in r, "table_lookup: non-ok is an ABSTAIN (with reason)")
        c.ok("candidates" in (r["data"] or {}), "table_lookup: abstain returns candidates to the model")
    # Missing required args → error, not raise.
    r = table_lookup("", "2022", msft)
    c.ok(is_envelope(r) and not r["ok"] and "error" in r, "table_lookup: empty metric → error envelope")
    # Garbage grids → no crash.
    r = table_lookup("revenue", "2022", None)
    c.ok(is_envelope(r), "table_lookup: None grids → envelope (no raise)")

    print("\n── read_document (grid-load adapter) ────────────────────────────")
    # Offline path: pass pre-built grids (no db_client). Kernel-style doc matching: a
    # distinctive substring is enough (the model rarely reproduces the extension).
    r = read_document("msft-10k_20220630", grids=msft)
    c.ok(is_envelope(r) and r["ok"], "read_document: pre-loaded grids → ok envelope")
    c.ok(len(r["data"]["grids"]) == len(msft), "read_document: returns all grid JSONs")
    c.ok(all("rows" in g and "periods" in g for g in r["data"]["grids"]),
         "read_document: each grid JSON has rows + periods")
    # No doc_id and no grids → error.
    r = read_document("", grids=None)
    c.ok(is_envelope(r) and not r["ok"], "read_document: no doc_id/grids → error envelope")
    # Live-ish call with a None db_client and no grids → ok envelope, empty grids (no raise).
    r = read_document("some-doc-id", db_client=None)
    c.ok(is_envelope(r), "read_document: no db_client → envelope (no raise)")
    # doc_id matching the grids' doc label scopes the result to that doc only.
    mixed = msft + amzn
    r = read_document("amzn-20221231.pdf", grids=mixed)
    c.ok(r["ok"] and all(g["doc"] == "amzn-20221231.pdf" for g in r["data"]["grids"]),
         "read_document: doc_id filters pre-loaded grids to the requested doc")
    c.ok(0 < len(r["data"]["grids"]) < len(mixed),
         "read_document: filtered set is a strict, non-empty subset")

    # Doc-miss with NO db_client: the old code fell back to ALL preloaded grids — the
    # model believed it read doc X while seeing other documents' tables (live
    # 2026-06-11). New contract: a self-healing error naming the loaded docs.
    r = read_document("nvda-20240128.pdf", grids=mixed)
    c.ok(is_envelope(r) and not r["ok"] and "not in the loaded scope" in (r.get("error") or "")
         and "amzn-20221231.pdf" in (r.get("error") or ""),
         "read_document: unknown doc → error naming loaded docs (no silent fallback)")

    # Harness never-blank guard (live Phase-2 bug): a doc that resolves but returns NO
    # content with the given args (table_grids=false, no page_range, no full_text) must come
    # back with an ACTIONABLE summary pointing to full_text=true / read_section — NOT a
    # silent empty ok envelope (which made the agent degrade to a blank answer). With no
    # db_client and table_grids=false there are no grids and no page_text → the guard fires.
    r = read_document("some-doc-id", db_client=None, table_grids=False)
    c.ok(is_envelope(r) and r["ok"] and not r["data"]["grids"] and not r["data"]["page_text"],
         "read_document: no-content read still returns an ok envelope (no raise)")
    c.ok("full_text=true" in (r.get("summary") or "") and "read_section" in (r.get("summary") or ""),
         "read_document: no-content read's summary points to full_text/read_section (never silently blank)")

    # Doc-miss WITH a db_client: the doc's grids load FRESH and JOIN the run's grid
    # scope, so a follow-up compute can use them (live: the model read the right doc
    # but compute said "document not in scope").
    from types import SimpleNamespace

    class _FakeQ:
        def __init__(self, rows): self._rows = rows
        def select(self, *_a, **_k): return self
        def eq(self, *_a, **_k): return self
        def limit(self, *_a, **_k): return self
        def execute(self): return SimpleNamespace(data=self._rows)

    class _FakeDB:
        def __init__(self, rows):
            self.client = SimpleNamespace(table=lambda _name: _FakeQ(rows))

    _tj = {"headers": ["section", "label", "2022", "2021"], "periods": ["2022", "2021"],
           "rows": [{"section": "Revenue", "label": "Total revenue",
                     "2022": "200", "2021": "100"}]}
    _rows = [{"content": "fresh table",
              "metadata": {"chunk_type": "table", "table_json": _tj, "page_number": 7}}]
    scope_list = list(msft)  # the live list the registry hands to compute
    r = read_document("fresh-doc.pdf", grids=msft, db_client=_FakeDB(_rows),
                      filename_by_doc={"uuid-fresh-1": "fresh-doc.pdf"},
                      scope_grids=scope_list)
    c.ok(is_envelope(r) and r["ok"] and len(r["data"]["grids"]) == 1
         and r["data"]["grids"][0]["doc"] == "fresh-doc.pdf",
         "read_document: doc-miss + db → loads that doc's grids fresh")
    c.ok(len(scope_list) == len(msft) + 1
         and any(getattr(g, "doc", None) == "fresh-doc.pdf" for g in scope_list),
         "read_document: fresh grids JOIN the run's compute scope")
    r = compute({"op": "growth_pct", "doc": "fresh-doc.pdf",
                 "row": {"section": "Revenue", "label": "Total revenue"},
                 "from_period": "2021", "to_period": "2022"}, scope_list)
    c.ok(r["ok"] and abs(((r["data"] or {}).get("value") or 0) - 100.0) < 1e-9,
         "compute: resolves in the freshly joined doc (no 'document not in scope')")
    # BUG-D: the DERIVED result (a growth %) is ledgered as a traceable param — the
    # gate redacted a CORRECT computed growth live because only operand cells shipped.
    _pk = [p for p in r["provenance"] if p.get("kind") == "param" and p.get("label") == "result"]
    c.ok(len(_pk) == 1 and _pk[0]["value"] == 100.0,
         "compute: derived result value ships in provenance (gate-traceable)")

    print("\n── compute: entity-axis selection (over='entity' + rows) ────────")
    # The schema must let the model express a cross-entity argmax (the cross-company
    # question class). Fixture: two sections, one metric, fixed period.
    seg = Grid({"headers": ["label", "2022"], "periods": ["2022"],
                "rows": [{"section": "AWS", "label": "Net sales", "2022": "80,096"},
                         {"section": "North America", "label": "Net sales", "2022": "315,880"}],
                "table_id": "seg"}, doc="amzn-2022", page=108)
    c.ok("over" in SCHEMAS["compute"]["input_schema"]["properties"]
         and "rows" in SCHEMAS["compute"]["input_schema"]["properties"],
         "compute schema exposes over + rows (entity axis expressible)")
    r = compute({"op": "argmax", "over": "entity", "period": "2022",
                 "rows": [{"row": {"section": "AWS", "label": "Net sales"}},
                          {"row": {"section": "North America", "label": "Net sales"}}]},
                [seg])
    c.ok(r["ok"] and r["data"]["binding"] == "North America",
         "compute: entity argmax binds the right entity"
         + ("" if r["ok"] else f" (error: {r.get('error')})"))
    c.ok(r["ok"] and len(r["data"]["candidates"]) == 2,
         "compute: entity argmax carries the full completeness trail (2 candidates)")

    print("\n── search_vault (retrieval adapter) ─────────────────────────────")
    # Stub manager mimicking RetrievalManager's two methods.
    class _Doc:
        def __init__(self, text, md):
            self.page_content = text
            self.metadata = md

    class _StubRM:
        def __init__(self):
            self.table_kwargs = None
            self.text_kwargs = None
        def retrieve(self, query, **kw):
            self.text_kwargs = kw  # capture what search_vault passed to text retrieval
            return [_Doc("net sales rose", {"filename": "amzn.pdf", "page_number": 41,
                                            "chunk_id": "c1", "score": 0.9})]
        def retrieve_table_chunks(self, query, **kw):
            self.table_kwargs = kw  # capture what search_vault passed
            return [_Doc("table", {"filename": "amzn.pdf", "page_number": 42, "chunk_id": "t1"})]

    r = search_vault("AWS net sales", _StubRM(), scope={"collection_id": "x"}, k=4, kind="both")
    c.ok(is_envelope(r) and r["ok"], "search_vault: stub manager → ok envelope")
    c.ok(len(r["provenance"]) >= 1 and r["provenance"][0]["kind"] == "span",
         "search_vault: provenance is chunk spans")

    # BUG-F (live, 2026-06-11): table retrieval must filter by FILENAME, not
    # collection_id. Ingest never stamps collection_id on chunks, and
    # retrieve_table_chunks prioritizes collection_id when present → the live filter
    # {chunk_type:table, collection_id:<id>} matched 0 chunks on every numeric query
    # (the model's primary "find the table" tool was silently dead). Assert
    # search_vault passes filenames and does NOT pass collection_id to table search.
    stub = _StubRM()
    search_vault("AWS net sales", stub,
                 scope={"collection_id": "x", "filenames": ["amzn.pdf", "msft.pdf"]},
                 k=4, kind="table")
    c.ok(stub.table_kwargs is not None
         and stub.table_kwargs.get("filename_filters") == ["amzn.pdf", "msft.pdf"]
         and not stub.table_kwargs.get("collection_id"),
         "search_vault(table): filters by FILENAME, not collection_id (BUG-F)")
    # ── G3 Step A: doc_id is the stable scope axis (vault isolation as DATA) ──────
    # When the scope carries real doc_ids, search_vault scopes by doc_id (NOT filename),
    # and the retriever turns a multi-doc scope into a `doc_id $in` filter.
    from src.components.retrieval import RetrievalManager
    stub = _StubRM()
    search_vault("net sales", stub,
                 scope={"collection_id": "x", "doc_ids": ["docA1", "docA2"],
                        "filenames": ["amzn.pdf"]},  # filenames must NOT win over doc_ids
                 k=4, kind="both")
    c.ok(stub.text_kwargs is not None
         and stub.text_kwargs.get("doc_ids") == ["docA1", "docA2"]
         and not stub.text_kwargs.get("filename_filters")
         and not stub.text_kwargs.get("filename_filter"),
         "search_vault(text): doc_ids scope wins over filename fallback (G3 Step A)")
    c.ok(stub.table_kwargs is not None and stub.table_kwargs.get("doc_ids") == ["docA1", "docA2"],
         "search_vault(table): scopes by doc_id $in when doc_ids present (G3 Step A)")
    # The doc_ids actually become a `doc_id $in` Pinecone filter in the retriever.
    f = RetrievalManager._build_filter(doc_ids=stub.text_kwargs.get("doc_ids"))
    c.ok(f == {"doc_id": {"$in": ["docA1", "docA2"]}},
         "search_vault → retriever builds `doc_id $in` filter (data-property isolation)")

    # ── G3 Step B: scope.filters → conjunctive metadata_filter on BOTH branches ──
    stub = _StubRM()
    search_vault("contracts", stub,
                 scope={"doc_ids": ["docA1", "docA2"],
                        "filters": {"doc_type": "legal_contract",
                                    "fiscal_year": {"$in": [2023]}}},
                 k=4, kind="both")
    mf = {"doc_type": "legal_contract", "fiscal_year": {"$in": [2023]}}
    c.ok(stub.text_kwargs.get("metadata_filter") == mf,
         "search_vault(text): scope.filters threads through as metadata_filter (Step B)")
    c.ok(stub.table_kwargs.get("metadata_filter") == mf,
         "search_vault(table): scope.filters threads through as metadata_filter (Step B)")
    # The merged Pinecone filter is the scope (doc_id $in) AND the narrowing — CONJUNCTIVE.
    merged = RetrievalManager._build_filter(
        doc_ids=stub.text_kwargs.get("doc_ids"),
        metadata_filter=stub.text_kwargs.get("metadata_filter"),
    )
    c.ok(merged == {"doc_id": {"$in": ["docA1", "docA2"]},
                    "doc_type": "legal_contract", "fiscal_year": {"$in": [2023]}},
         "Step B: merged filter = vault scope AND metadata narrowing (conjunctive)")
    # A metadata_filter can NEVER overwrite the scope key (no cross-vault widening).
    leak = RetrievalManager._build_filter(
        doc_ids=["docA1"], metadata_filter={"doc_id": "docB1", "doc_type": "x"},
    )
    c.ok(leak.get("doc_id") == {"$in": ["docA1"]} and leak.get("doc_type") == "x",
         "Step B: metadata_filter cannot replace the doc_id scope key (no leak)")
    # No filters → no metadata_filter key forced on (clean omission).
    stub = _StubRM()
    search_vault("x", stub, scope={"doc_ids": ["docA1"]}, k=4, kind="both")
    c.ok(not stub.text_kwargs.get("metadata_filter"),
         "search_vault: no scope.filters → metadata_filter is absent (no empty {})")

    # No manager → error, not raise.
    r = search_vault("x", None)
    c.ok(is_envelope(r) and not r["ok"] and "error" in r, "search_vault: no manager → error envelope")
    # A manager whose methods raise → tool still returns an envelope (never raises).
    class _BoomRM:
        def retrieve(self, *a, **k): raise RuntimeError("boom")
        def retrieve_table_chunks(self, *a, **k): raise RuntimeError("boom")
    r = search_vault("x", _BoomRM(), kind="both")
    c.ok(is_envelope(r), "search_vault: raising manager → envelope (no raise escapes)")

    print("\n── T1: retrieval quality grading (CRAG — external signal) ─────────")
    # Grade helpers are pure functions; test them directly first, then via the tool.
    from src.components.agent_core.tools.search import _grade_spans, _expand_query

    # _grade_spans: empty → grade=empty
    c.ok(_grade_spans([])["grade"] == "empty",
         "T1 _grade_spans: empty span list → grade=empty")
    # _grade_spans: all below threshold → weak
    low_spans = [{"score": 0.30}, {"score": 0.25}]
    c.ok(_grade_spans(low_spans)["grade"] == "weak",
         "T1 _grade_spans: all scores below threshold → grade=weak")
    # _grade_spans: top high but no gap vs median → weak (clustered, low signal)
    clustered = [{"score": 0.60}, {"score": 0.58}, {"score": 0.57}]
    c.ok(_grade_spans(clustered)["grade"] == "weak",
         "T1 _grade_spans: high top but clustered (gap<0.10) → grade=weak")
    # _grade_spans: single high-score span → strong (no median to compare)
    c.ok(_grade_spans([{"score": 0.90}])["grade"] == "strong",
         "T1 _grade_spans: single span above threshold → grade=strong")
    # _grade_spans: top high WITH gap ≥ 0.10 → strong
    clear = [{"score": 0.85}, {"score": 0.70}, {"score": 0.40}]
    c.ok(_grade_spans(clear)["grade"] == "strong",
         "T1 _grade_spans: top with clear gap (0.85 vs median 0.70 → 0.15 gap) → strong")
    # _grade_spans: no scores at all (table path) → treat as strong (no false alarm)
    c.ok(_grade_spans([{"snippet": "text", "chunk_id": "c1"}])["grade"] == "strong",
         "T1 _grade_spans: score-less spans (table path) → grade=strong (no false alarm)")

    # _expand_query: parenthetical stripped
    c.ok(_expand_query("governing law (dispute resolution)") == "governing law",
         "T1 _expand_query: trailing parenthetical stripped")
    # _expand_query: prepositional phrase stripped
    c.ok(_expand_query("net sales in fiscal year 2023") == "net sales",
         "T1 _expand_query: trailing 'in <phrase>' stripped")
    # _expand_query: single-word query → None (can't broaden further)
    c.ok(_expand_query("revenue") is None,
         "T1 _expand_query: single-word query → None (no broadening possible)")

    # Via the tool: strong scores → grade=strong in envelope, no repair, no extra retrieve.
    class _ScoreRM:
        """Returns docs with controllable scores to drive grade logic."""
        def __init__(self, text_scores, expand_scores=None):
            self._text_scores = text_scores
            self._expand_scores = expand_scores or []
            self.expand_called = False
            self.retrieve_calls = 0
        def retrieve(self, query, **kw):
            self.retrieve_calls += 1
            is_expand = self.retrieve_calls > 1
            scores = self._expand_scores if is_expand else self._text_scores
            return [_Doc("text", {"filename": "a.pdf", "page_number": i+1,
                                  "chunk_id": f"c{i}", "score": sc})
                    for i, sc in enumerate(scores)]
        def retrieve_table_chunks(self, query, **kw):
            return []

    # strong case: high score with gap → grade=strong, only ONE retrieve call
    rm = _ScoreRM(text_scores=[0.85, 0.40])
    r = search_vault("governing law", rm, scope={"collection_id": "x"}, kind="text")
    c.ok(is_envelope(r) and r["ok"], "T1: strong retrieval → ok envelope")
    c.ok(r["data"]["retrieval_quality"]["grade"] == "strong",
         "T1: high-score+gap result → grade=strong in envelope")
    c.ok(rm.retrieve_calls == 1,
         "T1: strong grade → no expansion (only 1 retrieve call)")

    # weak case: low scores → grade=weak, expansion attempted, repair named
    rm_weak = _ScoreRM(text_scores=[0.30, 0.25],
                       expand_scores=[0.75, 0.50])
    r = search_vault("governing law (dispute resolution)", rm_weak,
                     scope={"collection_id": "x"}, kind="text")
    c.ok(is_envelope(r) and r["ok"], "T1: weak retrieval → still ok envelope (graceful)")
    c.ok(r["data"]["retrieval_quality"]["grade"] in ("weak", "strong"),
         "T1: weak initial → grade re-evaluated after expansion")
    c.ok(rm_weak.retrieve_calls == 2,
         "T1: weak grade → one expansion retrieve fired (2 retrieve calls total)")
    c.ok("repair" in r["data"]["retrieval_quality"],
         "T1: weak result → repair key present in retrieval_quality")
    c.ok("governing law" in (r["data"]["retrieval_quality"].get("repair") or ""),
         "T1: repair names the expanded query")
    # The summary line reflects the grade
    c.ok("[" in r.get("summary", ""),
         "T1: summary includes grade bracket e.g. [strong] or [weak]")

    # empty case: nothing returned on first retrieve, expansion also empty → grade=empty
    class _EmptyRM:
        def retrieve(self, *a, **k): return []
        def retrieve_table_chunks(self, *a, **k): return []
    r = search_vault("xyzzy nonexistent topic", _EmptyRM(), kind="both")
    c.ok(r["ok"] and r["data"]["retrieval_quality"]["grade"] == "empty",
         "T1: zero chunks → grade=empty (not an error, still ok)")
    c.ok("repair" in r["data"]["retrieval_quality"],
         "T1: empty grade → repair guidance present")

    # LIVE gate owed (not run here — requires live Pinecone):
    # eval/routing_recall.py: a semantically-distant query against the real index
    # must return grade=weak; a precise query must return grade=strong.

    print("\n── S-E: ingestion fidelity_warning in search_vault spans ───────")
    # S-E: when a doc was ingested with low fidelity (grade != "good"), spans from that
    # doc must carry fidelity_warning=True so the agent sees the caveat.

    class _FidelityRM:
        """Returns one span with a doc_id so we can test the stamp."""
        def __init__(self, doc_id):
            self._doc_id = doc_id
        def retrieve(self, query, **kw):
            return [_Doc("net sales rose", {"filename": "a.pdf", "page_number": 1,
                                            "chunk_id": "c1", "score": 0.9,
                                            "doc_id": self._doc_id})]
        def retrieve_table_chunks(self, query, **kw):
            return []

    DOC_ID_LOW = "doc-low-fidelity"
    DOC_ID_GOOD = "doc-good-fidelity"
    DOC_ID_UNKNOWN = "doc-no-entry"

    # S-E.1: a low-fidelity doc's spans carry fidelity_warning=True.
    r = search_vault("net sales", _FidelityRM(DOC_ID_LOW),
                     scope={"collection_id": "x"},
                     fidelity_by_doc={DOC_ID_LOW: "partial", DOC_ID_GOOD: "good"},
                     kind="text")
    c.ok(is_envelope(r) and r["ok"], "S-E: search_vault with fidelity_by_doc → ok envelope")
    low_spans = [sp for sp in r["provenance"] if sp.get("doc_id") == DOC_ID_LOW]
    c.ok(low_spans and low_spans[0].get("fidelity_warning") is True,
         "S-E: span from low-fidelity doc (grade=partial) carries fidelity_warning=True")

    # S-E.2: a good-fidelity doc's spans do NOT carry fidelity_warning.
    r_good = search_vault("net sales", _FidelityRM(DOC_ID_GOOD),
                          scope={"collection_id": "x"},
                          fidelity_by_doc={DOC_ID_LOW: "partial", DOC_ID_GOOD: "good"},
                          kind="text")
    good_spans = [sp for sp in r_good["provenance"] if sp.get("doc_id") == DOC_ID_GOOD]
    c.ok(good_spans and not good_spans[0].get("fidelity_warning"),
         "S-E: span from good-fidelity doc does NOT carry fidelity_warning")

    # S-E.3: when fidelity_by_doc is None (flag off) → byte-identical (no fidelity_warning).
    r_off = search_vault("net sales", _FidelityRM(DOC_ID_LOW),
                         scope={"collection_id": "x"}, kind="text")
    off_spans = r_off["provenance"]
    c.ok(not any(sp.get("fidelity_warning") for sp in off_spans),
         "S-E: fidelity_by_doc=None → no fidelity_warning on any span (flag-off byte-identical)")

    # S-E.4: a doc_id not in fidelity_by_doc (unknown) → no warning (treat as good, don't alarm).
    r_unk = search_vault("net sales", _FidelityRM(DOC_ID_UNKNOWN),
                         scope={"collection_id": "x"},
                         fidelity_by_doc={DOC_ID_LOW: "partial", DOC_ID_GOOD: "good"},
                         kind="text")
    unk_spans = [sp for sp in r_unk["provenance"] if sp.get("doc_id") == DOC_ID_UNKNOWN]
    c.ok(unk_spans and not unk_spans[0].get("fidelity_warning"),
         "S-E: doc_id not in fidelity_by_doc → no fidelity_warning (unknown ≠ low)")

    # S-E.5: span_to_dict carries doc_id from chunk metadata.
    from src.components.agent_core.tools._envelope import span_to_dict as _s2d
    _fake_chunk = type("D", (), {
        "page_content": "some text",
        "metadata": {"filename": "x.pdf", "page_number": 1,
                     "chunk_id": "c99", "doc_id": "uuid-42"},
    })()
    _sp = _s2d(_fake_chunk)
    c.ok(_sp.get("doc_id") == "uuid-42",
         "S-E: span_to_dict reads doc_id from chunk metadata (the fidelity stamp axis)")

    print("\n── survey_collection (G5 broad-pass tool — demoted map_reduce) ──")
    from src.components.agent_core.tools import survey_collection
    from src.components.brain import map_reduce as _mr
    from src.components.brain.claims import PerDocExtract, Claim, EvidenceSpan

    # survey_collection retrieves per-doc (chat.py POSITIONAL signature: query,
    # filename_filter, page_filter, filename_filters, top_k, apply_threshold, use_reranker)
    # then runs the demoted Brain MAP step. To keep the gate $0/offline, we (a) feed a stub
    # RM that returns chunks and (b) monkeypatch Brain._map_all_docs so NO LLM is called —
    # we assert the tool's RETRIEVAL→ENVELOPE→PROVENANCE shaping, the part G5 added.
    class _SurveyRM:
        def __init__(self, with_chunks=True):
            self.with_chunks = with_chunks
            self.calls = []
        def retrieve(self, query, *args, **kw):
            self.calls.append((query, args, kw))
            if not self.with_chunks:
                return []
            fname = args[0] if args else kw.get("filename_filter", "doc.pdf")
            return [_Doc("the agreement is governed by the laws of India",
                         {"filename": fname, "page_number": 5, "chunk_id": "c1",
                          "doc_id": f"id-{fname}"})]

    # Canned MAP output (no LLM): two docs with cited claims, one empty, one errored.
    def _fake_map_all(query, doc_chunks, on_progress=None):
        out = []
        for doc_id, (fname, _chunks) in doc_chunks.items():
            out.append(PerDocExtract(
                doc_id=doc_id, filename=fname,
                claims=[Claim(
                    text=f"{fname}: governed by Indian law",
                    evidence=[EvidenceSpan(doc_id=doc_id, chunk_id="c1",
                                           verbatim_span="governed by the laws of India")],
                    confidence=0.9,
                )],
            ))
        return out

    _orig_map_all = _mr.Brain._map_all_docs
    try:
        _mr.Brain._map_all_docs = staticmethod(lambda q, dc, on_progress=None: _fake_map_all(q, dc))
        cfg = object()  # Brain(config) only stores it; the patched MAP never touches it
        rm = _SurveyRM()
        r = survey_collection("key terms and governing law", rm, cfg,
                              filenames=["a.pdf", "b.pdf"],
                              filename_by_doc={"id-a.pdf": "a.pdf", "id-b.pdf": "b.pdf"})
        c.ok(is_envelope(r) and r["ok"], "survey_collection: stub RM + MAP → ok envelope")
        c.ok(isinstance(r["data"].get("clusters"), list) and len(r["data"]["clusters"]) == 2,
             "survey_collection: returns one evidence cluster per relevant doc")
        c.ok(all(cl.get("claims") and "verbatim_span" in cl["claims"][0]
                 for cl in r["data"]["clusters"]),
             "survey_collection: each cluster carries cited claims (not a written answer)")
        c.ok(len(r["provenance"]) >= 2 and all(p["kind"] == "span" for p in r["provenance"]),
             "survey_collection: provenance is evidence spans (flows to ledger/gate)")
        c.ok(all(p.get("snippet") for p in r["provenance"]),
             "survey_collection: each span carries its verbatim snippet")
        # Retrieval used the breadth settings (apply_threshold=False, use_reranker=False).
        c.ok(rm.calls and rm.calls[0][1][4] is False and rm.calls[0][1][5] is False,
             "survey_collection: broad retrieval (no similarity floor, no reranker)")

        # No chunks anywhere → a CLEAN empty survey (ok=True, 0 clusters), not an error.
        r2 = survey_collection("nothing here", _SurveyRM(with_chunks=False), cfg,
                               filenames=["a.pdf"], filename_by_doc={"id-a.pdf": "a.pdf"})
        c.ok(is_envelope(r2) and r2["ok"] and r2["data"]["clusters"] == [],
             "survey_collection: no relevant passages → clean empty survey (ok, 0 clusters)")

        # never-raise contract: missing deps + a raising RM → error envelope, no exception.
        c.ok(is_envelope(survey_collection("", rm, cfg)) and not survey_collection("", rm, cfg)["ok"],
             "survey_collection: empty query → error envelope")
        c.ok(is_envelope(survey_collection("q", None, cfg)) and not survey_collection("q", None, cfg)["ok"],
             "survey_collection: no retrieval manager → error envelope")
        c.ok(is_envelope(survey_collection("q", rm, None)) and not survey_collection("q", rm, None)["ok"],
             "survey_collection: no config → error envelope")
        class _BoomSurveyRM:
            def retrieve(self, *a, **k): raise RuntimeError("boom")
        rboom = survey_collection("q", _BoomSurveyRM(), cfg,
                                  filenames=["a.pdf"], filename_by_doc={"id-a.pdf": "a.pdf"})
        # A per-doc retrieval that raises is swallowed (non-fatal) → clean empty survey.
        c.ok(is_envelope(rboom) and rboom["ok"] and rboom["data"]["clusters"] == [],
             "survey_collection: raising RM is non-fatal per-doc → envelope (no raise escapes)")
    finally:
        _mr.Brain._map_all_docs = _orig_map_all

    print("\n── list_metrics (the line-item index — anti-guessing) ───────────")
    # Live (2026-06-11) the model GUESSED labels it couldn't know ("Technology and
    # content" instead of Amazon's real "Technology and infrastructure") → every
    # compute/lookup failed → abstain. list_metrics shows the REAL labels so it picks
    # an exact reference. Real AMZN + GOOG FY23 grids in one scope.
    amzn = grids_for("amzn-20231231.pdf")
    goog = grids_for("goog-20231231.pdf")
    both = amzn + goog
    r = list_metrics("amzn-20231231.pdf", both, contains="technology", period="2023")
    labels = [m["label"] for m in r["data"]["metrics"]] if r["ok"] else []
    c.ok(is_envelope(r) and r["ok"] and "Technology and infrastructure" in labels,
         "list_metrics: surfaces AMZN's REAL R&D label 'Technology and infrastructure'")
    # the model can now distinguish the $ row (Operating Expenses) from the % twins.
    secs = {(m["label"], m["section"]) for m in r["data"]["metrics"]}
    c.ok(any(lab == "Technology and infrastructure" and "Operating" in sec for lab, sec in secs),
         "list_metrics: shows the section so the $ row is pickable (not the % twin)")
    # scoping: a GOOG query does not leak AMZN rows.
    r = list_metrics("goog-20231231.pdf", both, contains="research")
    gdocs = {m.get("label") for m in r["data"]["metrics"]} if r["ok"] else set()
    c.ok(r["ok"] and "Research and development" in gdocs,
         "list_metrics: GOOG scope surfaces 'Research and development'")
    # prose fragments are filtered OUT (the is_lineitem_label guard).
    c.ok(r["ok"] and not any("increased" in (m["label"] or "").lower()
                             for m in r["data"]["metrics"]),
         "list_metrics: prose fragments excluded (only real line-items)")
    # never-raise: unknown doc + empty scope → error envelopes.
    c.ok(is_envelope(list_metrics("nope.pdf", both)) and not list_metrics("nope.pdf", both)["ok"],
         "list_metrics: unknown doc → error envelope (names loaded docs)")
    c.ok(is_envelope(list_metrics("x", [])) and not list_metrics("x", [])["ok"],
         "list_metrics: empty scope → error envelope (no raise)")

    print("\n── search_knowledge (G8 legal KB adapter) ──────────────────────")
    # A stub KB retrieval manager: captures the metadata_filter search_knowledge built
    # and returns KB spans whose metadata carries citation/as_of (the version-in-force
    # stamps). $0/offline — same shape contract as search_vault's stub.
    class _StubKB:
        def __init__(self, docs=None):
            self.kw = None
            self._docs = docs if docs is not None else [
                _Doc("Art.21 — No person shall be deprived of his life or personal liberty…",
                     {"citation": "Constitution Art.21", "source_key": "constitution_of_india",
                      "instrument_type": "article", "section_or_article_id": "21",
                      "jurisdiction": "IN", "as_of_date": "2025-01-01",
                      "enacted_date": "1950-01-26", "chunk_id": "kb1", "page_number": 9}),
            ]
        def retrieve(self, query, **kw):
            self.kw = kw
            return list(self._docs)

    kb = _StubKB()
    r = search_knowledge("right to life", kb, jurisdiction="IN")
    c.ok(is_envelope(r) and r["ok"], "search_knowledge: stub KB manager → ok envelope")
    c.ok(r["provenance"] and r["provenance"][0]["kind"] == "knowledge"
         and r["provenance"][0]["citation"] == "Constitution Art.21",
         "search_knowledge: provenance is cited KB passages (citation carried)")
    # the span's `doc` (the gate's bracket marker) IS the citation, so the cite gate
    # binds `[Constitution Art.21]` exactly like a `[doc p.N]` vault marker.
    c.ok(r["provenance"][0]["doc"] == "Constitution Art.21",
         "search_knowledge: span.doc == citation (flows through the same cite gate)")
    # filters: source + instrument_type + jurisdiction → conjunctive metadata_filter.
    kb = _StubKB()
    search_knowledge("directors", kb, jurisdiction="IN",
                     source="companies_act_2013", instrument_type="act")
    c.ok(kb.kw and kb.kw.get("metadata_filter") == {
            "jurisdiction": "IN", "source_key": "companies_act_2013", "instrument_type": "act"},
         "search_knowledge: source/instrument/jurisdiction → conjunctive metadata_filter")
    # §G8.5 version-in-force: a repealed span is WITHHELD, never cited as current.
    repealed_doc = _Doc("(old text)", {"citation": "X s.1", "source_key": "x",
                                       "instrument_type": "act", "as_of_date": "2025-01-01",
                                       "enacted_date": "1990-01-01", "repealed": True,
                                       "chunk_id": "kb-r"})
    r = search_knowledge("q", _StubKB([repealed_doc]))
    c.ok(r["ok"] and r["data"]["passages"] == [] and r["data"]["withheld"] == 1,
         "search_knowledge: a repealed provision is withheld (wrong-cell → not cited)")
    # as_of PAST the snapshot horizon → withheld (we cannot vouch; never guess).
    r = search_knowledge("right to life", _StubKB(), as_of="2030-01-01")
    c.ok(r["ok"] and r["data"]["passages"] == [] and r["data"]["withheld"] == 1,
         "search_knowledge: as_of past the snapshot horizon → withheld (version-or-abstain)")
    # as_of INSIDE the window → the in-force provision is returned.
    r = search_knowledge("right to life", _StubKB(), as_of="2020-06-01")
    c.ok(r["ok"] and len(r["data"]["passages"]) == 1,
         "search_knowledge: as_of inside the vouchable window → in-force provision returned")
    # never-raise: empty query + missing manager + a raising manager → error envelopes.
    c.ok(is_envelope(search_knowledge("", kb)) and not search_knowledge("", kb)["ok"],
         "search_knowledge: empty query → error envelope")
    c.ok(is_envelope(search_knowledge("q", None)) and not search_knowledge("q", None)["ok"],
         "search_knowledge: no KB manager → error envelope")
    class _BoomKB:
        def retrieve(self, *a, **k): raise RuntimeError("boom")
    c.ok(is_envelope(search_knowledge("q", _BoomKB())) and not search_knowledge("q", _BoomKB())["ok"],
         "search_knowledge: raising KB manager → error envelope (never raises)")

    # ── G8.7 source-chip allow-list (the chips' server-side gate) ────────────────────
    # A mixed KB: one statute Article + one judgment. The allow-list must let only the
    # enabled instrument types through — a chip off ⇒ that source is UNREACHABLE.
    _art = _Doc("Art.21 — right to life…",
                {"citation": "Constitution Art.21", "source_key": "constitution_of_india",
                 "instrument_type": "article", "as_of_date": "2025-01-01",
                 "enacted_date": "1950-01-26", "chunk_id": "kb-art"})
    _judg = _Doc("Maneka Gandhi v. Union of India…",
                 {"citation": "Maneka Gandhi (1978)", "source_key": "sc_judgments",
                  "instrument_type": "judgment", "as_of_date": "2025-01-01",
                  "enacted_date": "1978-01-25", "chunk_id": "kb-judg"})
    mixed = _StubKB([_art, _judg])
    # No allow-list ⇒ both returned (default, byte-identical).
    r = search_knowledge("liberty", mixed)
    c.ok(r["ok"] and len(r["data"]["passages"]) == 2,
         "search_knowledge: no allow-list → all instrument types returned")
    # "Case law off" ⇒ only statutes allowed ⇒ the judgment is dropped (blocked).
    r = search_knowledge("liberty", _StubKB([_art, _judg]),
                         allowed_instrument_types=["article", "act", "regulation"])
    cits = {p["citation"] for p in r["data"]["passages"]}
    c.ok(r["ok"] and cits == {"Constitution Art.21"} and r["data"]["blocked"] == 1,
         "search_knowledge: statutes-only allow-list hides the judgment (chip gate)")
    # "Statutes off" ⇒ only caselaw allowed ⇒ the Article is dropped.
    r = search_knowledge("liberty", _StubKB([_art, _judg]),
                         allowed_instrument_types=["judgment"])
    cits = {p["citation"] for p in r["data"]["passages"]}
    c.ok(r["ok"] and cits == {"Maneka Gandhi (1978)"},
         "search_knowledge: caselaw-only allow-list hides the statute Article")
    # A model-requested instrument_type OUTSIDE the allow-list ⇒ 0 results, flagged
    # (the agent cannot bypass a disabled source by naming it).
    r = search_knowledge("liberty", _StubKB([_art, _judg]),
                         instrument_type="judgment",
                         allowed_instrument_types=["article", "act"])
    c.ok(r["ok"] and r["data"]["passages"] == []
         and r["data"].get("blocked_by_source_filter"),
         "search_knowledge: requesting a disabled instrument_type → 0 results (not bypassable)")

    print("\n── S-D: KB retrieval quality grade + section existence check ────")
    # S-D.1: strong KB match carries grade=strong in the envelope.
    strong_kb_doc = _Doc(
        "Art.21 — No person shall be deprived of his life or personal liberty…",
        {"citation": "Constitution Art.21", "source_key": "constitution_of_india",
         "instrument_type": "article", "section_or_article_id": "21",
         "jurisdiction": "IN", "as_of_date": "2025-01-01",
         "enacted_date": "1950-01-26", "chunk_id": "kb-strong",
         "score": 0.88},   # high score — stands out from neighbours
    )
    low_score_doc = _Doc(
        "Art.22 — protection against arbitrary arrest",
        {"citation": "Constitution Art.22", "source_key": "constitution_of_india",
         "instrument_type": "article", "section_or_article_id": "22",
         "jurisdiction": "IN", "as_of_date": "2025-01-01",
         "enacted_date": "1950-01-26", "chunk_id": "kb-low",
         "score": 0.40},
    )

    class _ScoredKB:
        def __init__(self, docs): self._docs = docs
        def retrieve(self, query, **kw): return list(self._docs)

    r = search_knowledge("right to life", _ScoredKB([strong_kb_doc, low_score_doc]))
    c.ok(is_envelope(r) and r["ok"], "S-D: strong KB hit → ok envelope")
    c.ok((r["data"] or {}).get("kb_retrieval_quality", {}).get("grade") == "strong",
         "S-D: high-score KB hit with gap → grade=strong in envelope")
    c.ok("repair" not in ((r["data"] or {}).get("kb_retrieval_quality") or {}),
         "S-D: strong grade → no repair key (no false alarm)")

    # S-D.2: weak KB match (low scores) → grade=weak + repair message.
    weak_doc = _Doc(
        "Some tangentially related text.",
        {"citation": "X s.1", "source_key": "x", "instrument_type": "act",
         "section_or_article_id": "1", "jurisdiction": "IN",
         "as_of_date": "2025-01-01", "enacted_date": "2000-01-01",
         "chunk_id": "kb-weak", "score": 0.28},
    )
    r_weak = search_knowledge("director duties Companies Act", _ScoredKB([weak_doc]))
    c.ok(is_envelope(r_weak) and r_weak["ok"], "S-D: weak KB hit → still ok envelope (graceful)")
    c.ok((r_weak["data"] or {}).get("kb_retrieval_quality", {}).get("grade") == "weak",
         "S-D: low-score KB result → grade=weak signals uncertain authority")
    c.ok("repair" in ((r_weak["data"] or {}).get("kb_retrieval_quality") or {}),
         "S-D: weak KB grade → repair guidance present in envelope")
    c.ok("[weak]" in (r_weak.get("summary") or ""),
         "S-D: weak grade reflected in summary line")

    # S-D.3: section_verified=True when the cited §/article id is in the snippet.
    r_sv = search_knowledge("right to life", _ScoredKB([strong_kb_doc]))
    passage = (r_sv["data"] or {}).get("passages", [{}])[0]
    c.ok(passage.get("section_verified") is True,
         "S-D: section_or_article_id '21' present in citation 'Art.21' → section_verified=True")

    # S-D.4: section_verified=False when the cited id is NOT in the snippet/citation.
    mismatch_doc = _Doc(
        "This section deals with something else entirely.",
        {"citation": "Constitution Art.99",   # citation says Art.99
         "source_key": "constitution_of_india",
         "instrument_type": "article",
         "section_or_article_id": "21",       # but id says 21 — mismatch
         "jurisdiction": "IN", "as_of_date": "2025-01-01",
         "enacted_date": "1950-01-26", "chunk_id": "kb-mismatch", "score": 0.80},
    )
    r_mm = search_knowledge("right to life", _ScoredKB([mismatch_doc]))
    mm_passage = (r_mm["data"] or {}).get("passages", [{}])[0]
    c.ok(mm_passage.get("section_verified") is False,
         "S-D: section id '21' absent from citation 'Art.99' and snippet → section_verified=False")

    # S-D.5: no section_or_article_id → section_verified=True (can't verify, don't block).
    no_id_doc = _Doc(
        "General commentary on the Act.",
        {"citation": "Companies Act 2013", "source_key": "companies_act_2013",
         "instrument_type": "act", "jurisdiction": "IN",
         "as_of_date": "2025-01-01", "enacted_date": "2013-09-12",
         "chunk_id": "kb-noid", "score": 0.75},
    )
    r_noid = search_knowledge("company secretary", _ScoredKB([no_id_doc]))
    noid_passage = (r_noid["data"] or {}).get("passages", [{}])[0]
    c.ok(noid_passage.get("section_verified") is True,
         "S-D: no section_or_article_id → section_verified=True (no false block)")

    # S-D.6: empty KB result → grade=empty.
    class _EmptyKB:
        def retrieve(self, *a, **k): return []
    r_empty = search_knowledge("nonexistent provision", _EmptyKB())
    c.ok(r_empty["ok"] and
         (r_empty["data"] or {}).get("kb_retrieval_quality", {}).get("grade") == "empty",
         "S-D: zero KB chunks → grade=empty (still ok envelope)")
    c.ok("repair" in ((r_empty["data"] or {}).get("kb_retrieval_quality") or {}),
         "S-D: empty KB grade → repair guidance present")

    print("\n── T6: every error names its repair (argument audit) ───────────")
    # T6 rule: every error_result must carry a message that names the EXACT repair the
    # model should take — not a bare "requires X". Validated per tool/site.

    # compute: bad op → names the valid ops list.
    r = compute({"op": "not_an_op", "row": {"label": "x"}}, amzn)
    c.ok(not r["ok"] and "Valid ops:" in (r.get("error") or ""),
         "T6 compute: invalid op error names the valid op list")
    # compute: non-dict spec → names the type received + valid ops.
    r = compute("string", amzn)
    c.ok(not r["ok"] and "Valid ops:" in (r.get("error") or ""),
         "T6 compute: non-dict spec error names valid ops (agent knows what to pass)")
    # compute: kernel error with candidates → candidates propagated in data, named in error.
    # Use a bad label that makes the kernel scan candidates.
    r_fail = compute(
        {"op": "value", "row": {"label": "ZZZNOMATCH"}, "period": "2022"}, amzn
    )
    c.ok(not r_fail["ok"] and "error" in r_fail,
         "T6 compute: kernel miss → error envelope (no raise)")
    # If the kernel returned candidates, the error message must name them.
    if (r_fail.get("data") or {}).get("candidates"):
        c.ok("Candidates" in (r_fail.get("error") or ""),
             "T6 compute: kernel miss with candidates → candidates named in error message")
    else:
        c.ok(True, "T6 compute: kernel miss with no candidates → bare error (acceptable)")

    # table_lookup: names which arg is missing and the repair steps.
    r = table_lookup("", "2022", msft)
    c.ok("missing:" in (r.get("error") or "") and "list_metrics" in (r.get("error") or ""),
         "T6 table_lookup: missing metric → error names the missing arg + repair (use list_metrics)")
    r = table_lookup("revenue", "", msft)
    c.ok("missing:" in (r.get("error") or "") and "read_document" in (r.get("error") or ""),
         "T6 table_lookup: missing period → error names the missing arg + repair (use read_document)")

    # list_metrics: no grids → tells model to call read_document first.
    r = list_metrics("amzn.pdf", [])
    c.ok("read_document" in (r.get("error") or ""),
         "T6 list_metrics: no grids → error tells model to call read_document first")
    # list_metrics: no doc_id → lists loaded docs.
    r = list_metrics("", amzn)
    c.ok((r.get("error") or "").startswith("list_metrics requires a 'doc_id'") and
         "amzn" in (r.get("error") or "").lower(),
         "T6 list_metrics: empty doc_id → error lists loaded docs so model can pick one")

    # search_vault: empty query → names what to pass.
    r = search_vault("", _StubRM())
    c.ok("specific phrase" in (r.get("error") or "") or "non-empty" in (r.get("error") or ""),
         "T6 search_vault: empty query → error gives example of what to pass")
    # search_vault: no manager → distinguishes internal from model error.
    r = search_vault("x", None)
    c.ok("routing bug" in (r.get("error") or "") or "retrieval_manager" in (r.get("error") or ""),
         "T6 search_vault: no manager → error flags it as routing bug (not a model mistake)")

    # search_knowledge: empty query → example format.
    r = search_knowledge("", _StubKB())
    c.ok("non-empty" in (r.get("error") or ""),
         "T6 search_knowledge: empty query → error names what to pass")
    # search_knowledge: no manager → distinguishes internal routing from model error.
    r = search_knowledge("x", None)
    c.ok("routing bug" in (r.get("error") or "") or "USE_KNOWLEDGE" in (r.get("error") or ""),
         "T6 search_knowledge: no manager → error flags it as routing bug")

    # survey_collection: empty vault scope → repair hint (check vault docs).
    r_sv = survey_collection("q", _SurveyRM(), object(), filenames=[])
    c.ok("search_vault" in (r_sv.get("error") or "") or "documents" in (r_sv.get("error") or ""),
         "T6 survey_collection: empty scope → error names repair (check vault contains docs)")

    print("\n── schemas (registry-ready, A2) ─────────────────────────────────")
    c.ok(set(SCHEMAS) == {"search_vault", "read_document", "list_metrics",
                          "table_lookup", "compute", "survey_collection", "search_knowledge",
                          # DOCUMENT_HARNESS Phase 1: the document-filesystem tools.
                          "list_documents", "search_text", "read_section"},
         "SCHEMAS: all tools registered (incl. harness ls/grep/sed)")
    c.ok(all("name" in s and "input_schema" in s for s in SCHEMAS.values()),
         "SCHEMAS: each has name + input_schema")
    # G5: survey_collection is deep-mode only — it must NOT leak into standard mode.
    from src.components.agent_core.registry import REGISTRY as _REG
    c.ok("survey_collection" in _REG.names("deep")
         and "survey_collection" not in _REG.names("standard"),
         "registry: survey_collection in deep mode only (breadth tool, not standard)")
    # G8: search_knowledge is OFF by default (byte-identical) — offered only when the run
    # opts in via include_knowledge (loop passes scope.kb_retrieval_manager is not None).
    _std_schema_names = {s["name"] for s in _REG.schemas("standard")}
    c.ok("search_knowledge" not in _REG.names("standard")
         and "search_knowledge" not in _std_schema_names,
         "registry: search_knowledge NOT offered by default (USE_KNOWLEDGE off = byte-identical)")
    c.ok("search_knowledge" in _REG.names("standard", include_knowledge=True)
         and "search_knowledge" in _REG.names("grid", include_knowledge=True)
         and "search_knowledge" not in _REG.names("fast", include_knowledge=True),
         "registry: search_knowledge offered in standard/grid (not fast) when knowledge on")

    print("\n" + "=" * 64)
    print(f"  PASS: {c.passed}   FAIL: {c.failed}")
    print("=" * 64)
    if c.failed == 0:
        print("  ✓ A1 tool-adapter gate GREEN (envelope · abstain · never-raise)")
        return 0
    print("  ✗ A1 gate FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
