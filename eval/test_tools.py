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
    compute, read_document, search_vault, table_lookup, SCHEMAS,
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
    # Offline path: pass pre-built grids (no db_client).
    r = read_document("msft-fixture", grids=msft)
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
        def retrieve(self, query, **kw):
            return [_Doc("net sales rose", {"filename": "amzn.pdf", "page_number": 41,
                                            "chunk_id": "c1", "score": 0.9})]
        def retrieve_table_chunks(self, query, **kw):
            return [_Doc("table", {"filename": "amzn.pdf", "page_number": 42, "chunk_id": "t1"})]

    r = search_vault("AWS net sales", _StubRM(), scope={"collection_id": "x"}, k=4, kind="both")
    c.ok(is_envelope(r) and r["ok"], "search_vault: stub manager → ok envelope")
    c.ok(len(r["provenance"]) >= 1 and r["provenance"][0]["kind"] == "span",
         "search_vault: provenance is chunk spans")
    # No manager → error, not raise.
    r = search_vault("x", None)
    c.ok(is_envelope(r) and not r["ok"] and "error" in r, "search_vault: no manager → error envelope")
    # A manager whose methods raise → tool still returns an envelope (never raises).
    class _BoomRM:
        def retrieve(self, *a, **k): raise RuntimeError("boom")
        def retrieve_table_chunks(self, *a, **k): raise RuntimeError("boom")
    r = search_vault("x", _BoomRM(), kind="both")
    c.ok(is_envelope(r), "search_vault: raising manager → envelope (no raise escapes)")

    print("\n── schemas (registry-ready, A2) ─────────────────────────────────")
    c.ok(set(SCHEMAS) == {"search_vault", "read_document", "table_lookup", "compute"},
         "SCHEMAS: all four tools registered")
    c.ok(all("name" in s and "input_schema" in s for s in SCHEMAS.values()),
         "SCHEMAS: each has name + input_schema")

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
