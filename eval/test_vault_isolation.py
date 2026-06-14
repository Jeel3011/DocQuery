"""Vault-isolation gate — G3 Step A/B (the silent-wrong gate).

The failure class G3 kills: a query scoped to **Vault A** retrieves a chunk from
**Vault B** (or an unfiltered query pulls a doc the user filtered out). G3 makes vault
isolation a PROPERTY OF THE DATA — chunks carry an ingest-stamped `doc_id`, retrieval
filters `doc_id $in [vault's docs]`, and any metadata_filter NARROWS that scope, never
replaces it.

This gate is fully OFFLINE — no Pinecone, no DB, no API, no model. A tiny in-memory
fake vector store applies Pinecone-style metadata filters (`$in`, scalar equality) over
a 2-vault fixture corpus, so we can assert:

  1. `_build_filter` emits `doc_id $in` for a multi-doc vault scope (the data axis).
  2. A query scoped to Vault A returns ONLY Vault-A chunks — never a Vault-B chunk
     (the cross-vault-leak gate, asserted with a 2-vault fixture).
  3. `search_vault` end-to-end (text + table) over the fake manager isolates the vault.
  4. A metadata_filter NARROWS within the vault (doc_type / fiscal_year) and is
     CONJUNCTIVE — it can never widen scope or overwrite a scope key (§5 risk #4).
  5. The legacy filename `$in` fallback still scopes when a vault has no doc_ids
     (un-stamped vectors degrade, never silently return everything).

Run: python -u eval/test_vault_isolation.py
"""

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.components.retrieval import RetrievalManager
from src.components.agent_core.tools import search_vault


# ── tiny checker (mirrors test_tools.py style) ──────────────────────────────────
class Checks:
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


# ── a 2-vault fixture corpus ────────────────────────────────────────────────────
# Vault A = {docA1 (FY2022 filing), docA2 (FY2023 contract)}
# Vault B = {docB1 (FY2023 filing)}  — a DIFFERENT vault, must never leak into A.
CORPUS = [
    # (doc_id, filename, chunk_type, doc_type, fiscal_year, text)
    ("docA1", "amzn-2022.pdf", "text",  "financial_filing", 2022, "amazon net sales 2022"),
    ("docA1", "amzn-2022.pdf", "table", "financial_filing", 2022, "amzn income statement"),
    ("docA2", "nda-acme.pdf",  "text",  "legal_contract",   2023, "governing law new york"),
    ("docB1", "msft-2023.pdf", "text",  "financial_filing", 2023, "microsoft net sales 2023"),
    ("docB1", "msft-2023.pdf", "table", "financial_filing", 2023, "msft income statement"),
]

VAULT_A = ["docA1", "docA2"]
VAULT_B = ["docB1"]


class _Doc:
    def __init__(self, text, md):
        self.page_content = text
        self.metadata = md


def _matches(md: dict, f: dict | None) -> bool:
    """Apply a Pinecone-style metadata filter (scalar eq + {'$in': [...]})."""
    if not f:
        return True
    for key, cond in f.items():
        val = md.get(key)
        if isinstance(cond, dict) and "$in" in cond:
            if val not in cond["$in"]:
                return False
        else:
            if val != cond:
                return False
    return True


class FakeManager:
    """A RetrievalManager-shaped fake that honors the real `_build_filter` output.

    It reuses RetrievalManager._build_filter (the code under test) to turn scope args
    into a Pinecone filter dict, then applies that filter over the in-memory corpus —
    so the isolation assertions exercise the ACTUAL filter-building logic.
    """

    def __init__(self):
        self.last_filter = None

    def _rows(self):
        for doc_id, fn, ct, dt, fy, text in CORPUS:
            yield _Doc(text, {
                "doc_id": doc_id, "filename": fn, "chunk_type": ct,
                "doc_type": dt, "fiscal_year": fy, "page_number": 1,
                "chunk_id": f"{doc_id}-{ct}-{hash(text) & 0xffff}",
            })

    def retrieve(self, query, *, doc_ids=None, filename_filters=None,
                 filename_filter=None, metadata_filter=None, top_k=8,
                 apply_threshold=False, use_reranker=True):
        f = RetrievalManager._build_filter(
            filename_filter=filename_filter, filename_filters=filename_filters,
            doc_ids=doc_ids, metadata_filter=metadata_filter,
        )
        self.last_filter = f
        return [d for d in self._rows()
                if d.metadata["chunk_type"] == "text" and _matches(d.metadata, f)][:top_k]

    def retrieve_table_chunks(self, query, *, doc_ids=None, filename_filters=None,
                              collection_id=None, doc_id=None, metadata_filter=None, k=8):
        # Mirror retrieve_table_chunks' own filter logic (chunk_type pinned to table).
        f = {"chunk_type": "table"}
        if doc_id:
            f["doc_id"] = doc_id
        elif doc_ids:
            f["doc_id"] = {"$in": doc_ids}
        elif filename_filters:
            f["filename"] = {"$in": filename_filters}
        if metadata_filter:
            for mk, mv in metadata_filter.items():
                if mk not in ("doc_id", "collection_id", "filename", "chunk_type"):
                    f[mk] = mv
        self.last_filter = f
        return [d for d in self._rows() if _matches(d.metadata, f)][:k]


def main():
    c = Checks()

    print("\n── 1. _build_filter: doc_id is the data axis ────────────────────")
    f = RetrievalManager._build_filter(doc_ids=VAULT_A)
    c.ok(f == {"doc_id": {"$in": ["docA1", "docA2"]}},
         "_build_filter(doc_ids) → {doc_id: {$in: [...]}} (vault scope as data)")
    # doc_id wins over the legacy filename fallback (priority cascade).
    f2 = RetrievalManager._build_filter(doc_ids=VAULT_A, filename_filters=["x.pdf"])
    c.ok("doc_id" in f2 and "filename" not in f2,
         "_build_filter: doc_id scope wins over filename fallback")

    print("\n── 2. cross-vault leak gate (the silent-wrong class) ────────────")
    mgr = FakeManager()
    # Vault-A scoped TEXT query must never surface docB1 (msft, Vault B).
    res = mgr.retrieve("net sales", doc_ids=VAULT_A)
    got_docs = {d.metadata["doc_id"] for d in res}
    c.ok(got_docs and got_docs <= set(VAULT_A),
         f"Vault-A query returns only Vault-A docs (got {sorted(got_docs)})")
    c.ok("docB1" not in got_docs,
         "Vault-A query NEVER returns the Vault-B chunk (no cross-vault leak)")
    # Symmetric: Vault-B query never returns Vault-A.
    resB = mgr.retrieve("net sales", doc_ids=VAULT_B)
    gotB = {d.metadata["doc_id"] for d in resB}
    c.ok(gotB == {"docB1"}, "Vault-B query returns only Vault-B (symmetric isolation)")

    print("\n── 3. search_vault end-to-end isolates the vault ────────────────")
    mgr = FakeManager()
    r = search_vault("net sales", mgr, scope={"doc_ids": VAULT_A}, k=8, kind="both")
    chunks = r["data"]["chunks"] if r.get("ok") else []
    leaked = [ch for ch in chunks if ch.get("source") == "msft-2023.pdf"
              or ch.get("doc") == "msft-2023.pdf"]
    c.ok(r["ok"] and not leaked,
         "search_vault(both, Vault A): no Vault-B (msft) chunk in results")
    c.ok(mgr.last_filter and mgr.last_filter.get("doc_id") == {"$in": VAULT_A},
         "search_vault: table search filters by doc_id $in (not collection_id)")

    print("\n── 4. metadata_filter NARROWS, never replaces, the scope ────────")
    mgr = FakeManager()
    # Within Vault A, filter to legal_contract → only docA2 (the NDA), not docA1.
    res = mgr.retrieve("law", doc_ids=VAULT_A, metadata_filter={"doc_type": "legal_contract"})
    got = {d.metadata["doc_id"] for d in res}
    c.ok(got == {"docA2"},
         "doc_type=legal_contract narrows Vault A to the contract only (docA2)")
    c.ok(mgr.last_filter.get("doc_id") == {"$in": VAULT_A},
         "metadata_filter is CONJUNCTIVE — vault scope (doc_id $in) still present")
    # A metadata_filter trying to OVERWRITE a scope key is dropped (no scope widening).
    f = RetrievalManager._build_filter(
        doc_ids=VAULT_A, metadata_filter={"doc_id": "docB1", "fiscal_year": 2023},
    )
    c.ok(f.get("doc_id") == {"$in": VAULT_A} and f.get("fiscal_year") == 2023,
         "metadata_filter cannot overwrite the doc_id scope key (no cross-vault widening)")
    # fiscal_year narrowing within the vault.
    res = mgr.retrieve("sales", doc_ids=VAULT_A, metadata_filter={"fiscal_year": 2022})
    got = {d.metadata["doc_id"] for d in res}
    c.ok(got == {"docA1"}, "fiscal_year=2022 narrows Vault A to the FY2022 filing (docA1)")

    print("\n── 5. legacy filename fallback still scopes (un-stamped vectors) ─")
    mgr = FakeManager()
    res = mgr.retrieve("net sales", filename_filters=["amzn-2022.pdf"])
    got = {d.metadata["doc_id"] for d in res}
    c.ok(got == {"docA1"},
         "no doc_ids → filename $in still scopes (degrades, not return-everything)")

    print("\n" + "=" * 64)
    print(f"  PASS: {c.passed}   FAIL: {c.failed}")
    print("=" * 64)
    if c.failed == 0:
        print("  ✓ G3 Step A/B vault-isolation gate GREEN (no cross-vault leak)")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
