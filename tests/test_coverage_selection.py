"""Unit tests for _select_with_coverage — pure function, no network needed."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from src.components.retrieval import RetrievalManager


def _make_docs(specs):
    """specs: list of (filename, content_hash, page_content)"""
    docs = []
    for fname, chash, text in specs:
        docs.append(Document(
            page_content=text,
            metadata={"filename": fname, "content_hash": chash, "chunk_id": f"{fname}::{chash}"},
        ))
    return docs


def _make_mgr():
    """Build a minimal RetrievalManager-like object that exposes only _select_with_coverage."""
    class FakeConfig:
        pass
    mgr = object.__new__(RetrievalManager)
    mgr.config = FakeConfig()
    return mgr


mgr = _make_mgr()

# ── Test 1: all 5 filenames represented in k_final=8 ──────────────────────────
files = ["a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf"]
specs = []
for rank, fname in enumerate(files):
    for i in range(4):
        specs.append((fname, f"{fname}-{i}", f"content about {fname} chunk {i} " + "word " * (rank * 3 + i * 2)))

docs = _make_docs(specs)
result = mgr._select_with_coverage(docs, k_final=8, min_per_file=1, jaccard_thresh=0.9)
assert len(result) == 8, f"Expected 8 docs, got {len(result)}"
covered = {d.metadata["filename"] for d in result}
assert covered == set(files), f"Not all files covered: {covered}"
print("Test 1 passed: 8 docs, all 5 files covered")

# ── Test 2: near-duplicate pair — only one survives ────────────────────────────
near_dup_text = "apple banana cherry date elderberry fig grape honeydew"
dup_specs = [
    ("x.pdf", "x1", near_dup_text),
    ("x.pdf", "x2", near_dup_text + " fig"),  # >90% token overlap with x1
    ("y.pdf", "y1", "completely different content here nothing shared at all really"),
]
dup_docs = _make_docs(dup_specs)
dup_result = mgr._select_with_coverage(dup_docs, k_final=3, min_per_file=1, jaccard_thresh=0.9)
x_docs = [d for d in dup_result if d.metadata["filename"] == "x.pdf"]
assert len(x_docs) == 1, f"Near-duplicate not deduped: got {len(x_docs)} x.pdf docs"
print("Test 2 passed: near-duplicate deduped to 1")

# ── Test 3: empty input returns [] ────────────────────────────────────────────
assert mgr._select_with_coverage([], k_final=5, min_per_file=1, jaccard_thresh=0.9) == []
print("Test 3 passed: empty input returns []")

# ── Test 4: k_final=0 returns [] ─────────────────────────────────────────────
assert mgr._select_with_coverage(docs, k_final=0, min_per_file=1, jaccard_thresh=0.9) == []
print("Test 4 passed: k_final=0 returns []")

print("\nAll tests passed.")
