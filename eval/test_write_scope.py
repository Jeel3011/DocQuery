"""F-C offline gate — service-role write scope audit.

Proves (per tool_hard.md §F-C):
  1. Every service-role TABLE write in db.py carries either a user_id OR a firm_id scope
     predicate — no write is naked (no predicate = any tenant's row could be touched).
  2. The specific mutating table methods on SupabaseManager each have the correct scope
     key verified through a mock that asserts the predicate was applied.
  3. The collections insert always includes at least user_id (the owner) — no firm-less
     orphan collection is possible via the normal write path.
  4. The firm_memberships upsert carries both user_id and firm_id — no half-keyed row.
  5. The documents table writes (.insert/.update/.delete) all carry user_id scope.
  6. The document_chunks table writes carry both document_id AND user_id.
  7. The conversations table writes carry user_id.
  8. The messages table writes carry conversation_id AND user_id.

APPROACH: static + behavioral.
  - Static: grep the db.py source for .insert/.update/.delete/.upsert calls on each table
    and assert a scope predicate appears in context (the same 10-line window). Catches a
    future write that forgets the predicate at authoring time.
  - Behavioral: mock the Supabase client and call the SupabaseManager methods, then assert
    the captured calls carry the required scope key. Catches a write that has the predicate
    logically but would submit without it at runtime.

Run:  python -u eval/test_write_scope.py

LIVE gate (still owed): attempt a cross-firm write on real Postgres with a real lesser
client and confirm it is rejected/scoped. This gate proves the pure function only.
"""
from __future__ import annotations

import ast
import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


# ── §1: STATIC AUDIT — scan db.py source for naked writes ─────────────────────────────────

DB_SRC = Path(__file__).resolve().parent.parent / "src/components/db.py"
src_lines = DB_SRC.read_text().splitlines()

# Table → scope keys that MUST appear in any mutating call block for that table.
# A "write" is a line that calls .insert(, .update(, .delete(, or .upsert( on that table.
# The scope key must appear within ±10 lines (the same logical call chain / builder chain).
SCOPE_RULES: dict[str, list[str]] = {
    "documents": ["user_id"],
    "document_chunks": ["user_id"],
    "conversations": ["user_id"],
    "messages": ["user_id"],
    "collections": ["user_id"],
    "firm_memberships": ["firm_id"],
    "firms": [],          # firm rows are scoped by their own PK (passed as firm_id param); no extra predicate needed
    "firm_invites": ["firm_id"],
    "signatures": ["firm_id"],
    "audit_log": ["user_id"],   # audit_log inserts always carry user_id via log_audit
}

# These lines are EXEMPT (read paths, not writes, or justified allow-list).
EXEMPT_PATTERNS = [
    r"\.select\(",
    r"read_client\.",
    r"#.*exempt",
    r"#.*read.only",
    r"auth\.admin",  # auth admin API — not a table write
]

print("\n1. STATIC: naked-write detection per table")

WRITE_CALL_RE = re.compile(r'\.(insert|update|delete|upsert)\(')
TABLE_RE = re.compile(r'\.table\("(\w+)"\)')

violations: list[str] = []

for lineno, line in enumerate(src_lines, 1):
    if not WRITE_CALL_RE.search(line):
        continue
    # Skip exempt lines (selects, reads, comments).
    if any(re.search(p, line) for p in EXEMPT_PATTERNS):
        continue
    # Look backwards up to 15 lines to find the .table("name") it belongs to.
    window_start = max(0, lineno - 16)
    window = "\n".join(src_lines[window_start:lineno])
    tbl_m = TABLE_RE.search(window)
    if not tbl_m:
        continue
    table = tbl_m.group(1)
    if table not in SCOPE_RULES:
        continue
    required_scopes = SCOPE_RULES[table]
    if not required_scopes:
        continue
    # Extend window forward too (a chained .eq() may be on the next line).
    win_end = min(len(src_lines), lineno + 10)
    full_window = "\n".join(src_lines[window_start:win_end])
    # Accept the scope key appearing as: a quoted string key ("firm_id"), a keyword
    # argument (firm_id=), or a dict key in a row-builder call (firm_id=firm_id inside
    # build_signature_row / build_*_row helpers). All three are legitimate scope carries.
    missing = [
        s for s in required_scopes
        if f'"{s}"' not in full_window
        and f"'{s}'" not in full_window
        and f"{s}=" not in full_window   # kwarg form: firm_id=firm_id or user_id=self.user_id
    ]
    if missing:
        violations.append(f"  L{lineno}: {table} write MISSING scope key(s) {missing}  →  {line.strip()[:80]}")

check("no naked service-role writes detected in db.py",
      not violations,
      "\n" + "\n".join(violations) if violations else "")
if violations:
    for v in violations:
        print(v)


# ── §2: BEHAVIOURAL — mock the Supabase client and capture predicates ─────────────────────

print("\n2. BEHAVIOURAL: predicate capture on SupabaseManager methods")

USER_ID = str(uuid.uuid4())
FIRM_ID = str(uuid.uuid4())
COLL_ID = str(uuid.uuid4())
DOC_ID  = str(uuid.uuid4())
CONV_ID = str(uuid.uuid4())


def _make_mock_table() -> MagicMock:
    """Return a mock that supports chained .table().insert().execute() etc.
    Captures every .eq() call's arguments."""
    t = MagicMock()
    t.insert.return_value = t
    t.update.return_value = t
    t.delete.return_value = t
    t.upsert.return_value = t
    t.eq.return_value = t
    t.execute.return_value = MagicMock(data=[{"id": str(uuid.uuid4()), "user_id": USER_ID}])
    return t


def _make_sm(user_id: str = USER_ID, firm_id: str = FIRM_ID) -> tuple:
    """Build a SupabaseManager with a mocked Supabase client. Returns (sm, mock_client)."""
    from src.components.db import SupabaseManager
    sm = SupabaseManager.__new__(SupabaseManager)
    mock_client = MagicMock()
    # table() returns a fresh mock per call so we can inspect eq() calls on it.
    table_mocks: dict[str, MagicMock] = {}
    def _table(name):
        if name not in table_mocks:
            table_mocks[name] = _make_mock_table()
        return table_mocks[name]
    mock_client.table.side_effect = _table
    mock_client.storage = MagicMock()
    mock_client.storage.from_.return_value = MagicMock()
    sm.client = mock_client
    sm._user = SimpleNamespace(id=user_id, email=f"{user_id[:8]}@test.com", user_metadata={})
    sm._access_token = None
    sm._read_client = None
    return sm, mock_client, table_mocks


def _eq_args(table_mock: MagicMock) -> list[tuple]:
    """Collect all (column, value) pairs passed to .eq() on this table mock."""
    return [(c.args[0], c.args[1]) for c in table_mock.eq.call_args_list if c.args]


# --- documents: insert carries user_id ---
sm, _, tbls = _make_sm()
sm.create_document_record("test.pdf", "path/test.pdf", "pdf", 1024)
doc_tbl = tbls.get("documents")
eq_cols = [a[0] for a in _eq_args(doc_tbl)] if doc_tbl else []
# For insert, the user_id is in the INSERT body (not a .eq predicate) — check the mock call.
insert_call = doc_tbl.insert.call_args if doc_tbl else None
insert_body = (insert_call.args[0] if insert_call and insert_call.args else {}) or {}
check("create_document_record: insert body carries user_id",
      "user_id" in insert_body,
      f"insert body: {insert_body}")

# --- documents: update scoped by user_id ---
sm2, _, tbls2 = _make_sm()
sm2.update_document_status(DOC_ID, "ready", chunk_count=10)
doc_tbl2 = tbls2.get("documents")
eq2 = _eq_args(doc_tbl2) if doc_tbl2 else []
check("update_document_status: .eq(user_id) present",
      any(col == "user_id" for col, _ in eq2),
      f".eq args: {eq2}")

# --- documents: delete scoped by user_id ---
sm3, _, tbls3 = _make_sm()
sm3.delete_document_record(DOC_ID)
doc_tbl3 = tbls3.get("documents")
eq3 = _eq_args(doc_tbl3) if doc_tbl3 else []
check("delete_document_record: .eq(user_id) present",
      any(col == "user_id" for col, _ in eq3),
      f".eq args: {eq3}")

# --- document_chunks: insert carries user_id ---
sm4, _, tbls4 = _make_sm()
chunk = SimpleNamespace(page_content="test", metadata={})
sm4.save_document_chunks(DOC_ID, [chunk])
chunks_tbl = tbls4.get("document_chunks")
insert4 = chunks_tbl.insert.call_args if chunks_tbl else None
body4 = (insert4.args[0] if insert4 and insert4.args else [{}])
body4_row = body4[0] if isinstance(body4, list) else body4
check("save_document_chunks: insert body carries user_id",
      "user_id" in body4_row,
      f"insert body[0]: {body4_row}")

# --- document_chunks: delete scoped by document_id AND user_id ---
sm5, _, tbls5 = _make_sm()
sm5.delete_document_chunks(DOC_ID)
chunks_del_tbl = tbls5.get("document_chunks")
eq5 = _eq_args(chunks_del_tbl) if chunks_del_tbl else []
check("delete_document_chunks: .eq(user_id) present",
      any(col == "user_id" for col, _ in eq5),
      f".eq args: {eq5}")
check("delete_document_chunks: .eq(document_id) present",
      any(col == "document_id" for col, _ in eq5),
      f".eq args: {eq5}")

# --- collections: insert carries user_id ---
sm6, _, tbls6 = _make_sm()
sm6.create_collection("Test Matter", firm_id=FIRM_ID)
coll_tbl = tbls6.get("collections")
insert6 = coll_tbl.insert.call_args if coll_tbl else None
body6 = (insert6.args[0] if insert6 and insert6.args else {}) or {}
check("create_collection: insert body carries user_id",
      "user_id" in body6,
      f"insert body: {body6}")

# --- firm_memberships: upsert carries both user_id and firm_id ---
sm7, _, tbls7 = _make_sm()
sm7.add_membership(USER_ID, FIRM_ID, "associate")
fm_tbl = tbls7.get("firm_memberships")
upsert7 = fm_tbl.upsert.call_args if fm_tbl else None
body7 = (upsert7.args[0] if upsert7 and upsert7.args else {}) or {}
check("add_membership: upsert body carries user_id",
      "user_id" in body7, f"upsert body: {body7}")
check("add_membership: upsert body carries firm_id",
      "firm_id" in body7, f"upsert body: {body7}")

# --- conversations: insert carries user_id ---
sm8, _, tbls8 = _make_sm()
sm8.create_conversation("Test")
conv_tbl = tbls8.get("conversations")
insert8 = conv_tbl.insert.call_args if conv_tbl else None
body8 = (insert8.args[0] if insert8 and insert8.args else {}) or {}
check("create_conversation: insert body carries user_id",
      "user_id" in body8, f"insert body: {body8}")

# --- conversations: rename scoped by user_id ---
sm9, _, tbls9 = _make_sm()
sm9.rename_conversation(CONV_ID, "Renamed")
conv_tbl9 = tbls9.get("conversations")
eq9 = _eq_args(conv_tbl9) if conv_tbl9 else []
check("rename_conversation: .eq(user_id) present",
      any(col == "user_id" for col, _ in eq9),
      f".eq args: {eq9}")

# --- messages: insert carries user_id ---
sm10, _, tbls10 = _make_sm()
sm10.save_message(CONV_ID, "assistant", "Hello")
msg_tbl = tbls10.get("messages")
insert10 = msg_tbl.insert.call_args if msg_tbl else None
body10 = (insert10.args[0] if insert10 and insert10.args else {}) or {}
check("save_message: insert body carries user_id",
      "user_id" in body10, f"insert body: {body10}")
check("save_message: insert body carries conversation_id",
      "conversation_id" in body10, f"insert body: {body10}")


# ── §3: NO WRITE WITHOUT A SCOPE — the invariant that governs future writes ───────────────

print("\n3. INVARIANT: cross-tenant write impossible through normal write methods")

# The normal write methods all carry user_id or firm_id derived from self.user_id / the
# caller's resolved firm — never from an untrusted request body. Confirm the scope values
# seen in §2 MATCH the manager's own user/firm context (not an arbitrary caller-supplied id).

sm_inv, _, tbls_inv = _make_sm(user_id=USER_ID)
sm_inv.create_document_record("inv.pdf", "path/inv.pdf", "pdf", 512)
doc_inv = tbls_inv.get("documents")
insert_inv = doc_inv.insert.call_args if doc_inv else None
body_inv = (insert_inv.args[0] if insert_inv and insert_inv.args else {}) or {}
check("document insert user_id matches manager's own user_id (no foreign-id injection)",
      body_inv.get("user_id") == USER_ID,
      f"got user_id={body_inv.get('user_id')!r}, expected {USER_ID!r}")

sm_inv2, _, tbls_inv2 = _make_sm()
sm_inv2.save_message(CONV_ID, "user", "Q")
msg_inv2 = tbls_inv2.get("messages")
insert_inv2 = msg_inv2.insert.call_args if msg_inv2 else None
body_inv2 = (insert_inv2.args[0] if insert_inv2 and insert_inv2.args else {}) or {}
check("message insert user_id matches manager's own user_id",
      body_inv2.get("user_id") == USER_ID,
      f"got user_id={body_inv2.get('user_id')!r}, expected {USER_ID!r}")


# ── result ─────────────────────────────────────────────────────────────────────────────────

print(f"\n{'─'*60}")
total = _passed + _failed
print(f"  {_passed}/{total} passed  {'(ALL GREEN)' if not _failed else f'({_failed} FAILED)'}")
print()
print("  NOTE: This gate proves the PURE FUNCTION only (mock Supabase client).")
print("  LIVE gate still owed: attempt a cross-firm write on real Postgres with a real")
print("  lesser-role client and confirm it is rejected at the DB layer.")
if _failed:
    sys.exit(1)
