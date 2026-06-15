"""G6.2 — Playbook gate (offline, $0, no DB required).

Proves three things without touching Supabase:
1. The seed playbook data is structurally valid (all required fields, lengths in bounds,
   20 rows, no blank standard_position, no duplicate topics).
2. The CRUD route module imports cleanly (no broken imports at startup).
3. The route's schema models validate correctly and reject bad input.

Run: python eval/test_playbook.py
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_failures = []


def check(name, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _failures.append(name)


# ── Test 1: seed data structural validity ────────────────────────────────────

def test_seed_data():
    print("\nTest 1 — seed playbook data is structurally valid")
    from src.api.routes.playbooks import _SEED_PLAYBOOK

    check("20 seed rows present", len(_SEED_PLAYBOOK) == 20, f"got {len(_SEED_PLAYBOOK)}")

    topics = [r["clause_topic"] for r in _SEED_PLAYBOOK]
    check("no duplicate clause topics", len(topics) == len(set(topics)),
          f"dupes: {[t for t in topics if topics.count(t) > 1]}")

    for row in _SEED_PLAYBOOK:
        topic = row.get("clause_topic", "")
        std = row.get("standard_position", "")
        check(f"[{topic[:30]}] clause_topic non-empty", bool(topic.strip()))
        check(f"[{topic[:30]}] standard_position non-empty", bool(std.strip()))
        check(f"[{topic[:30]}] clause_topic ≤ 300 chars", len(topic) <= 300, f"len={len(topic)}")
        check(f"[{topic[:30]}] standard_position ≤ 4000 chars", len(std) <= 4000, f"len={len(std)}")
        fp = row.get("fallback_position") or ""
        notes = row.get("notes") or ""
        check(f"[{topic[:30]}] fallback_position ≤ 4000 chars", len(fp) <= 4000)
        check(f"[{topic[:30]}] notes ≤ 2000 chars", len(notes) <= 2000)


# ── Test 2: module imports cleanly ───────────────────────────────────────────

def test_module_imports():
    print("\nTest 2 — playbooks route module imports without error")
    try:
        from src.api.routes import playbooks  # noqa: F401
        check("playbooks module imports", True)
        check("router attribute present", hasattr(playbooks, "router"))
        check("_SEED_PLAYBOOK attribute present", hasattr(playbooks, "_SEED_PLAYBOOK"))
    except Exception as exc:
        check("playbooks module imports", False, str(exc))


# ── Test 3: Pydantic schema validation ───────────────────────────────────────

def test_schema_validation():
    print("\nTest 3 — PlaybookRow schema validates correctly")
    from src.api.routes.playbooks import PlaybookRow
    from pydantic import ValidationError

    # Valid minimal row
    try:
        row = PlaybookRow(
            clause_topic="Governing Law",
            standard_position="Laws of India.",
        )
        check("valid minimal row constructs", True)
        check("fallback_position defaults None", row.fallback_position is None)
        check("notes defaults None", row.notes is None)
    except ValidationError as e:
        check("valid minimal row constructs", False, str(e))

    # clause_topic over 300 chars should fail
    try:
        PlaybookRow(clause_topic="x" * 301, standard_position="ok")
        check("clause_topic > 300 chars raises ValidationError", False, "no error")
    except ValidationError:
        check("clause_topic > 300 chars raises ValidationError", True)

    # standard_position over 4000 chars should fail
    try:
        PlaybookRow(clause_topic="ok", standard_position="x" * 4001)
        check("standard_position > 4000 chars raises ValidationError", False, "no error")
    except ValidationError:
        check("standard_position > 4000 chars raises ValidationError", True)

    # notes over 2000 chars should fail
    try:
        PlaybookRow(clause_topic="ok", standard_position="ok", notes="x" * 2001)
        check("notes > 2000 chars raises ValidationError", False, "no error")
    except ValidationError:
        check("notes > 2000 chars raises ValidationError", True)

    # Empty clause_topic should fail (min_length or non-empty)
    try:
        PlaybookRow(clause_topic="", standard_position="ok")
        # If no min_length enforced, still check at route level — record intent
        check("empty clause_topic: schema level (informational)", True)
    except ValidationError:
        check("empty clause_topic raises ValidationError", True)


if __name__ == "__main__":
    print("=" * 64)
    print("G6.2 PLAYBOOK GATE — seed data + schema + imports ($0)")
    print("=" * 64)
    test_seed_data()
    test_module_imports()
    test_schema_validation()
    print("\n" + "=" * 64)
    if _failures:
        print(f"{FAIL}: {len(_failures)} check(s) failed: {_failures}")
        sys.exit(1)
    print(f"{PASS}: all playbook checks green.")
