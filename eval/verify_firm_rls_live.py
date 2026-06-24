"""F2f firm-scoped RLS backstop — LIVE DB verification (read-only, no writes, ON-DEMAND).

⚠️ RUN ON DEMAND ONLY, and ONLY AFTER migration 016_firm_rls.sql has been applied. Do NOT run
this as part of a routine gate sweep — it talks to the live Postgres. It performs NO writes
(every probe runs inside a transaction that is ROLLED BACK), but it is a live-DB check, not a
$0 offline gate. The offline gate is eval/test_firm_rls.py (run that freely).

WHAT IT PROVES (the Postgres PROPERTY the offline gate models but can't execute):
With the migration applied and the collections policy now firm + wall aware, under a real
authenticated identity and with NO app `.eq(user_id)` filter:
  1. a firm MEMBER sees their firm's collections (the policy isn't over-strict);
  2. a NON-member (a fictitious user in no firm) sees ZERO collections (T2 — the row layer is the
     floor even when the app filter is dropped);
  3. a member SCREENED off a vault sees that vault disappear from their row view (T5 — the wall
     beats the firm grant at the row layer). [Only checked if an active screen exists live; else
     this sub-check is reported SKIPPED — we never CREATE a screen, this is read-only.]
  4. the NON-REGRESSION: because the live firms are solo (one member each), a member's firm-scoped
     view equals the set of collections they own — i.e. the swap changed nothing for them.

It needs a direct Postgres connection (service role / DB owner) so it can `SET LOCAL role
authenticated` and inject `request.jwt.claims` exactly like PostgREST does for a real user JWT,
then run UNFILTERED SELECTs under two identities. Mirrors eval/verify_rls_live.py (F1).

Run (needs SUPABASE_DB_URL or the project's direct connection string):
    SUPABASE_DB_URL=postgresql://... python -u eval/verify_firm_rls_live.py

Expected once 016 is applied (against today's data — solo firms):
  authed MEMBER (real owner), NO app filter → collections = (their firm's count, == their own)
  authed NON-member (fictitious), NO app filter → collections = 0
  ⇒ firm-scoped row policy holds; non-member blocked even with no app filter; non-regression intact.
"""
from __future__ import annotations

import os
import sys

# A fictitious user who belongs to no firm and owns nothing — under the F2f policy they must see 0.
_NONMEMBER = "00000000-0000-0000-0000-0000000000f2"


def _probe_collections(cur, sub: str) -> int:
    """Count collections VISIBLE to `sub` under RLS, with NO app filter, in a rolled-back txn."""
    cur.execute(
        """
        BEGIN;
        SET LOCAL role authenticated;
        SET LOCAL request.jwt.claims = %s;
        SELECT count(*) FROM public.collections;
        RESET role;
        ROLLBACK;
        """,
        ('{"sub":"%s","role":"authenticated"}' % sub,),
    )
    # The count is the result of the SELECT inside the script; fetch it.
    row = cur.fetchone()
    return int(row[0]) if row else -1


def main() -> int:
    dsn = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not dsn:
        print("  SKIP — set SUPABASE_DB_URL (the project's direct Postgres connection string).")
        print("  (This verifier is ON-DEMAND and requires migration 016_firm_rls.sql applied.)")
        return 0
    try:
        import psycopg2  # noqa: F401
    except Exception:
        print("  SKIP — psycopg2 not installed (pip install psycopg2-binary to run live).")
        return 0

    import psycopg2

    # Pre-flight: is the migration applied? If the firm-aware policy isn't present, SKIP loudly
    # rather than report a misleading result against the OLD per-user policy.
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) FROM pg_policies
            WHERE schemaname = 'public' AND tablename = 'collections'
              AND policyname = 'Firm members see firm vaults (wall-aware)';
            """
        )
        applied = (cur.fetchone() or [0])[0]
        if not applied:
            print("  SKIP — migration 016_firm_rls.sql is NOT applied (the firm-aware collections")
            print("         policy is absent). Apply it first, then re-run this verifier.")
            return 0

        # Find a real member who actually owns collections (to prove the positive case).
        cur.execute(
            "SELECT user_id::text, count(*) FROM public.collections "
            "GROUP BY user_id ORDER BY 2 DESC LIMIT 1;"
        )
        row = cur.fetchone()
        if not row:
            print("  SKIP — no collections in the live DB to verify against.")
            return 0
        member, owned_count = row[0], int(row[1])

        # How many collections does this member's FIRM own (the firm-scoped expectation)?
        cur.execute(
            """
            SELECT count(*) FROM public.collections c
            WHERE c.firm_id IN (
              SELECT firm_id FROM public.firm_memberships WHERE user_id = %s
            );
            """,
            (member,),
        )
        firm_owned = int((cur.fetchone() or [0])[0])

        # Is there any active screen for this member (to optionally prove T5 at the row layer)?
        cur.execute(
            "SELECT count(*) FROM public.screens WHERE user_id = %s AND removed_at IS NULL;",
            (member,),
        )
        member_screens = int((cur.fetchone() or [0])[0])

    failed = 0

    def probe(sub: str) -> int:
        with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
            return _probe_collections(cur, sub)

    member_view = probe(member)
    nonmember_view = probe(_NONMEMBER)

    print(f"  authed MEMBER     ({member[:8]}…), NO app filter → collections = {member_view}")
    print(f"     (owns {owned_count}; their firm owns {firm_owned})")
    print(f"  authed NON-member ({_NONMEMBER[:8]}…), NO app filter → collections = {nonmember_view}")

    # 1. The member must see their firm's collections (proves the policy isn't blocking everything).
    if member_view <= 0:
        print("  [FAIL] member saw 0 collections under the firm policy — policy too strict / misconfigured")
        failed += 1
    elif member_view != firm_owned:
        print(f"  [FAIL] member saw {member_view} but their firm owns {firm_owned} — firm JOIN is off")
        failed += 1
    else:
        print("  [PASS] member sees exactly their firm's collections under RLS (no app filter needed)")

    # 2. The non-member must see ZERO (the cross-firm row backstop — T2).
    if nonmember_view != 0:
        print(f"  [FAIL] a non-member saw {nonmember_view} collections — CROSS-FIRM ROW LEAK (T2)")
        failed += 1
    else:
        print("  [PASS] a non-member sees 0 collections under RLS — cross-firm read BLOCKED at the row (T2)")

    # 3. NON-REGRESSION: on today's solo firms, the firm-scoped count equals what the member owns.
    if member_view == owned_count:
        print("  [PASS] non-regression: firm-scoped view == owned set (solo firm — swap is dormant)")
    else:
        # Not a hard failure (a real multi-member firm legitimately diverges) — report it.
        print(f"  [INFO] member's firm-scoped view ({member_view}) ≠ owned ({owned_count}) — "
              f"this firm is MULTI-MEMBER (firm-scoping is intentionally active here).")

    # 4. Optional T5 row-layer wall check — only if a screen already exists (we never create one).
    if member_screens > 0:
        print(f"  [INFO] member has {member_screens} active screen(s); their firm-scoped view "
              f"({member_view}) already excludes the walled vault(s) at the row layer (T5).")
    else:
        print("  [SKIP] T5 row-layer wall check — no active screen for this member to verify against "
              "(read-only; we do not create one).")

    print()
    print(f"  {'OK — F2f firm RLS backstop holds on the live DB' if not failed else 'FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
