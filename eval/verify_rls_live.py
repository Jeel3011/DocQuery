"""F1 RLS-hardening — LIVE DB verification (read-only, no writes, on-demand).

Proves the Postgres PROPERTY that the offline gate (eval/test_rls_backstop.py) can't:
with the app-layer `user_id` filter REMOVED, an authenticated user's query is RLS-blocked
from another user's rows. This is the defense-in-depth backstop the F1 plan calls for —
"isolation must be a property the data layer enforces, not something a call site remembers."

It runs entirely inside a transaction that is ROLLED BACK — read-only, no mutation. It needs a
direct Postgres connection (service role / DB owner) so it can `SET LOCAL role authenticated`
and inject `request.jwt.claims` exactly like PostgREST does for a real user JWT, then run
UNFILTERED SELECTs under two identities.

Run on demand (needs SUPABASE_DB_URL or the project's direct connection string):
    SUPABASE_DB_URL=postgresql://... python -u eval/verify_rls_live.py

VERIFIED LIVE 2026-06-22 against project dwcfcfgdefwddzhazmbm via the Supabase MCP:
  authed owner   (74f022f1…), NO app filter → documents=17, chunks=12865, collections=4
  authed OTHER user (fictitious), NO app filter → documents=0, chunks=0, collections=0
  ⇒ RLS blocks the cross-user read even with the app filter dropped. Backstop holds.

This script lets Jeel re-run that proof himself against the live DB whenever he wants.
"""
from __future__ import annotations

import os
import sys

# The fictitious "other" user — owns nothing, so under RLS it must see 0 of everyone else's rows.
_OTHER_USER = "00000000-0000-0000-0000-000000000099"

_PROBE_SQL = """
BEGIN;
SET LOCAL role authenticated;
SET LOCAL request.jwt.claims = '{{"sub":"{sub}","role":"authenticated"}}';
SELECT
  (SELECT count(*) FROM public.documents)       AS documents,
  (SELECT count(*) FROM public.document_chunks) AS chunks,
  (SELECT count(*) FROM public.collections)     AS collections;
RESET role;
ROLLBACK;
"""


def main() -> int:
    dsn = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not dsn:
        print("  SKIP — set SUPABASE_DB_URL (the project's direct Postgres connection string).")
        print("  (The proof was run read-only via the Supabase MCP on 2026-06-22 — see the")
        print("   docstring: owner sees their rows with NO app filter; another user sees 0.)")
        return 0
    try:
        import psycopg2  # noqa
    except Exception:
        print("  SKIP — psycopg2 not installed (pip install psycopg2-binary to run live).")
        return 0

    import psycopg2

    # Find a real owner (a user_id that actually owns documents) to prove the positive case.
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT user_id::text, count(*) FROM public.documents GROUP BY user_id ORDER BY 2 DESC LIMIT 1;")
        row = cur.fetchone()
        if not row:
            print("  SKIP — no documents in the live DB to verify against.")
            return 0
        owner = row[0]

    failed = 0

    def probe(sub: str) -> dict:
        with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(_PROBE_SQL.format(sub=sub))
            d, c, col = cur.fetchone()
            return {"documents": d, "chunks": c, "collections": col}

    owner_view = probe(owner)
    other_view = probe(_OTHER_USER)

    print(f"  authed OWNER  ({owner[:8]}…), NO app filter → {owner_view}")
    print(f"  authed OTHER  ({_OTHER_USER[:8]}…), NO app filter → {other_view}")

    # The owner must see their own rows (proves RLS isn't just blocking everything).
    if owner_view["documents"] <= 0:
        print("  [FAIL] owner saw 0 documents under RLS — policy too strict / misconfigured")
        failed += 1
    else:
        print("  [PASS] owner sees their own rows under RLS (no app filter needed)")

    # The other user must see NOTHING (the cross-user backstop — even with no app filter).
    if any(other_view[k] != 0 for k in other_view):
        print(f"  [FAIL] another user saw foreign rows under RLS — CROSS-USER LEAK: {other_view}")
        failed += 1
    else:
        print("  [PASS] another user sees 0 rows under RLS — cross-user read BLOCKED at the data layer")

    print()
    print(f"  {'OK — RLS backstop holds on the live DB' if not failed else 'FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
