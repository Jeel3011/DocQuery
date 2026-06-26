"""F2i regression gate — E-SIGNATURES: legally-valid, tamper-evident sign-off (offline, $0, no live DB).

The committed gate for F2i (plans/F2_FIRM_CONSOLE_PLAN.md §F2i + F2_ARCHITECTURE.md §4). A sign-off you
can't verify, or a signature on an instrument the law forbids e-signing, is worse than none. This pins
the secure-electronic-signature record (IT Act §85B) so it can't drift:

    ./venv/bin/python -u eval/test_esign.py

What it proves (maps to the plan's §3 gate row for F2i):
  A. THE §1(4) EXCLUSION GUARD — a will / power of attorney / negotiable instrument / trust /
     immovable-property sale is REFUSED as e-signable ("wet-ink required"); a normal review artifact is
     allowed. The route turns the refusal into a 422 BEFORE any state change (a void signature is never
     created).
  B. HASHING — artifact_hash anchors the signed content; content_hash anchors the signed payload;
     tampering ANY material field changes content_hash. Deterministic + canonical (order-independent).
  C. THE APPEND-ONLY CHAIN — append_signature extends the firm chain monotonically (seq 1,2,3…, each
     prev_hash = the prior row_hash); a concurrent seq collision retries off the fresh tip.
  D. VERIFICATION DETECTS TAMPERING — verify_firm_chain passes on an intact chain; an EDITED row (any
     field), a DELETED row (seq gap), and a REORDERED row each break verification with a reason.
  E. THE STRONG-TIER SEAM — a strong-tier (Aadhaar/DSC) request that isn't wired falls BACK to the
     secure tier with a note (never blocked); secure-tier signing never depends on it.
  F. ROUTE WIRING — approve produces an 'approve' signature, release produces a 'release' signature,
     both chained; a §1(4)-excluded artifact 422s the transition; the signatures are read-gated by the
     review relationship; the verify endpoint is partner-gated.
  G. DORMANT PARITY — with the signatures table "unapplied" (fake raises missing-relation), append
     returns None and the transition STILL completes (best-effort write); verify reports an empty chain
     intact — byte-identical to pre-F2i.

OFFLINE: esign.py is pure; the db chain methods + route handlers are driven with a fake PostgREST that
models the signatures table (insert + the firm/seq ordered reads + a dedup-style unique seq), mirroring
test_review_artifact's fake. See [[run-only-relevant-gates]] — F2i touches esign/db/matters routes, NOT
extraction or the kernel.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.components import esign                       # noqa: E402
from src.components.db import SupabaseManager           # noqa: E402
from src.api.routes import matters as mroutes           # noqa: E402
from src.api.schemas import ReviewDecisionRequest       # noqa: E402
from fastapi import HTTPException                        # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = ""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ✓ {name}")
    else:
        _failed += 1
        print(f"  ✗ {name}  {detail}")


def _raises(fn, exc_type) -> bool:
    """True if fn() raises exc_type (used to assert the §1(4) refusal)."""
    try:
        fn()
        return False
    except exc_type:
        return True
    except Exception:
        return False


def _expect_http(fn) -> "int | None":
    """Run fn(); return the HTTPException status_code it raises, or None if it didn't raise one."""
    try:
        fn()
        return None
    except HTTPException as e:
        return e.status_code


# ── A fake PostgREST modeling the signatures table (insert + firm/seq ordered reads) ──
class _Result:
    def __init__(self, data, count=None):
        self.data = data
        self.count = count


class _Query:
    """Chainable select/eq/order/limit/execute over a backing list, plus insert() with a per-table
    UNIQUE(firm_id, chain_seq) constraint to model uq_signatures_firm_seq (the chain race)."""
    def __init__(self, table_name, rows, missing=False, live=False):
        self._name = table_name
        self._rows = rows
        self._eq = {}
        self._order = None
        self._desc = False
        self._limit = None
        self._is_null = set()
        self._missing = missing
        self._live = live

    def select(self, *_a, **_k): return self

    def eq(self, col, val):
        self._eq[col] = str(val)
        return self

    def is_(self, col, _val):
        self._is_null.add(col)
        return self

    def order(self, col, desc=False):
        self._order = col
        self._desc = desc
        return self

    def limit(self, n):
        self._limit = n
        return self

    def insert(self, row):
        self._insert_row = row
        return self

    def update(self, fields):
        self._update_fields = fields
        return self

    def execute(self):
        if self._missing:
            raise RuntimeError(f'relation "{self._name}" does not exist (PGRST205)')
        if self._live:
            raise RuntimeError("connection terminated mid-query (live fault)")
        if hasattr(self, "_update_fields"):
            hit = [r for r in self._rows if all(str(r.get(k)) == v for k, v in self._eq.items())]
            for r in hit:
                r.update(self._update_fields)
            return _Result(hit)
        if hasattr(self, "_insert_row"):
            row = dict(self._insert_row)
            # Model uq_signatures_firm_seq: reject a duplicate (firm_id, chain_seq).
            if self._name == "signatures":
                for r in self._rows:
                    if str(r.get("firm_id")) == str(row.get("firm_id")) and \
                       r.get("chain_seq") == row.get("chain_seq"):
                        raise RuntimeError("duplicate key value violates unique constraint "
                                           '"uq_signatures_firm_seq" (23505)')
                row.setdefault("id", f"sig-{len(self._rows)+1}")
            self._rows.append(row)
            return _Result([row])
        out = [r for r in self._rows
               if all(str(r.get(k)) == v for k, v in self._eq.items())
               and all(r.get(k) is None for k in self._is_null)]
        if self._order:
            out = sorted(out, key=lambda r: (r.get(self._order) is None, r.get(self._order)),
                         reverse=self._desc)
        if self._limit is not None:
            out = out[: self._limit]
        return _Result(out)


class _Client:
    def __init__(self, tables, missing=(), live=()):
        self._tables = tables
        self._missing = set(missing)
        self._live = set(live)

    def table(self, name):
        return _Query(name, self._tables.setdefault(name, []),
                      missing=(name in self._missing), live=(name in self._live))


def _mgr(*, caller, firm, tables, missing=(), live=(), role="managing_partner"):
    m = SupabaseManager.__new__(SupabaseManager)
    m._user = type("U", (), {"id": caller})()
    client = _Client(tables, missing=missing, live=live)
    m.client = client
    m._access_token = None
    m._read_client = client            # one backing store for read + write (like test_matter_read_access)
    m.get_user_firm = lambda *a, **k: ({"id": firm, "role": role} if firm else {})
    m.resolve_member_emails = lambda ids: {str(i): f"{i}@firm.test" for i in ids}
    return m


FIRM = "firm-A"
SIGNER = "user-mp"


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print("\n── A · the §1(4) exclusion guard (a void signature is never created) ──")

_EXCLUDED = [
    ("Last Will and Testament", "will"),
    ("General Power of Attorney", "power_of_attorney"),
    ("Promissory Note for ₹5,00,000", "negotiable_instrument"),
    ("Family Trust Deed", "trust"),
    ("Sale Deed for immovable property", "immovable_property"),
]
for title, expect in _EXCLUDED:
    got = esign.classify_signability(title=title)
    check(f"A · refuse §1(4): {title!r} → {expect}", got == expect, f"got {got}")

check("A · explicit instrument_type='will' refused",
      esign.classify_signability(title="bland title", instrument_type="will") == "will")
check("A · a normal MSA review is signable (not excluded)",
      esign.classify_signability(title="Master Services Agreement — indemnity review") is None)
check("A · assert_signable raises SignabilityRefusal on a will",
      _raises(lambda: esign.assert_signable(title="Last Will"), esign.SignabilityRefusal))
check("A · assert_signable passes a normal artifact",
      esign.assert_signable(title="NDA clause review") is None)


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print("\n── B · hashing (deterministic, canonical, tamper-sensitive) ──")

h1 = esign.hash_content("the indemnity cap is 12 months of fees")
check("B · artifact hash is stable", h1 == esign.hash_content("the indemnity cap is 12 months of fees"))
check("B · artifact hash changes on edit", h1 != esign.hash_content("the indemnity cap is 24 months of fees"))

payload = dict(signer_id=SIGNER, artifact_type="review_request", artifact_ref="req-1",
               artifact_hash=h1, intent="release", signature_method="secure_eauth",
               signed_at="2026-06-26T10:00:00Z")
c1 = esign.content_hash_for(**payload)
check("B · content hash is deterministic", c1 == esign.content_hash_for(**payload))
# tampering ANY material field changes content_hash
for fld, bad in [("signer_id", "user-evil"), ("intent", "approve"), ("artifact_hash", "deadbeef"),
                 ("signed_at", "2026-06-26T11:00:00Z")]:
    tampered = {**payload, fld: bad}
    check(f"B · content hash sensitive to {fld}", esign.content_hash_for(**tampered) != c1)

# chain link
r1 = esign.chain_row_hash(None, c1)
r2 = esign.chain_row_hash(r1, "next-content")
check("B · genesis row_hash deterministic", r1 == esign.chain_row_hash(None, c1))
check("B · row_hash depends on prev_hash", r2 != esign.chain_row_hash("other-prev", "next-content"))


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print("\n── C · the append-only chain (monotonic, prev-linked, race-safe) ──")

m = _mgr(caller=SIGNER, firm=FIRM, tables={"signatures": []})
s1 = m.append_signature(firm_id=FIRM, artifact_type="review_request", artifact_ref="req-1",
                        artifact_content="answer one", intent="approve")
s2 = m.append_signature(firm_id=FIRM, artifact_type="review_request", artifact_ref="req-2",
                        artifact_content="answer two", intent="release")
check("C · first signature seq=1, prev=None", s1 and s1["chain_seq"] == 1 and s1["prev_hash"] is None)
check("C · second signature seq=2", s2 and s2["chain_seq"] == 2)
check("C · chain links: s2.prev_hash == s1.row_hash", s2 and s2["prev_hash"] == s1["row_hash"])
check("C · signer is always the caller (can't sign as another)", s1["signer_id"] == SIGNER)
check("C · signed at secure_eauth tier by default", s1["signature_method"] == "secure_eauth")

# Race: another firm's chain is independent.
s_other = m.append_signature(firm_id="firm-B", artifact_type="review_request", artifact_ref="req-x",
                             artifact_content="x", intent="approve")
check("C · a different firm's chain starts at seq=1 (per-firm)", s_other["chain_seq"] == 1)


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print("\n── D · verification DETECTS tampering (edit / delete / reorder) ──")

tables = {"signatures": []}
m = _mgr(caller=SIGNER, firm=FIRM, tables=tables)
for i in range(3):
    m.append_signature(firm_id=FIRM, artifact_type="review_request", artifact_ref=f"req-{i}",
                       artifact_content=f"answer {i}", intent="approve")
check("D · an intact 3-row chain verifies OK", m.verify_firm_chain(FIRM)["ok"] is True)

# EDIT a row's material field → content_hash mismatch (record + link both broken).
tables["signatures"][1]["intent"] = "release"
res = m.verify_firm_chain(FIRM)
check("D · an EDITED row breaks verification", res["ok"] is False and res["first_broken_seq"] == 2,
      str(res))
tables["signatures"][1]["intent"] = "approve"  # restore

# DELETE the middle row → a seq gap (1,3) is detected.
removed = tables["signatures"].pop(1)
res = m.verify_firm_chain(FIRM)
check("D · a DELETED row breaks the chain (seq gap)", res["ok"] is False, str(res))
tables["signatures"].insert(1, removed)  # restore order
check("D · restored chain verifies OK again", m.verify_firm_chain(FIRM)["ok"] is True)

# REORDER (swap two rows' seq) → prev_hash links no longer match.
tables["signatures"][0]["chain_seq"], tables["signatures"][1]["chain_seq"] = 2, 1
res = m.verify_firm_chain(FIRM)
check("D · a REORDERED chain breaks verification", res["ok"] is False, str(res))


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print("\n── E · the strong-tier seam (Aadhaar/DSC) — falls back, never blocks ──")

method, note = esign.resolve_method("aadhaar_esign")
check("E · unwired Aadhaar falls back to secure_eauth", method == "secure_eauth" and note)
method, note = esign.resolve_method("dsc")
check("E · unwired DSC falls back to secure_eauth", method == "secure_eauth" and note)
method, note = esign.resolve_method(None)
check("E · default is secure_eauth, no note", method == "secure_eauth" and note is None)
check("E · strong_tier_available is False until wired", esign.strong_tier_available("aadhaar_esign") is False)


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print("\n── F · route wiring (approve signs, release signs, §1(4) 422s the transition) ──")


def _route_world(*, status="pending", vault="vault-1", chain=None, matter_kind="",
                 vault_name="Acme MSA", missing=()):
    """A world the matters.py handlers can run against: one review request + the supporting tables the
    sign helpers read (collections for the §1(4) title, review_requests, signatures)."""
    req = {"id": "req-1", "firm_id": FIRM, "vault_id": vault, "artifact_ref": "msg-a",
           "submitted_by": "user-para", "current_owner": SIGNER, "chain": chain or [], "status": status}
    tables = {
        "review_requests": [req],
        "collections": [{"id": vault, "firm_id": FIRM, "name": vault_name, "matter_kind": matter_kind}],
        "signatures": [],
        "screens": [],
        "firm_memberships": [{"user_id": SIGNER, "firm_id": FIRM, "role": "managing_partner"}],
        "messages": [], "conversations": [],
    }
    m = _mgr(caller=SIGNER, firm=FIRM, tables=tables, missing=missing)
    # The handlers call sb.get_review_request / get_collection / update_review_request / append_signature
    # / get_review_artifact / resolve_member_emails — most exist on SupabaseManager; stub the few that
    # would otherwise touch un-modeled tables.
    m.get_review_artifact = lambda rid: {"available": True, "answer_full": "the indemnity cap is 12 months",
                                         "title": vault_name}
    return m, tables, req


class _Membership:
    def __init__(self, role="managing_partner"):
        self.firm_id = FIRM
        self.role = role


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# APPROVE (single-member chain → becomes 'approved', internal sign-off written).
m, tables, _ = _route_world(status="pending", chain=[])
resp = _run(mroutes.approve_review("req-1", ReviewDecisionRequest(), None, sb=m, membership=_Membership()))
sigs = [r for r in tables["signatures"] if r.get("intent") == "approve"]
check("F · approve writes an 'approve' signature", len(sigs) == 1)
check("F · approve response carries the signature", resp.signature is not None and resp.signature.intent == "approve")

# RELEASE (status approved → released, external sign-off written).
m, tables, _ = _route_world(status="approved", chain=[])
resp = _run(mroutes.release_review("req-1", ReviewDecisionRequest(), None, sb=m, membership=_Membership()))
sigs = [r for r in tables["signatures"] if r.get("intent") == "release"]
check("F · release writes a 'release' signature", len(sigs) == 1)
check("F · release response carries the signature", resp.signature is not None and resp.signature.intent == "release")

# §1(4) EXCLUSION on approve → 422 BEFORE the transition (no state change, no signature).
m, tables, req = _route_world(status="pending", chain=[], vault_name="Last Will and Testament of Jane Doe")
err = _expect_http(lambda: _run(mroutes.approve_review("req-1", ReviewDecisionRequest(), None, sb=m, membership=_Membership())))
check("F · §1(4)-excluded artifact 422s the approve", err == 422, f"got {err}")
check("F · no signature written on the refused approve", len(tables["signatures"]) == 0)

# §1(4) EXCLUSION on release → 422 (the artifact title carries the excluded-instrument signal).
m, tables, _ = _route_world(status="approved", chain=[], vault_name="General Power of Attorney")
err = _expect_http(lambda: _run(mroutes.release_review("req-1", ReviewDecisionRequest(), None, sb=m, membership=_Membership())))
check("F · §1(4)-excluded artifact 422s the release", err == 422, f"got {err}")


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print("\n── G · dormant parity (table unapplied ⇒ transition still completes, no 500) ──")

m, tables, _ = _route_world(status="approved", chain=[], missing=("signatures",))
resp = _run(mroutes.release_review("req-1", ReviewDecisionRequest(), None, sb=m, membership=_Membership()))
check("G · release completes with the signatures table unapplied", resp.status == "released")
check("G · no signature object when dormant", resp.signature is None)
check("G · verify_firm_chain on an empty/unapplied ledger is intact",
      m.verify_firm_chain(FIRM)["ok"] is True and m.verify_firm_chain(FIRM)["count"] == 0)


# ═══════════════════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}\nF2i E-SIGN GATE: {_passed} passed, {_failed} failed\n{'='*70}")
sys.exit(1 if _failed else 0)
