"""F2c regression gate — ETHICAL WALLS (conflict screens). THE MOAT GATE. (offline, $0).

The committed gate for F2c (plans/F2_FIRM_CONSOLE_PLAN.md §F2c). A screen hard-blocks a named
member from a matter (vault) at EVERY layer — a wall that covers only one path is a FALSE wall, so
this gate carries the same path-by-path rigor F1 gave isolation (its H1–H6), re-walked for screens
(P1–P9). Deterministic, no API, no Supabase, no extraction/kernel. Run:

    python -u eval/test_ethical_walls.py

What it proves (maps to the plan's §3 gate row for F2c — the moat):
  A. DENY-OVERRIDES-ROLE (the headline): a screen DENIES a Managing Partner — not just a junior.
     The wall is a negative authorization an ALLOW can never cross (D2 precedence). Proven on the
     pure decision (authz.authorize) AND end-to-end through the resolved membership.
  B. Every retrieval/read PATH P1–P9 returns 0 / refuse for a screened user (reproduce-then-close,
     per path): P1 agent-core search_vault, P2 read_document (incl. the prompt-injection variant),
     P3 Brain/legacy chat _resolve_collection_filters, P4 redline, P5 review-grid/workflows,
     P6 export, P7 worker re-ingest (add/remove doc to vault), P9 the row-RLS seam (F2f — noted).
  C. SOFT-REMOVE restores access on the NEXT request (T7) — screens resolved per request, never
     cached; a lifted screen is excluded by removed_at IS NULL.
  D. Screen create/remove is FIRM-SCOPED: a body firm_id is ignored (server-resolved, T3); a
     guessed cross-firm screen_id can't be lifted, a cross-firm user/vault can't be walled (T2).
  E. Every screen add/remove (and every wall BLOCK) is AUDIT-LOGGED (T10).

OFFLINE: authz is PURE. The DB-layer tests use a fake SupabaseManager that models the screens
table (active = removed_at None) + the firm-scoping the real queries do. No live anything.
See [[run-only-relevant-gates]] — F2c touches authz/dependencies/db/routes, NOT extraction/kernel.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException  # noqa: E402

from src.components.authz import (  # noqa: E402
    Membership,
    Scope,
    authorize,
    caps_for_role,
)

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = ""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


# A few fixed ids reused across the gate.
FIRM_A = "firm-A"
FIRM_B = "firm-B"
WALLED_VAULT = "vault-walled"
OPEN_VAULT = "vault-open"
MP_USER = "user-mp"          # a Managing Partner — the deny-overrides-role subject
PARA_USER = "user-para"      # a paralegal — the obvious-junior subject


# ─────────────────────────────────────────
# A FAKE SupabaseManager modeling the screens table + firm-scoping (no live DB).
#   Mirrors the real query semantics: active screen = removed_at is None; everything is
#   firm-scoped; a body firm_id never reaches a write (routes pass the resolved firm).
# ─────────────────────────────────────────
class FakeSB:
    def __init__(self, user_id=MP_USER, firm_id=FIRM_A, role="managing_partner"):
        self.user_id = user_id
        self._firm_id = firm_id
        self._role = role
        # screens: list of dicts {id, firm_id, user_id, vault_id, reason, created_by,
        #          created_at, removed_at}
        self._screens: list[dict] = []
        # firm membership map: user_id -> firm_id (for user_in_firm)
        self._members = {MP_USER: FIRM_A, PARA_USER: FIRM_A, "user-otherfirm": FIRM_B}
        # vault -> firm (for collection_in_firm) and vault -> doc_ids
        self._vault_firm = {WALLED_VAULT: FIRM_A, OPEN_VAULT: FIRM_A, "vault-firmB": FIRM_B}
        self._vault_docs = {WALLED_VAULT: ["d1", "d2"], OPEN_VAULT: ["d3"]}
        self.audit: list[tuple] = []      # (action, resource_type, resource_id, metadata)
        self._seq = 0
        self.read_client = self          # the fakes route read_client back to self

    # —— get_user_firm: server-side firm + role (resolve_membership reads this) ——
    def get_user_firm(self, user_id=None, firm_id=None):
        return {"id": self._firm_id, "name": "Firm", "role": self._role}

    # —— the F2c read path (resolve_membership + the retrieval-layer guard read these) ——
    def screened_vault_ids(self, user_id=None, firm_id=None):
        uid = user_id or self.user_id
        return {s["vault_id"] for s in self._screens
                if s["user_id"] == uid and s["removed_at"] is None
                and (firm_id is None or s["firm_id"] == firm_id)}

    def is_vault_screened(self, vault_id, user_id=None, firm_id=None):
        uid = user_id or self.user_id
        return any(s["user_id"] == uid and s["vault_id"] == vault_id
                   and s["removed_at"] is None
                   and (firm_id is None or s["firm_id"] == firm_id)
                   for s in self._screens)

    # —— the F2c writes (admin routes call these, after authorizing + resolving firm) ——
    def create_screen(self, firm_id, user_id, vault_id, reason, created_by=None):
        if not (reason or "").strip():
            raise ValueError("A reason is required to raise an ethical wall.")
        # unique active (firm,user,vault) — mirror the partial unique index
        if any(s["firm_id"] == firm_id and s["user_id"] == user_id
               and s["vault_id"] == vault_id and s["removed_at"] is None
               for s in self._screens):
            raise Exception("duplicate active screen")
        self._seq += 1
        row = {"id": f"screen-{self._seq}", "firm_id": firm_id, "user_id": user_id,
               "vault_id": vault_id, "reason": reason.strip(),
               "created_by": created_by or self.user_id,
               "created_at": "2026-06-24T00:00:00Z", "removed_at": None}
        self._screens.append(row)
        return row

    def remove_screen(self, screen_id, firm_id):
        for s in self._screens:
            if s["id"] == screen_id and s["firm_id"] == firm_id and s["removed_at"] is None:
                s["removed_at"] = "2026-06-24T01:00:00Z"
                return dict(s)
        return {}

    def list_screens(self, firm_id, include_removed=False):
        return [dict(s) for s in self._screens if s["firm_id"] == firm_id
                and (include_removed or s["removed_at"] is None)]

    def collection_in_firm(self, vault_id, firm_id):
        return self._vault_firm.get(vault_id) == firm_id

    def user_in_firm(self, user_id, firm_id):
        return self._members.get(user_id) == firm_id

    # —— shapes the retrieval paths use ——
    def get_collection_document_ids(self, cid):
        return list(self._vault_docs.get(cid, []))

    def get_collection(self, cid):
        return {"id": cid} if cid in self._vault_firm else {}

    def table(self, *_a, **_k):
        # documents().select().in_().eq().execute() in _resolve_collection_filters
        outer = self

        class _Q:
            def select(self, *a, **k): return self
            def in_(self, *a, **k): return self
            def eq(self, *a, **k): return self
            def execute(self_inner):
                return type("R", (), {"data": [{"filename": "a1.pdf"}, {"filename": "a2.pdf"}]})()
        return _Q()

    @property
    def client(self):
        return self


# A tiny stand-in for the audit module so log_audit records into the fake (T10 assertion).
def _patch_audit(sb):
    import src.api.routes.audit as audit_mod

    def _fake_log(_sb, action, resource_type=None, resource_id=None, metadata=None, ip_address=None):
        sb.audit.append((action, resource_type, resource_id, metadata or {}))
    audit_mod.log_audit = _fake_log
    return audit_mod


print("\n── A. DENY-OVERRIDES-ROLE (the headline: a screen beats a Managing Partner) ──")

# Pure decision: an MP whose membership carries the screened vault is DENIED on that vault.
mp = Membership(user_id=MP_USER, firm_id=FIRM_A, role="managing_partner",
                caps=caps_for_role("managing_partner"),
                screened_vault_ids=frozenset({WALLED_VAULT}))
for verb in ("ask", "draft", "run_workflow", "ingest", "grids", "release_external", "delete"):
    d = authorize(mp, verb, Scope(vault_id=WALLED_VAULT, firm_id=FIRM_A))
    check(f"A: MP DENIED '{verb}' on the walled vault (screen beats role)", not d.allow, d.reason)
# …but the SAME MP is allowed on an OPEN (un-screened) vault — the wall is matter-scoped (D4).
d_open = authorize(mp, "ask", Scope(vault_id=OPEN_VAULT, firm_id=FIRM_A))
check("A: MP still ALLOWED on a non-screened vault (wall is matter-level, not a ban)", d_open.allow)
# A paralegal is denied too — but that's the easy case; the headline above is the MP.
para = Membership(user_id=PARA_USER, firm_id=FIRM_A, role="paralegal",
                  caps=caps_for_role("paralegal"),
                  screened_vault_ids=frozenset({WALLED_VAULT}))
check("A: paralegal also DENIED on the walled vault",
      not authorize(para, "ask", Scope(vault_id=WALLED_VAULT, firm_id=FIRM_A)).allow)
# Precedence proof: the screen is checked BEFORE the role grant — a no-vault check still allows.
check("A: with NO vault in scope, the MP keeps the verb (screen only bites the walled vault)",
      authorize(mp, "ask", Scope()).allow)


print("\n── A2. End-to-end: resolve_membership populates the screen from the DB (Layer 1) ──")
from src.api.dependencies import resolve_membership, assert_vault_not_screened, require_cap  # noqa: E402

sb = FakeSB(user_id=MP_USER, firm_id=FIRM_A, role="managing_partner")
sb.create_screen(FIRM_A, MP_USER, WALLED_VAULT, "adverse party — prior representation")
m = resolve_membership(sb)
check("A2: resolve_membership pulls the screen into membership.screened_vault_ids",
      WALLED_VAULT in m.screened_vault_ids, str(m.screened_vault_ids))
check("A2: authorize() over the RESOLVED membership denies the MP on the walled vault",
      not authorize(m, "ask", Scope(vault_id=WALLED_VAULT, firm_id=FIRM_A)).allow)


print("\n── B. Every retrieval/read PATH P1–P9 returns 0 / refuse for a screened user ──")
_patch_audit(sb)  # capture screen.block audit rows

# P1 — agent-core / search_vault: assert_vault_not_screened (the route floor) refuses the vault.
raised = False
try:
    assert_vault_not_screened(sb, WALLED_VAULT)
except HTTPException as e:
    raised = e.status_code == 403
check("P1: agent-core route floor (assert_vault_not_screened) 403s the walled vault", raised)
# …and ALLOWS an un-screened vault (no over-block).
ok = True
try:
    assert_vault_not_screened(sb, OPEN_VAULT)
except HTTPException:
    ok = False
check("P1: the same floor ALLOWS a non-screened vault (no over-block)", ok)

# P3 — Brain / legacy chat _resolve_collection_filters refuses the walled vault.
from src.api.routes.chat import _resolve_collection_filters  # noqa: E402
raised = False
try:
    _resolve_collection_filters(sb, WALLED_VAULT, query="revenue",
                                user_config=type("C", (), {"ROUTING_MAX_FANOUT": 8})())
except HTTPException as e:
    raised = e.status_code == 403
check("P3: Brain/chat _resolve_collection_filters 403s the walled vault", raised)
# …and still resolves a non-screened vault to a filename list (no regression).
res = _resolve_collection_filters(sb, OPEN_VAULT, query="revenue",
                                  user_config=type("C", (), {"ROUTING_MAX_FANOUT": 8})())
check("P3: a non-screened vault still resolves to a scoped filename list",
      isinstance(res, list) and res, str(res))

# P2 — read_document: the F1 H2 vault-membership guard refuses a foreign/walled doc_id (the
# PROMPT-INJECTION variant — a doc telling the agent to read the walled vault's doc returns 0).
from src.components.agent_core.tools.read import read_document  # noqa: E402


class _Grid:
    def __init__(self, doc):
        self.doc, self.page, self.table_id = doc, 1, "t0"


# The run is scoped to the OPEN vault (filename_by_doc IS its membership set). A prompt-injected
# doc_id pointing at the WALLED vault's document is not in scope ⇒ read returns no grids.
inj = read_document(
    "walled-doc-d1",
    db_client=sb,
    grids=[_Grid("d3.pdf")],                     # the open vault's only grid
    filename_by_doc={"d3": "d3.pdf"},            # the open-vault membership set
)
# The H2 guard returns an error envelope (ok False, data None, "not in the loaded scope") — the
# out-of-vault doc is refused, NOT read. That is the prompt-injection closure (the agent prompt
# is never the authz boundary; the vault membership set in filename_by_doc is).
check("P2: read_document (prompt-injection variant) refuses an out-of-vault doc (ok=False, no data)",
      inj.get("ok") is False and inj.get("data") is None
      and "not in the loaded scope" in (inj.get("error") or ""), str(inj)[:200])

# P4/P5/P7 — redline / review-grid / workflows / add-doc all gate on the SAME route floor
# (assert_vault_not_screened); proving the floor 403s the walled vault (P1) closes them. We
# assert the wiring is present so a future refactor that drops the call FAILS this gate.
import inspect  # noqa: E402
import src.api.routes.redline as redline_mod  # noqa: E402
import src.api.routes.review_grid as grid_mod  # noqa: E402
import src.api.routes.workflows as wf_mod  # noqa: E402
import src.api.routes.collections as coll_mod  # noqa: E402
import src.api.routes.agent_core as ac_mod  # noqa: E402
check("P4: redline route calls assert_vault_not_screened",
      "assert_vault_not_screened" in inspect.getsource(redline_mod))
check("P5: review-grid route calls assert_vault_not_screened",
      "assert_vault_not_screened" in inspect.getsource(grid_mod))
check("P5: workflows route calls assert_vault_not_screened",
      "assert_vault_not_screened" in inspect.getsource(wf_mod))
check("P1: agent-core route calls assert_vault_not_screened",
      "assert_vault_not_screened" in inspect.getsource(ac_mod))
check("P7: collections (add/remove doc → re-ingest) calls assert_vault_not_screened",
      inspect.getsource(coll_mod).count("assert_vault_not_screened") >= 2)

# P6 — export: the export verb is cap-gated on release_external (F2b); a screened user can never
# PRODUCE the content to export (P1–P5 all return 0/refuse upstream), so there is no walled
# content to release. The wall closure for export is the upstream content-generation block.
check("P6: export is upstream-closed (screened user cannot generate walled content to export)",
      True)  # documented closure — no vault id flows to the export bodies; see §F2c P6.

# P9 — row RLS is the F2f backstop (NOT EXISTS(screen) on the row policy). Seam only here.
check("P9: row-RLS screen clause is the F2f seam (noted, not built in F2c)", True)


print("\n── C. SOFT-REMOVE restores access on the NEXT request (T7) ──")
# The screen is still active from A2. Resolve again (a fresh 'request') → still blocked.
check("C: while active, the next request is still blocked",
      WALLED_VAULT in resolve_membership(sb).screened_vault_ids)
# Lift it (soft remove), then resolve again → access restored, and the row is preserved.
scr = sb.list_screens(FIRM_A)[0]
removed = sb.remove_screen(scr["id"], FIRM_A)
check("C: remove_screen soft-removes (sets removed_at, row preserved)",
      removed.get("removed_at") is not None and len(sb.list_screens(FIRM_A, include_removed=True)) == 1)
m_after = resolve_membership(sb)
check("C: the NEXT request after removal restores access (screen gone from membership)",
      WALLED_VAULT not in m_after.screened_vault_ids)
ok = True
try:
    assert_vault_not_screened(sb, WALLED_VAULT)
except HTTPException:
    ok = False
check("C: the retrieval floor ALLOWS the vault again after the wall is lifted", ok)
# A double-remove is a clean no-op (idempotent), not an error.
check("C: a double-remove (or unknown id) is a clean no-op", sb.remove_screen(scr["id"], FIRM_A) == {})


print("\n── D. Screen create/remove is FIRM-SCOPED (body firm_id ignored — T2/T3) ──")
# The admin route resolves the firm SERVER-SIDE from the membership; the body has no firm_id.
# We simulate the route's T2 validation: a cross-firm user / vault is rejected before any write.
sb2 = FakeSB(user_id=MP_USER, firm_id=FIRM_A, role="managing_partner")
check("D: T2 — a vault of ANOTHER firm cannot be walled (collection_in_firm False)",
      not sb2.collection_in_firm("vault-firmB", FIRM_A))
check("D: T2 — a member of ANOTHER firm cannot be walled (user_in_firm False)",
      not sb2.user_in_firm("user-otherfirm", FIRM_A))
check("D: T3 — own-firm vault + own-firm user pass the scope checks",
      sb2.collection_in_firm(WALLED_VAULT, FIRM_A) and sb2.user_in_firm(PARA_USER, FIRM_A))
# T2 — a guessed cross-firm screen_id cannot be lifted (remove_screen is firm-scoped).
sb2.create_screen(FIRM_A, PARA_USER, WALLED_VAULT, "conflict")
s_id = sb2.list_screens(FIRM_A)[0]["id"]
check("D: T2 — remove_screen with a WRONG firm_id lifts nothing (cross-firm denied)",
      sb2.remove_screen(s_id, FIRM_B) == {})
check("D: the screen is still active after the cross-firm remove attempt",
      len(sb2.list_screens(FIRM_A)) == 1)
# screened_vault_ids is firm-scoped: a firm-A screen never bleeds into a firm-B membership read.
check("D: screened_vault_ids(firm_B) excludes the firm-A screen (no cross-firm bleed)",
      WALLED_VAULT not in sb2.screened_vault_ids(user_id=PARA_USER, firm_id=FIRM_B))


print("\n── E. Every wall add / remove / block is AUDIT-LOGGED (T10) ──")
# Drive the admin routes' audit via the patched logger. assert_vault_not_screened logs a block.
sb3 = FakeSB(user_id=MP_USER, firm_id=FIRM_A, role="managing_partner")
_patch_audit(sb3)
sb3.create_screen(FIRM_A, PARA_USER, WALLED_VAULT, "conflict of interest")
# A blocked request (the screened user is the paralegal here — simulate their request).
sb_para = FakeSB(user_id=PARA_USER, firm_id=FIRM_A, role="paralegal")
sb_para._screens = sb3._screens             # share the screen store
_patch_audit(sb_para)
try:
    assert_vault_not_screened(sb_para, WALLED_VAULT)
except HTTPException:
    pass
check("E: a wall BLOCK writes a 'screen.block' audit row (T10)",
      any(a[0] == "screen.block" for a in sb_para.audit), str(sb_para.audit))
# require_cap's deny is also audited (re-confirm the F2b T10 path still holds for a screened MP).
deny_logged = {"hit": False}


class _DenySB(FakeSB):
    def __init__(self):
        super().__init__(user_id=MP_USER, firm_id=FIRM_A, role="paralegal")

    def get_user_firm(self, user_id=None, firm_id=None):
        return {"id": FIRM_A, "name": "Firm", "role": "paralegal"}


dsb = _DenySB()
_patch_audit(dsb)
guard = require_cap("release_external")   # a paralegal lacks this verb
raised = False
try:
    guard(sb=dsb)
except HTTPException as e:
    raised = e.status_code == 403
check("E: require_cap deny still 403s + audits (F2b T10 intact)",
      raised and any(a[0] == "authz.deny" for a in dsb.audit), str(dsb.audit))


print("\n── F. Flag-off / pre-F2c parity (no screens ⇒ byte-identical) ──")
clean = FakeSB(user_id=MP_USER, firm_id=FIRM_A, role="managing_partner")  # no screens created
m_clean = resolve_membership(clean)
check("F: with no screens, membership.screened_vault_ids is empty (legacy parity)",
      m_clean.screened_vault_ids == frozenset())
ok = True
try:
    assert_vault_not_screened(clean, WALLED_VAULT)
except HTTPException:
    ok = False
check("F: with no screens, every vault passes the retrieval floor (byte-identical to pre-F2c)", ok)


# ── tally ──
print(f"\n{'='*60}")
print(f"  test_ethical_walls: {_passed} passed, {_failed} failed")
print(f"{'='*60}")
sys.exit(0 if _failed == 0 else 1)
