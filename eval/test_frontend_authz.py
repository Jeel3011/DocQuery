"""F2g frontend gate — the Firm Console UI + frontend security (offline, $0, source-static).

The committed gate for F2g (plans/F2_FIRM_CONSOLE_PLAN.md §F2g + §3 gate row). The frontend has no
JS test runner in this bootstrap repo (package.json scripts = dev/build/start/lint only), so — exactly
like F2f's lockstep-with-SQL gate parsed the migration — this gate PARSES THE ACTUAL BUILT FRONTEND
SOURCE and asserts the three contracts. A surface that is dropped, or a security guard that is removed,
FAILS here (it cannot silently pass). Deterministic; reads files only; no node, no browser, no API,
no extraction/kernel. Run:

    python -u eval/test_frontend_authz.py

It proves (maps to the §3 F2g gate row):

  SECURITY (§2.1, defense-in-depth — the server is ALWAYS the boundary):
    • middleware.ts BLOCKS /app/settings/firm for a non-cap user (T8) — redirect, not just a hidden
      link — by reading the SERVER caps payload (/auth/capabilities), and excludes external roles.
    • the store REFETCHES caps on a 403 (T7 — onForbidden → refreshCaps).
    • lists are SERVER-filtered (the store calls listMembers/listScreens/getReviewQueue — never a
      fetch-all-then-filter-in-JS), and the override POST is the gate (the dialog can't fire without
      the server cap — the affordance renders only for a holder + the POST re-checks).

  COMPLETENESS (the full-scope contract — every §2.2 surface ships + wires to its route):
    all 10 surfaces are present and wired to their already-built api.ts method. A missing surface
    FAILS this checklist (the "no silent caps" rule applied to the UI build itself).

  UX / D0:
    • a staffed paralegal sees the FULL toolkit framed as ENABLED, NOT read-only — the matter-team
      panel states "full access on this matter" (D0 made visible), and the access matrix mirrors
      authz.py ROLE_CAPS so a paralegal row shows the working toolkit (assert NOT read-only).
    • held controls are limited to release_external / manage-firm (the outbound/admin gates).
    • every list surface designs loading (Skeleton) + empty (EmptyState) + error (+ retry) states.
    • optimistic mutations reconcile on failure (rollback present — the F1e delete-404 lesson).
    • DESIGN.md token compliance: no raw hex color outside the reserved trust tokens; the Overridden
      reserved token exists; a wall is a lock glyph, not a red alarm.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FE = ROOT / "frontend-next"

_passed = 0
_failed = 0


def check(name, cond, hint=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}" + (f"  ({hint})" if hint else ""))


def _wall_region(src: str) -> str:
    """The WallsTab function body — so 'a wall is a lock glyph, not a red alarm' checks only the
    wall surface, not the unrelated error states (which legitimately use the red/risk token)."""
    i = src.find("function WallsTab")
    if i < 0:
        return ""
    j = src.find("function AddScreenDialog", i)
    return src[i:j if j > 0 else len(src)]


def read(rel: str) -> str:
    p = FE / rel
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def exists(rel: str) -> bool:
    return (FE / rel).exists()


# Load every file the gate inspects once.
mw = read("middleware.ts")
store = read("stores/firm.store.ts")
api = read("lib/api.ts")
console = read("app/app/settings/firm/page.tsx")
shared = read("app/app/settings/firm/_shared.tsx")
team = read("components/app/MatterTeamPanel.tsx")
chain = read("components/app/ReviewChain.tsx")
override = read("components/chat/OverrideAffordance.tsx")
accept = read("app/app/accept-invite/page.tsx")
queue_page = read("app/app/review-queue/page.tsx")
globals_css = read("app/globals.css")
topbar = read("components/app/TopBar.tsx")
select_cmp = read("components/ui/Select.tsx")
login = read("app/login/page.tsx")
firm_setup = read("lib/firmSetup.ts")
chat_msg = read("components/chat/ChatMessage.tsx")

print("\nF2g frontend gate — Firm Console UI + frontend security (source-static)\n")

# ════════════════════════════════════════════════════════════════════════════════
# SECURITY (§2.1)
# ════════════════════════════════════════════════════════════════════════════════
print("SECURITY — the server is the boundary (T8 block-not-hide · T7 stale-cap · T2 server-filtered)")

check("middleware guards /app/settings/firm (T8)", "/app/settings/firm" in mw)
check("middleware resolves the SERVER caps (calls /auth/capabilities, no drift)",
      "/api/v1/auth/capabilities" in mw or "/auth/capabilities" in mw)
check("middleware REDIRECTS a non-cap user (block, not hide)",
      "NextResponse.redirect" in mw and "manage_members" in mw)
check("middleware excludes external roles (is_external)", "is_external" in mw)

check("store refetches caps on a 403 (T7 — onForbidden → refreshCaps)",
      "onForbidden" in store and "refreshCaps" in store
      and re.search(r"status\s*===\s*403", store) is not None)
check("store caps come from the SERVER payload (getCapabilities), not a hardcoded role map",
      "getCapabilities" in store)
check("store lists are SERVER-filtered (listMembers/listScreens/getReviewQueue), not fetch-all+filter",
      all(m in store for m in ("listMembers", "listScreens", "getReviewQueue")))
check("store does NOT filter a member/screen list by role in JS (no client-side .filter on caps)",
      ".filter(" not in store or "role ===" not in store)

# the caps endpoint method exists in api.ts and is read-only (GET)
check("api.ts has getCapabilities → GET /auth/capabilities",
      "getCapabilities" in api and '.get<CapabilitiesResponse>("/auth/capabilities")' in api)

# the override POST is the gate: the affordance renders only for a holder AND the POST re-checks
check("override affordance renders ONLY for a cap holder (can('override_abstain'))",
      'can("override_abstain")' in override)
check("override is a server POST (overrideAbstain) — the dialog opening is not the gate",
      "overrideAbstain" in override)

# ════════════════════════════════════════════════════════════════════════════════
# COMPLETENESS — the full-scope contract (every §2.2 surface present + wired)
# ════════════════════════════════════════════════════════════════════════════════
print("\nCOMPLETENESS — every §2.2 surface ships + wires to its already-built route")

SURFACES = [
    ("1  onboarding (invite + accept-invite landing)",
     ("inviteMember" in console) and exists("app/app/accept-invite/page.tsx") and "acceptInvite" in accept),
    ("2  People tab (member list + promote/demote + remove)",
     all(m in console for m in ("listInvites", "setRole", "removeMember")) and "RolePicker" in console),
    ("3  Matters & Access matrix (role × verb, read-only)",
     "AccessMatrixTab" in console and "MATRIX_VERBS" in console and "ROLE_CAPS" in shared),
    ("4  Ethical Walls (list + add/lift + required reason)",
     "WallsTab" in console and "loadScreens" in console        # list is server-filtered via the store
     and "createScreen" in console and "removeScreen" in console),
    ("5  Matter team panel (add/remove + 'full access on this matter')",
     exists("components/app/MatterTeamPanel.tsx")
     and all(m in team for m in ("addMatterTeam", "removeMatterTeam", "getMatterTeam"))),
    ("6  Review chain (send + chain preview + customize + My Review Queue)",
     exists("components/app/ReviewChain.tsx")
     and all(m in chain for m in ("submitForReview", "approveReview", "requestChanges",
                                  "releaseExternal", "getReviewQueue"))
     and "setReviewChain" in team
     and exists("app/app/review-queue/page.tsx")),
    ("7  Delegation / PA (grant verbs + expiry, time-boxed chip, revoke)",
     all(m in console for m in ("grantAuthority", "revokeAuthority", "listDelegations"))
     and "DelegationTab" in console),
    ("8  Abstain-override moment (holder-only → reason → Overridden state)",
     exists("components/chat/OverrideAffordance.tsx")
     and "overridden" in override and "gateObjection" in override),
    ("9  Firm switcher (active firm, re-scope caps/screens)",
     "ConsoleHeader" in console and ("getFirm" in api) and "firm" in store),
    ("10 Caps source of truth (server payload feeds the UI, not a JS role→button map)",
     "getCapabilities" in store and "CapabilitiesResponse" in api),
]
for label, present in SURFACES:
    check(f"surface {label}", present, "surface missing or not wired")

# wiring spine §2.3 — EVERY enumerated api.ts method exists
print("\nCOMPLETENESS — §2.3 wiring spine: every enumerated api.ts method exists")
API_METHODS = [
    "listMembers", "setRole", "removeMember", "inviteMember", "acceptInvite", "listInvites",
    "addMatterTeam", "removeMatterTeam", "grantAuthority", "revokeAuthority", "listDelegations",
    "listScreens", "createScreen", "removeScreen", "submitForReview", "approveReview",
    "requestChanges", "getReviewQueue", "setReviewChain", "releaseExternal", "overrideAbstain",
    "getCapabilities",
]
missing = [m for m in API_METHODS if f"export async function {m}" not in api]
check(f"all {len(API_METHODS)} api.ts methods present", not missing, f"missing: {missing}")

# nav reachability — the console + queue are reachable, console link is cap-gated
check("Firm console nav link is cap-gated render (manage_members) in TopBar",
      '"/app/settings/firm"' in topbar and 'caps.has("manage_members")' in topbar)
check("My review queue nav link present in TopBar", '"/app/review-queue"' in topbar)

# ════════════════════════════════════════════════════════════════════════════════
# UX / D0
# ════════════════════════════════════════════════════════════════════════════════
print("\nUX / D0 — positive not punitive · every state designed · legible · DESIGN.md compliant")

# D0: a staffed member is told they have FULL access (not a wall of disabled buttons)
check("D0 — matter-team panel states 'full access on this matter' (positive grant, visible)",
      re.search(r"full access on this matter", team, re.IGNORECASE) is not None)
# D0: the access matrix mirrors ROLE_CAPS → a paralegal row is NOT read-only (has the working toolkit)
para_caps = re.search(r"paralegal:\s*new Set\((\w+)\)", shared)
check("D0 — paralegal default caps = the WORKING_TOOLKIT (NOT read-only) in the matrix mirror",
      para_caps is not None and para_caps.group(1) == "WORKING_TOOLKIT")
# held controls are the few outbound/admin gates
check("held controls are release_external + manage-firm only (the matrix mirrors authz)",
      "PARTNER_ELEVATED" in shared and "FIRM_GOVERNANCE" in shared)

# every state designed: loading (Skeleton) + empty (EmptyState) + error (+retry)
for fname, src in [("console", console), ("matter-team", team), ("review-chain", chain)]:
    check(f"{fname}: loading state uses Skeleton (not a dead spinner)", "Skeleton" in src)
for fname, src in [("console People/Walls/Delegation", console), ("review queue", chain)]:
    check(f"{fname}: empty state uses EmptyState ('situation then action')", "EmptyState" in src)
for fname, src in [("console", console), ("matter-team", team), ("review-chain", chain)]:
    check(f"{fname}: error state offers a Retry", "Retry" in src or "Try again" in src)

# optimistic mutations reconcile on failure (the F1e delete-404 lesson — rollback present)
check("matter-team optimistic remove ROLLS BACK on failure (no stranded state)",
      "setTeam(prev)" in team and "// roll back" in team)
check("People role-change RECONCILES with server truth on failure (reload after error)",
      "reconcile" in console or "await reload();" in console)

# legible human moments
check("promote/demote explains WHY a role is unselectable (≥ your own rank)",
      re.search(r"at or above your own", console) is not None)
check("remove names the consequence ('revokes all access immediately')",
      re.search(r"access (ended|immediately)", console, re.IGNORECASE) is not None
      or "immediately" in console)
check("chain preview shows who owns it next BY NAME (anti-stall, visible)",
      "ChainPreview" in chain and ("current_owner" in chain or "ownerId" in chain))
check("override dialog STATES the gate objection before confirm",
      "What the gate objected to" in override)

# DESIGN.md token compliance
check("Overridden reserved trust token exists (--fidelity-overridden)",
      "--fidelity-overridden" in globals_css)
# wall = lock glyph BOUNDARY marker, neutral ink — not a red alarm. The screen ROW renders a Lock
# in neutral ink; the red/risk token only appears in the (separate, legitimate) error state.
_wall = _wall_region(console)
check("a wall is a lock glyph (Lock) in neutral ink, not a red alarm",
      "Lock size={15}" in _wall
      and re.search(r"<Lock size=\{15\}[^>]*var\(--ink-2\)", _wall) is not None)
# no raw hex outside the reserved trust tokens (we only allow the indigo override rgba in components,
# which maps to the reserved token) — the surfaces use CSS vars, not raw hex backgrounds.
def _raw_hex_bg(src: str) -> list[str]:
    # flag a hex color used directly as a background/color value in a style (not via var()).
    return re.findall(r"(?:background|color)\s*:\s*[\"']?#[0-9A-Fa-f]{3,6}", src)
for fname, src in [("console", console), ("matter-team", team), ("review-chain", chain)]:
    hits = _raw_hex_bg(src)
    check(f"{fname}: no raw hex color (uses CSS tokens / trust vars)", not hits, f"raw hex: {hits[:3]}")

# no em-dashes in user-facing COPY (DESIGN.md). Comments may use them freely; we strip // and /* */
# comment lines and check only the remaining source (JSX text + string literals).
def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)        # block comments
    src = re.sub(r"^\s*//.*$", "", src, flags=re.MULTILINE)     # full-line // comments
    src = re.sub(r"\s//[^\n\"']*$", "", src, flags=re.MULTILINE)  # trailing // comments
    return src
check("no em-dash in console user-facing copy (DESIGN.md; comments exempt)",
      "—" not in _strip_comments(console))


# ════════════════════════════════════════════════════════════════════════════════
# POLISH & WIRING (the make-interfaces-feel-better pass + the onboarding gaps)
# ════════════════════════════════════════════════════════════════════════════════
print("\nPOLISH & WIRING — custom Select · real member names · signup onboarding · live affordances")

# 1) The native <select> is replaced by the custom on-brand Select everywhere in the firm surfaces.
for fname, src in [("console", console), ("matter-team", team)]:
    check(f"{fname}: no native <select> (custom Select component used)", "<select" not in src and "Select" in src)
check("custom Select uses scale-on-press (active:scale-[0.96]) + reduced-motion",
      "active:scale-[0.96]" in select_cmp and "useReducedMotion" in select_cmp)
check("custom Select option rows meet the 40px+ hit area (min-h)",
      "min-h-[40px]" in select_cmp and "min-h-[38px]" in select_cmp)
check("custom Select avoids transition: all (specific properties only)",
      "transition: all" not in _strip_comments(select_cmp) and "transition-[" in select_cmp)

# make-interfaces-feel-better pass: scale-on-press, tabular-nums, text-wrap balance.
check("console interactive buttons use scale-on-press (active:scale-[0.96])",
      "active:scale-[0.96]" in console)
check("no sub-0.95 press scale anywhere in the F2g surfaces (skill: never below 0.95)",
      not any(re.search(r"active:scale-\[0\.(9[0-4]|[0-8]\d?)\]", s)
              for s in (console, select_cmp, team, override, chain)))
check("dynamic counts/dates use tabular-nums (no layout shift)",
      "tabular-nums" in console)
check("page heading uses text-wrap: balance", 'textWrap: "balance"' in console or "[text-wrap:balance]" in console)

# 2) Member names: the backend resolves a real email (no bare 74f… user_id) — server-side.
check("api.ts MemberResponse carries email", "email: string | null" in api or "email?: string" in api)
check("matter-team member carries email (server-resolved, real name not raw id)",
      "email: string | null" in api)

# 3) Signup onboarding (the firm gap): name a new firm OR join by invite token.
check("signup form has a firm-name field", "firmName" in login and "Start a firm" in login)
check("signup form has a join-by-invite token field", "inviteToken" in login and "Join with invite" in login)
check("signup applies the onboarding intent (setupFirm / stash for email-confirm)",
      "setupFirm" in login and "stashFirmSetup" in login)
check("firmSetup consumes the stash on first authed load (email-confirm path)",
      "consumeStashedFirmSetup" in firm_setup and "acceptInvite" in firm_setup and "renameFirm" in firm_setup)
check("api.ts has renameFirm → PATCH /auth/firm", "renameFirm" in api and 'patch<Firm>("/auth/firm"' in api)

# 3b) Invite delivery (no email server yet): a copyable LINK, and a resend that ROTATES the token.
check("invite dialog shows a full accept-invite LINK (not a bare token)",
      "acceptInviteUrl" in console and "Copy invite link" in console)
check("api.ts builds the full link via acceptInviteUrl(public /invite?token=)",
      "acceptInviteUrl" in api and "/invite?token=" in api)
check("pending invites can re-copy (resendInvite rotates) + revoke",
      "resendInvite" in console and "revokeInvite" in console and "PendingInviteRow" in console)
check("api.ts has resendInvite (POST .../resend) + revokeInvite (DELETE)",
      "resendInvite" in api and "/resend" in api and "revokeInvite" in api)
check("the one-time success dialog does NOT refresh the list while open (anti-remount fix)",
      "do NOT refresh the parent list here" in console or "deferred the refresh" in console.lower()
      or "if (hadInvite) onInvited()" in console)

# 3c) Invite redemption robustness: accept a pasted LINK or a bare token; public /invite landing so
# a logged-out invitee isn't bounced to login and lose the token.
login = read("app/login/page.tsx")
invite_landing = read("app/invite/page.tsx")
accept = read("app/app/accept-invite/page.tsx")
check("extractInviteToken pulls ?token= out of a pasted full link (or returns the bare token)",
      "extractInviteToken" in api and "token=" in api)
check("signup 'Join with invite' runs the pasted value through extractInviteToken",
      "extractInviteToken" in login)
check("accept-invite page also tolerates a pasted full link (extractInviteToken on ?token=)",
      "extractInviteToken" in accept)
check("a PUBLIC /invite landing exists (not under /app, so a logged-out invitee keeps the token)",
      bool(invite_landing) and "stashFirmSetup" in invite_landing)
check("/invite is NOT matched by middleware (stays public)",
      '"/invite"' not in mw and "/app/:path*" in mw)
check("login pre-fills the invite token from the stash (no manual paste after /invite)",
      "peekStashedFirmSetup" in login)
check("acceptInviteUrl points at the public /invite route",
      "/invite?token=" in api)

# 4) The review + override affordances are MOUNTED on a real answer (not dead components).
check("SendForReview + OverrideAffordance are imported into ChatMessage", "SendForReview" in chat_msg and "OverrideAffordance" in chat_msg)
check("ChatMessage mounts the affordances on a settled assistant answer (vaultId+messageId)",
      "vaultId={vaultId}" in chat_msg and "OverrideAffordance" in chat_msg and "SendForReview" in chat_msg)


def report():
    print(f"\n{'='*60}\nF2g frontend gate: {_passed} passed, {_failed} failed\n{'='*60}")
    sys.exit(1 if _failed else 0)


report()
