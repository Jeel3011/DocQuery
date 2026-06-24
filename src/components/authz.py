"""F2b — the central authorization decision (PBAC).

ONE function every route trusts. `authorize(membership, verb, scope) -> Decision` is the single
gatekeeper (D2): a role→capability matrix gated by attribute conditions, with an explicit DENY
(ethical wall) that OVERRIDES any role grant (deny-overrides precedence — XACML/Istio standard +
legally correct). Enforced as `require_cap(verb)` middleware (src/api/dependencies.py), never as 40
hand-written `if`s scattered across features.

This module is PURE: no DB, no network, no LLM, no FastAPI. It takes a resolved membership (firm +
role + caps + screens + delegations) and returns a Decision. The per-request RESOLUTION
(server-side, never from the JWT/body — T7/T3) lives in dependencies.resolve_membership; the
ENFORCEMENT (403 before the handler) lives in dependencies.require_cap. Keeping the decision pure
is what makes the gate `eval/test_authz.py` $0/offline and what makes the eventual swap to an
external policy engine (OPA/Cerbos — enterprise, logged-not-built) mechanical.

PRIME PRINCIPLE (§0.5 D0): the product exists to make EVERYONE productive — juniors and paralegals
included. The matrix is NOT "who may use the tool"; everyone on a matter has the full working
toolkit (ask · draft · run-agent · grids · ingest). The only two controls are the matter boundary
(the ethical wall + scoping) and what LEAVES the firm (`release_external`, which flows up a review
chain). Paralegals and assistants are NEVER read-only.

Lockstep: the role names here MUST match `src/api/schemas.py:ROLES` and the CHECK constraints in
`docs/migrations/012_firm_roles.sql`. The gate asserts it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.api.schemas import ROLES, ROLE_RANK


# ─────────────────────────────────────────
# CAPABILITIES — the verb set (§0.8 final capability list).
#   Each verb is one thing a member can attempt. Routes gate on these names, so they are a stable
#   contract; renaming one is a coordinated change across authz.py + every require_cap() call.
# ─────────────────────────────────────────
CAPABILITIES: frozenset[str] = frozenset({
    "create_vault",         # create a new matter/vault
    "ingest",               # upload + ingest documents into a vault
    "ask",                  # ask the agent / Brain a question over a vault
    "draft",                # generate a draft / redline
    "run_workflow",         # run a review-grid / workflow template
    "grids",                # build/read extraction grids
    "send_for_review",      # submit work UP the review chain (everyone on a matter — D0/D5)
    "release_external",     # send work OUTSIDE the firm (the one tightly-held verb — D5/ABA-512)
    "manage_matter_team",   # staff a matter (add team members to it — D3)
    "manage_members",       # firm-level: invite/role-change/remove members
    "view_billing",         # see the firm's billing
    "delete",               # delete a matter/vault/document
    "sign_certificate",     # sign a deliverable certificate (e-sign seam, F2i)
    "edit_playbooks",       # edit firm playbooks
    "publish_to_firm_brain",# publish a vetted answer/doc to the firm-wide brain (F6)
    "run_sentinel",         # run the covenant/obligation sentinel (F4)
    "override_abstain",     # override an agent abstain (the high-trust moment — F2d)
})


# ─────────────────────────────────────────
# ROLE_CAPS — the LOCKED §0.8 default capability matrix (productivity-first, D0).
#   dict[role, set[verb]]. These are EDITABLE per-firm defaults (a firm could tune them), but this
#   is the day-one matrix the gate pins. Each tier's line is documented inline so the exact line a
#   role draws is auditable here, not inferred.
# ─────────────────────────────────────────

# The full working toolkit — what EVERYONE on a matter can do (D0). The matter boundary + the
# review chain are the controls; this set is never the place we restrict a team member's work.
_WORKING_TOOLKIT: frozenset[str] = frozenset({
    "create_vault", "ingest", "ask", "draft", "run_workflow", "grids", "send_for_review",
})

# Firm-governance verbs — held by the partner tier (run the firm, not the matter work).
_FIRM_GOVERNANCE: frozenset[str] = frozenset({
    "manage_members", "view_billing", "edit_playbooks", "publish_to_firm_brain",
})

# The high-trust / outbound verbs — release work, sign, override, delete, run the sentinel,
# staff a matter. Partners hold the full set; senior associates hold a productivity subset.
_PARTNER_ELEVATED: frozenset[str] = frozenset({
    "release_external", "manage_matter_team", "delete", "sign_certificate",
    "override_abstain", "run_sentinel",
})

ROLE_CAPS: dict[str, frozenset[str]] = {
    # Managing Partner — runs the firm. Everything.
    "managing_partner": _WORKING_TOOLKIT | _FIRM_GOVERNANCE | _PARTNER_ELEVATED,
    # Senior Partner — same authority as MP for our purposes (the last-MP guard, not the matrix,
    # is what protects the firm from losing its final owner).
    "senior_partner":   _WORKING_TOOLKIT | _FIRM_GOVERNANCE | _PARTNER_ELEVATED,
    # Partner — full firm + matter authority (release, sign, override, staff, govern).
    "partner":          _WORKING_TOOLKIT | _FIRM_GOVERNANCE | _PARTNER_ELEVATED,
    # Senior Associate — the full working toolkit PLUS: may staff their matters (D3), may release
    # their own / reviewed-juniors' work (D5 — the matrix grants the verb; the review-chain in F2e
    # bounds it to their own + reviewed work), may delete their own matters. NOT firm-governance,
    # NOT sign/override (those stay with partners).
    "senior_associate": _WORKING_TOOLKIT | frozenset({"release_external", "manage_matter_team", "delete"}),
    # Associate — full working toolkit. release_external routes UP the chain (no direct verb).
    "associate":        _WORKING_TOOLKIT,
    # Paralegal — D0: FULL working toolkit (ask/draft/agent/grids/ingest). NOT read-only. Only
    # external release routes up the chain.
    "paralegal":        _WORKING_TOOLKIT,
    # Assistant — D0: FULL working toolkit, same as paralegal. NOT read-only.
    "assistant":        _WORKING_TOOLKIT,
    # Client (external) — deny-by-default (T8). Only view + ask, and only on the single vault
    # explicitly shared with them. Never ingest/draft/release/manage, never the console.
    "client":           frozenset({"ask"}),
    # Guest (external) — deny-by-default (T8). View only on the single shared vault. No verbs here
    # ("view" is not a capability — it is the absence of any mutating cap; a guest's read access is
    # governed by the shared-vault scope + RLS, not by a verb grant).
    "guest":            frozenset(),
}

# External roles get the smallest surface (T8): deny-by-default, console blocked, single shared
# vault only. Listed explicitly so callers (and the frontend route-guard) have one source of truth.
EXTERNAL_ROLES: frozenset[str] = frozenset({"client", "guest"})


@dataclass(frozen=True)
class Decision:
    """The result of an authorization check. `allow` is the binary outcome; `reason` is a
    human-readable explanation used as the 403 detail (so a denied user sees WHY) and the audit
    metadata (T10 — every deny-of-consequence is logged with its reason)."""
    allow: bool
    reason: str


@dataclass
class Membership:
    """A per-request, server-resolved membership (built by dependencies.resolve_membership — T7).

    PURE input to authorize(): no DB handle, no JWT. `caps` is derived from `role` via ROLE_CAPS at
    resolution time (so a firm's edited matrix flows through). `screens` and `delegations` are the
    seams for F2c (ethical walls) and F2e (delegation) — empty now, wired later; authorize() already
    honors them so those slices need no change here.
    """
    user_id: str
    firm_id: str
    role: str
    caps: frozenset[str] = field(default_factory=frozenset)
    # F2c seam: set of vault_ids this user is SCREENED OFF (ethical wall — a hard DENY). Empty now.
    screened_vault_ids: frozenset[str] = field(default_factory=frozenset)
    # F2d seam: set of verbs granted by an active, un-expired delegation (D6). Empty now.
    delegated_verbs: frozenset[str] = field(default_factory=frozenset)
    # F2e seam (D0/D3): the vaults this user is STAFFED on (a matter team member). This is a
    # PRODUCTIVITY grant, never a restriction — being on a matter grants the FULL working toolkit on
    # it (the member's role already carries the toolkit; see has_full_toolkit_on). The matter
    # boundary + the ethical wall are the only controls. Empty ⇒ byte-identical to pre-F2e.
    matter_vault_ids: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_external(self) -> bool:
        return self.role in EXTERNAL_ROLES

    def is_on_matter(self, vault_id: str | None) -> bool:
        """Is this member STAFFED on `vault_id` (D3)? The positive matter-access grant."""
        return bool(vault_id) and str(vault_id) in self.matter_vault_ids

    def has_full_toolkit_on(self, vault_id: str | None) -> frozenset[str]:
        """The capabilities this member can exercise ON A MATTER they are staffed on (D0 — the
        productivity assertion). It is their FULL working toolkit (ask/draft/run_workflow/grids/
        ingest/send_for_review/create_vault) intersected with what their role already grants — NEVER
        a read-only subset. Staffing is what grants ACCESS to the matter; it never strips a cap.
        Returns the empty set (no access) when the member is screened off the vault (the wall wins)
        or not staffed on it. PURE — no I/O."""
        if vault_id is not None and str(vault_id) in self.screened_vault_ids:
            return frozenset()                      # the ethical wall overrides matter access
        if not self.is_on_matter(vault_id):
            return frozenset()
        return self.caps & _WORKING_TOOLKIT


def caps_for_role(role: str) -> frozenset[str]:
    """The default capability set for a role (ROLE_CAPS). Unknown role → no caps (fail closed)."""
    return ROLE_CAPS.get(role, frozenset())


@dataclass(frozen=True)
class Scope:
    """The RESOLVED scope of an action (never the request body — T2/T3). Built server-side by the
    route from the target object's own firm_id/vault_id. authorize() trusts only this.

      - vault_id: the matter/vault the action targets (for the ethical-wall check).
      - firm_id:  the target object's firm, resolved server-side; asserted == the caller's firm.
      - target_role: for manage_members — the role being GRANTED (invite role / new role on a
        change). The self-escalation guard (T1) keys off this: you can't grant a role ≥ your own.
      - target_current_role: for manage_members — the target member's CURRENT role (None for a new
        invite). The last-MP guard keys off this: demoting/removing a sole managing_partner.
      - is_removal: True when the action removes the member (target_role is then their current role).
      - mp_count: for the last-MP guard — how many members currently hold managing_partner
        (db.count_firm_role), so orphaning the firm is blockable without a DB call inside authorize.
      - is_self: True when a manage_members action targets the caller themselves.
    """
    vault_id: Optional[str] = None
    firm_id: Optional[str] = None
    target_role: Optional[str] = None
    target_current_role: Optional[str] = None
    is_removal: bool = False
    mp_count: Optional[int] = None
    is_self: bool = False


def authorize(membership: Membership, verb: str, scope: Scope = Scope()) -> Decision:
    """The central decision. Fixed deny-overrides precedence (D2):

        (0) verb sanity / cross-firm scope          → deny (fail closed)
        (1) explicit screen DENY (ethical wall)     → deny  [F2c populates screens]
        (1.5) active delegation grant               → allow [F2e populates delegated_verbs]
        (2) role grants the verb on this scope      → allow
        (3) else                                    → deny

    PURE: no DB, no I/O. The caller resolves `membership` and `scope` server-side first. Returns a
    Decision with a human `reason` for every branch (403 detail + audit).
    """
    # (0a) unknown verb → fail closed. A typo'd require_cap must never silently allow.
    if verb not in CAPABILITIES:
        return Decision(False, f"Unknown capability '{verb}'.")

    # (0b) cross-firm confused-deputy (T2/T3): if a scope firm is resolved, it MUST be the caller's
    # own firm. The route resolves scope.firm_id from the TARGET object server-side; a body-supplied
    # firm_id never reaches here, and even if a wrong one did, this denies it.
    if scope.firm_id is not None and str(scope.firm_id) != str(membership.firm_id):
        return Decision(False, "Cross-firm action denied: the target belongs to another firm.")

    # (1) explicit screen DENY — the ethical wall. A negative authorization that an ALLOW can never
    # cross (deny-overrides). Checked BEFORE any grant. [F2c populates screened_vault_ids.]
    if scope.vault_id is not None and scope.vault_id in membership.screened_vault_ids:
        return Decision(False, "Access denied by an ethical wall (conflict screen) on this matter.")

    # manage_members carries the privilege-escalation guards (T1), applied before the grant check so
    # a role-holder who HAS manage_members still cannot escalate. Runs whenever the action names a
    # target (a grant via target_role, or a removal/demotion via target_current_role/is_removal).
    if verb == "manage_members" and (
        scope.target_role is not None or scope.target_current_role is not None or scope.is_removal
    ):
        guard = _manage_members_guard(membership, scope)
        if guard is not None:
            return guard

    # (1.5) active delegation grant (D6 — a PA acting for a senior). Time-boxed/revocable; the
    # resolver only includes verbs from a currently-valid delegation. [F2e populates delegated_verbs.]
    if verb in membership.delegated_verbs:
        return Decision(True, f"Allowed via an active delegation for '{verb}'.")

    # (2) role grant: does the caller's resolved capability set include this verb?
    if verb in membership.caps:
        return Decision(True, f"Allowed: role '{membership.role}' grants '{verb}'.")

    # (3) default deny.
    if membership.is_external:
        return Decision(False, "External users may only view (and, for clients, ask on) the shared matter.")
    return Decision(False, f"Your role '{membership.role}' does not have permission to '{verb}'.")


def _manage_members_guard(membership: Membership, scope: Scope) -> Optional[Decision]:
    """T1 privilege-escalation guards for a manage_members action. Returns a DENY Decision if the
    action is forbidden, or None if it clears the guards (the normal grant check then applies).

      - Self-escalation: a member can never GRANT a role at or ABOVE their own rank (lower rank
        index = more senior). This blocks both raising yourself and minting a peer who could then
        act against you. (An MP may grant a peer MP — the normal add-a-co-owner flow.)
      - Last-MP guard: demoting OR removing a managing_partner is denied when they are the last one
        (mp_count <= 1) — the firm must never be left without an owner. Keyed on the target's
        CURRENT role, so a demotion (target_role != MP) is caught too.
    """
    actor_rank = ROLE_RANK.get(membership.role, len(ROLES))

    # Self-escalation: only relevant when GRANTING a role (an invite or a role-change).
    if scope.target_role is not None:
        if scope.target_role not in ROLE_RANK:
            return Decision(False, f"Unknown target role '{scope.target_role}'.")
        target_rank = ROLE_RANK[scope.target_role]
        # target_rank < actor_rank → more senior than the actor: always denied.
        if target_rank < actor_rank:
            return Decision(False, "You cannot grant a role more senior than your own.")
        # target_rank == actor_rank → a peer: allowed ONLY for managing_partner (co-owner).
        if target_rank == actor_rank and membership.role != "managing_partner":
            return Decision(False, "You cannot grant a role at your own level of seniority.")

    # Last-MP guard: would this demote/remove the firm's final managing_partner? The target is
    # currently an MP, the action either removes them or changes them to a non-MP role, and they
    # are the only MP left.
    is_currently_mp = scope.target_current_role == "managing_partner"
    demotes_or_removes = scope.is_removal or (
        scope.target_role is not None and scope.target_role != "managing_partner"
    )
    if is_currently_mp and demotes_or_removes and (scope.mp_count or 0) <= 1:
        return Decision(False, "Cannot demote or remove the firm's last Managing Partner.")

    return None
