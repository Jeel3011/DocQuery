"""F2e — the REVIEW CHAIN builder (pure, offline, $0).

The chain of command a piece of work flows UP for review and external release (D5). This module is
PURE — no DB, no network, no LLM, no FastAPI — exactly like `authz.py`. It takes the matter's
members (each with a role), the submitter, and an OPTIONAL custom chain, and returns the ordered
reviewer list (`chain`) + the first `current_owner`. The DB writers + the route enforce + persist
it; keeping the builder pure is what makes the gate `eval/test_review_chain.py` $0/offline.

THE TWO ROUTING MODES (D5):
  - DEFAULT = up by RANK on the matter (zero setup). The work flows from the submitter UP through
    the matter's more-senior members, ending at the most-senior (a partner who can release_external).
    "Up by rank" = ascending seniority: among the matter's members MORE SENIOR than the submitter,
    ordered from least-senior-above to most-senior (one step at a time up the reporting line — the
    Cflow "sequential routing by hierarchy" pattern). ROLE_RANK is 0 = most senior, so "more senior
    than the submitter" = a strictly SMALLER rank index, and we walk those from largest index (just
    above the submitter) to smallest (the top).
  - CUSTOM per matter — a matter lead sets `matter_review_config.chain` (an explicit ordered list of
    reviewer user_ids) for a big/complex matter (e.g. Associate → Senior Associate → Compliance →
    Partner). When present, the chain routes in that EXACT defined order (the builder does not
    re-sort it) — the submitter is dropped if they appear (you don't review your own work), and an
    empty result falls back to the rank default so a request is never ownerless.

THE ANTI-STALL INVARIANT (D5 — the #1 documented review failure is "work stalls because no one owns
the next step"): the builder's output is consumed so that EVERY review_request always names a
`current_owner`. `first_owner()` returns the first reviewer in the chain, or — if there is no one
above the submitter to review (a solo senior / a flat boutique) — the most-senior member of the
matter (never the submitter, never None when the matter has ≥1 other member). The route guarantees a
non-null owner; this module gives it the right person.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.api.schemas import ROLE_RANK, ROLES


# The partner tier — the roles that may RELEASE_EXTERNAL at the end of the chain (the chain's
# terminal step). Mirrors authz.ROLE_CAPS (release_external is held by the partner tier + senior
# associate; but the chain's *external release* end-point is a PARTNER — D5/ABA-512). A senior
# associate may release their own / reviewed-junior work per the matrix, but the chain TOP is a
# partner. Kept here as the chain's notion of "can close the chain externally".
_RELEASE_TIER: frozenset[str] = frozenset({"managing_partner", "senior_partner", "partner"})


def _rank(role: Optional[str]) -> int:
    """Rank index for a role (0 = most senior). Unknown role → most-junior (len(ROLES)), so a
    stray/legacy role never accidentally sorts to the top of a review chain."""
    return ROLE_RANK.get(role or "", len(ROLES))


@dataclass(frozen=True)
class Member:
    """A matter's staffed member, for chain-building. PURE input — user_id + their firm role."""
    user_id: str
    role: str


def build_chain(
    members: list[Member],
    submitted_by: str,
    custom_chain: Optional[list[str]] = None,
) -> list[str]:
    """Build the ordered reviewer list a piece of work flows UP for review (D5).

    members      — the matter's staffed team (each with a role). The submitter is typically among
                   them; they are never put in their own review chain.
    submitted_by — the user_id who sent the work up.
    custom_chain — an optional explicit ordered list of reviewer user_ids (matter_review_config). When
                   non-empty, routes in THAT defined order (the submitter removed); else the default.

    Returns a list of reviewer user_ids in the order the work visits them. May be empty only when the
    submitter is the matter's most-senior member and there is no one above them — the caller (the
    route) then assigns the most-senior member as the owner so a request is never ownerless.
    """
    if custom_chain:
        # CUSTOM: route in the matter lead's EXACT defined order. Drop the submitter (no self-review)
        # and any duplicates while preserving order. We do NOT re-sort by rank — the order IS the spec.
        seen: set = set()
        ordered: list[str] = []
        for uid in custom_chain:
            u = str(uid)
            if u == str(submitted_by) or u in seen:
                continue
            seen.add(u)
            ordered.append(u)
        if ordered:
            return ordered
        # A custom chain that resolves to nobody (e.g. only the submitter listed) falls through to
        # the rank default — never leave a request without a chain.

    # DEFAULT: up by rank among the members MORE SENIOR than the submitter. "More senior" = a
    # strictly smaller rank index than the submitter's. We walk them from least-senior-above
    # (largest index below the submitter's) to most-senior (smallest index) — one step up at a time.
    sub_rank = _submitter_rank(members, submitted_by)
    seniors = [m for m in members
               if str(m.user_id) != str(submitted_by) and _rank(m.role) < sub_rank]
    # Ascending seniority = descending rank index: largest index (just above) first, smallest (top) last.
    seniors.sort(key=lambda m: _rank(m.role), reverse=True)
    return [str(m.user_id) for m in seniors]


def first_owner(
    members: list[Member],
    submitted_by: str,
    custom_chain: Optional[list[str]] = None,
) -> Optional[str]:
    """Who owns the FIRST review step (the anti-stall current_owner at submit). The first reviewer in
    the chain; or — when no one is above the submitter (a solo senior / flat matter) — the matter's
    MOST-SENIOR OTHER member, so the request is never ownerless. Returns None only when the matter has
    no member other than the submitter (the route then keeps the submitter as a degenerate owner — a
    one-person matter can't stall on someone else)."""
    chain = build_chain(members, submitted_by, custom_chain)
    if chain:
        return chain[0]
    # No one above the submitter: hand it to the most-senior OTHER member (never the submitter), so
    # the chain still has an owner. None only if the submitter is the matter's only member.
    others = [m for m in members if str(m.user_id) != str(submitted_by)]
    if not others:
        return None
    others.sort(key=lambda m: _rank(m.role))   # most senior first (smallest rank index)
    return str(others[0].user_id)


def escalation_owner(
    submitted_by: str,
    vault_owner: Optional[str] = None,
    firm_members: Optional[list[Member]] = None,
) -> Optional[str]:
    """The anti-stall fallback owner when the MATTER TEAM has no senior to review the work (D5).

    This is the locked decision for the "lone junior on a matter" case: when `first_owner` finds no
    reviewer on the matter team (e.g. only the submitting paralegal is staffed), the work must NOT
    loop back to the submitter (that is a silent stall — the submitter "reviewing" their own work).
    Route UP, in order:
      1. the VAULT OWNER (the partner/MP who owns the matter) — they staffed it, the work is theirs;
      2. else the firm's MOST-SENIOR partner-tier member (the firm's backstop reviewer);
      3. else None (no one but the submitter exists firm-wide — a true one-person firm).
    The submitter is NEVER returned here — that is the bug this function exists to prevent.
    """
    sub = str(submitted_by)
    # 1. The vault owner — unless they ARE the submitter (an owner submitting their own work has no
    #    one above them on this matter; that legitimately falls through to the firm backstop / self).
    if vault_owner and str(vault_owner) != sub:
        return str(vault_owner)
    # 2. The firm's most-senior partner-tier member who is not the submitter.
    seniors = [m for m in (firm_members or [])
               if str(m.user_id) != sub and m.role in _RELEASE_TIER]
    if seniors:
        seniors.sort(key=lambda m: _rank(m.role))   # most senior first (smallest rank index)
        return str(seniors[0].user_id)
    # 3. No senior anywhere in the firm other than the submitter.
    return None


def next_owner(chain: list[str], current_owner: str) -> Optional[str]:
    """The next reviewer after `current_owner` in an already-built chain (used on APPROVE to advance
    one step up). Returns None when `current_owner` is the LAST in the chain — the chain is cleared
    and the request moves to APPROVED (the partner who can release_external owns the terminal step)."""
    chain = [str(c) for c in (chain or [])]
    cur = str(current_owner)
    if cur not in chain:
        return None
    i = chain.index(cur)
    if i + 1 < len(chain):
        return chain[i + 1]
    return None


def chain_end_owner(members: list[Member], chain: list[str]) -> Optional[str]:
    """The owner of the APPROVED (terminal-internal) state — the member who may RELEASE_EXTERNAL at
    the chain's end. The most-senior member of the matter who is in the release tier (a partner). The
    chain ends at a partner by construction (the rank default walks up to the top); this resolves the
    release owner explicitly so the APPROVED state always names a partner who can close the chain.
    Falls back to the last chain member, then the most-senior member, so APPROVED is never ownerless."""
    # Prefer the most-senior partner-tier member of the matter.
    partners = [m for m in members if m.role in _RELEASE_TIER]
    if partners:
        partners.sort(key=lambda m: _rank(m.role))   # most senior first
        return str(partners[0].user_id)
    # No partner on the matter: the last person in the chain owns the (internal) approval; if the
    # chain is empty, the most-senior member. Never None when the matter has any member.
    if chain:
        return str(chain[-1])
    if members:
        ms = sorted(members, key=lambda m: _rank(m.role))
        return str(ms[0].user_id)
    return None


def can_release_external(role: Optional[str]) -> bool:
    """Does this role sit in the partner tier that may RELEASE_EXTERNAL at the chain's end (D5)?
    The chain's terminal external-release step is a PARTNER cap — a junior/associate's release routes
    UP the chain, it never closes it. (authz.ROLE_CAPS also grants release_external to senior
    associates for their own/reviewed work; this is specifically the chain-end external-release tier.)"""
    return (role or "") in _RELEASE_TIER


def _submitter_rank(members: list[Member], submitted_by: str) -> int:
    """The submitter's rank index among the matter's members. If the submitter is not in the member
    list (shouldn't happen — the route staffs them or they own the vault), treat them as most-junior
    so the whole matter is "above" them and the chain routes up to the top (fail-safe, never empty)."""
    for m in members:
        if str(m.user_id) == str(submitted_by):
            return _rank(m.role)
    return len(ROLES)
