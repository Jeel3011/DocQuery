"""Practice-aware self-config (F1c — plans/F1_VAULT_PLAN.md §1, the "vault is the router"
differentiator).

Once a vault is TYPED (F1a gave it a `matter_kind`), the type should *do* something. This
module is the static, **$0, no-LLM** map that turns a matter_kind into a starting
configuration:

  - **review-grid columns** — the clause/term set a reviewer of THIS kind of matter expects
    (an M&A SPA grid ≠ a lending facility grid ≠ a litigation pleadings grid). These seed the
    review grid (`agent_core/review_grid.py` GridColumn shape) so the user opens a typed vault
    and the right columns are already there — overridable, never forced.
  - **KB scope** — which shared Knowledge Base sources chip in by default (the `sources`
    vocabulary the agent-core route already gates on: {"vault","statutes","caselaw"}; F1c only
    sets the *default*, the user's chips still win).
  - **flagship pin** — which flagship surface this kind foregrounds (§F1c: lending→F5 Covenant
    Cockpit, litigation→F7/F8, obligation-heavy→F4 Sentinel). The vault *is* the router.

DISCIPLINE (the moat held):
  This is a STATIC table. No model call, no per-question branching — a small recombinable
  basis (the GRAND_PLAN "skills/memory, not handlers" rule). It produces DEFAULTS the user
  overrides; it never makes a legal claim, never touches a document, never gates retrieval by
  itself (the KB scope it returns is fed into the EXISTING server-side chip gate, not a new
  path). An unknown / None matter_kind falls back to a neutral generic template — never raises.

The `matter_kind` values are the 9 from F1a (`schemas.MATTER_KINDS`), kept in lockstep here;
the F1c gate asserts every kind has a template and the keys match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# The KB-source vocabulary the agent-core route already understands (G8.7 chips). F1c only
# chooses the DEFAULT subset per matter_kind; the route still gates the backend from the
# user's actual chips, so a wrong default here can never widen scope past what the user picks.
KB_VAULT = "vault"          # the user's own documents (always on for every kind)
KB_STATUTES = "statutes"    # Indian statutes / regulations / circulars (the legal KB)
KB_CASELAW = "caselaw"      # judgments / case law

# Flagship surface ids (UNIFIED_FEATURES_PLAN F4–F8). A "pin" is a hint to the UI about which
# surface to foreground for this kind of matter — NOT a capability grant (F2 owns who-can-do).
# These features are largely not built yet; the pin is forward-declared so the vault router is
# correct the day each flagship lands. "review_grid" is the always-available default surface.
FLAGSHIP_REVIEW_GRID = "review_grid"        # the law-first head-to-head (built)
FLAGSHIP_COVENANT_COCKPIT = "covenant_cockpit"   # F5 — finance/lending numeric compliance
FLAGSHIP_OBLIGATION_SENTINEL = "obligation_sentinel"  # F4 — clause-traced duties/dates
FLAGSHIP_ARGUMENT_ENGINE = "argument_engine"     # F7 — litigation adversarial reasoning
FLAGSHIP_CASE_FILE = "case_file"            # F8 — litigation two-sided living record


@dataclass(frozen=True)
class TemplateColumn:
    """A seed review-grid column. Mirrors `agent_core/review_grid.GridColumn` field-for-field
    (key/label/prompt/kind/risk_rubric) so a template column maps 1:1 onto a GridColumn the
    review-grid engine runs — no translation, no second shape. `kind` is "clause" (the legal
    default) or "numeric"; the review grid validates it the same way it validates a user column.
    """
    key: str
    label: str
    prompt: str
    kind: str = "clause"
    risk_rubric: Optional[str] = None


@dataclass(frozen=True)
class PracticeTemplate:
    """The starting config for one matter_kind. Pure data — produced by `template_for`, served
    by the route, rendered by the create dialog. Every field is a DEFAULT the user overrides.
    """
    matter_kind: Optional[str]
    label: str                                  # human name for the kind (display)
    grid_columns: List[TemplateColumn] = field(default_factory=list)
    kb_scope: List[str] = field(default_factory=list)   # subset of {vault,statutes,caselaw}
    flagship: str = FLAGSHIP_REVIEW_GRID        # which surface to foreground
    summary: Optional[str] = None               # one-line "why this template" for the UI

    def to_dict(self) -> Dict:
        """Serialize for the JSON response / the UI. Columns flatten to the GridColumn shape so
        the frontend can hand them straight to a review-grid request."""
        return {
            "matter_kind": self.matter_kind,
            "label": self.label,
            "grid_columns": [
                {
                    "key": c.key,
                    "label": c.label,
                    "prompt": c.prompt,
                    "kind": c.kind,
                    "risk_rubric": c.risk_rubric,
                }
                for c in self.grid_columns
            ],
            "kb_scope": list(self.kb_scope),
            "flagship": self.flagship,
            "summary": self.summary,
        }


# ──────────────────────────────────────────────────────────────────────────────
# The static per-matter_kind table. Each column's `prompt` is the cite-or-abstain extraction
# instruction the per-cell agent runs (review_grid.py contract: "find X, quote it exactly, or
# abstain"). The risk_rubric, when set, classifies the finding (standard / non-standard). KB
# scope: legal matters default statutes on; finance-shaped matters (lending) keep it on (Indian
# SEBI/RBI circulars) but the user can pare it. Case law (caselaw) defaults ON only for the
# adversarial kinds (litigation/arbitration) where precedent is load-bearing.
# ──────────────────────────────────────────────────────────────────────────────

def _generic_columns() -> List[TemplateColumn]:
    """A neutral starter set for an untyped / unknown matter — the safe fall-back."""
    return [
        TemplateColumn(
            key="parties",
            label="Parties",
            prompt="Identify the named parties to this document. Quote the exact text naming them, or abstain.",
        ),
        TemplateColumn(
            key="effective_date",
            label="Effective date",
            prompt="Find the document's effective / commencement date. Quote the exact clause, or abstain.",
        ),
        TemplateColumn(
            key="governing_law",
            label="Governing law",
            prompt="Find the governing-law clause. Quote it exactly, or abstain.",
        ),
        TemplateColumn(
            key="termination",
            label="Termination",
            prompt="Find the termination clause and any notice period. Quote the exact text, or abstain.",
        ),
    ]


_TEMPLATES: Dict[str, PracticeTemplate] = {
    "litigation": PracticeTemplate(
        matter_kind="litigation",
        label="Litigation",
        summary="Pleadings and case documents — claims, reliefs, limitation, and forum.",
        grid_columns=[
            TemplateColumn("cause_of_action", "Cause of action",
                           "Identify the cause(s) of action or grounds pleaded. Quote the exact text, or abstain."),
            TemplateColumn("relief_sought", "Relief sought",
                           "Find the relief / prayer sought. Quote it exactly, or abstain."),
            TemplateColumn("forum_jurisdiction", "Forum / jurisdiction",
                           "Find the court or forum and the jurisdiction clause. Quote it exactly, or abstain."),
            TemplateColumn("limitation", "Limitation",
                           "Find any limitation period or date that bears on whether the claim is time-barred. Quote it, or abstain.",
                           risk_rubric="non_standard if the claim appears outside the limitation period; else standard"),
            TemplateColumn("key_dates", "Key dates",
                           "Identify the key procedural dates (filing, hearing, order). Quote the exact text, or abstain."),
        ],
        kb_scope=[KB_VAULT, KB_STATUTES, KB_CASELAW],
        flagship=FLAGSHIP_ARGUMENT_ENGINE,
    ),
    "m&a": PracticeTemplate(
        matter_kind="m&a",
        label="M&A",
        summary="Acquisition and transaction agreements — consideration, conditions, and protections.",
        grid_columns=[
            TemplateColumn("parties", "Parties",
                           "Identify the buyer, seller, and target. Quote the exact text naming them, or abstain."),
            TemplateColumn("consideration", "Consideration",
                           "Find the purchase price / consideration and how it is paid. Quote the exact clause, or abstain.",
                           kind="numeric"),
            TemplateColumn("conditions_precedent", "Conditions precedent",
                           "Find the conditions precedent to closing. Quote the exact text, or abstain."),
            TemplateColumn("reps_warranties", "Reps & warranties",
                           "Find the representations and warranties and any cap on liability for breach. Quote the exact text, or abstain."),
            TemplateColumn("indemnity", "Indemnity",
                           "Find the indemnity clause and any cap. Quote the exact text, or abstain."),
            TemplateColumn("change_of_control", "Change of control",
                           "Find any change-of-control clause. Quote the exact text, or abstain."),
            TemplateColumn("governing_law", "Governing law",
                           "Find the governing-law clause. Quote it exactly, or abstain.",
                           risk_rubric="standard if New York / Delaware / England / India; else non_standard"),
        ],
        kb_scope=[KB_VAULT, KB_STATUTES],
        flagship=FLAGSHIP_REVIEW_GRID,
    ),
    "lending": PracticeTemplate(
        matter_kind="lending",
        label="Lending",
        summary="Facility and security documents — the covenant set the cockpit monitors.",
        grid_columns=[
            TemplateColumn("facility_amount", "Facility amount",
                           "Find the facility / commitment amount. Quote the exact figure and clause, or abstain.",
                           kind="numeric"),
            TemplateColumn("interest_rate", "Interest rate",
                           "Find the interest rate / margin. Quote the exact clause, or abstain.",
                           kind="numeric"),
            TemplateColumn("financial_covenants", "Financial covenants",
                           "Find the financial covenants (e.g. leverage, DSCR, interest cover) and their thresholds. Quote the exact text, or abstain."),
            TemplateColumn("events_of_default", "Events of default",
                           "Find the events of default. Quote the exact text, or abstain."),
            TemplateColumn("security", "Security",
                           "Find the security / collateral granted. Quote the exact text, or abstain."),
            TemplateColumn("repayment", "Repayment",
                           "Find the repayment schedule and final maturity date. Quote the exact text, or abstain."),
        ],
        kb_scope=[KB_VAULT, KB_STATUTES],
        flagship=FLAGSHIP_COVENANT_COCKPIT,
    ),
    "arbitration": PracticeTemplate(
        matter_kind="arbitration",
        label="Arbitration",
        summary="Arbitration agreements and awards — seat, rules, tribunal, and scope.",
        grid_columns=[
            TemplateColumn("arbitration_clause", "Arbitration clause",
                           "Find the arbitration agreement / clause. Quote it exactly, or abstain."),
            TemplateColumn("seat_venue", "Seat / venue",
                           "Find the seat and venue of arbitration. Quote the exact text, or abstain."),
            TemplateColumn("rules", "Rules",
                           "Find the arbitral rules that govern (e.g. SIAC, ICC, UNCITRAL). Quote the exact text, or abstain."),
            TemplateColumn("tribunal", "Tribunal",
                           "Find how the tribunal is constituted (number / appointment of arbitrators). Quote the exact text, or abstain."),
            TemplateColumn("governing_law", "Governing law",
                           "Find the governing-law clause (of the contract and of the arbitration agreement, if distinct). Quote it exactly, or abstain."),
        ],
        kb_scope=[KB_VAULT, KB_STATUTES, KB_CASELAW],
        flagship=FLAGSHIP_ARGUMENT_ENGINE,
    ),
    "ip": PracticeTemplate(
        matter_kind="ip",
        label="Intellectual property",
        summary="IP agreements — ownership, licence scope, and field of use.",
        grid_columns=[
            TemplateColumn("ip_owner", "IP owner",
                           "Identify who owns the intellectual property. Quote the exact text, or abstain."),
            TemplateColumn("licence_grant", "Licence grant",
                           "Find the licence grant and whether it is exclusive. Quote the exact text, or abstain.",
                           risk_rubric="non_standard if the grant is perpetual, irrevocable, or royalty-free; else standard"),
            TemplateColumn("field_of_use", "Field of use",
                           "Find the field-of-use / territory restriction. Quote the exact text, or abstain."),
            TemplateColumn("assignment", "Assignment",
                           "Find any assignment of IP. Quote the exact text, or abstain."),
            TemplateColumn("royalties", "Royalties",
                           "Find the royalty / fee terms. Quote the exact clause, or abstain.", kind="numeric"),
        ],
        kb_scope=[KB_VAULT, KB_STATUTES],
        flagship=FLAGSHIP_REVIEW_GRID,
    ),
    "regulatory": PracticeTemplate(
        matter_kind="regulatory",
        label="Regulatory",
        summary="Filings and approvals — obligations, deadlines, and the regulator.",
        grid_columns=[
            TemplateColumn("regulator", "Regulator",
                           "Identify the regulator / authority involved. Quote the exact text, or abstain."),
            TemplateColumn("obligation", "Obligation",
                           "Find the regulatory obligation(s) imposed. Quote the exact text, or abstain."),
            TemplateColumn("deadline", "Deadline",
                           "Find any filing / compliance deadline. Quote the exact text and date, or abstain.",
                           risk_rubric="non_standard if a deadline appears to have passed; else standard"),
            TemplateColumn("penalty", "Penalty",
                           "Find any penalty for non-compliance. Quote the exact text, or abstain."),
        ],
        kb_scope=[KB_VAULT, KB_STATUTES],
        flagship=FLAGSHIP_OBLIGATION_SENTINEL,
    ),
    "employment": PracticeTemplate(
        matter_kind="employment",
        label="Employment",
        summary="Employment and HR agreements — terms, restraints, and termination.",
        grid_columns=[
            TemplateColumn("role_compensation", "Role & compensation",
                           "Find the role and the compensation. Quote the exact clause, or abstain."),
            TemplateColumn("term", "Term",
                           "Find the term of employment (and any probation). Quote the exact text, or abstain."),
            TemplateColumn("non_compete", "Non-compete",
                           "Find any non-compete / non-solicit restraint and its duration. Quote the exact text, or abstain.",
                           risk_rubric="non_standard if the restraint is unusually long or broad; else standard"),
            TemplateColumn("termination", "Termination",
                           "Find the termination clause and notice period. Quote the exact text, or abstain."),
            TemplateColumn("confidentiality", "Confidentiality",
                           "Find the confidentiality / IP-assignment clause. Quote the exact text, or abstain."),
        ],
        kb_scope=[KB_VAULT, KB_STATUTES],
        flagship=FLAGSHIP_OBLIGATION_SENTINEL,
    ),
    "advisory": PracticeTemplate(
        matter_kind="advisory",
        label="Advisory",
        summary="General advisory documents — a broad clause starter set.",
        grid_columns=_generic_columns(),
        kb_scope=[KB_VAULT, KB_STATUTES],
        flagship=FLAGSHIP_REVIEW_GRID,
    ),
    "compliance": PracticeTemplate(
        matter_kind="compliance",
        label="Compliance",
        summary="Policies and compliance documents — obligations and the controls that meet them.",
        grid_columns=[
            TemplateColumn("obligation", "Obligation",
                           "Find the compliance obligation(s) imposed. Quote the exact text, or abstain."),
            TemplateColumn("control", "Control",
                           "Find the control / measure that meets the obligation. Quote the exact text, or abstain."),
            TemplateColumn("owner", "Owner",
                           "Identify who is responsible for the obligation. Quote the exact text, or abstain."),
            TemplateColumn("review_date", "Review date",
                           "Find any review / renewal date. Quote the exact text and date, or abstain."),
        ],
        kb_scope=[KB_VAULT, KB_STATUTES],
        flagship=FLAGSHIP_OBLIGATION_SENTINEL,
    ),
}

# The neutral fall-back template (None / unknown matter_kind). NOT in _TEMPLATES so the gate
# can assert _TEMPLATES.keys() == MATTER_KINDS exactly.
_GENERIC_TEMPLATE = PracticeTemplate(
    matter_kind=None,
    label="General",
    summary="An untyped vault — a neutral starter set. Pick a matter type to tailor it.",
    grid_columns=_generic_columns(),
    kb_scope=[KB_VAULT, KB_STATUTES],
    flagship=FLAGSHIP_REVIEW_GRID,
)


def template_for(matter_kind: Optional[str]) -> PracticeTemplate:
    """Return the practice template for a matter_kind. $0, no LLM, never raises.

    An unknown / None kind returns the neutral generic template (the safe fall-back) — so a
    legacy/untyped vault still gets a sensible starting config without any special-casing at
    the call site.
    """
    if not matter_kind:
        return _GENERIC_TEMPLATE
    return _TEMPLATES.get(matter_kind.lower(), _GENERIC_TEMPLATE)


def all_kinds() -> List[str]:
    """The matter kinds that have a template — used by the gate to assert lockstep with
    `schemas.MATTER_KINDS`."""
    return list(_TEMPLATES.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Matter-kind INFERENCE (F1c, optional) — $0, no LLM. Extends the structural
# `classify_document` signal (financial_filing|legal_contract|mixed|generic, from
# data_ingestion.py — G1) with cheap filename keyword hints to SUGGEST a matter_kind. The
# user always overrides; a suggestion is a starting point, never an assignment. We deliberately
# do NOT build an LLM classifier — the structural class + filename keywords are enough to seed
# the picker, and they cost nothing.
# ──────────────────────────────────────────────────────────────────────────────

# Filename keyword → matter_kind. First match wins, in this precedence order (more specific
# transaction/matter words before generic ones). Hints are matched as WHOLE WORDS / WHOLE
# PHRASES (a filename is tokenized on non-alphanumerics; a single-word hint must be a token, a
# multi-word hint must appear as consecutive tokens) — NEVER a raw substring, so "suit" can't
# match "suitable", "sha" can't match "sharma", "vs"/"v" can't match inside other words. These
# are HINTS, not authority — a doc that matches nothing yields None (no suggestion, never a
# wrong guess). Multi-word hints are written space-separated; abbreviations are single tokens.
_KIND_NAME_HINTS: List[tuple] = [
    ("litigation", ("complaint", "petition", "plaint", "writ", "pleading", "affidavit",
                    "summons", "litigation", "lawsuit", "vs", "versus")),
    ("arbitration", ("arbitration", "arbitral", "award", "siac", "icc", "uncitral", "lcia")),
    ("m&a", ("merger", "acquisition", "share purchase", "spa", "asset purchase", "apa",
             "stock purchase", "ma", "scheme of arrangement", "shareholders agreement", "sha")),
    ("lending", ("facility", "loan", "credit agreement", "debenture", "security",
                 "guarantee", "mortgage", "hypothecation", "promissory")),
    ("ip", ("license", "licence", "licensing", "trademark", "patent", "copyright",
            "royalty", "ip assignment", "technology transfer", "franchise")),
    ("employment", ("employment", "offer letter", "appointment", "non compete",
                    "noncompete", "hr policy", "severance", "consultancy")),
    ("regulatory", ("regulator", "sebi", "rbi", "filing", "circular", "approval",
                    "notification", "regulatory")),
    ("compliance", ("compliance", "policy", "code of conduct", "kyc", "aml",
                    "data protection", "privacy policy")),
    # "advisory" is a deliberate non-hint — it's the catch-all the user picks, not one we guess.
]

# How the structural class constrains the suggestion. A FINANCIAL filing is almost never a
# law matter we type here; a LEGAL contract is the natural home for the keyword hints; mixed /
# generic stay open to the hints but never force a kind on their own.
_LEGALISH_CLASSES = {"legal_contract", "mixed", "generic"}


def suggest_matter_kind(doc_type: Optional[str], filename: Optional[str] = None) -> Optional[str]:
    """Suggest a `matter_kind` from the structural doc class + filename keywords. $0, no LLM.

    `doc_type` is the `classify_document` output (financial_filing|legal_contract|mixed|generic).
    Returns a kind from `MATTER_KINDS` or None (no confident suggestion → the picker stays
    unselected). Never raises. The user always overrides — this only seeds the default.

    Rules:
      - A financial filing yields no law-matter suggestion (None) — its kind, if any, is the
        user's call; we don't mis-type a 10-K as a "matter".
      - Otherwise the first kind with a WHOLE-WORD / WHOLE-PHRASE hit wins (precedence order in
        `_KIND_NAME_HINTS`). Matching is on tokens, never substrings (no false positives).
      - No keyword match → None (never a wrong guess).
    """
    try:
        if doc_type == "financial_filing":
            return None
        if doc_type is not None and doc_type not in _LEGALISH_CLASSES:
            return None
        name = (filename or "").lower()
        if not name:
            return None
        # Tokenize the filename into a list of word tokens (order preserved for phrase hits).
        tokens = [t for t in re.split(r"[^a-z0-9]+", name) if t]
        if not tokens:
            return None
        token_set = set(tokens)
        joined = " ".join(tokens)  # normalized single-spaced form for multi-word phrase hits
        for kind, hints in _KIND_NAME_HINTS:
            for h in hints:
                if " " in h:
                    # multi-word phrase: match as consecutive tokens (bounded by word edges)
                    if f" {h} " in f" {joined} ":
                        return kind
                elif h in token_set:
                    # single word: must be a whole token, never a substring
                    return kind
        return None
    except Exception:  # noqa: BLE001 — inference never breaks a caller
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Metadata-only CONFLICT SCAN (F1c) — the ethical-wall screen. THE WALL: this touches ZERO
# document content. It compares only PARTY NAMES (already-typed metadata on the
# `collections.parties` JSONB) across a firm's other matters. The matcher below is a pure
# function over party-name strings so it is testable offline ($0) and provably content-free —
# its only inputs are party-name strings and matter labels, never a chunk, never a file.
#
# An ADVERSE collision is the screen we care about: a party we represent on THIS matter (role
# "client"/"self"/"company"/etc.) appears as an OPPOSING party on another matter in the firm,
# or vice-versa. We also surface a softer "same party, other matter" note (the firm already
# acts for/against this name somewhere) so the admin can judge — but only the adverse case is
# flagged at the higher tone. Non-blocking by construction: this returns FINDINGS, the route
# turns them into a banner; nothing here blocks the create.
# ──────────────────────────────────────────────────────────────────────────────

# Role buckets — which side of a matter a party sits on. Roles are free text (MatterParty.role
# is optional), so we classify by WHOLE-WORD tokens (never substrings — a substring match would
# misfire, e.g. "lender" inside "lender-counterparty", or "our"/"we" inside unrelated words).
# An unbucketed role is "neutral": we do NOT guess a side, so it can never produce a false
# "adverse". A role token in BOTH sets (none today, but a guard against future edits) resolves
# to "neutral" too — ambiguity must never upgrade to adverse. Single-token keywords only; a
# multi-word role like "opposing party" still matches because "opposing" is one of its tokens.
_OUR_SIDE_ROLE_TOKENS = frozenset({
    "client", "self", "company", "borrower", "buyer", "purchaser",
    "licensor", "employer", "applicant", "petitioner", "plaintiff",
    "claimant", "lender", "principal",
})
_OTHER_SIDE_ROLE_TOKENS = frozenset({
    "opposing", "opponent", "adverse", "counterparty", "defendant",
    "respondent", "seller", "vendor", "guarantor", "licensee", "employee",
})


def _normalize_party(name: str) -> str:
    """Normalize a party name for comparison: lowercase, collapse whitespace, drop common
    corporate suffixes/punctuation so 'Acme Corp.' and 'ACME corporation' match. Conservative —
    we'd rather miss a fuzzy match than fabricate one (a false conflict screen is its own harm)."""
    s = (name or "").lower().strip()
    # strip trailing punctuation and collapse internal whitespace
    s = " ".join(s.replace(",", " ").replace(".", " ").split())
    for suffix in (" private limited", " pvt ltd", " pvt limited", " ltd", " limited",
                   " llp", " llc", " inc", " corp", " corporation", " co", " plc",
                   " gmbh", " sa", " ag", " bv"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    return s


# Split a role into lowercase word tokens ("opposing party" → {"opposing","party"};
# "lender-counterparty" → {"lender","counterparty"}). Non-alphanumerics are separators.
def _role_tokens(role: Optional[str]) -> set:
    return set(re.split(r"[^a-z0-9]+", (role or "").lower())) - {""}


def _side_of(role: Optional[str]) -> str:
    """Classify a party role into 'our' / 'other' / 'neutral' by WHOLE-WORD token match.

    'neutral' when the role is empty, unrecognized, OR ambiguous (tokens in both sets) — so an
    unknown or contradictory role never produces a false 'adverse'. This is the conservative
    direction: a missed adverse downgrades to a still-surfaced same-party note (the name match
    is always reported); it never silently drops the finding."""
    toks = _role_tokens(role)
    if not toks:
        return "neutral"
    our = bool(toks & _OUR_SIDE_ROLE_TOKENS)
    other = bool(toks & _OTHER_SIDE_ROLE_TOKENS)
    if our and not other:
        return "our"
    if other and not our:
        return "other"
    return "neutral"  # none matched, or both (ambiguous) → don't guess a side


def find_conflicts(new_parties: List[Dict], existing_matters: List[Dict]) -> List[Dict]:
    """Compare the new matter's parties against other matters' parties — metadata only.

    `new_parties`     : the parties on the matter being created — list of {name, role?}.
    `existing_matters`: other matters in the firm — list of
                        {collection_id, name, matter_kind, parties: [{name, role?}, ...]}.
                        Each carries ONLY metadata (no documents) — that is the wall.

    Returns a list of finding dicts (possibly empty), each:
        {party, matched_party, collection_id, matter_name, matter_kind,
         severity: "adverse" | "same_party", new_side, existing_side}
    `severity == "adverse"` ⇒ our-side here vs other-side there (or the reverse) = a true
    ethical-wall hit → the higher-tone banner. `same_party` = the firm already touches this
    name elsewhere on the same/neutral side → an informational note. Pure, never raises.
    """
    findings: List[Dict] = []
    try:
        norm_new = [(_normalize_party(p.get("name", "")), _side_of(p.get("role")),
                     p.get("name", "")) for p in (new_parties or []) if p.get("name")]
        if not norm_new:
            return findings
        for m in (existing_matters or []):
            for ep in (m.get("parties") or []):
                if not ep.get("name"):
                    continue
                e_norm = _normalize_party(ep.get("name", ""))
                e_side = _side_of(ep.get("role"))
                for n_norm, n_side, n_raw in norm_new:
                    if not n_norm or n_norm != e_norm:
                        continue
                    # adverse iff the two sides are opposed (our↔other). Neutral on either
                    # side downgrades to an informational same-party note.
                    adverse = {n_side, e_side} == {"our", "other"}
                    findings.append({
                        "party": n_raw,
                        "matched_party": ep.get("name"),
                        "collection_id": m.get("collection_id") or m.get("id"),
                        "matter_name": m.get("name"),
                        "matter_kind": m.get("matter_kind"),
                        "severity": "adverse" if adverse else "same_party",
                        "new_side": n_side,
                        "existing_side": e_side,
                    })
    except Exception:  # noqa: BLE001 — a conflict scan must never break create
        return findings
    return findings
