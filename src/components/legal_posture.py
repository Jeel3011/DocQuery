"""F2l — Legal posture: the code that makes the legal artifacts (MSA / DPA / Privacy / AI-addendum) TRUE.

plans/F2_FIRM_CONSOLE_PLAN.md §F2l + F2_ARCHITECTURE.md §7. This is the PURE, dependency-free core of the
last F2 slice. The documents in docs/legal/ are TEMPLATES a real lawyer reviews before launch (we do NOT
self-certify legal advice). This module is the part that makes their load-bearing promises ENFORCED, not
just prose — because *"a DPA you can't honor is worse than none"* (Callidus). It has no Supabase / FastAPI
dependency, so the gate runs offline ($0).

Two load-bearing guarantees the product must keep, made real here:

  1. "NO TRAINING ON YOUR DATA" (DPA §3, AI-addendum). A real flag, OFF by default (config.MODEL_TRAINING_
     ON_CUSTOMER_DATA = False), and a GUARDED path — assert_no_training() — that any "improve the model on
     Customer Data" seam MUST call and that REFUSES while the flag is off. There is no such training path in
     the product today (Customer Data flows to the LLM vendor only for inference, never for training), so
     the guard protects the SEAM: a future fine-tune / feedback-export feature is forced through it and is
     refused unless the firm has explicitly, deliberately opted in (per the DPA's written-consent term).
     The enforcement POINT is documented (see assert_no_training) so it can't be quietly bypassed.

  2. AN ACCURATE SUBPROCESSOR LIST (DPA §5). *"An undisclosed subprocessor is a breach every time data
     flows through it."* So the disclosed list is NOT hand-maintained (it would drift). It is DERIVED from
     the live config — the same vendor decisions the running code makes — so disclosed_subprocessors(config)
     reflects what the code actually calls (Supabase, Pinecone, the LLM vendor per agent-core config, and
     the email-transport seam from F2j). subprocessor_drift() is the assertion the gate runs: the disclosed
     list must equal the code-derived list, catching any drift the moment a vendor is added/removed.

The DPA's erasure-on-termination clause (DPA §4) is honored by F2k's erase endpoint
(src/components/db.py::erase_personal_data + routes/dpdp.py POST /dpdp/erase) — dpa_erasure_clause_backed_by()
names that proof so the gate can assert the clause is callable, not aspirational.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


# ── 1. "No training on your data" — the real, enforced switch ───────────────────────────────────────

class TrainingUseRefused(RuntimeError):
    """Raised when a code path attempts to use Customer Data for model training/fine-tuning while the
    no-train posture is in force (the default). This is a HARD refusal, not a warning — the DPA promise
    that Customer Data is never used to train a model without written consent is enforced here, so a
    future feature can never silently start training on tenant data."""


def model_training_allowed(config: Any) -> bool:
    """True ONLY if the deployment has explicitly opted IN to training on Customer Data (the DPA's
    written-consent term). False by default — the compliant ship state. The single predicate every
    training seam consults; reads config.MODEL_TRAINING_ON_CUSTOMER_DATA (default False)."""
    return bool(getattr(config, "MODEL_TRAINING_ON_CUSTOMER_DATA", False))


def assert_no_training(config: Any, *, purpose: str = "model training/fine-tuning on Customer Data") -> None:
    """THE ENFORCEMENT POINT. Any path that would use Customer Data to train, fine-tune, or otherwise
    "improve the model" MUST call this first; it raises TrainingUseRefused unless the firm has explicitly
    opted in. Because no such path exists in the product today, this guards the SEAM — wiring a future
    fine-tune / feedback-export feature without calling this would itself be the bug the gate (and code
    review) catches. The DPA §3 / AI-addendum promise is true by construction: off ⇒ refused."""
    if not model_training_allowed(config):
        raise TrainingUseRefused(
            f"Refused: {purpose}. Customer Data is processed only on the firm's documented instructions "
            "and is never used for model training without the firm's explicit, written opt-in "
            "(config.MODEL_TRAINING_ON_CUSTOMER_DATA is off — the DPA §3 / AI-use addendum default)."
        )


# ── 2. The subprocessor list — DERIVED from the code, not hand-maintained (so it can't drift) ────────

@dataclass(frozen=True)
class Subprocessor:
    """One disclosed subprocessor. `purpose` = what Customer Data flows to it for; `data` = the category
    of Customer Data it touches; `derived_from` = the config/code fact that proves it's really in use
    (the anti-drift anchor)."""
    key: str
    name: str
    purpose: str
    data: str
    region_note: str
    derived_from: str


# The LLM vendor is resolved by model id the SAME way agent_core/model.py::build_model routes it
# (`claude*` → Anthropic, else → OpenAI). Deriving the vendor from the live model id (not a hardcoded
# name) is what keeps the disclosure honest when the orchestrator vendor is switched via config
# (the §3.1 multi-vendor decision: OpenAI now on Jeel's key, Claude at production).
def llm_vendor_for_model(model_id: str) -> tuple[str, str]:
    """Return (vendor_key, vendor_name) for an orchestrator/LLM model id, mirroring build_model's
    prefix routing. Anthropic for `claude*`, OpenAI otherwise (gpt-*/o-*). This is the single rule both
    the live model construction and the subprocessor disclosure share, so they cannot disagree."""
    mid = (model_id or "").strip().lower()
    if mid.startswith("claude"):
        return ("anthropic", "Anthropic (Claude API)")
    return ("openai", "OpenAI (GPT API)")


def _llm_model_ids(config: Any) -> list[str]:
    """Every model id the product sends prompts to (orchestrator + Brain + classifier + embeddings).
    Embeddings AND chat both transmit document text to the vendor, so both count for disclosure."""
    return [
        getattr(config, "AGENT_MODEL_STANDARD", "") or "",
        getattr(config, "AGENT_MODEL_DEEP", "") or "",
        getattr(config, "AGENT_MODEL_CLASSIFIER", "") or "",
        getattr(config, "LLM_MODEL_NAME", "") or "",
        getattr(config, "REDUCE_LLM_MODEL", "") or "",
        getattr(config, "VERIFY_LLM_MODEL", "") or "",
        getattr(config, "EMBEDDING_MODEL_NAME", "") or "",
    ]


def disclosed_subprocessors(config: Any) -> list[Subprocessor]:
    """The subprocessor list DERIVED from the live config — exactly the vendors the running code calls.
    This is the source of truth the DPA §5 disclosure (docs/legal/DPA.md) must match; subprocessor_drift()
    asserts they're identical. Always-on infrastructure (Supabase, Pinecone) plus every distinct LLM
    vendor the configured model ids route to, plus the email-transport seam (declared as a SEAM while
    F2j ships in-app-only — disclosed now so enabling email is a config flip, not an undisclosed flow)."""
    subs: list[Subprocessor] = [
        Subprocessor(
            key="supabase",
            name="Supabase (Postgres + Storage)",
            purpose="Primary datastore — documents, conversations, audit log, firm/matter records, file blobs.",
            data="All Customer Data at rest (document content, metadata, generated answers).",
            region_note="Hosted region per the project; India-region hosting is the DPDP-aligned target.",
            derived_from="config.SUPABASE_URL is set; src/components/db.py creates the Supabase client.",
        ),
        Subprocessor(
            key="pinecone",
            name="Pinecone (vector index)",
            purpose="Vector store for retrieval — embeddings of document chunks + metadata.",
            data="Embedding vectors and chunk metadata derived from Customer Data.",
            region_note="Serverless index; namespace-isolated per user/firm.",
            derived_from="config.PINECONE_INDEX_NAME / PINECONE_API_KEY; src/components/embeddings.py upserts.",
        ),
    ]

    # LLM vendor(s) — derived from the live model ids the same way build_model routes them. De-duped by
    # vendor so two OpenAI model ids disclose ONE OpenAI subprocessor (the data flow is to the vendor).
    seen_vendors: set[str] = set()
    for mid in _llm_model_ids(config):
        if not mid:
            continue
        vkey, vname = llm_vendor_for_model(mid)
        if vkey in seen_vendors:
            continue
        seen_vendors.add(vkey)
        subs.append(Subprocessor(
            key=vkey,
            name=vname,
            purpose="LLM inference — prompts containing document excerpts are sent for answer generation "
                    "and embedding. Inference only; never training (see the no-train switch).",
            data="Document excerpts + the user's question, transmitted per request for inference.",
            region_note="API processing region per the vendor; covered by the vendor's zero-retention / "
                        "no-training-on-API-data terms.",
            derived_from=f"config model id {mid!r} routes to this vendor via build_model's prefix rule.",
        ))

    # Email transport — the F2j seam. Disclosed as a SEAM even though v1 is in-app-only (no transport
    # wired), because the moment a transport is configured it becomes a real data flow; disclosing it now
    # means turning email on is a config change, NOT a new undisclosed subprocessor.
    subs.append(Subprocessor(
        key="email_transport",
        name="Email transport provider (seam — not yet active)",
        purpose="Outbound notification email (review-awaiting / approved / returned). In-app inbox only in "
                "v1; no transport configured, so no Customer Data flows here yet.",
        data="Recipient address + notification text (no document content) — only once a transport is set.",
        region_note="Provider TBD at the point email is enabled; disclosed in advance to avoid drift.",
        derived_from="src/components/notifications.py::email_status_for is the documented transport seam "
                    "(v1 returns 'skipped'; enabling email is a one-line flip).",
    ))
    return subs


def disclosed_subprocessor_keys(config: Any) -> set[str]:
    """The set of subprocessor keys the code actually uses — the anti-drift fingerprint the DPA must match."""
    return {s.key for s in disclosed_subprocessors(config)}


# The interchangeable LLM-vendor pair: the orchestrator can route to either depending on the configured
# model id (OpenAI now on Jeel's key, Anthropic/Claude at production — the §3.1 multi-vendor decision).
# Disclosing BOTH in the DPA is correct forward-disclosure, not stale drift: a vendor switch via config
# must never produce an UNDISCLOSED flow. So a declared-but-currently-inactive LLM vendor is allowed in
# the 'extra' direction, AS LONG AS at least one of the pair is actually in use (the LLM flow IS happening,
# just to the other vendor). The 'missing' direction stays strict for everyone — used-but-undisclosed is
# always the breach risk.
_LLM_VENDOR_KEYS: frozenset[str] = frozenset({"openai", "anthropic"})


def subprocessor_drift(config: Any, declared_keys: set[str]) -> dict[str, set[str]]:
    """Compare a DECLARED subprocessor key set (e.g. docs/legal/DPA.md's disclosure) against the
    CODE-DERIVED set. Returns {'missing': used by code but NOT disclosed → DPA breach risk, 'extra':
    disclosed but NOT used by code → stale}. Both must be empty for the gate to pass — EXCEPT the known
    dual-LLM-vendor case: disclosing both OpenAI and Anthropic is intentional forward-disclosure (the
    orchestrator routes to one OR the other by config), so a declared-but-inactive LLM vendor is NOT
    counted as drift provided the LLM flow really exists (≥1 of the pair is in use). Everything else,
    including any undisclosed-but-used vendor, is strict — the disclosure is a CHECKED artifact."""
    code_keys = disclosed_subprocessor_keys(config)
    missing = code_keys - declared_keys           # strict: anything used must be disclosed
    extra = declared_keys - code_keys
    # Forgive a declared-but-inactive LLM vendor IFF the LLM flow is genuinely present (the other vendor
    # is in use). If NO LLM vendor is in use at all, a declared one IS stale and stays flagged.
    if code_keys & _LLM_VENDOR_KEYS:
        extra = extra - _LLM_VENDOR_KEYS
    return {"missing": missing, "extra": extra}


# The keys the DPA template (docs/legal/DPA.md §5) discloses. Kept here as the doc's machine-readable
# mirror so the gate can assert doc-vs-code parity without brittle markdown parsing; if a vendor is
# added to disclosed_subprocessors() without adding it here (and to the doc), the drift check fails.
DPA_DISCLOSED_SUBPROCESSOR_KEYS: frozenset[str] = frozenset(
    {"supabase", "pinecone", "openai", "anthropic", "email_transport"}
)


# ── 3. The DPA erasure-on-termination clause — backed by F2k's real erase path ───────────────────────

@dataclass(frozen=True)
class ClauseProof:
    """A legal clause whose promise is backed by real code. `clause` = the contractual obligation;
    `backed_by` = the function/route that honors it; `notes` = the caveat (e.g. soft-delete + preserved
    records). The gate asserts `backed_by` is importable/callable so the clause is not aspirational."""
    clause: str
    backed_by: str
    notes: str


def dpa_erasure_clause_backed_by() -> ClauseProof:
    """The DPA's erasure-on-termination / data-subject-erasure clause (DPA §4) is honored by F2k's erase
    endpoint. Named here so the gate can assert the proof exists (importable) — a DPA erasure clause with
    no working erase behind it is exactly the unhonorable promise the architecture warns against."""
    return ClauseProof(
        clause="DPA §4 — on a data-principal erasure request or contract termination, the Processor erases "
               "the firm's Customer Data, preserving only the immutable records of processing required by law.",
        backed_by="src/components/db.py::SupabaseManager.erase_personal_data "
                  "(routes/dpdp.py POST /dpdp/erase + /dpdp/admin/erase)",
        notes="Soft-delete of personal CONTENT (documents/conversations/messages); the audit_log and the "
              "F2i signature hash-chain are PRESERVED (Rule 6.5 + non-repudiation). See src/components/dpdp.py "
              "ERASABLE_CONTENT vs PRESERVED_RECORDS — the same distinction the DPA states.",
    )


# ── The four artifacts F2l ships (templates, lawyer-reviewed before launch) ──────────────────────────
# Named so the gate can assert each template file exists + carries its honesty caveat. Paths are relative
# to the repo root (docs/legal/).
LEGAL_ARTIFACTS: dict[str, str] = {
    "MSA": "docs/legal/MSA.md",
    "DPA": "docs/legal/DPA.md",
    "PRIVACY_POLICY": "docs/legal/PRIVACY_POLICY.md",
    "AI_USE_ADDENDUM": "docs/legal/AI_USE_ADDENDUM.md",
}

# The caveat every artifact must carry verbatim (the "we don't self-certify" honesty rule).
TEMPLATE_CAVEAT_MARKER: str = "TEMPLATE — NOT LEGAL ADVICE"
