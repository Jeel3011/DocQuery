"""F2i — E-signatures: legally-valid sign-off (IT Act 2000).

plans/F2_FIRM_CONSOLE_PLAN.md §F2i + F2_ARCHITECTURE.md §4. This module is the PURE core of the
e-signature slice — the hashing, the chain link, the §1(4) exclusion classifier, and the signed-payload
canonicalisation. It has NO Supabase / FastAPI / LLM dependency (so the gate runs offline, $0); db.py
owns the append-and-chain write, routes/matters.py owns the sign-on-approve / sign-on-release wiring.

The three legal pillars (all cited in the architecture doc):

  1. SECURE ELECTRONIC SIGNATURE (the default, IT Act §3/§5/§10A, presumption §85B). A signature is
     "secure" when the signer's identity is authenticated (here: the JWT-verified member) AND the
     record is tamper-evident (here: content_hash over the signed payload + a per-firm append-only
     hash chain). No Aadhaar/DSC dependency — valid on its own. `signature_method='secure_eauth'`.

  2. THE §1(4) EXCLUSION GUARD. IT Act §1(4) lists instruments that CANNOT be electronically signed —
     a negotiable instrument, a power of attorney, a will / testamentary disposition, a trust, and a
     contract for sale/conveyance of immovable property. An e-sign on these is VOID. `classify_signability`
     detects them (by artifact title / declared instrument type) and the route REFUSES to e-sign,
     returning "wet-ink required" — a hard 422, because a void signature is worse than an honest "we
     can't sign this here."

  3. THE STRONG-TIER SEAM (Aadhaar-eSign / DSC, §67A/§85B/§85C statutory presumptions). Architected as
     a `signature_method` value + a callable seam (`strong_tier_available`) — NOT forced on every
     internal click (the anti-burden rule). When a paying firm needs court-grade signing, the seam is
     where it plugs in; until then it advertises itself as unavailable and the secure tier is used.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

# ── §1(4) EXCLUSIONS — instruments that may NOT be electronically signed (void if e-signed) ─────────
# Each entry: the canonical instrument key → the human reason shown to the user. The classifier maps a
# free-text artifact title / declared type onto one of these via keyword signatures. Conservative by
# design: we REFUSE on a match (better a false "wet-ink required" than a void signature).
EXCLUDED_INSTRUMENTS: dict[str, str] = {
    "negotiable_instrument": "Negotiable instruments (cheques, promissory notes, bills of exchange) "
                             "cannot be electronically signed under IT Act §1(4) — wet-ink required.",
    "power_of_attorney": "A power of attorney cannot be electronically signed under IT Act §1(4) — "
                         "wet-ink (and where applicable, notarisation/registration) required.",
    "will": "A will or testamentary disposition cannot be electronically signed under IT Act §1(4) — "
            "wet-ink required.",
    "trust": "A trust deed cannot be electronically signed under IT Act §1(4) — wet-ink required.",
    "immovable_property": "A contract for the sale or conveyance of immovable property cannot be "
                          "electronically signed under IT Act §1(4) — wet-ink (and registration) required.",
}

# Keyword signatures → excluded instrument key. Matched case-insensitively as whole-ish phrases over the
# artifact title + declared instrument type. Ordered most-specific first so e.g. "promissory note" wins
# over a bare "note". These are intentionally broad on the EXCLUSION side (fail safe = refuse).
_EXCLUSION_PATTERNS: list[tuple[str, str]] = [
    (r"\bpromissory note\b", "negotiable_instrument"),
    (r"\bbill of exchange\b", "negotiable_instrument"),
    (r"\bcheque\b|\bcheck\b", "negotiable_instrument"),
    (r"\bnegotiable instrument\b", "negotiable_instrument"),
    (r"\bpower of attorney\b|\bgpa\b|\bspa\b", "power_of_attorney"),
    (r"\blast will\b|\btestament\b|\bcodicil\b|\bwill\b", "will"),
    (r"\btrust deed\b|\bdeed of trust\b", "trust"),
    (r"\bsale deed\b|\bconveyance\b|\bimmovable propert", "immovable_property"),
    (r"\bsale of land\b|\bproperty sale agreement\b", "immovable_property"),
]

# The IT Act signature tiers (the `signature_method` column values).
METHOD_SECURE_EAUTH = "secure_eauth"      # tier-2 default: JWT identity + tamper-evident record (§85B)
METHOD_AADHAAR_ESIGN = "aadhaar_esign"    # strong tier (§67A/§85B): Aadhaar eSign — the SEAM
METHOD_DSC = "dsc"                         # strong tier: digital signature certificate — the SEAM
VALID_METHODS = frozenset({METHOD_SECURE_EAUTH, METHOD_AADHAAR_ESIGN, METHOD_DSC})

# The legal acts a signature can attest (the `intent` column).
INTENT_APPROVE = "approve"   # internal sign-off (a reviewer in the chain approves)
INTENT_RELEASE = "release"   # external release (the work leaves the firm)
VALID_INTENTS = frozenset({INTENT_APPROVE, INTENT_RELEASE})


class SignabilityRefusal(Exception):
    """Raised (and mapped to HTTP 422 by the route) when an artifact is a §1(4)-excluded instrument.
    Carries the instrument key + the human reason so the UI can show "wet-ink required" precisely."""

    def __init__(self, instrument: str, reason: str):
        self.instrument = instrument
        self.reason = reason
        super().__init__(reason)


def classify_signability(*, title: str = "", instrument_type: str = "") -> Optional[str]:
    """Return the EXCLUDED-instrument key if the artifact is categorically non-e-signable under IT Act
    §1(4), else None (signable at the secure tier). `instrument_type` is an explicit declaration (if the
    caller knows it); `title` is the free-text artifact/document title (best-effort keyword match).

    Precedence: an explicit, recognised `instrument_type` wins over title heuristics (a caller that
    declares 'will' is believed even if the title is bland). Conservative — any match REFUSES."""
    decl = (instrument_type or "").strip().lower().replace(" ", "_")
    if decl in EXCLUDED_INSTRUMENTS:
        return decl
    hay = f"{title or ''} {instrument_type or ''}".lower()
    for pattern, key in _EXCLUSION_PATTERNS:
        if re.search(pattern, hay):
            return key
    return None


def assert_signable(*, title: str = "", instrument_type: str = "") -> None:
    """Raise SignabilityRefusal if the artifact is §1(4)-excluded; return silently if signable. The
    route calls this BEFORE writing any signature row — a void signature must never be created."""
    key = classify_signability(title=title, instrument_type=instrument_type)
    if key:
        raise SignabilityRefusal(key, EXCLUDED_INSTRUMENTS[key])


def hash_content(content: str) -> str:
    """sha256 hex of an artifact's content — the tamper anchor stored as `artifact_hash`. Re-hashing the
    artifact later and comparing to this proves whether the signed thing changed. Stable for empty/None
    (an empty artifact still gets a well-defined hash, so 'signed nothing' is itself detectable)."""
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def _canonical_payload(*, signer_id: str, artifact_type: str, artifact_ref: str, artifact_hash: str,
                       intent: str, signature_method: str, signed_at: str) -> str:
    """The exact bytes that are signed — a canonical (sorted-key, no-whitespace) JSON of the legally
    material fields. Canonical so the hash is reproducible for verification regardless of dict order."""
    return json.dumps(
        {
            "signer_id": str(signer_id),
            "artifact_type": artifact_type,
            "artifact_ref": str(artifact_ref),
            "artifact_hash": artifact_hash,
            "intent": intent,
            "signature_method": signature_method,
            "signed_at": signed_at,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def content_hash_for(*, signer_id: str, artifact_type: str, artifact_ref: str, artifact_hash: str,
                     intent: str, signature_method: str, signed_at: str) -> str:
    """sha256 of the canonical signed payload → the `content_hash` (record-tamper anchor). Tampering ANY
    material field (who/what/when/which artifact/intent/method) changes this hash."""
    return hashlib.sha256(
        _canonical_payload(
            signer_id=signer_id, artifact_type=artifact_type, artifact_ref=artifact_ref,
            artifact_hash=artifact_hash, intent=intent, signature_method=signature_method,
            signed_at=signed_at,
        ).encode("utf-8")
    ).hexdigest()


def chain_row_hash(prev_hash: Optional[str], content_hash: str) -> str:
    """The append-only chain link: sha256(prev_hash + content_hash). prev_hash is the previous row's
    row_hash for the SAME firm (None/'' for the genesis row). Deleting or reordering any row breaks
    every subsequent row_hash → tamper-evident at the LEDGER level, not just per-record."""
    return hashlib.sha256(f"{prev_hash or ''}{content_hash}".encode("utf-8")).hexdigest()


def build_signature_row(
    *,
    firm_id: str,
    signer_id: str,
    artifact_type: str,
    artifact_ref: str,
    artifact_content: str,
    intent: str,
    signed_at: str,
    chain_seq: int,
    prev_row_hash: Optional[str],
    signature_method: str = METHOD_SECURE_EAUTH,
    signer_name: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Assemble a complete, chain-linked signature row ready for insert. Computes artifact_hash,
    content_hash, and row_hash deterministically. The caller (db.append_signature) supplies chain_seq +
    prev_row_hash from the firm's current chain tip under a lock so the chain stays monotonic.

    Validates intent/method up front (a bad enum is a programming error, raised loudly — distinct from
    the §1(4) refusal which is a user-facing 422 handled earlier in the route)."""
    if intent not in VALID_INTENTS:
        raise ValueError(f"invalid signature intent: {intent!r}")
    if signature_method not in VALID_METHODS:
        raise ValueError(f"invalid signature_method: {signature_method!r}")

    artifact_hash = hash_content(artifact_content)
    c_hash = content_hash_for(
        signer_id=signer_id, artifact_type=artifact_type, artifact_ref=artifact_ref,
        artifact_hash=artifact_hash, intent=intent, signature_method=signature_method,
        signed_at=signed_at,
    )
    r_hash = chain_row_hash(prev_row_hash, c_hash)
    return {
        "firm_id": str(firm_id),
        "signer_id": str(signer_id),
        "artifact_type": artifact_type,
        "artifact_ref": str(artifact_ref),
        "artifact_hash": artifact_hash,
        "intent": intent,
        "signature_method": signature_method,
        "signer_name": signer_name,
        "ip_address": ip_address,
        "user_agent": user_agent,
        "signed_at": signed_at,
        "chain_seq": chain_seq,
        "content_hash": c_hash,
        "prev_hash": prev_row_hash,
        "row_hash": r_hash,
        "metadata": metadata or {},
    }


def verify_signature_against(row: dict, *, artifact_content: Optional[str] = None) -> dict:
    """Verify ONE signature row in isolation. Recomputes content_hash from the row's stored material
    fields and checks it matches row['content_hash'] (record integrity); recomputes row_hash from
    prev_hash+content_hash (chain-link integrity); and, if the current artifact content is supplied,
    re-hashes it and checks it matches the stored artifact_hash (artifact integrity — the tamper test).

    Returns {ok, record_intact, link_intact, artifact_intact, reason}. ok = all applicable checks pass.
    This is the per-row half; verify_chain (db.py) walks the whole firm chain for ordering/deletion."""
    out = {"ok": False, "record_intact": False, "link_intact": False,
           "artifact_intact": None, "reason": ""}
    try:
        recomputed_content = content_hash_for(
            signer_id=row["signer_id"], artifact_type=row["artifact_type"],
            artifact_ref=row["artifact_ref"], artifact_hash=row["artifact_hash"],
            intent=row["intent"], signature_method=row["signature_method"],
            signed_at=row["signed_at"],
        )
        out["record_intact"] = (recomputed_content == row.get("content_hash"))
        recomputed_row = chain_row_hash(row.get("prev_hash"), recomputed_content)
        out["link_intact"] = (recomputed_row == row.get("row_hash"))
        if artifact_content is not None:
            out["artifact_intact"] = (hash_content(artifact_content) == row.get("artifact_hash"))
    except (KeyError, TypeError) as e:
        out["reason"] = f"malformed signature row: {e}"
        return out

    checks = [out["record_intact"], out["link_intact"]]
    if out["artifact_intact"] is not None:
        checks.append(out["artifact_intact"])
    out["ok"] = all(checks)
    if not out["ok"]:
        broken = []
        if not out["record_intact"]:
            broken.append("record tampered (content_hash mismatch)")
        if not out["link_intact"]:
            broken.append("chain link broken (row_hash mismatch)")
        if out["artifact_intact"] is False:
            broken.append("artifact changed since signing (artifact_hash mismatch)")
        out["reason"] = "; ".join(broken)
    return out


def strong_tier_available(method: str) -> bool:
    """The Aadhaar-eSign / DSC SEAM. Returns whether the requested strong-tier method is wired in this
    deployment. v1: always False (the seam is architected, not connected) — so a strong-tier request
    falls back to the secure tier with a clear note, and the secure tier is NEVER blocked on a missing
    strong-tier integration (the anti-burden rule). When a firm connects Aadhaar-eSign/DSC, this is the
    single switch that flips (config/env), no schema or call-site change."""
    import os
    if method == METHOD_AADHAAR_ESIGN:
        return os.getenv("AADHAAR_ESIGN_ENABLED", "false").lower() == "true"
    if method == METHOD_DSC:
        return os.getenv("DSC_SIGNING_ENABLED", "false").lower() == "true"
    return False


def resolve_method(requested: Optional[str]) -> tuple[str, Optional[str]]:
    """Resolve the effective signature_method for a sign request. A strong-tier request that isn't wired
    falls back to secure_eauth with an explanatory note (returned so the route can surface it); an
    unknown method also falls back. Returns (method, note_or_None)."""
    req = (requested or METHOD_SECURE_EAUTH).strip().lower()
    if req == METHOD_SECURE_EAUTH:
        return METHOD_SECURE_EAUTH, None
    if req in (METHOD_AADHAAR_ESIGN, METHOD_DSC):
        if strong_tier_available(req):
            return req, None
        return METHOD_SECURE_EAUTH, (
            f"{req} is not connected in this deployment; signed at the secure-eauth tier "
            "(IT Act §85B), which is legally valid without it."
        )
    return METHOD_SECURE_EAUTH, f"unknown method {req!r}; defaulted to secure-eauth."
