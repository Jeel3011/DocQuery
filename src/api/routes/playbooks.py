"""G6.2 — Playbook CRUD route.

`GET/POST/PUT/DELETE /playbooks` — manage firm standard positions (clause topic →
standard position + fallback + notes). Scoped per-user with RLS; optionally scoped
to a specific vault (collection_id). The G6.3 redline tool reads these rows when
comparing a target clause against the firm's position.

`POST /playbooks/seed` — load the Indian NDA/vendor starter playbook for the
authenticated user (idempotent: skips any topic already present for that user).
No model burn — stored data only.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PlaybookRow(BaseModel):
    id: Optional[str] = None
    collection_id: Optional[str] = None
    clause_topic: str = Field(..., max_length=300)
    standard_position: str = Field(..., max_length=4000)
    fallback_position: Optional[str] = Field(default=None, max_length=4000)
    notes: Optional[str] = Field(default=None, max_length=2000)


class PlaybookRowOut(PlaybookRow):
    id: str
    user_id: str
    is_seed: bool
    created_at: str
    updated_at: str


# ── Indian NDA / vendor-contract starter playbook ────────────────────────────
# 20 clause topics covering the main negotiation points in Indian commercial
# contracts (NDA + vendor / services agreements). Each row = one firm standard
# position a reviewer would check against. Topics mirror the G4 grid presets.

_SEED_PLAYBOOK: List[dict] = [
    {
        "clause_topic": "Governing Law",
        "standard_position": "The agreement shall be governed by and construed in accordance with the laws of India, without regard to conflict-of-law principles.",
        "fallback_position": "Laws of Singapore or England & Wales acceptable for cross-border agreements with MNC counterparties.",
        "notes": "Indian courts preferred for enforcement. Avoid US/EU governing law unless the counterparty is domiciled there.",
    },
    {
        "clause_topic": "Dispute Resolution / Arbitration Seat",
        "standard_position": "All disputes shall be referred to binding arbitration under the Arbitration and Conciliation Act, 1996 (as amended), with the seat at Mumbai or New Delhi. The tribunal shall consist of a sole arbitrator unless the amount in dispute exceeds ₹5 crore, in which case three arbitrators.",
        "fallback_position": "Singapore International Arbitration Centre (SIAC) acceptable for transactions above ₹25 crore involving foreign parties.",
        "notes": "Avoid ICC or LCIA if the transaction is India-domestic — cost and enforceability concerns.",
    },
    {
        "clause_topic": "Confidentiality / NDA Term",
        "standard_position": "Confidentiality obligations survive termination for a period of three (3) years.",
        "fallback_position": "Two (2) years acceptable for low-sensitivity engagements; five (5) years required for IP or trade-secret sharing.",
        "notes": "Perpetual confidentiality is non-standard and likely unenforceable under Indian law for general business information.",
    },
    {
        "clause_topic": "Permitted Disclosures",
        "standard_position": "Receiving party may disclose confidential information only to employees and advisors with a need to know, subject to written confidentiality obligations no less restrictive than those in this agreement.",
        "fallback_position": None,
        "notes": "Ensure sub-processors and offshore affiliates are explicitly covered.",
    },
    {
        "clause_topic": "Indemnity Scope",
        "standard_position": "Each party indemnifies the other only for direct losses arising from its own material breach or gross negligence. No indemnity for consequential, indirect, or speculative losses.",
        "fallback_position": "Mutual indemnity for IP infringement claims by third parties acceptable in addition to the above.",
        "notes": "Broad indemnities (including loss of profit, goodwill, or business) should be resisted and flagged.",
    },
    {
        "clause_topic": "Liability Cap",
        "standard_position": "Each party's aggregate liability under or in connection with this agreement shall not exceed the total fees paid or payable in the twelve (12) months immediately preceding the claim.",
        "fallback_position": "A fixed cap of ₹50 lakh acceptable for low-value service agreements; unlimited cap unacceptable.",
        "notes": "Cap should be mutual. Uncapped liability for fraud, wilful misconduct, or death/personal injury is standard carve-out.",
    },
    {
        "clause_topic": "Exclusion of Consequential Loss",
        "standard_position": "Neither party shall be liable for any indirect, consequential, special, incidental, or punitive damages, including loss of profits, loss of data, or loss of business opportunity, even if advised of the possibility.",
        "fallback_position": None,
        "notes": "Mutual exclusion. Carve-outs for data-breach fines or regulatory penalties may be required for DPDPA compliance.",
    },
    {
        "clause_topic": "Term and Renewal",
        "standard_position": "Initial term of one (1) year, renewing automatically for successive one-year periods unless either party provides thirty (30) days' written notice of non-renewal prior to the end of the then-current term.",
        "fallback_position": "Auto-renewal acceptable for SaaS/subscription agreements; require explicit renewal notice for high-value services.",
        "notes": "Flag evergreen clauses without a cap on renewal periods.",
    },
    {
        "clause_topic": "Termination for Convenience",
        "standard_position": "Either party may terminate this agreement for convenience upon thirty (30) days' written notice. Vendor/service provider shall be entitled to fees for services rendered up to the termination date.",
        "fallback_position": "Sixty (60) days acceptable where significant transition costs are involved.",
        "notes": "Termination fees or break charges for convenience termination should be flagged and quantified.",
    },
    {
        "clause_topic": "Termination for Cause",
        "standard_position": "Either party may terminate immediately upon written notice if the other party commits a material breach and fails to cure such breach within fifteen (15) days of written notice specifying the breach.",
        "fallback_position": "Thirty (30) days cure period acceptable for complex operational breaches.",
        "notes": "Insolvency, liquidation, or regulatory action should trigger immediate termination rights without cure period.",
    },
    {
        "clause_topic": "Intellectual Property Ownership",
        "standard_position": "All IP created specifically for and paid for by the client under this agreement vests in the client upon payment in full. Background IP of each party remains owned by that party.",
        "fallback_position": "License-back of background IP to the client for use of the deliverable is acceptable in lieu of assignment.",
        "notes": "Work-for-hire provisions under Indian Copyright Act, 1957 require explicit written assignment — verbal or implied assignment is insufficient.",
    },
    {
        "clause_topic": "Data Protection / DPDPA Compliance",
        "standard_position": "Each party shall comply with applicable data protection laws, including the Digital Personal Data Protection Act, 2023 (DPDPA). The vendor shall process personal data only on documented instructions of the client and implement appropriate technical and organisational measures.",
        "fallback_position": None,
        "notes": "DPDPA fiduciary / processor obligations apply where personal data of Indian residents is processed. Cross-border transfer restrictions apply.",
    },
    {
        "clause_topic": "Non-Solicitation",
        "standard_position": "During the term and for twelve (12) months following termination, neither party shall solicit or hire the other party's employees or contractors who were directly involved in the performance of this agreement, without the other party's prior written consent.",
        "fallback_position": "Six (6) months acceptable for short-duration agreements.",
        "notes": "Non-solicitation of employees is generally enforceable in India; outright non-compete is harder to enforce.",
    },
    {
        "clause_topic": "Force Majeure",
        "standard_position": "Neither party shall be liable for failure to perform obligations caused by events beyond its reasonable control (force majeure), provided the affected party: (a) gives prompt written notice; (b) uses reasonable efforts to mitigate; and (c) resumes performance as soon as reasonably practicable. If force majeure continues for more than sixty (60) days, either party may terminate without liability.",
        "fallback_position": None,
        "notes": "Pandemic / epidemic should be listed. Avoid open-ended force-majeure extensions without a termination right.",
    },
    {
        "clause_topic": "Assignment",
        "standard_position": "Neither party may assign or transfer this agreement or any rights hereunder without the prior written consent of the other party, not to be unreasonably withheld. Assignment to an affiliate or in connection with a merger or acquisition of all or substantially all assets is permitted on notice.",
        "fallback_position": None,
        "notes": "Change-of-control clauses (requiring consent on acquisition) should be flagged and negotiated.",
    },
    {
        "clause_topic": "Entire Agreement / Supersession",
        "standard_position": "This agreement (including all schedules and annexures) constitutes the entire agreement between the parties with respect to its subject matter and supersedes all prior negotiations, representations, warranties, and agreements.",
        "fallback_position": None,
        "notes": "Verify that all schedules are attached and referenced. Oral side-agreements are void.",
    },
    {
        "clause_topic": "Notices",
        "standard_position": "All notices shall be in writing and delivered by: (a) registered post with acknowledgement due; (b) reputed courier; or (c) email with confirmation of receipt, to the addresses set out in this agreement. Notice is effective on the date of receipt (or next business day if received after 6 pm IST).",
        "fallback_position": None,
        "notes": "Email-only notice is acceptable provided a reply-acknowledgement or read-receipt mechanism is in place.",
    },
    {
        "clause_topic": "Amendments",
        "standard_position": "No amendment or modification of this agreement shall be effective unless in writing and duly executed by authorised representatives of both parties.",
        "fallback_position": None,
        "notes": "Verbal or email-only amendments are unenforceable. Require wet-ink or DocuSign signatures.",
    },
    {
        "clause_topic": "Stamp Duty",
        "standard_position": "This agreement shall be stamped as required under applicable state stamp-duty legislation. The cost of stamping shall be borne by [party — to be agreed].",
        "fallback_position": None,
        "notes": "Unstamped agreements are inadmissible as evidence in Indian courts. Confirm applicable state and applicable duty.",
    },
    {
        "clause_topic": "Anti-Bribery and Corruption",
        "standard_position": "Each party warrants that it has not made and will not make any improper payment, gift, or benefit in connection with this agreement in violation of applicable laws, including the Prevention of Corruption Act, 1988 and the Foreign Corrupt Practices Act (where applicable).",
        "fallback_position": None,
        "notes": "Required for any agreement involving government entities or foreign counterparties. Trigger for immediate termination on breach.",
    },
]


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/playbooks", response_model=List[PlaybookRowOut])
async def list_playbooks(
    collection_id: Optional[str] = None,
    sb=Depends(get_current_user),
):
    """List all playbook rows for the authenticated user, optionally filtered by vault."""
    try:
        q = sb.client.table("playbooks").select("*").eq("user_id", sb.user_id)
        if collection_id:
            q = q.eq("collection_id", collection_id)
        res = q.order("clause_topic").execute()
        return res.data or []
    except Exception as exc:
        logger.warning("playbooks list failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/playbooks", response_model=PlaybookRowOut, status_code=201)
async def create_playbook_row(
    body: PlaybookRow,
    sb=Depends(get_current_user),
):
    """Create a new playbook row (clause topic + standard position)."""
    try:
        row = {
            "user_id": sb.user_id,
            "collection_id": body.collection_id,
            "clause_topic": body.clause_topic,
            "standard_position": body.standard_position,
            "fallback_position": body.fallback_position,
            "notes": body.notes,
            "is_seed": False,
        }
        res = sb.client.table("playbooks").insert(row).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert returned no data")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("playbooks create failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.put("/playbooks/{row_id}", response_model=PlaybookRowOut)
async def update_playbook_row(
    row_id: str,
    body: PlaybookRow,
    sb=Depends(get_current_user),
):
    """Update an existing playbook row (must belong to the authenticated user)."""
    try:
        patch = {
            "clause_topic": body.clause_topic,
            "standard_position": body.standard_position,
            "fallback_position": body.fallback_position,
            "notes": body.notes,
        }
        res = (
            sb.client.table("playbooks")
            .update(patch)
            .eq("id", row_id)
            .eq("user_id", sb.user_id)
            .execute()
        )
        if not res.data:
            raise HTTPException(status_code=404, detail="Playbook row not found")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("playbooks update failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/playbooks/{row_id}", status_code=204)
async def delete_playbook_row(
    row_id: str,
    sb=Depends(get_current_user),
):
    """Delete a playbook row (must belong to the authenticated user)."""
    try:
        sb.client.table("playbooks").delete().eq("id", row_id).eq("user_id", sb.user_id).execute()
    except Exception as exc:
        logger.warning("playbooks delete failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/playbooks/seed", status_code=201)
async def seed_playbook(
    collection_id: Optional[str] = None,
    sb=Depends(get_current_user),
):
    """Load the Indian NDA/vendor starter playbook (idempotent — skips existing topics).

    Returns {inserted: N, skipped: N} so the caller knows what changed.
    """
    try:
        # Fetch existing topics for this user (+ optional vault scope) to skip dupes.
        q = sb.client.table("playbooks").select("clause_topic").eq("user_id", sb.user_id)
        if collection_id:
            q = q.eq("collection_id", collection_id)
        existing_res = q.execute()
        existing_topics = {r["clause_topic"] for r in (existing_res.data or [])}

        to_insert = [
            {
                "user_id": sb.user_id,
                "collection_id": collection_id,
                "clause_topic": row["clause_topic"],
                "standard_position": row["standard_position"],
                "fallback_position": row.get("fallback_position"),
                "notes": row.get("notes"),
                "is_seed": True,
            }
            for row in _SEED_PLAYBOOK
            if row["clause_topic"] not in existing_topics
        ]

        inserted = 0
        if to_insert:
            sb.client.table("playbooks").insert(to_insert).execute()
            inserted = len(to_insert)

        skipped = len(_SEED_PLAYBOOK) - inserted
        logger.info("playbook seed: inserted=%d skipped=%d user=%s", inserted, skipped, sb.user_id)
        return {"inserted": inserted, "skipped": skipped}
    except Exception as exc:
        logger.warning("playbooks seed failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
