"""Legal document-type catalog (LEGAL_TASK_CATALOG_PLAN §2.1) — the DRAFT verb's data.

A `DocType` is **declarative DATA** (one catalog row), NOT code — the same bet the
workflow gallery (`workflows.py`) makes for the REVIEW/COMPARE/OUTPUT verbs. The ONE
drafting path (`/query/brain/stream` mode="draft" → `run_agent` + `DRAFT_PROMPT_SUFFIX`
gate) interprets it. A catalog row changes only:
  1. the ordered STRUCTURE the draft must follow (the India-correct section list),
  2. the REQUIRED INPUTS the agent must source from the vault or bracket as `[INPUT NEEDED]`,
  3. the governing-law AUTHORITY refs to cite from G8 (version-in-force), and
  4. the practice-area / jurisdiction the card is filed under.

Everything else — retrieval, the cite-or-abstain gate, the ledger, tracing — is shared,
untouched, and already proven. **A doc type that needs a new loop branch is a DESIGN BUG,
not a feature** (the §0/§4 "one engine, many surfaces" rule).

The unit of GROWTH is a catalog row + a fixture (`eval/test_doc_catalog.py`). Per the
plan's non-negotiable (§3.1, the §-gate): **do NOT ship a card without its fixture** — a
wrong drafted citation is the legal equivalent of a confident-wrong number.

Indian vocabulary is deliberate and load-bearing (§3.1): a *plaint* (not "complaint"), a
*written statement* (not "answer"), *Order XI interrogatories*, *MOA/AOA*. Wrong
terminology = instantly not-credible to an Indian lawyer; the catalog encodes the right
names so the draft inherits them.

`render_draft_request(dt, facts)` is a PURE function: DocType + matter facts → the
`{doc_type, instructions}` the draft route already composes into `body.question`. It
branches on DATA (the row's fields), never on `dt.id` — adding a row never touches it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# The catalog entry — declarative data (the §2.1 DocType schema).
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DocType:
    id: str                          # "legal_notice" | "writ_petition" | "spa" — the card/route key
    title: str                       # gallery card header (the Harvey `<Verb> <DocType>` label)
    practice_area: str               # "Litigation" | "Transactional" | "Financial services" | "Compliance (India)"
    jurisdiction: str = "IN"         # India-first; the vocabulary moat
    description: str = ""            # one line for the gallery card

    # The India-correct ordered structure: the sections/clauses the draft MUST follow,
    # in order. This is what carries the vocabulary moat into the deliverable.
    structure: List[str] = field(default_factory=list)

    # Facts the agent must SOURCE FROM THE VAULT or bracket as a visible [INPUT NEEDED]
    # (never invent). Each is a short label the form/agent resolves.
    required_inputs: List[str] = field(default_factory=list)

    # Governing statute / sections to cite from G8 (version-in-force). The draft states
    # "under Section X of the Companies Act…" only by RETRIEVING X, never from memory.
    authority_refs: List[str] = field(default_factory=list)

    # The firm's gold-standard template/playbook id (→ F5 Firm Brain), when one exists.
    # None ⇒ structure-only draft (no house template to graft).
    template_ref: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Render — DocType + matter facts → the draft route's (doc_type, instructions).
# PURE, branches on DATA only. The draft route (agent_core.py mode="draft") already
# composes `"Produce a client-ready {doc_type}. Instructions: {instructions}"` into the
# question and runs it through DRAFT_PROMPT_SUFFIX's gate — this just builds a structured,
# India-correct, cite-or-bracket instruction string for it.
# ──────────────────────────────────────────────────────────────────────────────
def render_draft_request(dt: DocType, facts: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Return `{"doc_type", "instructions"}` for the existing draft route.

    Forgiving: a missing fact is emitted as a `[INPUT NEEDED: <label>]` bracket (never a
    KeyError, never an invented value) — the same discipline the draft gate enforces, made
    explicit in the request so the agent brackets rather than hallucinates.
    """
    facts = facts or {}

    lines: List[str] = []
    lines.append(f"Jurisdiction: {dt.jurisdiction} (use Indian legal terminology and procedure).")

    if dt.structure:
        lines.append(
            "Follow this exact document structure, in order, one '##' section per item:\n"
            + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(dt.structure))
        )

    if dt.required_inputs:
        resolved = []
        for label in dt.required_inputs:
            val = facts.get(label)
            if val is None or str(val).strip() == "":
                # No provided fact → the agent must source it from the vault or BRACKET it.
                resolved.append(f"  - {label}: source from the vault and cite, or write [INPUT NEEDED: {label}]")
            else:
                resolved.append(f"  - {label}: {val}")
        lines.append("Required facts (cite each to a vault span or bracket it — never invent):\n"
                     + "\n".join(resolved))

    if dt.authority_refs:
        lines.append(
            "Cite governing law ONLY by retrieving it from the knowledge base, version-in-force "
            "(never paraphrase from memory): " + "; ".join(dt.authority_refs) + "."
        )

    if dt.template_ref:
        lines.append(f"Use the firm template/playbook '{dt.template_ref}' as the gold standard where available.")

    return {"doc_type": dt.title, "instructions": "\n\n".join(lines)}


# ══════════════════════════════════════════════════════════════════════════════
# THE CATALOG — the WEDGE SET first (§1 coverage discipline): litigation pleadings +
# pre-litigation notices + commercial-contract draft + finance. India-first vocabulary.
# Grow coverage by adding a row + a fixture, never engine code.
# ══════════════════════════════════════════════════════════════════════════════
_C: List[DocType] = [

    # ── Litigation & Dispute Resolution (§1.A — largest slice of Indian work) ──────
    DocType(
        id="legal_notice",
        title="Legal Notice",
        practice_area="Litigation",
        description="A pre-litigation legal notice (demand / cease-and-desist) under the relevant cause of action.",
        structure=[
            "Heading (sender's advocate, on behalf of the client)",
            "Addressee (the noticee)",
            "Facts giving rise to the cause of action",
            "The legal grievance and the breach asserted",
            "The demand / relief sought, with a time period to comply",
            "Consequence on non-compliance (intention to initiate legal proceedings)",
            "Reservation of rights and signature",
        ],
        required_inputs=["sender_name", "noticee_name", "facts", "demand", "compliance_period"],
        authority_refs=["the substantive statute under which the claim arises (cite as in force)"],
    ),
    DocType(
        id="plaint",
        title="Plaint",
        practice_area="Litigation",
        description="A civil plaint instituting a suit (CPC Order VII) — not a US 'complaint'.",
        structure=[
            "Cause title (in the Court of …, suit no., parties as plaintiff and defendant)",
            "Description of the parties",
            "Jurisdiction of the court (pecuniary and territorial)",
            "Facts constituting the cause of action and when it arose",
            "Cause of action paragraph",
            "Valuation for court-fee and jurisdiction",
            "Limitation (the suit is within time)",
            "Relief(s) claimed (the prayer)",
            "Verification and list of documents",
        ],
        required_inputs=["court", "plaintiff", "defendant", "facts", "cause_of_action", "relief", "valuation"],
        authority_refs=["Code of Civil Procedure, 1908 (Order VII)", "the substantive statute relied on"],
    ),
    DocType(
        id="written_statement",
        title="Written Statement",
        practice_area="Litigation",
        description="The defendant's written statement in answer to a plaint (CPC Order VIII) — not an 'answer'.",
        structure=[
            "Cause title (titled 'Written Statement on behalf of the defendant')",
            "Preliminary objections (maintainability, limitation, jurisdiction, cause of action)",
            "Para-wise reply to the plaint (admissions and denials)",
            "Additional pleas / set-off or counter-claim, if any",
            "Prayer (for dismissal of the suit)",
            "Verification",
        ],
        required_inputs=["court", "suit_reference", "plaint_allegations", "defences"],
        authority_refs=["Code of Civil Procedure, 1908 (Order VIII)"],
    ),
    DocType(
        id="writ_petition",
        title="Writ Petition",
        practice_area="Litigation",
        description="A writ petition under Article 226 (High Court) / Article 32 (Supreme Court).",
        structure=[
            "Cause title (in the High Court of … / Supreme Court of India, writ jurisdiction)",
            "Details of the petitioner and the respondent (State / authority)",
            "Facts of the case",
            "The impugned action / order and the fundamental or legal right infringed",
            "Grounds (why the action is illegal, arbitrary, or ultra vires)",
            "Whether an alternative remedy exists and why it is not efficacious",
            "Interim relief sought",
            "Main prayer (the writ sought — mandamus / certiorari / prohibition)",
            "Verification and affidavit",
        ],
        required_inputs=["court", "petitioner", "respondent", "impugned_action", "right_infringed", "grounds", "relief"],
        authority_refs=["Constitution of India, Article 226 / Article 32 (as in force)"],
    ),
    DocType(
        id="reply_to_legal_notice",
        title="Reply to Legal Notice",
        practice_area="Litigation",
        description="A reply on behalf of the noticee, denying or answering a legal notice.",
        structure=[
            "Advocate's letterhead (name, Bar enrolment no., office address)",
            "Subject: 'Reply to Legal Notice dated … on behalf of …'",
            "Opening (under instructions from the client; receipt and noting of the notice)",
            "Para-wise denial of the allegations (denied in toto or specifically denied)",
            "The client's version of facts, chronologically, with supporting annexures",
            "Legal arguments (statutes and contract clauses relied on)",
            "Reservation of rights (the reply is without prejudice)",
            "Signature of the advocate / client",
        ],
        required_inputs=["client_name", "original_notice_reference", "denials", "client_facts"],
        authority_refs=["the substantive statute and contract clauses relied on (cite as in force)"],
    ),
    DocType(
        id="counter_claim",
        title="Counter-Claim",
        practice_area="Litigation",
        description="A defendant's counter-claim under CPC Order VIII Rule 6A — an independent cross-suit raised in the written statement.",
        structure=[
            "Cause title (counter-claim in the suit, defendant as counter-claimant)",
            "Statement that the defendant raises this by way of counter-claim (Order VIII Rule 6A)",
            "Facts constituting the counter-claim and the cause of action",
            "Jurisdiction and that the counter-claim is within the court's pecuniary limits",
            "Valuation and court-fee on the counter-claim",
            "Limitation (the counter-claim is within time)",
            "Relief claimed on the counter-claim (the prayer)",
            "Verification (counter-claim is governed by the rules applicable to a plaint)",
        ],
        required_inputs=["court", "suit_reference", "counter_claim_facts", "relief", "valuation"],
        authority_refs=["Code of Civil Procedure, 1908 (Order VIII Rule 6A)"],
    ),
    DocType(
        id="interlocutory_application",
        title="Interlocutory Application",
        practice_area="Litigation",
        description="An interlocutory application (interim injunction / stay / condonation of delay) under CPC Order XXXIX or the relevant provision.",
        structure=[
            "Cause title (the application in the pending suit/petition, with the IA number)",
            "Description of the parties (applicant and non-applicant)",
            "The interim relief sought and the provision invoked (e.g. Order XXXIX Rules 1-2)",
            "Facts and the urgency justifying interim orders",
            "Prima facie case, balance of convenience and irreparable injury",
            "Prayer for the specific interim relief",
            "Supporting affidavit and verification",
        ],
        required_inputs=["court", "suit_reference", "applicant", "interim_relief", "grounds"],
        authority_refs=["Code of Civil Procedure, 1908 (Order XXXIX / the provision invoked, as in force)"],
    ),
    DocType(
        id="interrogatories",
        title="Interrogatories",
        practice_area="Litigation",
        description="Written interrogatories for the examination of the opposite party under CPC Order XI (Form No. 2, Appendix C).",
        structure=[
            "Cause title (interrogatories in the suit, by which party to which party)",
            "Note at the foot stating which party must answer which interrogatories",
            "The numbered interrogatories (limited to facts in issue — not law or inference)",
            "Statement that they are delivered with the leave of the court",
            "Direction that the interrogatories be answered by affidavit within ten days",
        ],
        required_inputs=["court", "suit_reference", "delivering_party", "answering_party", "questions"],
        authority_refs=["Code of Civil Procedure, 1908 (Order XI Rule 1; Form No. 2, Appendix C)"],
    ),
    DocType(
        id="execution_petition",
        title="Execution Petition",
        practice_area="Litigation",
        description="A petition to execute a decree under CPC Order XXI Rule 11(2) — the mandatory tabular form.",
        structure=[
            "Cause title (execution petition before the court which passed the decree / Section 38 court)",
            "Tabular particulars (Order XXI Rule 11(2)): suit number; decree-holder and judgment-debtor; date of decree; whether an appeal is pending; amount of costs and interest due to date",
            "Statement of non-compliance by the judgment-debtor",
            "The mode of execution sought (arrest / attachment and sale / delivery of possession)",
            "Prayer (the precise mode of assistance from the court)",
            "Schedule of property to be attached, if any, and verification",
        ],
        required_inputs=["court", "decree_reference", "decree_holder", "judgment_debtor", "amount_due", "mode_of_execution"],
        authority_refs=["Code of Civil Procedure, 1908 (Order XXI Rule 11(2); Section 38)"],
    ),
    DocType(
        id="special_leave_petition",
        title="Special Leave Petition (SLP)",
        practice_area="Litigation",
        description="A special leave petition to the Supreme Court under Article 136 (Form No. 28, Supreme Court Rules, 2013).",
        structure=[
            "Cause title (in the Supreme Court of India, petition for special leave to appeal, Article 136)",
            "Synopsis and list of dates (chronological summary up to the impugned order)",
            "Details of the petitioner and respondent",
            "Facts of the case",
            "The impugned judgment/order appealed against",
            "Questions of law of general importance arising for consideration",
            "Grounds for grant of special leave",
            "Relief sought (special leave to appeal and interim relief)",
            "Certificate, affidavit, and signature of the Advocate-on-Record",
        ],
        required_inputs=["impugned_order", "petitioner", "respondent", "questions_of_law", "grounds", "relief"],
        authority_refs=["Constitution of India, Article 136 (as in force)", "Supreme Court Rules, 2013 (Form No. 28)"],
    ),
    DocType(
        id="written_arguments",
        title="Written Arguments",
        practice_area="Litigation",
        description="Written submissions / written arguments filed at the conclusion of the hearing.",
        structure=[
            "Cause title (written arguments on behalf of the party, in the suit/petition)",
            "Brief statement of facts and the issues for determination",
            "Issue-wise submissions with the supporting evidence and exhibits referenced",
            "Legal submissions with citations to the authorities relied on",
            "Answer to the opposite party's contentions",
            "Conclusion and the relief prayed for",
        ],
        required_inputs=["court", "suit_reference", "party", "issues", "submissions"],
        authority_refs=["the case law and statutory provisions relied on (cite as in force)"],
    ),

    # ── Corporate / M&A / Transactional (§1.B) ────────────────────────────────────
    DocType(
        id="nda",
        title="Non-Disclosure Agreement (NDA)",
        practice_area="Transactional",
        description="A mutual or one-way NDA governing confidential information shared between parties.",
        structure=[
            "Parties and effective date",
            "Definition of Confidential Information",
            "Purpose / permitted use",
            "Obligations of confidentiality and exclusions",
            "Term and survival of obligations",
            "Return or destruction of information",
            "Remedies (including injunctive relief)",
            "Governing law and dispute resolution",
            "Boilerplate (notices, assignment, entire agreement)",
        ],
        required_inputs=["disclosing_party", "receiving_party", "purpose", "term", "governing_law"],
        authority_refs=["Indian Contract Act, 1872 (as in force)"],
    ),
    DocType(
        id="spa",
        title="Share Purchase Agreement (SPA)",
        practice_area="Transactional",
        description="A share purchase agreement for the sale/acquisition of shares in an Indian company.",
        structure=[
            "Parties (seller, purchaser, the company)",
            "Definitions and interpretation",
            "Sale and purchase of the sale shares and consideration",
            "Conditions precedent to closing",
            "Closing mechanics and deliverables",
            "Representations and warranties (of seller and purchaser)",
            "Indemnification (caps, baskets, survival)",
            "Covenants (conduct between signing and closing)",
            "Termination",
            "Governing law and dispute resolution",
            "Boilerplate",
            "Schedules (disclosure schedule, capitalization)",
        ],
        required_inputs=["seller", "purchaser", "company", "sale_shares", "consideration", "governing_law"],
        authority_refs=["Companies Act, 2013 (share transfer provisions, as in force)", "Indian Contract Act, 1872"],
    ),
    DocType(
        id="board_resolution",
        title="Board Resolution",
        practice_area="Transactional",
        description="A resolution of the board of directors of an Indian company under the Companies Act, 2013.",
        structure=[
            "Name of the company and reference to the board meeting (date, place)",
            "Recital of the matter placed before the board",
            "The resolution ('RESOLVED THAT …')",
            "Authorisation (the director/officer authorised to act and file)",
            "Certified true copy attestation",
        ],
        required_inputs=["company", "meeting_date", "matter", "authorised_person"],
        authority_refs=["Companies Act, 2013 (the section governing the matter resolved, as in force)"],
    ),
    DocType(
        id="shareholders_agreement",
        title="Shareholders' Agreement (SHA)",
        practice_area="Transactional",
        description="A shareholders' agreement governing the rights of shareholders inter se in an Indian company.",
        structure=[
            "Parties (the shareholders and the company) and definitions",
            "Governance — board composition, quorum, and shareholder meetings",
            "Reserved matters / affirmative-vote items requiring investor consent",
            "Information and inspection rights",
            "Transfer restrictions — right of first refusal / right of first offer",
            "Tag-along (co-sale) rights",
            "Drag-along rights",
            "Anti-dilution protection",
            "Exit mechanics (IPO, strategic sale, buy-back)",
            "Representations and warranties",
            "Deadlock and dispute resolution",
            "Governing law (the SHA must not conflict with the AOA)",
        ],
        required_inputs=["company", "shareholders", "reserved_matters", "transfer_restrictions", "governing_law"],
        authority_refs=["Companies Act, 2013 (Section 58(2), share transfer; as in force)", "Indian Contract Act, 1872"],
    ),
    DocType(
        id="term_sheet",
        title="Term Sheet",
        practice_area="Transactional",
        description="An investment term sheet (LOI / MOU) — non-binding on commercial terms, binding on confidentiality, exclusivity and governing law.",
        structure=[
            "Parties (the company, the investor) and the proposed transaction",
            "Type of security and the investment amount (with any tranches)",
            "Pre-money / post-money valuation and the resulting shareholding (fully diluted)",
            "Liquidation preference",
            "Board composition and investor governance / reserved matters",
            "Anti-dilution and pre-emption rights",
            "Conditions precedent and due-diligence",
            "Exclusivity (no-shop) and confidentiality (binding)",
            "Binding vs non-binding nature of the clauses",
            "Governing law and dispute resolution (binding)",
        ],
        required_inputs=["company", "investor", "investment_amount", "valuation", "security_type"],
        authority_refs=["Indian Contract Act, 1872 (as in force)", "Companies Act, 2013 (issue of securities)"],
    ),
    DocType(
        id="moa",
        title="Memorandum of Association (MOA)",
        practice_area="Transactional",
        description="The memorandum of association — the company's constitution under Section 4, Companies Act, 2013.",
        structure=[
            "Name clause",
            "Registered office clause (the State of the registered office)",
            "Object clause (the objects for which the company is incorporated)",
            "Liability clause (limited by shares / by guarantee)",
            "Capital clause (the authorised share capital and its division)",
            "Subscription clause (subscribers' names, addresses and shares taken)",
        ],
        required_inputs=["company_name", "registered_office_state", "objects", "authorised_capital", "subscribers"],
        authority_refs=["Companies Act, 2013 (Section 4; Schedule I, Tables A-E; as in force)"],
    ),
    DocType(
        id="aoa",
        title="Articles of Association (AOA)",
        practice_area="Transactional",
        description="The articles of association — the internal regulations of the company under Section 5, Companies Act, 2013.",
        structure=[
            "Interpretation and the table adopted (Table F, etc.)",
            "Share capital and variation of rights",
            "Calls on shares, lien, transfer and transmission of shares",
            "Alteration of capital and buy-back",
            "General meetings and proceedings at general meetings",
            "Directors — appointment, powers, and proceedings of the board",
            "Dividends, reserves and accounts",
            "Winding up and indemnity",
        ],
        required_inputs=["company_name", "share_capital_rules", "board_rules", "table_adopted"],
        authority_refs=["Companies Act, 2013 (Section 5, Section 14 alteration; Schedule I Table F; as in force)"],
    ),
    DocType(
        id="msa",
        title="Master Services Agreement (MSA)",
        practice_area="Transactional",
        description="A framework master services agreement under which project-specific Statements of Work (SOWs) are executed.",
        structure=[
            "Parties, effective date, and the framework nature of the agreement",
            "Statement of Work (SOW) mechanism — how project scope is added",
            "Service levels (SLA) and service credits",
            "Fees, invoicing and payment",
            "Intellectual property ownership and licence",
            "Confidentiality and data protection (DPDP Act data-processing obligations)",
            "Indemnity (IP infringement, breach, negligence)",
            "Limitation of liability (cap, with carve-outs for fraud / gross negligence)",
            "Term, termination and exit / transition assistance",
            "Governing law and dispute resolution",
        ],
        required_inputs=["service_provider", "customer", "services", "fees", "governing_law"],
        authority_refs=["Indian Contract Act, 1872 (Sections 124-125 indemnity; as in force)", "Digital Personal Data Protection Act, 2023"],
    ),
    DocType(
        id="saas_agreement",
        title="SaaS / Subscription Agreement",
        practice_area="Transactional",
        description="A software-as-a-service / subscription agreement for access to a hosted platform.",
        structure=[
            "Parties and definitions (the platform, the subscriber)",
            "Grant of the subscription (the right to access and use)",
            "Subscription term, fees and auto-renewal",
            "Acceptable use restrictions",
            "Service levels, uptime and support",
            "Customer data, security and data protection (DPDP Act compliance)",
            "Intellectual property and feedback licence",
            "Warranties and disclaimers",
            "Indemnity and limitation of liability",
            "Suspension, termination and data return on exit",
            "Governing law and dispute resolution",
        ],
        required_inputs=["provider", "subscriber", "subscription_scope", "fees", "governing_law"],
        authority_refs=["Indian Contract Act, 1872 (as in force)", "Digital Personal Data Protection Act, 2023", "Information Technology Act, 2000"],
    ),
    DocType(
        id="dpa",
        title="Data Processing Agreement (DPA)",
        practice_area="Transactional",
        description="A data processing agreement allocating Data Fiduciary / Data Processor obligations under the DPDP Act, 2023.",
        structure=[
            "Parties and their roles (Data Fiduciary and Data Processor)",
            "Subject-matter, nature and purpose of the processing",
            "Categories of personal data and data principals",
            "Processor obligations (process only on instructions; confidentiality; security safeguards)",
            "Sub-processing and flow-down obligations",
            "Personal data breach notification timelines",
            "Cross-border transfer restrictions",
            "Assistance with data-principal rights and audits",
            "Return or deletion of personal data on termination",
            "Governing law",
        ],
        required_inputs=["data_fiduciary", "data_processor", "processing_purpose", "data_categories"],
        authority_refs=["Digital Personal Data Protection Act, 2023 (as in force)"],
    ),
    DocType(
        id="due_diligence_request_list",
        title="Due-Diligence Request List",
        practice_area="Transactional",
        description="A diligence request list itemising the documents to be produced into the data room for a transaction.",
        structure=[
            "Heading (the transaction, the target, the requesting party)",
            "Corporate and constitutional documents requested (MOA/AOA, statutory registers)",
            "Capitalisation, shareholding and securities records",
            "Material contracts and commercial arrangements",
            "Financial statements, tax and audit records",
            "Litigation, disputes and regulatory matters",
            "Intellectual property, employment and property records",
            "Instructions (format, timeline and the data-room index)",
        ],
        required_inputs=["target", "transaction", "requested_categories"],
        authority_refs=["the statutes governing the requested records (Companies Act, 2013; tax and labour laws; as in force)"],
    ),

    # ── Banking & Finance (§1.C — ties to the F3 Covenant flagship) ────────────────
    DocType(
        id="intercreditor_agreement",
        title="Intercreditor Agreement",
        practice_area="Financial services",
        description="An intercreditor agreement ranking and regulating the rights of multiple lenders.",
        structure=[
            "Parties (the creditors, the debtor, the security trustee/agent)",
            "Definitions and interpretation",
            "Ranking and priority of the debts and security",
            "Permitted payments and payment blockage / standstill",
            "Turnover of recoveries",
            "Enforcement of security and the role of the security trustee",
            "Application of proceeds (the waterfall)",
            "Subordination of intra-group / shareholder debt",
            "Amendments, waivers and decision-making among creditors",
            "Governing law and dispute resolution",
        ],
        required_inputs=["senior_creditor", "junior_creditor", "debtor", "ranking", "governing_law"],
        authority_refs=["Indian Contract Act, 1872", "the applicable security enforcement statute (e.g. SARFAESI, 2002, as in force)"],
    ),
    DocType(
        id="facility_agreement",
        title="Facility Agreement",
        practice_area="Financial services",
        description="A loan / facility agreement between a lender and a borrower (term or working-capital facility).",
        structure=[
            "Parties (the lender(s), the borrower, the agent/security trustee) and definitions",
            "The facility — commitment amount, purpose and availability period",
            "Conditions precedent to drawdown",
            "Drawdown mechanics (drawdown notice; repeating representations)",
            "Interest, fees and repayment / prepayment",
            "Representations and warranties",
            "Financial covenants and information undertakings",
            "General undertakings (positive and negative covenants)",
            "Events of default and acceleration",
            "Security and the security documents",
            "Governing law and dispute resolution",
        ],
        required_inputs=["lender", "borrower", "facility_amount", "purpose", "governing_law"],
        authority_refs=["Indian Contract Act, 1872 (as in force)", "Reserve Bank of India directions applicable to the facility"],
    ),
    DocType(
        id="deed_of_guarantee",
        title="Deed of Guarantee",
        practice_area="Financial services",
        description="A deed of guarantee securing a borrower's obligations to a lender (a contract of guarantee under the Contract Act).",
        structure=[
            "Parties (the guarantor, the lender / beneficiary, and the principal borrower)",
            "Recitals (the underlying facility being guaranteed)",
            "The guarantee and indemnity (guarantee of the secured obligations)",
            "Continuing nature of the guarantee",
            "Demand and payment mechanics",
            "Preservation of the guarantor's liability (waiver of discharge defences)",
            "Representations and warranties of the guarantor",
            "Subrogation deferral",
            "Governing law and dispute resolution",
        ],
        required_inputs=["guarantor", "lender", "principal_borrower", "guaranteed_obligations"],
        authority_refs=["Indian Contract Act, 1872 (Sections 126-147, contract of guarantee; as in force)"],
    ),
    DocType(
        id="deed_of_hypothecation",
        title="Deed of Hypothecation",
        practice_area="Financial services",
        description="A deed of hypothecation creating a charge over movable assets in favour of the lender / security trustee.",
        structure=[
            "Parties (the borrower / hypothecator and the lender / security trustee)",
            "Recitals (the secured facility)",
            "Description of the hypothecated (secured) assets",
            "Creation of the charge by way of hypothecation (first / exclusive charge)",
            "The secured obligations",
            "Covenants of the hypothecator (insurance, maintenance, no further charge)",
            "Enforcement rights on an event of default",
            "Registration of the charge (Companies Act / CERSAI)",
            "Governing law",
        ],
        required_inputs=["hypothecator", "lender", "hypothecated_assets", "secured_obligations"],
        authority_refs=["Indian Contract Act, 1872", "Companies Act, 2013 (Section 77, registration of charges); SARFAESI Act, 2002 (as in force)"],
    ),
    DocType(
        id="debenture_trust_deed",
        title="Debenture Trust Deed",
        practice_area="Financial services",
        description="A debenture trust deed between the issuer and the debenture trustee for the benefit of debenture holders (SEBI format).",
        structure=[
            "Parties (the issuer and the debenture trustee, in a fiduciary capacity)",
            "Recitals and the terms of the issue of debentures",
            "Description of the security created (charge / mortgage / pledge / hypothecation)",
            "Description of the assets over which the security is created",
            "Covenants of the issuer (financial covenants, reporting)",
            "Powers, duties and remuneration of the debenture trustee",
            "Events of default and enforcement of security",
            "Meetings of debenture holders and modification (consent of the majority)",
            "Governing law",
        ],
        required_inputs=["issuer", "debenture_trustee", "debenture_terms", "security_assets"],
        authority_refs=["Companies Act, 2013 (Section 71, debentures)", "SEBI (Debenture Trustees) Regulations, 1993 (as in force)"],
    ),
    DocType(
        id="commitment_letter",
        title="Commitment Letter",
        practice_area="Financial services",
        description="A lender's commitment letter setting out the indicative terms on which it is willing to provide a facility.",
        structure=[
            "Addressee (the borrower) and the proposed facility",
            "The commitment (amount, type and tenor of the facility)",
            "Indicative pricing (interest, fees) and the term sheet attached",
            "Conditions to the commitment (conditions precedent, due diligence)",
            "Market-flex / material adverse change rights",
            "Confidentiality and exclusivity (binding)",
            "Expiry of the commitment and governing law",
        ],
        required_inputs=["lender", "borrower", "facility_amount", "indicative_terms"],
        authority_refs=["Indian Contract Act, 1872 (as in force)"],
    ),

    # ── Real Estate / Property / Conveyancing (§1.D) ──────────────────────────────
    DocType(
        id="sale_deed",
        title="Sale Deed",
        practice_area="Real estate",
        description="A sale deed conveying title to immovable property from seller to purchaser (Transfer of Property Act).",
        structure=[
            "Title and parties (the vendor and the purchaser)",
            "Recitals (the vendor's title and devolution)",
            "Consideration and receipt of the sale price",
            "Operative conveyance (the transfer of the property)",
            "Schedule of the property (description and boundaries)",
            "Covenants for title and quiet enjoyment",
            "Indemnity against defects in title",
            "Possession and delivery",
            "Stamp duty and registration particulars",
            "Execution, attestation and registration",
        ],
        required_inputs=["vendor", "purchaser", "property_description", "consideration"],
        authority_refs=["Transfer of Property Act, 1882 (Section 54, sale)", "Registration Act, 1908", "the applicable State Stamp Act"],
    ),
    DocType(
        id="agreement_to_sell",
        title="Agreement to Sell",
        practice_area="Real estate",
        description="An agreement to sell immovable property — an executory contract to convey, not a conveyance.",
        structure=[
            "Parties (the proposed seller and buyer)",
            "Recitals (the seller's title)",
            "Agreement to sell and the agreed consideration",
            "Payment schedule (advance / earnest money and balance)",
            "Conditions to completion (title clearance, approvals)",
            "Time for execution of the sale deed",
            "Default and forfeiture / refund of earnest money",
            "Schedule of the property",
            "Governing law and dispute resolution",
        ],
        required_inputs=["seller", "buyer", "property_description", "consideration", "completion_date"],
        authority_refs=["Transfer of Property Act, 1882", "Indian Contract Act, 1872; Specific Relief Act, 1963 (as in force)"],
    ),
    DocType(
        id="lease_deed",
        title="Lease Deed / Leave-and-Licence",
        practice_area="Real estate",
        description="A lease deed or leave-and-licence agreement granting the right to occupy immovable property.",
        structure=[
            "Parties (the lessor / licensor and the lessee / licensee)",
            "Demised premises and the schedule of the property",
            "Term, commencement and renewal / lock-in",
            "Rent / licence fee, escalation and security deposit",
            "Permitted use and restrictions",
            "Maintenance, repairs and outgoings",
            "Lessee / licensee covenants and lessor covenants",
            "Termination, forfeiture and handover",
            "Stamp duty and registration",
            "Governing law and dispute resolution",
        ],
        required_inputs=["lessor", "lessee", "premises", "term", "rent"],
        authority_refs=["Transfer of Property Act, 1882 (Section 105, lease)", "Registration Act, 1908; the applicable State rent / stamp law"],
    ),
    DocType(
        id="gift_deed",
        title="Gift Deed",
        practice_area="Real estate",
        description="A gift deed transferring property voluntarily and without consideration (Transfer of Property Act).",
        structure=[
            "Parties (the donor and the donee)",
            "Recitals (the donor's title and natural love and affection / donative intent)",
            "The gift (voluntary transfer without consideration)",
            "Acceptance of the gift by the donee",
            "Schedule of the property gifted",
            "Delivery of possession",
            "Stamp duty and registration particulars",
            "Execution, attestation and registration",
        ],
        required_inputs=["donor", "donee", "property_description"],
        authority_refs=["Transfer of Property Act, 1882 (Section 122, gift)", "Registration Act, 1908; the applicable State Stamp Act"],
    ),
    DocType(
        id="development_agreement",
        title="Development Agreement",
        practice_area="Real estate",
        description="A development / joint-development agreement between a landowner and a developer.",
        structure=[
            "Parties (the landowner and the developer) and recitals of title",
            "Grant of development rights",
            "Consideration / revenue-or-area sharing ratio",
            "Developer's obligations (approvals, construction, timelines)",
            "Landowner's obligations (title, possession, power of attorney)",
            "Permissions, sanctions and RERA registration",
            "Allocation of the built-up area / units",
            "Default, termination and dispute resolution",
            "Schedule of the property and stamp duty / registration",
        ],
        required_inputs=["landowner", "developer", "property_description", "sharing_ratio"],
        authority_refs=["Transfer of Property Act, 1882; Indian Contract Act, 1872", "Real Estate (Regulation and Development) Act, 2016 (as in force)"],
    ),
    DocType(
        id="title_opinion",
        title="Title Search & Opinion",
        practice_area="Real estate",
        description="A title search report and legal opinion on the marketability of title to immovable property.",
        structure=[
            "Subject property and the scope of the search",
            "Documents examined and the period of search",
            "Devolution / chain of title",
            "Encumbrances, charges and litigation noted",
            "Statutory approvals and revenue records examined",
            "Requisitions / defects and how to cure them",
            "Opinion on marketability of title",
        ],
        required_inputs=["property_description", "documents_examined", "search_period"],
        authority_refs=["Transfer of Property Act, 1882; Registration Act, 1908; the applicable State land-revenue and RERA laws (as in force)"],
    ),

    # ── Employment & Labour (§1.E) ────────────────────────────────────────────────
    DocType(
        id="employment_agreement",
        title="Employment Agreement",
        practice_area="Employment",
        description="An employment agreement between an Indian employer and an employee.",
        structure=[
            "Parties, designation and date of joining",
            "Term and probation",
            "Duties, place of work and reporting",
            "Remuneration, benefits and statutory contributions (PF / ESIC / gratuity)",
            "Working hours and leave",
            "Confidentiality and intellectual property assignment",
            "Non-solicitation (and any permissible restrictive covenant)",
            "Code of conduct and POSH compliance",
            "Termination and notice period",
            "Governing law and dispute resolution",
        ],
        required_inputs=["employer", "employee", "designation", "remuneration", "notice_period"],
        authority_refs=["Indian Contract Act, 1872; the applicable Shops & Establishments Act and labour codes; Code on Wages, 2019 (as in force)"],
    ),
    DocType(
        id="appointment_letter",
        title="Appointment / Offer Letter",
        practice_area="Employment",
        description="An appointment / offer letter extending employment on stated terms.",
        structure=[
            "Addressee and the offer of appointment to the role",
            "Date of joining and place of posting",
            "Compensation (CTC) and benefits break-up",
            "Probation and confirmation",
            "Key terms (working hours, leave, notice period)",
            "Conditions of the offer (background verification, documents)",
            "Reference to the detailed employment agreement and policies",
            "Acceptance",
        ],
        required_inputs=["candidate_name", "role", "compensation", "joining_date"],
        authority_refs=["the applicable Shops & Establishments Act; Code on Wages, 2019 (as in force)"],
    ),
    DocType(
        id="posh_policy",
        title="POSH Policy",
        practice_area="Employment",
        description="A workplace policy on Prevention of Sexual Harassment under the SH Act, 2013.",
        structure=[
            "Policy statement, scope and applicability (employees, interns, contract workers)",
            "Definition of sexual harassment at the workplace",
            "Constitution of the Internal Committee (Presiding Officer; at least half women members)",
            "Complaint mechanism (written complaint to the Internal Committee)",
            "Inquiry procedure and timelines (inquiry completed within 90 days)",
            "Confidentiality and protection against retaliation",
            "Interim relief and recommendations / action on the report",
            "Safeguards against false or malicious complaints",
            "Awareness, training and display obligations of the employer",
        ],
        required_inputs=["organisation", "internal_committee_members"],
        authority_refs=["Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013 (as in force)"],
    ),
    DocType(
        id="show_cause_notice",
        title="Show-Cause Notice / Charge Sheet",
        practice_area="Employment",
        description="A show-cause notice or charge sheet initiating disciplinary proceedings (domestic enquiry), observing natural justice.",
        structure=[
            "Addressee (the delinquent employee) and reference to the standing orders",
            "The day, time and place of the alleged incident",
            "Short summary of the incident and the persons involved",
            "The specific articles of charge / misconduct alleged (not vague)",
            "The standing-order / conduct rule alleged to be breached",
            "Call to show cause / submit a written explanation within a stated time",
            "Notice that a domestic enquiry may follow (natural justice; right to be heard and to cross-examine)",
            "Signature of the disciplinary authority",
        ],
        required_inputs=["employee", "incident", "charges", "reply_period"],
        authority_refs=["the applicable Standing Orders (Industrial Employment (Standing Orders) Act, 1946) and the principles of natural justice (as in force)"],
    ),

    # ── IP & Technology (§1.F) ────────────────────────────────────────────────────
    DocType(
        id="trademark_application",
        title="Trademark Application",
        practice_area="IP & Technology",
        description="A trademark application on Form TM-A under the Trade Marks Act, 1999.",
        structure=[
            "Applicant details and address for service in India",
            "The mark (word / device) and its representation",
            "Class(es) of goods/services and the specification (NICE classification)",
            "Date of first use or 'proposed to be used'",
            "Statement of use / user affidavit, if claiming prior use",
            "Claim to priority, if any",
            "Verification and signature (Form TM-A)",
        ],
        required_inputs=["applicant", "mark", "classes", "goods_services"],
        authority_refs=["Trade Marks Act, 1999; Trade Marks Rules, 2017 (Form TM-A; as in force)"],
    ),
    DocType(
        id="trademark_examination_reply",
        title="Reply to Trademark Examination Report",
        practice_area="IP & Technology",
        description="A reply to the examination report objections raised under Sections 9 and 11 of the Trade Marks Act, 1999.",
        structure=[
            "Application and mark reference, and the examination report objected to",
            "Response to Section 9 objections (distinctiveness / descriptiveness)",
            "Response to Section 11 objections (conflicting earlier marks)",
            "Legal arguments under the Trade Marks Act, 1999 with authorities",
            "Evidence of distinctiveness / use (sales figures, advertising spend, affidavit)",
            "Request for acceptance and advertisement (reply within 30 days, Rule 29)",
        ],
        required_inputs=["application_reference", "mark", "objections", "arguments"],
        authority_refs=["Trade Marks Act, 1999 (Sections 9 and 11); Trade Marks Rules, 2017 (Rule 29; as in force)"],
    ),
    DocType(
        id="ip_licence",
        title="IP Licence Agreement",
        practice_area="IP & Technology",
        description="A licence of intellectual property (trademark / copyright / patent / technology) from a licensor to a licensee.",
        structure=[
            "Parties and the licensed intellectual property",
            "Grant of licence (exclusive / non-exclusive; field and territory)",
            "Permitted use and restrictions",
            "Royalties / licence fee and reporting",
            "Quality control and the licensor's standards",
            "Ownership, improvements and acknowledgement of the licensor's rights",
            "Warranties and indemnity (non-infringement)",
            "Term, termination and effect of termination",
            "Governing law and dispute resolution",
        ],
        required_inputs=["licensor", "licensee", "licensed_ip", "royalty", "territory"],
        authority_refs=["the relevant IP statute (Trade Marks Act, 1999 / Copyright Act, 1957 / Patents Act, 1970); Indian Contract Act, 1872 (as in force)"],
    ),

    # ── Tax (§1.G — finance-adjacent) ─────────────────────────────────────────────
    DocType(
        id="gst_scn_reply",
        title="Reply to GST Show-Cause Notice",
        practice_area="Tax",
        description="A reply (Form DRC-06) to a GST show-cause notice (DRC-01) issued under Section 73/74 of the CGST Act.",
        structure=[
            "Reference to the show-cause notice (DRC-01) and the demand proposed",
            "Preliminary objections (time-bar under Section 73/74; jurisdiction)",
            "Para-wise reply to the allegations",
            "Factual basis and evidence (invoices, payment proof, returns)",
            "Legal submissions under the CGST Act and rules, with authorities",
            "Request for a personal hearing (Section 75(4))",
            "Prayer (that the proposed demand be dropped) and list of documents",
        ],
        required_inputs=["scn_reference", "demand", "objections", "evidence"],
        authority_refs=["Central Goods and Services Tax Act, 2017 (Sections 73, 74, 75(4); Form DRC-06; as in force)"],
    ),
    DocType(
        id="advance_ruling_application",
        title="Advance Ruling Application",
        practice_area="Tax",
        description="An application for an advance ruling before the Authority for Advance Ruling (GST / income-tax).",
        structure=[
            "Applicant details and the registration / PAN",
            "The transaction or activity on which the ruling is sought",
            "The specific question(s) framed for the advance ruling",
            "Statement of the relevant facts",
            "The applicant's interpretation and grounds",
            "Statement that the question is not pending in any other proceeding",
            "Verification and the prescribed fee",
        ],
        required_inputs=["applicant", "transaction", "questions"],
        authority_refs=["Central Goods and Services Tax Act, 2017 (Chapter XVII, advance ruling) / Income-tax Act, 1961 (as in force)"],
    ),
    DocType(
        id="tax_opinion",
        title="Tax Opinion",
        practice_area="Tax",
        description="A written tax opinion / memorandum on the tax treatment of a transaction.",
        structure=[
            "Scope of the opinion and the question presented",
            "Statement of the relevant facts and assumptions",
            "The applicable statutory provisions and rules",
            "Analysis and the relevant case law / circulars",
            "Risks, caveats and alternative positions",
            "Conclusion / opinion on the tax treatment",
        ],
        required_inputs=["transaction", "question", "facts"],
        authority_refs=["the relevant tax statute (Income-tax Act, 1961 / CGST Act, 2017) and binding circulars (as in force)"],
    ),

    # ── Regulatory / Compliance / In-House (§1.H) ─────────────────────────────────
    DocType(
        id="litigation_hold_notice",
        title="Litigation-Hold Notice",
        practice_area="Compliance (India)",
        description="A litigation-hold (legal-hold) notice directing custodians to preserve documents relevant to anticipated or pending litigation.",
        structure=[
            "Recipients (the custodians) and the matter to which the hold relates",
            "Statement that litigation is reasonably anticipated / pending",
            "Scope of the documents and data to be preserved (categories, date range)",
            "Suspension of routine deletion / destruction and auto-purge",
            "Custodian acknowledgement and obligations",
            "Point of contact and the duration of the hold",
        ],
        required_inputs=["custodians", "matter", "preservation_scope"],
        authority_refs=["the evidentiary preservation obligations under the Bharatiya Sakshya Adhiniyam, 2023 / applicable procedure (as in force)"],
    ),
    DocType(
        id="regulatory_response",
        title="Response to a Regulator",
        practice_area="Compliance (India)",
        description="A response / corrective-action reply to a regulator's notice or inspection observation (SEBI / RBI / sectoral regulator).",
        structure=[
            "Reference to the regulator's notice / observation and its date",
            "Background and the entity's regulatory status",
            "Point-wise response to each observation / allegation",
            "Corrective and preventive action taken or proposed (with timelines)",
            "Supporting evidence and annexures",
            "Assurance of compliance and request for closure",
        ],
        required_inputs=["regulator", "notice_reference", "observations", "corrective_action"],
        authority_refs=["the regulations administered by the regulator (e.g. SEBI / RBI directions; as in force)"],
    ),
    DocType(
        id="board_minutes",
        title="Board-Meeting Minutes",
        practice_area="Compliance (India)",
        description="Minutes of a board meeting recorded under the Companies Act, 2013 and Secretarial Standard SS-1.",
        structure=[
            "Heading (company name, meeting number, date, time and place)",
            "Directors and invitees present, and the quorum",
            "Confirmation of the minutes of the previous meeting",
            "Items of business taken up and the deliberations",
            "Resolutions passed (with the manner of approval)",
            "Time of conclusion and signature of the chairperson",
        ],
        required_inputs=["company", "meeting_date", "directors_present", "business_items"],
        authority_refs=["Companies Act, 2013 (Section 118); Secretarial Standard SS-1 (as in force)"],
    ),

    # ── Family / Private Client (§1.I) ────────────────────────────────────────────
    DocType(
        id="will",
        title="Will",
        practice_area="Family",
        description="A will (testament) disposing of the testator's property, executed under the Indian Succession Act, 1925.",
        structure=[
            "Declaration (the testator's name, age, soundness of mind and that this is the last will)",
            "Revocation of all earlier wills and codicils",
            "Appointment of the executor",
            "Specific bequests (named property to named beneficiaries)",
            "Residuary bequest",
            "Provisions for minors / contingent beneficiaries, if any",
            "Date and signature of the testator",
            "Attestation by two witnesses (a beneficiary should not attest)",
        ],
        required_inputs=["testator", "executor", "bequests", "beneficiaries"],
        authority_refs=["Indian Succession Act, 1925 (Sections 59-63, execution and attestation of wills; as in force)"],
    ),
    DocType(
        id="power_of_attorney",
        title="Power of Attorney",
        practice_area="Family",
        description="A power of attorney authorising an attorney to act on behalf of the principal (general or special).",
        structure=[
            "Parties (the principal / donor and the attorney / donee)",
            "Recitals and the purpose of the power",
            "The powers granted (general or specifically enumerated)",
            "Whether the power is durable and any limitations",
            "Ratification clause",
            "Revocation and duration",
            "Execution, attestation, stamping and (where required) registration",
        ],
        required_inputs=["principal", "attorney", "powers"],
        authority_refs=["Powers-of-Attorney Act, 1882; Indian Contract Act, 1872; Registration Act, 1908 (as in force)"],
    ),
    DocType(
        id="trust_deed",
        title="Trust Deed",
        practice_area="Family",
        description="A trust deed settling property on trust for named beneficiaries / objects.",
        structure=[
            "Parties (the settlor / author of the trust and the trustees)",
            "Declaration of trust and the trust property",
            "Objects / beneficiaries of the trust",
            "Powers and duties of the trustees",
            "Appointment, retirement and removal of trustees",
            "Application of income and corpus",
            "Accounts and audit",
            "Irrevocability / revocation and winding up",
            "Execution, stamping and registration",
        ],
        required_inputs=["settlor", "trustees", "trust_property", "beneficiaries"],
        authority_refs=["Indian Trusts Act, 1882; Registration Act, 1908 (as in force)"],
    ),
    DocType(
        id="partition_deed",
        title="Partition Deed",
        practice_area="Family",
        description="A partition deed dividing joint / coparcenary property among the co-owners / coparceners.",
        structure=[
            "Parties (the co-owners / coparceners) and recitals of the joint holding",
            "Statement of the joint / coparcenary property to be partitioned",
            "The agreed shares of each party",
            "Allotment of specific properties to each party by metes and bounds",
            "Mutual release and relinquishment of claims over the allotted portions",
            "Possession and the schedule of properties allotted",
            "Stamp duty and registration particulars",
            "Execution and attestation",
        ],
        required_inputs=["co_owners", "joint_property", "shares"],
        authority_refs=["Hindu Succession Act, 1956 (coparcenary); Transfer of Property Act, 1882; Registration Act, 1908 (as in force)"],
    ),
]


# The registry. The draft route / card picker reads this map.
CATALOG: Dict[str, DocType] = {d.id: d for d in _C}

# Practice-area display order (the card grid groups in this order; matches the gallery).
PRACTICE_ORDER = [
    "Litigation", "Transactional", "Financial services", "Real estate",
    "Employment", "IP & Technology", "Tax", "Compliance (India)", "Family",
]


def list_doc_types() -> List[DocType]:
    """All catalog rows, grouped by practice-area order then title (the card-grid order)."""
    order = {p: i for i, p in enumerate(PRACTICE_ORDER)}
    return sorted(CATALOG.values(), key=lambda d: (order.get(d.practice_area, 99), d.title))


def get_doc_type(doc_type_id: str) -> Optional[DocType]:
    return CATALOG.get(doc_type_id)


def doc_type_card(d: DocType) -> Dict[str, Any]:
    """The card-grid view (the §2.3 picker renders from this — no internal authority/template
    plumbing leaked; the form collects required_inputs, the draft fills the structure)."""
    return {
        "id": d.id,
        "title": d.title,
        "practice_area": d.practice_area,
        "jurisdiction": d.jurisdiction,
        "description": d.description,
        "required_inputs": d.required_inputs,
        "verb": "Draft",   # the §0 primitive verb this card launches
    }
