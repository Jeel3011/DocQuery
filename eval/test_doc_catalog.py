"""LEGAL_TASK_CATALOG §2.1/§3.1 gate — the DRAFT doc-type catalog, OFFLINE ($0, no live calls).

Proves the §0/§4 bet that the doc-type catalog is the unit of GROWTH: a DRAFT card =
a `DocType` DATA row that the ONE drafting path interprets, NOT new code. And it enforces
the plan's hard non-negotiable (§3.1, the area gate): **no card ships without its fixture**
— for EVERY catalog row this asserts the four properties that make a drafted card trustworthy:

  (a) CORRECT INDIAN STRUCTURE & VOCABULARY — the row uses Indian doc types and procedure
      (plaint not "complaint", written statement not "answer", Order XI/Order VII/VIII,
      MOA/AOA, Article 226/32). Wrong terminology = not-credible to an Indian lawyer (§3.1).
  (b) CITE-OR-BRACKET — `render_draft_request` with a MISSING fact emits a visible
      `[INPUT NEEDED: …]` bracket, never an invented value (the no-hallucinated-fact rule).
      With the fact provided, it is carried through, not bracketed.
  (c) AUTHORITY VERSION-IN-FORCE — every row declares governing-law `authority_refs` and the
      rendered instruction tells the agent to cite law by RETRIEVING it, never from memory.
  (d) SAME GATE, NO SOFTER PATH — the rendered request is shaped for the existing draft route,
      and a draft built from it is bound by the SAME `gate_sectioned` that redacts an uncited
      sentence. A wrong drafted citation is the legal equivalent of a confident-wrong number.

Run: python -u eval/test_doc_catalog.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from src.components.agent_core.doc_catalog import (
    DocType, CATALOG, PRACTICE_ORDER,
    render_draft_request, list_doc_types, get_doc_type, doc_type_card,
)
from src.components.agent_core.gates import gate_sectioned
from src.components.agent_core.ledger import EvidenceLedger


class Check:
    def __init__(self):
        self.passed = self.failed = 0

    def ok(self, cond, label):
        if cond:
            self.passed += 1; print(f"  [PASS] {label}")
        else:
            self.failed += 1; print(f"  [FAIL] {label}")


# Per-doc-type fixture: the Indian-vocabulary tokens the structure MUST contain (the moat),
# and one required-input label to probe the cite-or-bracket discipline against.
# Adding a catalog row WITHOUT adding a fixture row here fails the coverage check below —
# that IS the "no card without its fixture" rule, enforced at $0.
_FIXTURES = {
    # ── A. Litigation ──────────────────────────────────────────────────────────
    "legal_notice":            {"vocab": ["cause of action", "noticee", "compliance"], "probe": "demand"},
    "plaint":                  {"vocab": ["Cause title", "plaintiff", "prayer", "Verification"], "probe": "relief"},
    "written_statement":       {"vocab": ["written statement", "admissions and denials", "set-off"], "probe": "defences"},
    "writ_petition":           {"vocab": ["writ", "petitioner", "respondent", "mandamus"], "probe": "grounds"},
    "reply_to_legal_notice":   {"vocab": ["Bar enrolment", "denial of the allegations", "without prejudice"], "probe": "client_name"},
    "counter_claim":           {"vocab": ["counter-claim", "Order VIII Rule 6A", "prayer", "Verification"], "probe": "court"},
    "interlocutory_application": {"vocab": ["interim relief", "balance of convenience", "irreparable injury"], "probe": "interim_relief"},
    "interrogatories":         {"vocab": ["interrogatories", "leave of the court", "answered by affidavit"], "probe": "questions"},
    "execution_petition":      {"vocab": ["decree-holder", "judgment-debtor", "mode of execution", "attachment"], "probe": "mode_of_execution"},
    "special_leave_petition":  {"vocab": ["special leave", "Article 136", "Synopsis and list of dates", "Advocate-on-Record"], "probe": "questions_of_law"},
    "written_arguments":       {"vocab": ["Issue-wise submissions", "authorities relied on", "relief prayed"], "probe": "submissions"},

    # ── B. Corporate / M&A / Transactional ─────────────────────────────────────
    "nda":                     {"vocab": ["Confidential Information", "injunctive relief", "survival"], "probe": "purpose"},
    "spa":                     {"vocab": ["sale shares", "Conditions precedent", "Indemnification", "disclosure schedule"], "probe": "consideration"},
    "board_resolution":        {"vocab": ["RESOLVED THAT", "board meeting", "Certified true copy"], "probe": "matter"},
    "shareholders_agreement":  {"vocab": ["Reserved matters", "right of first refusal", "Tag-along", "Drag-along"], "probe": "reserved_matters"},
    "term_sheet":              {"vocab": ["valuation", "Liquidation preference", "Exclusivity (no-shop)", "non-binding"], "probe": "investment_amount"},
    "moa":                     {"vocab": ["Name clause", "Object clause", "Liability clause", "Subscription clause"], "probe": "objects"},
    "aoa":                     {"vocab": ["Table adopted", "transfer and transmission of shares", "general meetings"], "probe": "company_name"},
    "msa":                     {"vocab": ["Statement of Work", "service credits", "Limitation of liability", "DPDP"], "probe": "services"},
    "saas_agreement":          {"vocab": ["subscription", "uptime and support", "data protection"], "probe": "subscription_scope"},
    "dpa":                     {"vocab": ["Data Fiduciary", "Data Processor", "breach notification", "Cross-border transfer"], "probe": "processing_purpose"},
    "due_diligence_request_list": {"vocab": ["data-room", "statutory registers", "Capitalisation"], "probe": "requested_categories"},

    # ── C. Banking & Finance ───────────────────────────────────────────────────
    "intercreditor_agreement": {"vocab": ["priority", "standstill", "waterfall", "Subordination"], "probe": "ranking"},
    "facility_agreement":      {"vocab": ["drawdown", "Conditions precedent", "Financial covenants", "Events of default"], "probe": "facility_amount"},
    "deed_of_guarantee":       {"vocab": ["guarantor", "guarantee and indemnity", "Continuing", "Subrogation"], "probe": "guaranteed_obligations"},
    "deed_of_hypothecation":   {"vocab": ["hypothecation", "charge", "secured", "registration of the charge"], "probe": "hypothecated_assets"},
    "debenture_trust_deed":    {"vocab": ["debenture trustee", "fiduciary", "security created", "debenture holders"], "probe": "security_assets"},
    "commitment_letter":       {"vocab": ["commitment", "Market-flex", "Exclusivity", "term sheet"], "probe": "facility_amount"},

    # ── D. Real Estate ─────────────────────────────────────────────────────────
    "sale_deed":               {"vocab": ["vendor", "conveyance", "Schedule of the property", "Stamp duty"], "probe": "consideration"},
    "agreement_to_sell":       {"vocab": ["earnest money", "title clearance", "sale deed"], "probe": "completion_date"},
    "lease_deed":              {"vocab": ["lessor", "Demised premises", "security deposit", "lock-in"], "probe": "rent"},
    "gift_deed":               {"vocab": ["donor", "without consideration", "Acceptance of the gift"], "probe": "property_description"},
    "development_agreement":   {"vocab": ["development rights", "RERA", "built-up area"], "probe": "sharing_ratio"},
    "title_opinion":           {"vocab": ["chain of title", "Encumbrances", "marketability of title"], "probe": "property_description"},

    # ── E. Employment & Labour ─────────────────────────────────────────────────
    "employment_agreement":    {"vocab": ["probation", "PF / ESIC", "Non-solicitation", "POSH"], "probe": "remuneration"},
    "appointment_letter":      {"vocab": ["appointment", "CTC", "Probation and confirmation"], "probe": "compensation"},
    "posh_policy":             {"vocab": ["sexual harassment", "Internal Committee", "within 90 days", "retaliation"], "probe": "organisation"},
    "show_cause_notice":       {"vocab": ["standing orders", "articles of charge", "domestic enquiry", "natural justice"], "probe": "charges"},

    # ── F. IP & Technology ─────────────────────────────────────────────────────
    "trademark_application":   {"vocab": ["mark", "NICE classification", "proposed to be used", "Form TM-A"], "probe": "classes"},
    "trademark_examination_reply": {"vocab": ["Section 9", "Section 11", "distinctiveness", "Rule 29"], "probe": "objections"},
    "ip_licence":              {"vocab": ["Grant of licence", "Royalties", "Quality control", "non-infringement"], "probe": "licensed_ip"},

    # ── G. Tax ─────────────────────────────────────────────────────────────────
    "gst_scn_reply":           {"vocab": ["DRC-01", "time-bar", "personal hearing", "Section 75(4)"], "probe": "demand"},
    "advance_ruling_application": {"vocab": ["advance ruling", "not pending in any other proceeding", "Verification"], "probe": "questions"},
    "tax_opinion":             {"vocab": ["question presented", "case law", "alternative positions"], "probe": "facts"},

    # ── H. Regulatory / Compliance / In-House ──────────────────────────────────
    "litigation_hold_notice":  {"vocab": ["custodians", "reasonably anticipated", "preserved", "auto-purge"], "probe": "preservation_scope"},
    "regulatory_response":     {"vocab": ["regulator", "Corrective and preventive action", "request for closure"], "probe": "corrective_action"},
    "board_minutes":           {"vocab": ["quorum", "Resolutions passed", "chairperson"], "probe": "business_items"},

    # ── I. Family / Private Client ─────────────────────────────────────────────
    "will":                    {"vocab": ["testator", "Revocation of all earlier wills", "executor", "Attestation by two witnesses"], "probe": "bequests"},
    "power_of_attorney":       {"vocab": ["attorney", "powers granted", "Ratification", "Revocation"], "probe": "powers"},
    "trust_deed":              {"vocab": ["settlor", "Declaration of trust", "trustees", "beneficiaries"], "probe": "trust_property"},
    "partition_deed":          {"vocab": ["coparceners", "metes and bounds", "relinquishment"], "probe": "shares"},
}


def main() -> int:
    c = Check()

    # ── 0. THE CATALOG — the wedge set, grouped, India-first ───────────────────────
    print("── 0. the catalog: wedge-set doc types across practice areas ─────")
    dts = list_doc_types()
    c.ok(len(dts) >= 6, f"the catalog has the wedge set of doc types (got {len(dts)})")
    areas = set(d.practice_area for d in dts)
    c.ok({"Litigation", "Transactional", "Financial services"}.issubset(areas),
         "doc types span Litigation / Transactional / Financial services (the wedge)")
    c.ok(all(d.jurisdiction == "IN" for d in dts), "every doc type is India-first (jurisdiction=IN)")
    order = {p: i for i, p in enumerate(PRACTICE_ORDER)}
    seq = [order.get(d.practice_area, 99) for d in dts]
    c.ok(seq == sorted(seq), "list_doc_types is grouped by practice-area order (the card grid)")

    # ── 1. NO CARD WITHOUT ITS FIXTURE (§3.1) — every row is covered here ──────────
    print("\n── 1. coverage: every catalog row has a fixture ─────────────────")
    missing_fixture = [d.id for d in dts if d.id not in _FIXTURES]
    c.ok(not missing_fixture,
         f"every catalog row has a fixture (uncovered: {missing_fixture or 'none'})")
    stale_fixture = [k for k in _FIXTURES if k not in CATALOG]
    c.ok(not stale_fixture, f"no fixture references a removed doc type (stale: {stale_fixture or 'none'})")

    # ── 2. PER-DOC-TYPE: structure, vocabulary, authority, cite-or-bracket ─────────
    print("\n── 2. per-doc-type: Indian structure · vocab · authority · bracket ──")
    for d in dts:
        fx = _FIXTURES.get(d.id, {})

        # (a) structure present and ordered.
        c.ok(len(d.structure) >= 4, f"[{d.id}] declares an ordered structure (>=4 sections)")

        # (a) Indian vocabulary moat — the right names appear in the structure.
        blob = " ".join(d.structure)
        missing_vocab = [v for v in fx.get("vocab", []) if v.lower() not in blob.lower()]
        c.ok(not missing_vocab,
             f"[{d.id}] structure uses the correct Indian terms (missing: {missing_vocab or 'none'})")

        # (c) authority refs declared (version-in-force cited from G8, not memory).
        c.ok(len(d.authority_refs) >= 1, f"[{d.id}] declares governing-law authority refs")

        # (b) cite-or-bracket: render with the probe fact MISSING → a visible [INPUT NEEDED] bracket.
        probe = fx.get("probe")
        if probe:
            req_missing = render_draft_request(d, facts={})
            c.ok(f"[INPUT NEEDED: {probe}]" in req_missing["instructions"],
                 f"[{d.id}] a missing required fact brackets [INPUT NEEDED], never invents")
            # …and with the fact PROVIDED, it is carried through, not bracketed.
            req_filled = render_draft_request(d, facts={probe: "PROVIDED_VALUE"})
            c.ok("PROVIDED_VALUE" in req_filled["instructions"]
                 and f"[INPUT NEEDED: {probe}]" not in req_filled["instructions"],
                 f"[{d.id}] a provided required fact is carried through (not bracketed)")

        # (c) the rendered instruction forbids law-from-memory AND is ACTIONABLE: it names the
        # `search_knowledge` tool (G8), instructs an as-of retrieval, and gives the unverified
        # fallback — so the agent retrieves the real provision, never paraphrases from training.
        instr = render_draft_request(d, facts={})["instructions"]
        c.ok("never from memory" in instr or "never paraphrase the section from memory" in instr,
             f"[{d.id}] instruction forbids stating law from memory")
        c.ok("search_knowledge" in instr,
             f"[{d.id}] instruction names the search_knowledge tool (law comes from G8, §3.3)")
        c.ok("as_of" in instr and "[AUTHORITY UNVERIFIED:" in instr,
             f"[{d.id}] instruction directs an as-of retrieval + brackets an unverified authority")

    # ── 3. THE RENDER FEEDS THE EXISTING DRAFT ROUTE (one engine) ──────────────────
    print("\n── 3. render → the draft route's (doc_type, instructions) ───────")
    spa = get_doc_type("spa")
    req = render_draft_request(spa, facts={"seller": "Acme Promoters Pvt Ltd", "consideration": "INR 50 crore"})
    c.ok(set(req.keys()) == {"doc_type", "instructions"},
         "render_draft_request returns exactly the {doc_type, instructions} the route composes")
    c.ok(req["doc_type"] == "Share Purchase Agreement (SPA)", "doc_type is the human-facing card title")
    c.ok("Acme Promoters Pvt Ltd" in req["instructions"] and "INR 50 crore" in req["instructions"],
         "provided matter facts are folded into the instructions")
    c.ok("[INPUT NEEDED: purchaser]" in req["instructions"],
         "an unprovided required fact is still bracketed (no silent omission)")

    # the card view exposes the form fields, not the internal authority/template plumbing.
    card = doc_type_card(spa)
    c.ok(card["verb"] == "Draft" and "required_inputs" in card
         and "authority_refs" not in card and "template_ref" not in card,
         "doc_type_card exposes the form (required_inputs) for the picker, not internal refs")

    # ── 4. SAME GATE, NO SOFTER PATH — a draft from a card is bound like any answer ─
    print("\n── 4. the draft gate redacts an uncited line (same gate) ────────")
    # A draft built for an SPA card with one GROUNDED, cited sentence and one UNCITED claim.
    # The per-section gate must keep the cited one and redact the uncited one — the exact
    # contract every answer/deep/workflow draft gets; a drafted CARD is not exempt.
    led = EvidenceLedger()
    led.record("read", 1, [{"kind": "span", "doc": "spa-draft", "page": 3,
                            "quote": "The total consideration shall be INR 50,00,00,000.",
                            "display": "INR 50 crore"}])
    draft = ("## Consideration\nThe total consideration is INR 50 crore [spa-draft p.3].\n\n"
             "## Governing Law\nThis Agreement is governed by the laws of Singapore.")
    outcome = gate_sectioned(draft, led)
    rd = outcome.redacted_draft or draft
    c.ok("INR 50 crore" in rd, "the gate PRESERVES the grounded, cited drafted clause")
    c.ok(not outcome.passed and "Singapore" not in rd,
         "an uncited drafted clause is redacted — no softer gate for a doc-type card")

    # ── 5. ROUTE FIELD CONTRACT — render returns only keys the draft route reads ───
    # (Regression guard, mirroring test_workflows §5: the draft route reads body.doc_type
    # and body.instructions; render must return exactly those, so a renamed field can't
    # silently break a live draft. Asserted structurally here at $0.)
    print("\n── 5. render contract: only {doc_type, instructions} ────────────")
    for d in dts:
        keys = set(render_draft_request(d).keys())
        c.ok(keys == {"doc_type", "instructions"},
             f"[{d.id}] render returns exactly the route's fields") if d.id == dts[0].id else None
    bad = [d.id for d in dts if set(render_draft_request(d).keys()) != {"doc_type", "instructions"}]
    c.ok(not bad, f"every row renders only the route's fields (offenders: {bad or 'none'})")

    # ── 6. GOLDEN-MATTER FIXTURES (§4 area gate) — a known matter, end-to-end ──────
    # The per-row checks above prove structure/vocab/cite-or-bracket in isolation. The
    # plan's area gate (§4) wants a KNOWN MATTER per shipped doc type: realistic facts →
    # assert the composed draft request has all four properties together (structure carried ·
    # provided facts folded · missing facts bracketed · law-by-G8-retrieval), then the SAME
    # gate binds a draft built from it. One representative doc type per wedge practice area.
    print("\n── 6. golden-matter fixtures: a known matter, end-to-end ─────────")
    _GOLDEN = [
        # (doc_type, matter facts present, a required label deliberately OMITTED to prove
        #  it brackets, a structure section that must appear, a vocab token that must appear)
        ("nda", {"disclosing_party": "Veridian Labs Pvt Ltd", "receiving_party": "Nodal AI Inc",
                 "purpose": "evaluating a data-licensing partnership", "governing_law": "India, seat at Bengaluru"},
         "term", "Definition of Confidential Information", "Confidential Information"),
        ("plaint", {"plaintiff": "Meera Iyer", "relief": "recovery of INR 12,40,000 with interest"},
         "court", "prayer", "plaintiff"),
        ("facility_agreement", {"facility_amount": "INR 200 crore term loan",
                                "governing_law": "India"},
         "borrower", "Events of default", "drawdown"),
        ("will", {"testator": "Rustom F. Mistry", "bequests": "the Colaba flat to my daughter Aban"},
         "executor", "Attestation by two witnesses", "testator"),
    ]
    for did, facts, omit_label, must_section, must_vocab in _GOLDEN:
        d = get_doc_type(did)
        if d is None:
            c.ok(False, f"[golden:{did}] doc type exists"); continue
        instr = render_draft_request(d, facts=facts)["instructions"]

        # (a) structure carried — the named India-correct section appears in the request.
        c.ok(must_section in instr, f"[golden:{did}] structure carries the '{must_section}' section")
        c.ok(must_vocab in instr, f"[golden:{did}] Indian vocabulary '{must_vocab}' is present")
        # (b) provided matter facts are folded through verbatim.
        a_value = next(iter(facts.values()))
        c.ok(a_value in instr, f"[golden:{did}] a provided matter fact is folded into the draft request")
        # (b) a required fact NOT in the matter is bracketed, never invented.
        c.ok(f"[INPUT NEEDED: {omit_label}]" in instr,
             f"[golden:{did}] the missing required fact '{omit_label}' brackets [INPUT NEEDED]")
        # (c) the law is sourced from G8 by retrieval, never memory (actionable instruction).
        c.ok("search_knowledge" in instr,
             f"[golden:{did}] the matter draft cites law by G8 retrieval (search_knowledge)")

    # (d) the SAME gate binds a golden-matter draft — a grounded, cited figure survives, an
    # uncited asserted figure is redacted (the no-softer-path rule, on a realistic facility
    # draft). A figure is what the citation gate scrutinizes; a wrong drafted number is the
    # legal equivalent of a confident-wrong cell.
    led_g = EvidenceLedger()
    led_g.record("read", 1, [{"kind": "cell", "doc": "facility-draft", "page": 2,
                              "value": 2000000000.0, "label": "Facility amount",
                              "display": "INR 200 crore"}])
    gd = ("## The facility — commitment amount\nThe facility amount is INR 200 crore "
          "[facility-draft p.2].\n\n## Interest\nThe applicable interest rate is 9.5% per annum.")
    og = gate_sectioned(gd, led_g)
    rg = og.redacted_draft or gd
    c.ok("INR 200 crore" in rg, "[golden] the gate preserves the grounded, cited facility figure")
    c.ok(not og.passed and "9.5%" not in rg,
         "[golden] an uncited drafted figure is redacted (same gate, no exemption for a card draft)")

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULT: {c.passed} passed, {c.failed} failed")
    print(f"{'='*60}")
    return 0 if c.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
