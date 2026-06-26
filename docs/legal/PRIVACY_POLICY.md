# Privacy Policy — TEMPLATE

> **TEMPLATE — NOT LEGAL ADVICE.** A working template for the DocQuery product. **A qualified lawyer
> must review and adapt it before it is published.** We do not self-certify it.

This notice explains what personal data DocQuery processes, why, for how long, and how a data principal
exercises their rights under the DPDP Act 2023 / Rules 2025. Within a law firm's use of DocQuery, the
**firm is the Data Fiduciary** and DocQuery is the **Data Processor** acting on the firm's instructions
(see the [DPA](DPA.md)).

## 1. What we process
- **Account data:** name, email, firm membership and role.
- **Customer Data:** the documents, conversations, queries, and generated answers submitted to or
  created within the service. The firm owns this data (see the [MSA](MSA.md) §2).
- **Processing records:** audit logs of who did what and when (records of processing, retained for the
  legal period).

## 2. Why we process it (purposes)
To provide the document-analysis and review service on the firm's instructions: ingesting and indexing
documents, answering queries with citations, running the review/approval chain, and keeping the audit
record. **We do not use Customer Data to train or improve models** without the firm's explicit, written
opt-in consent (see the [DPA](DPA.md) §3 and the [AI-use addendum](AI_USE_ADDENDUM.md)).

## 3. Retention
- **Customer Data:** retained for the term of the firm's subscription; erased on a valid erasure request
  or on termination (see §5 and the [DPA](DPA.md) §4).
- **Processing records (audit log):** retained for **at least one year** (DPDP Rule 6.5) as a record of
  what was done — these are not erasable personal content.

## 4. Subprocessors
We share Customer Data with the subprocessors disclosed in the [DPA](DPA.md) §5 (Supabase, Pinecone, the
LLM API vendor, and the email-transport seam), each only for the purpose stated there. The list is kept
accurate against the product's actual configuration.

## 5. Your rights (DPDP §§11–13) and how to exercise them
- **Access / export (§11):** obtain a summary of your personal data and processing — in-product via
  `GET /dpdp/export`.
- **Correction / erasure within 90 days (§12):** request erasure of your personal content — in-product
  via `POST /dpdp/erase`. Erasure soft-deletes personal content; the immutable records of processing are
  retained as required by law.
- **Grievance (§13):** raise a grievance to the firm's **named grievance officer**, completed within 90
  days. The officer's identity and contact are set per firm and shown in-product.

## 6. Security
TLS in transit, encryption at rest, role-based access control with ethical walls and row-level security,
masking for lower-privilege views, and an incident-response runbook. Details in the [DPA](DPA.md) §6.

## 7. The named grievance officer
The DPDP grievance officer is **set per firm** (name + contact), captured at filing time on each
grievance. *A real deployment fills in the officer's details; this template marks the field.*

---

*Honesty note: this Privacy Policy is a template a lawyer must review before publication. The rights in
§5 are backed by working endpoints (F2k), and the no-training and subprocessor statements are enforced
in code (F2l) — they describe what the product actually does.*
