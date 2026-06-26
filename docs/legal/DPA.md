# Data Processing Agreement (DPA) — TEMPLATE

> **TEMPLATE — NOT LEGAL ADVICE.** A working template for the DocQuery product. **A qualified lawyer
> must review and adapt it before use.** We do not self-certify it. It is written down so the product
> can be *built to honor it* — and the load-bearing clauses below (no-training-on-your-data,
> subprocessor disclosure, erasure) are each backed by real, tested code, not just prose. *"A DPA you
> can't honor is worse than none."*

**Roles.** Under the DPDP Act 2023 / Rules 2025, the **Customer (law firm) is the Data Fiduciary** and
**Provider (DocQuery) is the Data Processor.** Provider processes personal data **only on the
Customer's documented instructions.**

## 1. Subject matter & duration
Provider processes Customer Data to provide the DocQuery service for the term of the MSA. On
termination, §4 (erasure) applies.

## 2. Confidentiality (the §127 "agent of attorney" duty, contractualized)
Every person and subprocessor with access to Customer Data is bound to confidentiality on a
need-to-know basis. Provider's personnel are NDA-bound. This mirrors the Evidence Act §126/§127 duty:
the processor is an agent of the attorney and inherits the privilege-protection duty.

## 3. Purpose limitation — NO TRAINING ON YOUR DATA (load-bearing, enforced in code)
Provider processes Customer Data **solely to provide the service on the Customer's instructions.**
Provider does **NOT** use Customer Data for its own purposes, for analytics, or **to train, fine-tune,
or otherwise improve any model** — **unless** the Customer gives **explicit, written, opt-in consent**,
in which case any such use is **time-boxed and tenant-confined.**

> **This is enforced, not promised.** The product ships with the no-train posture **on by default**
> (`config.MODEL_TRAINING_ON_CUSTOMER_DATA = False`). Any code path that would use Customer Data for
> training must call `src/components/legal_posture.py::assert_no_training`, which **refuses** while the
> posture is in force. There is no training path in the product today; the guard protects the seam so
> one can never be added silently. Customer Data flows to the LLM subprocessor (§5) for **inference
> only.**

## 4. Erasure & data-subject rights (load-bearing, backed by F2k)
On a data-principal erasure request (DPDP §12) or on termination, Provider **erases the Customer's
personal content** (documents, conversations, messages), preserving only the **immutable records of
processing the law requires** — the audit log (DPDP Rule 6.5, retained ≥ 1 year) and the signature
hash-chain (non-repudiation). Erasing those records of processing would destroy the legal evidence and
break other members' sign-offs; the law requires their retention, so erasure is a soft-delete of
content plus a tombstone proving the erasure was honored.

> **Backed by:** `src/components/db.py::erase_personal_data` (routes `POST /dpdp/erase`,
> `POST /dpdp/admin/erase`). The same `ERASABLE_CONTENT` vs `PRESERVED_RECORDS` distinction stated here
> is the one the code enforces. Access/export (DPDP §11) is `GET /dpdp/export`; grievance (§13) routes
> to the firm's named officer.

## 5. Subprocessors (load-bearing — an undisclosed subprocessor is a breach)
Provider uses the subprocessors below. **This list is derived from the product's actual configuration**
(`src/components/legal_posture.py::disclosed_subprocessors`) so it cannot silently drift from what the
code calls; a drift check (`subprocessor_drift`) fails the build if the disclosure and the code disagree.

| Subprocessor (key) | Purpose | Customer Data it touches |
|---|---|---|
| **Supabase** (`supabase`) | Primary datastore — documents, conversations, audit log, firm/matter records, file blobs | All Customer Data at rest |
| **Pinecone** (`pinecone`) | Vector index for retrieval — embeddings of document chunks | Embedding vectors + chunk metadata derived from Customer Data |
| **LLM API vendor** (`openai` / `anthropic`) | LLM inference — prompts containing document excerpts are sent for answer generation + embedding (inference only, never training) | Document excerpts + the user's question, per request |
| **Email transport** (`email_transport`) | Outbound notification email (seam — in-app inbox only in v1; no transport active yet) | Recipient address + notification text once enabled (no document content) |

The LLM vendor is whichever vendor the configured orchestrator model id routes to (OpenAI for `gpt-*`,
Anthropic for `claude-*`) — disclosed both so a vendor switch via configuration never produces an
undisclosed flow. Provider will give Customer prior notice of a new subprocessor.

## 6. Security measures (the actual Rule-6 controls — not overstated)
Provider maintains: encryption in transit (TLS) and at rest; access control (the firm RBAC + ethical
walls + row-level security); masking for lower-privilege/external views; logging of processing
(retained ≥ 1 year, DPDP Rule 6.5); an incident-response runbook ([DPDP_BREACH_RUNBOOK.md](../DPDP_BREACH_RUNBOOK.md));
and this processor contract. *We list only controls the product actually has — claiming a control we
lack would make this DPA evidence against us.*

## 7. Breach notification
Provider will notify Customer without undue delay on becoming aware of a personal-data breach, with the
information the Customer (as Fiduciary) needs to meet its DPDP Rule-7 obligation to notify the Board and
affected principals within 72 hours. The runbook provides the templates; **the notification duty is the
Customer's; Provider enables it.**

---

*Honesty note: this DPA is a template a lawyer must review. Its three load-bearing clauses — no-training
(§3), accurate subprocessor disclosure (§5), and working erasure (§4) — are each backed by code and a
test (`eval/test_legal_posture.py`), so they are commitments the product actually keeps.*
