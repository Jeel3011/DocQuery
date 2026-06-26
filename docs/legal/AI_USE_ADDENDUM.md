# AI-Use Addendum — TEMPLATE

> **TEMPLATE — NOT LEGAL ADVICE.** A working template for the DocQuery product. **A qualified lawyer
> must review and adapt it before use.** We do not self-certify it. It is written to match the
> ABA Formal Opinion 512 / BCI-aligned honesty: we describe what the AI does and does not do, and we
> build the product to keep these statements true.

This addendum supplements the [MSA](MSA.md) and [DPA](DPA.md) and governs the AI-assisted features of
DocQuery.

## 1. AI output is assistive, not authoritative
DocQuery uses retrieval-augmented generation to help qualified professionals analyze documents. **Its
output may be incomplete or incorrect and is not legal advice.** It does not replace the professional
judgment of the firm's lawyers. The firm remains responsible for all work product.

## 2. The abstain / verify posture
The product is built to **abstain rather than guess**: answers are grounded in cited source passages,
and the verification gates redact or abstain when a claim cannot be traced to the source. This is a
deliberate honesty posture — the product prefers "I can't determine this from the documents" to a
confident wrong answer.

## 3. Human supervision (the review chain)
External release of work product flows **up a review chain of command** (junior → senior → partner),
with each step owning the next so nothing stalls. AI-assisted output is reviewed and approved by a
qualified person before it leaves the firm — the ABA-512 supervision/accountability model, built into
the product (F2e).

## 4. No training on your data (load-bearing — enforced in code)
**Customer Data is used for inference only and is never used to train, fine-tune, or improve any
model** without the firm's explicit, written, opt-in consent. This is the same commitment as the
[DPA](DPA.md) §3, and it is **enforced**: the product ships with training off by default, and any
training seam is guarded by `assert_no_training` (which refuses while the posture is in force). Prompts
are sent to the LLM subprocessor for inference per request; the vendor's API terms (zero-retention /
no-training-on-API-data) apply.

## 5. Transparency of sources
Answers carry citations to the source passages they are grounded in, so a reviewer can trace any claim
to its origin. The product is built so that a number or a clause that cannot be cited is not asserted.

## 6. Honesty about limits (no over-claiming)
We do not claim a fixed accuracy percentage, we do not claim the AI is a lawyer, and we do not claim
compliance certifications we do not hold. The DPDP framework is enforceable from approximately mid-2027;
we design to it now without overstating present legal obligations. BCI has not issued an AI-specific
rule; we follow the supervision principle without pretending a rule exists.

---

*Honesty note: this addendum is a template a lawyer must review. Its statements about no-training (§4),
the abstain/verify posture (§2), and human review (§3) describe behavior the product actually
implements and tests — not marketing.*
