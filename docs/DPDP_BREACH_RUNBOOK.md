# DPDP Breach-Notification Runbook (DPDP Rules 2025, Rule 7)

> **F2k — `plans/F2_FIRM_CONSOLE_PLAN.md` §F2k + `F2_ARCHITECTURE.md` §6.1.**
>
> **Honest framing first.** Under the DPDP Act 2023 + Rules 2025 the **firm is the Data Fiduciary**
> and **DocQuery is the Data Processor**. The **legal duty to notify is the firm's** — DocQuery
> *enables* it (we detect, we surface, we provide the records and the templates; the firm decides and
> notifies). This runbook is the firm's playbook, pre-written so no one improvises during an incident.
> It is a **template a qualified lawyer / the firm's DPO must review and adopt** — it is not legal
> advice, and adopting it does not by itself discharge any obligation. **₹200 crore / incident** is
> the cost of improvising, which is why this exists.

The Processor's contractual duty (DPA, F2l): on becoming aware of a breach affecting the firm's data,
**notify the firm without undue delay** with the facts below, and assist the firm's own notification.

---

## The four phases (Rule 7)

```
   DETECT ─────▶ CONTAIN ─────▶ NOTIFY ─────▶ RECORD
 (discover)    (stop spread)  (Board + the   (ledger:
                              affected         proof it
                              principals,      happened +
                              ≤72h)            was handled)
```

### 1. DETECT — establish "discovery"
The clock starts on **discovery** (Rule 7 requires notice *on becoming aware*). Capture:
- **When** the breach was discovered (timestamp) and **by whom**.
- **What** was exposed: which data, whose data (how many data principals), which systems.
- **How**: root cause if known, else "under investigation".
- Open an incident record immediately — do not wait for full root-cause before notifying.

**Sources we provide:** the `audit_log` (Rule 6.4/6.5, retained ≥ 1 year) is the access/processing
record to reconstruct *what was touched*; the F2i `signatures` chain proves sign-off integrity; the
ethical-wall + firm-RLS layers bound *whose* data a compromised actor could reach.

### 2. CONTAIN — stop the spread
- Revoke the compromised credential / offboard the actor (F2d instant revocation — caps drop on the
  next request; delegations revoked).
- Raise an ethical-wall **screen** (F2c) on affected matters if a person should be cut off pending
  investigation.
- Rotate keys / tokens as applicable; preserve evidence (do **not** delete logs — they are the record).

### 3. NOTIFY — Board on discovery, principals within 72 hours
- **Notify the Data Protection Board of India on discovery** (intimation; follow with details).
- **Notify each affected data principal within 72 hours** with the **required content** (Rule 7):
  a plain-language description, the **data exposed**, the **likely consequences**, the **protective
  steps** taken/recommended, and **contact details** (the firm's **named grievance officer**, §13 —
  set via `PUT /dpdp/firm/grievance-officer`).
- Use the templates below verbatim where they fit; a lawyer adapts specifics.

### 4. RECORD — write it into the ledger (the proof)
- Record the incident, the notifications sent (to whom, when), and the resolution in the immutable
  record (the `audit_log`; F3 will hash-chain it like the F2i signature chain). **A breach you can't
  prove you handled reads, to a regulator, like one you didn't.**
- File / link any resulting data-principal grievances (`/dpdp/grievances`) so follow-through is tracked
  against the §13 90-day window.

---

## Pre-written templates (adapt, lawyer-review before use)

### Template A — Data Protection Board intimation (on discovery)
```
To: Data Protection Board of India
Subject: Personal Data Breach Intimation — [FIRM NAME]

We are notifying the Board of a personal data breach discovered on [DATE/TIME].
- Nature of the breach: [brief description]
- Data principals affected: [approx. number / categories]
- Personal data involved: [categories of data]
- Likely consequences: [assessment, or "under assessment"]
- Measures taken / proposed to contain and remedy: [containment steps]
- Point of contact: [Grievance Officer name, email, phone]
A detailed report will follow as the investigation progresses.
```

### Template B — Notice to an affected data principal (within 72 hours)
```
Subject: Important notice about your personal data

Dear [NAME],

We are writing to inform you of a security incident discovered on [DATE] that may have
affected your personal data held by [FIRM NAME].

What happened: [plain-language description].
What data was involved: [the specific data categories exposed].
What this could mean for you: [likely consequences].
What we have done: [containment + remediation steps].
What you can do: [recommended protective steps — e.g. change credentials, watch for phishing].

If you have questions or wish to raise a concern, contact our Grievance Officer:
  [Officer name] — [email] — [phone]

You may also exercise your data rights (access, correction, erasure) at any time.

[FIRM NAME]
```

### Template C — Internal incident log entry (the RECORD step)
```
Incident ID: [id]
Discovered: [timestamp] by [person]
Scope: [systems, data, # principals]
Root cause: [known / under investigation]
Containment: [actions + timestamps]
Board notified: [timestamp]
Principals notified: [count] at [timestamp] via [channel]
Grievance officer: [name]
Resolution: [summary]  Closed: [timestamp]
```

---

## What the product gives the firm (the enabling surfaces)

| Need | Surface |
|---|---|
| Reconstruct what was accessed | `audit_log` (Rule 6.5, retained ≥ 1 year — `AUDIT_LOG_RETENTION_DAYS`) |
| Prove sign-off integrity | F2i `signatures` hash-chain (`GET /firm/signatures/verify`) |
| Cut off a compromised actor | F2d offboard (instant revoke) + F2c screen |
| Named contact for principals | F2k grievance officer (`/dpdp/firm/grievance-officer`) |
| Export / erase a person's data | F2k `/dpdp/export`, `/dpdp/erase` (§11/§12) |
| Track grievances to the 90-day window | F2k `/dpdp/grievances` (`due_at` = filed + 90d) |

**The duty is the firm's; we make it executable.**
