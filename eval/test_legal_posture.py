"""F2l regression gate — LEGAL POSTURE: the code that makes the legal artifacts TRUE (offline, $0).

The committed gate for F2l (plans/F2_FIRM_CONSOLE_PLAN.md §F2l + F2_ARCHITECTURE.md §7) — the LAST F2
slice. F2l is mostly ASSERTIONS OVER REAL CONFIG, not a new subsystem: the documents (MSA/DPA/Privacy/
AI-addendum) are templates a real lawyer reviews; the work is making their load-bearing clauses TRUE in
code. Run:

    ./venv/bin/python -u eval/test_legal_posture.py

What it proves (maps to the plan's §3 gate row for F2l):
  A. NO-TRAIN FLAG — config.MODEL_TRAINING_ON_CUSTOMER_DATA is OFF by default, AND a guarded path
     (legal_posture.assert_no_training) actually REFUSES training use while off, and ALLOWS it only when
     the deployment explicitly opts in. The DPA §3 promise is enforced, not just prose.
  B. SUBPROCESSOR LIST MATCHES THE CODE — the disclosed subprocessor list is DERIVED from the live
     config (Supabase, Pinecone, the LLM vendor per the agent-core model id, the F2j email seam), and the
     DPA's declared keys match the code-derived keys (subprocessor_drift is empty) — so the disclosure
     can't silently drift from what the code calls. A vendor added in code but not disclosed FAILS.
  C. DPA ERASURE CLAUSE BACKED BY F2k — the erase endpoint (F2k) is real and callable: it removes
     personal CONTENT while preserving the audit + signature chains, satisfying the DPA §4 erasure clause.
  D. ARTIFACTS SHIPPED HONESTLY — all four templates exist as tracked files and each carries the
     "TEMPLATE — NOT LEGAL ADVICE" caveat (we don't self-certify) + the load-bearing clauses are present.

OFFLINE: legal_posture.py is pure; the erase proof is driven with a tiny fake PostgREST (mirroring the
F2k gate). See [[run-only-relevant-gates]] — F2l touches legal_posture/config/docs + reuses the F2k erase
path, NOT extraction or the kernel.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.components import legal_posture as lp                 # noqa: E402
from src.components.config import Config                       # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = ""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ✓ {name}")
    else:
        _failed += 1
        print(f"  ✗ {name}  {detail}")


# ── A tiny fake PostgREST for the erase proof (C) — models content + signatures + tombstone ──
class _Result:
    def __init__(self, data, count=None):
        self.data = data
        self.count = count


class _Query:
    def __init__(self, rows, missing=False):
        self._rows = rows
        self._missing = missing
        self._filters = []
        self._neq = []
        self._update = None
        self._insert = None

    def select(self, *a, **k):
        return self

    def eq(self, col, val):
        self._filters.append((col, val))
        return self

    def neq(self, col, val):
        self._neq.append((col, val))
        return self

    def is_(self, col, val):
        return self

    def in_(self, col, vals):
        self._filters.append((col, ("__in__", list(vals))))
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def update(self, patch):
        self._update = patch
        return self

    def insert(self, row):
        self._insert = row
        return self

    def _matches(self, r):
        for col, val in self._filters:
            if isinstance(val, tuple) and val and val[0] == "__in__":
                if r.get(col) not in val[1]:
                    return False
            elif r.get(col) != val:
                return False
        for col, val in self._neq:
            if r.get(col) == val:
                return False
        return True

    def execute(self):
        if self._missing:
            raise Exception('relation "data_erasures" does not exist')
        sel = [r for r in self._rows if self._matches(r)]
        if self._update is not None:
            for r in sel:
                r.update(self._update)
            return _Result(sel)
        if self._insert is not None:
            row = dict(self._insert)
            row.setdefault("id", f"er-{len(self._rows)+1}")
            self._rows.append(row)
            return _Result([row])
        return _Result(sel)


class _FakeSB:
    def __init__(self, tables, missing=frozenset()):
        self._tables = tables
        self._missing = set(missing)

    def table(self, name):
        return _Query(self._tables.setdefault(name, []), missing=(name in self._missing))


class _FakeManager:
    """Just enough to call the real SupabaseManager.erase_personal_data unbound."""
    def __init__(self, sb, user_id):
        self.client = sb
        self.user_id = user_id


# ════════════════════════════════════════════════════════════════════════════════════════════════
print("\nA. NO-TRAIN FLAG — off by default AND actually refuses training use (DPA §3)")

cfg = Config()
check("A1 flag OFF by default (no training is the ship state)",
      lp.model_training_allowed(cfg) is False,
      f"got {lp.model_training_allowed(cfg)!r}")

# A2: a guarded training path is REFUSED while off.
refused = False
try:
    lp.assert_no_training(cfg, purpose="fine-tune a model on the firm's contracts")
except lp.TrainingUseRefused:
    refused = True
check("A2 assert_no_training RAISES while the flag is off (the enforcement point)", refused)

# A3: when the deployment explicitly opts in, the same guard ALLOWS it (the opt-in seam is real).
class _OptIn:
    MODEL_TRAINING_ON_CUSTOMER_DATA = True
allowed = True
try:
    lp.assert_no_training(_OptIn(), purpose="consented, time-boxed fine-tune")
except lp.TrainingUseRefused:
    allowed = False
check("A3 explicit opt-in ALLOWS training (written-consent seam works)", allowed)
check("A3b model_training_allowed reflects the opt-in", lp.model_training_allowed(_OptIn()) is True)

# A4: the flag is sourced from config, not hardcoded — a missing attr fails closed (no training).
class _NoAttr:
    pass
check("A4 missing config attr fails CLOSED (no training)",
      lp.model_training_allowed(_NoAttr()) is False)


# ════════════════════════════════════════════════════════════════════════════════════════════════
print("\nB. SUBPROCESSOR LIST — derived from the code, matches the DPA disclosure (no drift)")

subs = lp.disclosed_subprocessors(cfg)
keys = lp.disclosed_subprocessor_keys(cfg)
check("B1 always-on infra disclosed (Supabase + Pinecone)",
      {"supabase", "pinecone"} <= keys, f"got {keys}")
check("B2 the LLM vendor is disclosed (derived from the live model id)",
      ("openai" in keys) or ("anthropic" in keys), f"got {keys}")
check("B3 the F2j email-transport seam is disclosed (so enabling it isn't a new undisclosed flow)",
      "email_transport" in keys, f"got {keys}")

# B4: the LLM vendor is DERIVED the same way build_model routes the model id (drift-proof).
check("B4 gpt-* model id routes to OpenAI (mirrors build_model)",
      lp.llm_vendor_for_model("gpt-5.4")[0] == "openai")
check("B4b claude-* model id routes to Anthropic (mirrors build_model)",
      lp.llm_vendor_for_model("claude-opus-4-8")[0] == "anthropic")

# B5: with the LIVE config, the derived vendor matches what agent_core/build_model would construct.
from src.components.agent_core import model as ac_model        # noqa: E402
live_std = getattr(cfg, "AGENT_MODEL_STANDARD", "") or ""
derived_vendor = lp.llm_vendor_for_model(live_std)[0]
# build_model routes claude*→Anthropic else→OpenAI; mirror that expectation
expected_vendor = "anthropic" if live_std.lower().startswith("claude") else "openai"
check("B5 derived LLM vendor matches build_model's routing for the LIVE model id",
      derived_vendor == expected_vendor, f"model={live_std!r} derived={derived_vendor} expected={expected_vendor}")

# B6: THE DRIFT CHECK — the DPA's declared keys must equal the code-derived keys.
drift = lp.subprocessor_drift(cfg, set(lp.DPA_DISCLOSED_SUBPROCESSOR_KEYS))
check("B6 no subprocessor used-by-code-but-not-disclosed (would be a DPA breach)",
      drift["missing"] == set(), f"missing={drift['missing']}")
check("B6b no subprocessor disclosed-but-unused (stale disclosure)",
      drift["extra"] == set(), f"extra={drift['extra']}")

# B7: the drift check actually CATCHES a drop — if the DPA forgot to declare Pinecone, it must fail.
broken = lp.subprocessor_drift(cfg, set(lp.DPA_DISCLOSED_SUBPROCESSOR_KEYS) - {"pinecone"})
check("B7 drift check CATCHES an undisclosed vendor (reproduce-then-confirm)",
      "pinecone" in broken["missing"], f"got {broken}")

# B7b: the dual-LLM-vendor forgiveness is NARROW — disclosing BOTH vendors is OK (forward-disclosure)
# because the live config uses one of them, but it does NOT forgive an undisclosed non-LLM vendor.
both_llm = lp.subprocessor_drift(cfg, set(lp.DPA_DISCLOSED_SUBPROCESSOR_KEYS))
check("B7b disclosing both LLM vendors is NOT flagged as stale (intentional forward-disclosure)",
      both_llm["extra"] == set() and both_llm["missing"] == set(), f"got {both_llm}")
# B7c: but an undisclosed-but-USED LLM vendor would still be caught (strict 'missing' direction).
drop_used = lp.subprocessor_drift(cfg, set(lp.DPA_DISCLOSED_SUBPROCESSOR_KEYS) - keys)
check("B7c an LLM vendor the code actually uses, if undisclosed, IS caught (no over-forgiveness)",
      (keys & {"openai", "anthropic"}) <= drop_used["missing"], f"got {drop_used}")

# B8: every disclosed subprocessor carries a `derived_from` anchor (the proof it's really in use).
check("B8 every subprocessor names the config/code fact it's derived from",
      all(s.derived_from for s in subs))


# ════════════════════════════════════════════════════════════════════════════════════════════════
print("\nC. DPA ERASURE CLAUSE — backed by F2k's real, callable erase path")

proof = lp.dpa_erasure_clause_backed_by()
check("C1 the erasure clause names a concrete backing path",
      "erase_personal_data" in proof.backed_by)

# C2: the named path is importable + callable (not aspirational).
from src.components.db import SupabaseManager                  # noqa: E402
check("C2 SupabaseManager.erase_personal_data is importable",
      callable(getattr(SupabaseManager, "erase_personal_data", None)))

# C3: drive the REAL erase over a fake DB — it removes content AND preserves audit + signatures.
from src.components import dpdp                                # noqa: E402
tables = {
    "documents": [
        {"id": "d1", "user_id": "u1", "status": "ready", "filename": "secret.pdf", "storage_path": "p"},
        {"id": "d2", "user_id": "u1", "status": "ready", "filename": "deal.pdf", "storage_path": "p2"},
        {"id": "d3", "user_id": "other", "status": "ready", "filename": "keep.pdf", "storage_path": "p3"},
    ],
    "conversations": [{"id": "c1", "user_id": "u1", "title": "My matter"}],
    "messages": [{"id": "m1", "user_id": "u1", "content": "privileged text", "sources": ["x"]}],
    "audit_log": [{"id": "a1", "user_id": "u1", "action": "query"}],   # PRESERVED record of processing
    "signatures": [{"id": "s1", "firm_id": "f1", "chain_seq": 0, "row_hash": "h0"}],  # PRESERVED chain
    "data_erasures": [],
}
sb = _FakeSB(tables)
mgr = _FakeManager(sb, "u1")
res = SupabaseManager.erase_personal_data(mgr, principal_id="u1", firm_id="f1", requested_by="u1")

check("C3 erase reports the content it soft-deleted",
      res.get("documents") == 2 and res.get("conversations") == 1 and res.get("messages") == 1,
      f"got {res}")
check("C3b the principal's documents are blanked (content erased)",
      all(d["filename"] == "[erased]" and d["status"] == "erased"
          for d in tables["documents"] if d["user_id"] == "u1"))
check("C3c another user's documents are UNTOUCHED (scoped to the principal)",
      any(d["filename"] == "keep.pdf" for d in tables["documents"] if d["user_id"] == "other"))
check("C3d the message content is blanked + sources cleared",
      tables["messages"][0]["content"] == "[erased]" and tables["messages"][0]["sources"] == [])
check("C4 PRESERVED — the audit_log row is untouched (record of processing retained)",
      tables["audit_log"][0]["action"] == "query")
check("C4b PRESERVED — the signature chain row is untouched (non-repudiation kept)",
      tables["signatures"][0]["row_hash"] == "h0")
check("C4c PRESERVED matches dpdp.PRESERVED_RECORDS (the DPA §4 = code distinction)",
      set(res.get("preserved", [])) == set(dpdp.PRESERVED_RECORDS))
check("C5 a tombstone was written (proof the erasure was honored)",
      len(tables["data_erasures"]) == 1 and res.get("erasure_id"))

# C6: dormant parity — the erase still blanks content when the 019 tombstone table is unapplied.
tables2 = {
    "documents": [{"id": "d1", "user_id": "u9", "status": "ready", "filename": "x.pdf", "storage_path": "p"}],
    "conversations": [], "messages": [], "audit_log": [], "signatures": [],
}
sb2 = _FakeSB(tables2, missing={"data_erasures"})
mgr2 = _FakeManager(sb2, "u9")
res2 = SupabaseManager.erase_personal_data(mgr2, principal_id="u9", firm_id="f1", requested_by="u9")
check("C6 dormant (019 unapplied) — content still erased, no crash, tombstone skipped",
      res2.get("documents") == 1 and res2.get("erasure_id") is None
      and tables2["documents"][0]["filename"] == "[erased]")


# ════════════════════════════════════════════════════════════════════════════════════════════════
print("\nD. ARTIFACTS — all four templates shipped, honest caveat + load-bearing clauses present")

artifacts = lp.LEGAL_ARTIFACTS
check("D0 four artifacts named (MSA/DPA/Privacy/AI-addendum)",
      set(artifacts) == {"MSA", "DPA", "PRIVACY_POLICY", "AI_USE_ADDENDUM"})

texts = {}
for kind, rel in artifacts.items():
    p = _ROOT / rel
    exists = p.exists()
    check(f"D1 {kind} template file exists ({rel})", exists)
    texts[kind] = p.read_text() if exists else ""

for kind, txt in texts.items():
    check(f"D2 {kind} carries the honesty caveat ('{lp.TEMPLATE_CAVEAT_MARKER}')",
          lp.TEMPLATE_CAVEAT_MARKER in txt)

# D3: the DPA's load-bearing clauses are present in the prose (so the doc states what the code enforces).
dpa = texts.get("DPA", "")
check("D3 DPA states no-training-on-your-data (load-bearing)",
      ("training" in dpa.lower()) and ("no" in dpa.lower()))
check("D3b DPA discloses every code-derived subprocessor by name/key",
      all(s.key in dpa or s.name.split(" (")[0] in dpa for s in subs),
      "a subprocessor the code uses is not named in the DPA")
check("D3c DPA states the erasure-on-termination clause",
      "eras" in dpa.lower())
check("D3d DPA says customer retains data ownership (cross-ref MSA) ",
      "ownership" in dpa.lower() or "MSA" in dpa)

msa = texts.get("MSA", "")
check("D4 MSA states customer retains data ownership (load-bearing)",
      "retains all right" in msa.lower() or "customer retains" in msa.lower())

ai = texts.get("AI_USE_ADDENDUM", "")
check("D5 AI addendum states no-training + abstain/verify posture",
      "training" in ai.lower() and "abstain" in ai.lower())

priv = texts.get("PRIVACY_POLICY", "")
check("D6 Privacy Policy states the data-subject rights + named grievance officer",
      "erasure" in priv.lower() and "grievance officer" in priv.lower())


# ════════════════════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  RESULT: {_passed} passed, {_failed} failed")
print(f"{'='*70}")
sys.exit(0 if _failed == 0 else 1)
