"""F1c regression gate — practice-aware self-config (offline, no API, no Supabase).

The committed gate for F1c (plans/F1_VAULT_PLAN.md §1): once a vault is TYPED (F1a), the type
DOES something — it suggests a practice template (review-grid columns + KB scope + flagship pin),
the structural classifier suggests a matter_kind, and a METADATA-ONLY conflict scan screens
parties across the firm. Deterministic, $0. Run:

    python -u eval/test_practice_config.py

What it proves (each reproduces-then-closes the property it guards):
  1. TEMPLATE COVERAGE + LOCKSTEP: every matter_kind in schemas.MATTER_KINDS has a template,
     and ONLY those (a drift = a typed vault with no template, or a dead template). Each
     template's grid columns map 1:1 onto the GridColumn shape the review grid runs; KB scope is
     a subset of the real chip vocabulary {vault,statutes,caselaw}; the flagship is a known id.
  2. PER-KIND CORRECTNESS: a sample of kinds load the EXPECTED columns / KB / flagship
     (lending→covenant_cockpit, litigation→argument_engine+caselaw, regulatory→sentinel) —
     so "the vault is the router" is asserted, not assumed. Generic/unknown → the neutral
     fall-back, never a raise.
  3. INFERENCE: classify_document's structural class + filename keywords suggest the right
     matter_kind ($0, no LLM); a financial filing suggests nothing (we don't mis-type a 10-K);
     an unmatched name suggests nothing (never a wrong guess).
  4. CONFLICT SCAN — flags a PLANTED adverse collision, AND (the wall) touches 0 document
     content: a recording fake DB client proves scan_conflicts reads ONLY the `collections`
     metadata table — never `documents` / `document_chunks`. The route serializes findings.
  5. WALL / NON-REGRESSION: an empty-party scan is a no-op (no DB hit); a no-firm user yields
     no findings; the create response shape is byte-identical when no parties are given.

It does NOT touch Pinecone, the kernel, or extraction — F1c adds NO retrieval surface.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import schemas  # noqa: E402
from src.components import practice_templates as pt  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = ""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


# ── 1. template coverage + lockstep with schemas.MATTER_KINDS ─────────────────────────────
_KB_VOCAB = {"vault", "statutes", "caselaw"}
_FLAGSHIPS = {
    pt.FLAGSHIP_REVIEW_GRID, pt.FLAGSHIP_COVENANT_COCKPIT,
    pt.FLAGSHIP_OBLIGATION_SENTINEL, pt.FLAGSHIP_ARGUMENT_ENGINE, pt.FLAGSHIP_CASE_FILE,
}

check("every MATTER_KIND has a template (and only those)",
      set(pt.all_kinds()) == set(schemas.MATTER_KINDS),
      f"templates={sorted(pt.all_kinds())} vs kinds={sorted(schemas.MATTER_KINDS)}")

for kind in schemas.MATTER_KINDS:
    tpl = pt.template_for(kind)
    d = tpl.to_dict()
    cols_ok = len(d["grid_columns"]) >= 3 and all(
        c.get("key") and c.get("label") and c.get("prompt")
        and c.get("kind") in ("clause", "numeric")
        for c in d["grid_columns"]
    )
    keys = [c["key"] for c in d["grid_columns"]]
    check(f"[{kind}] template well-formed (cols 1:1 GridColumn shape)",
          cols_ok and len(keys) == len(set(keys)), str(d["grid_columns"]))
    check(f"[{kind}] kb_scope ⊆ chip vocab and includes vault",
          set(d["kb_scope"]) <= _KB_VOCAB and "vault" in d["kb_scope"], str(d["kb_scope"]))
    check(f"[{kind}] flagship is a known surface", d["flagship"] in _FLAGSHIPS, d["flagship"])

# Each template column maps onto a real schemas.GridColumnRequest (the review-grid contract).
sample = pt.template_for("m&a")
try:
    reqs = [schemas.GridColumnRequest(**{
        "key": c.key, "label": c.label, "prompt": c.prompt, "kind": c.kind,
        "risk_rubric": c.risk_rubric,
    }) for c in sample.grid_columns]
    cols_validate = len(reqs) == len(sample.grid_columns)
except Exception as exc:  # noqa: BLE001
    cols_validate = False
    print(f"    (GridColumnRequest validation error: {exc})")
check("template columns validate as GridColumnRequest", cols_validate)


# ── 2. per-kind correctness — the vault IS the router ─────────────────────────────────────
lending = pt.template_for("lending")
check("lending → covenant cockpit + numeric facility column",
      lending.flagship == pt.FLAGSHIP_COVENANT_COCKPIT
      and any(c.kind == "numeric" and "facility" in c.key for c in lending.grid_columns),
      f"{lending.flagship}")

litig = pt.template_for("litigation")
check("litigation → argument engine + caselaw KB on",
      litig.flagship == pt.FLAGSHIP_ARGUMENT_ENGINE and "caselaw" in litig.kb_scope,
      f"{litig.flagship} {litig.kb_scope}")

reg = pt.template_for("regulatory")
check("regulatory → obligation sentinel",
      reg.flagship == pt.FLAGSHIP_OBLIGATION_SENTINEL, reg.flagship)

# generic / unknown → neutral fall-back, never a raise
gen = pt.template_for(None)
unk = pt.template_for("not-a-real-kind")
check("None matter_kind → generic fall-back (matter_kind=None, columns present)",
      gen.matter_kind is None and len(gen.grid_columns) >= 3)
check("unknown matter_kind → generic fall-back (never raises)",
      unk.matter_kind is None and len(unk.grid_columns) >= 3)


# ── 3. matter-kind inference ($0, structural; no LLM) ─────────────────────────────────────
check("infer: legal_contract + 'share_purchase_agreement' → m&a",
      pt.suggest_matter_kind("legal_contract", "Project_Falcon_share_purchase_agreement.pdf") == "m&a")
check("infer: legal_contract + 'facility agreement' → lending",
      pt.suggest_matter_kind("legal_contract", "USD_50m_facility_agreement.pdf") == "lending")
check("infer: legal_contract + 'arbitration award' → arbitration",
      pt.suggest_matter_kind("legal_contract", "final_arbitration_award.pdf") == "arbitration")
check("infer: financial_filing → no suggestion (don't mis-type a 10-K)",
      pt.suggest_matter_kind("financial_filing", "amzn-20221231.pdf") is None)
check("infer: legal_contract + unmatched name → None (no wrong guess)",
      pt.suggest_matter_kind("legal_contract", "untitled_document.pdf") is None)
check("infer: empty filename → None", pt.suggest_matter_kind("legal_contract", "") is None)
# WHOLE-WORD matching (these would FALSE-POSITIVE under the old substring matcher):
check("infer: 'suitable_terms' does NOT match 'suit' → None (no substring false-positive)",
      pt.suggest_matter_kind("legal_contract", "suitable_terms.pdf") is None)
check("infer: 'sharma_employment' → employment, NOT m&a via 'sha' substring",
      pt.suggest_matter_kind("legal_contract", "sharma_employment.pdf") == "employment")
check("infer: multi-word phrase 'code of conduct' → compliance (consecutive tokens)",
      pt.suggest_matter_kind("legal_contract", "company_code_of_conduct_2024.pdf") == "compliance")


# ── 4. conflict-scan matcher (pure function) — adverse vs same-party ──────────────────────
new_parties = [{"name": "Acme Corp", "role": "client"}]
existing = [
    {"collection_id": "c-adv", "name": "Beta v Acme", "matter_kind": "litigation",
     "parties": [{"name": "ACME Corporation", "role": "opposing party"}]},
    {"collection_id": "c-same", "name": "Acme advisory", "matter_kind": "advisory",
     "parties": [{"name": "Acme Corp.", "role": "client"}]},
    {"collection_id": "c-none", "name": "Gamma deal", "matter_kind": "m&a",
     "parties": [{"name": "Gamma Inc", "role": "seller"}]},
]
findings = pt.find_conflicts(new_parties, existing)
adverse = [f for f in findings if f["severity"] == "adverse"]
same = [f for f in findings if f["severity"] == "same_party"]
check("conflict: planted adverse collision is flagged (norm across suffix/case)",
      len(adverse) == 1 and adverse[0]["collection_id"] == "c-adv", str(findings))
check("conflict: same-side reuse is a same_party note, not adverse",
      len(same) == 1 and same[0]["collection_id"] == "c-same", str(same))
check("conflict: unrelated party yields no finding",
      all(f["collection_id"] != "c-none" for f in findings))
check("conflict: empty parties → no findings", pt.find_conflicts([], existing) == [])

# Role side-classification — WHOLE-WORD, ambiguity-safe (locks the substring-bug fix):
check("role: 'client' → our side", pt._side_of("client") == "our")
check("role: 'opposing party' → other side (phrase token)", pt._side_of("opposing party") == "other")
check("role: 'trustee' → neutral ('we'/'our' must NOT substring-match)",
      pt._side_of("trustee") == "neutral", pt._side_of("trustee"))
check("role: contradictory role (both sides) → neutral, never a false adverse",
      pt._side_of("lender-counterparty") == "neutral", pt._side_of("lender-counterparty"))
check("role: empty / None → neutral",
      pt._side_of("") == "neutral" and pt._side_of(None) == "neutral")
# A same-name match with a NEUTRAL/unknown role still surfaces (as same_party), never dropped:
neutral_find = pt.find_conflicts(
    [{"name": "Acme Corp", "role": "trustee"}],
    [{"collection_id": "c-x", "name": "Other", "matter_kind": "advisory",
      "parties": [{"name": "Acme Corp", "role": "some-unknown-role"}]}])
check("conflict: unknown-role name match still surfaces as same_party (never dropped)",
      len(neutral_find) == 1 and neutral_find[0]["severity"] == "same_party", str(neutral_find))


# ── 5. THE WALL — scan_conflicts touches ONLY metadata (collections), never content ───────
# A recording fake DB client logs every table NAME any query reads. The wall holds iff the only
# table the conflict scan ever touches is `collections` — a read of `documents` or
# `document_chunks` would be a content leak and fails the gate.
class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, name, rec, rows):
        self._name = name
        self._rec = rec
        self._rows = rows
        rec["tables_touched"].append(name)  # record EVERY table read

    def __post_init_rows(self):
        if not hasattr(self, "_result_rows"):
            self._result_rows = list(self._rows.get(self._name, []))

    def select(self, cols):
        self._rec["selected_cols"] = cols
        self.__post_init_rows()
        return self

    def eq(self, *a, **k):
        return self

    def neq(self, col, val):
        # Mirror Postgres `.neq` so the gate actually exercises exclude_collection_id (the
        # real DB filters this server-side; the fake must too, or exclusion is untested).
        self.__post_init_rows()
        self._result_rows = [r for r in self._result_rows if r.get(col) != val]
        return self

    def limit(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def execute(self):
        self.__post_init_rows()
        return _FakeResult(self._result_rows)


class _FakeClient:
    def __init__(self, rec, rows):
        self._rec = rec
        self._rows = rows

    def table(self, name):
        return _FakeTable(name, self._rec, self._rows)


from src.components.db import SupabaseManager  # noqa: E402


class _User:
    id = "user-1"


# `collections` rows = the firm's OTHER matters (metadata only). A poisoned `documents`/
# `document_chunks` row would be returned IF the scan ever read those tables — proving the wall
# by contradiction: if the scan touched content, these rows would surface and the table list
# would include a content table.
_firm_rows = {
    "collections": [
        {"id": "c-adv", "name": "Beta v Acme", "matter_kind": "litigation",
         "parties": [{"name": "Acme Corporation", "role": "opposing"}]},
        {"id": "c-self", "name": "This very matter", "matter_kind": "m&a",
         "parties": [{"name": "Acme Corp", "role": "client"}]},
    ],
    "documents": [{"id": "POISON-doc", "filename": "secret.pdf"}],
    "document_chunks": [{"id": "POISON-chunk", "content": "secret content"}],
}


def _make_fake(firm):
    rec = {"tables_touched": []}
    sb = SupabaseManager.__new__(SupabaseManager)
    sb.client = _FakeClient(rec, _firm_rows)
    sb._user = _User()
    # get_user_firm uses read_client; stub it to return our fake firm without a network call.
    sb.get_user_firm = lambda user_id=None: ({"id": firm} if firm else {})  # type: ignore
    return sb, rec


# (a) firm present → planted adverse collision flagged, excluding the matter itself
sb, rec = _make_fake("firm-9")
hits = sb.scan_conflicts([{"name": "Acme Corp", "role": "client"}],
                         firm_id="firm-9", exclude_collection_id="c-self")
check("scan_conflicts flags the planted firm collision",
      any(h["severity"] == "adverse" and h["collection_id"] == "c-adv" for h in hits), str(hits))
check("scan_conflicts excludes the matter itself (no self-conflict)",
      all(h["collection_id"] != "c-self" for h in hits), str(hits))

# THE WALL assertion: only `collections` was ever read — no content table.
touched = set(rec["tables_touched"])
check("WALL: conflict scan touched ONLY `collections` (0 document content)",
      touched == {"collections"}, f"tables touched = {sorted(touched)}")
check("WALL: scan returned no POISON content row",
      all("POISON" not in str(h.get("matched_party", "")) for h in hits)
      and all("POISON" not in str(h) for h in hits), str(hits))
# Belt: the metadata-only SELECT never asked for a content column.
check("WALL: scan selected metadata columns only (no content/file cols)",
      "parties" in (rec.get("selected_cols") or "")
      and "content" not in (rec.get("selected_cols") or "")
      and "filename" not in (rec.get("selected_cols") or ""),
      rec.get("selected_cols"))

# (b) no firm → no firm-wide matters to screen → [] and NO DB read at all
sb2, rec2 = _make_fake(None)
hits2 = sb2.scan_conflicts([{"name": "Acme Corp", "role": "client"}], firm_id=None)
check("no-firm user → no findings", hits2 == [])
check("no-firm user → no DB table touched", rec2["tables_touched"] == [])

# (c) empty parties → no-op, no DB read
sb3, rec3 = _make_fake("firm-9")
check("empty parties → no findings", sb3.scan_conflicts([], firm_id="firm-9") == [])
check("empty parties → no DB table touched", rec3["tables_touched"] == [])


# ── 6. route serialization — response shapes (no live server) ─────────────────────────────
from src.api.routes.collections import _collection_response  # noqa: E402

# create-with-parties path: conflicts threaded → CollectionResponse carries them + has_adverse
row = {"id": "v1", "name": "Deal", "matter_kind": "m&a", "status": "active", "parties": []}
resp = _collection_response(row, doc_count=0, conflicts=[
    {"party": "Acme Corp", "matched_party": "ACME Corporation", "collection_id": "c-adv",
     "matter_name": "Beta v Acme", "matter_kind": "litigation", "severity": "adverse",
     "new_side": "our", "existing_side": "other"},
])
check("response: create-with-conflicts sets has_adverse + conflicts",
      resp.has_adverse is True and resp.conflicts and resp.conflicts[0].party == "Acme Corp",
      str(resp.has_adverse))

# legacy path: no conflicts arg → byte-identical shape (conflicts/has_adverse None)
resp2 = _collection_response(row, doc_count=0)
check("response: legacy create (no conflicts) → None (byte-identical shape)",
      resp2.conflicts is None and resp2.has_adverse is None)

# PracticeTemplateResponse serializes from the module dict
from src.api.schemas import PracticeTemplateResponse  # noqa: E402
ptr = PracticeTemplateResponse(**pt.template_for("lending").to_dict())
check("PracticeTemplateResponse serializes a template",
      ptr.flagship == pt.FLAGSHIP_COVENANT_COCKPIT and len(ptr.grid_columns) >= 3)


print()
print(f"  {_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
