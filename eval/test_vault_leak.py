"""F-F regression gate — the cross-vault-leak invariant on the answer path (tool_hard.md
Part II, F-F). Offline, $0, no live DB/Pinecone/model.

The answer path must NEVER surface a chunk from outside the active vault. Spans carry
`collection_id` from ingest (F1b, `_envelope.span_to_dict`), so a leak is detectable at
span granularity: a ledger span whose `collection_id` differs from the run's active
`scope.collection_id`. This gate asserts:

  1. `EvidenceLedger.foreign_spans(active)` flags exactly the foreign-vault spans, and
     `drop_foreign_spans` removes them (and nothing else).
  2. A span with NO collection_id (pre-F1b / non-vault source) is NOT a leak.
  3. Cells/params (kernel/grounding outputs, no vault id) are never leaks.
  4. The loop drops the leaked span BEFORE rendering `sources` and emits a `vault_leak`
     gate event — a leak is neither shown nor silently cited.
  5. A clean single-vault ledger is a no-op (byte-identical wrap-up).

Run:  python -u eval/test_vault_leak.py

The LIVE companion (run by Jeel on a populated DB — a real query in firm A's vault asserts
every cited span's collection_id belongs to firm A) is the F-F live gate; this offline gate
is the fast regression guard for the invariant logic itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.components.agent_core.ledger import EvidenceLedger  # noqa: E402

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def _span(collection_id, doc="d.pdf", page=1):
    return {"kind": "span", "doc": doc, "page": page, "chunk_id": f"{doc}-{page}",
            "snippet": "…", "collection_id": collection_id}


def _cell(value, collection_id=None):
    p = {"kind": "cell", "label": "Revenue", "period": "2022", "value": value,
         "doc": "d.pdf", "page": 1}
    if collection_id is not None:
        p["collection_id"] = collection_id
    return p


ACTIVE = "vault-A"
FOREIGN = "vault-B"


# ── 1. foreign_spans flags exactly the foreign-vault spans ────────────────────────────
led = EvidenceLedger()
led.record("search_vault", 1, [_span(ACTIVE), _span(ACTIVE)])
led.record("search_vault", 2, [_span(FOREIGN)])
foreign = led.foreign_spans(ACTIVE)
check("foreign_spans flags the 1 cross-vault span", len(foreign) == 1,
      f"got {len(foreign)}")
check("foreign_spans names the right vault",
      bool(foreign) and foreign[0].payload.get("collection_id") == FOREIGN)

# ── 2. a span with NO collection_id is NOT a leak (pre-F1b / non-vault source) ─────────
led2 = EvidenceLedger()
led2.record("search_vault", 1, [_span(ACTIVE), _span(None)])
check("unstamped span (collection_id=None) is NOT flagged as a leak",
      len(led2.foreign_spans(ACTIVE)) == 0,
      f"got {len(led2.foreign_spans(ACTIVE))}")

# ── 3. cells/params never count as leaks (no vault id, kernel outputs) ─────────────────
led3 = EvidenceLedger()
led3.record("compute", 1, [_cell(513983.0)])
led3.record("table_lookup", 1, [_cell(99.0, collection_id=FOREIGN)])  # even if stamped foreign
check("cells are never counted as vault leaks (only spans carry vault provenance)",
      len(led3.foreign_spans(ACTIVE)) == 0,
      f"got {len(led3.foreign_spans(ACTIVE))}")

# ── 4. drop_foreign_spans removes ONLY the leak, keeps everything else ─────────────────
led4 = EvidenceLedger()
led4.record("search_vault", 1, [_span(ACTIVE, doc="a1.pdf")])
led4.record("search_vault", 2, [_span(FOREIGN, doc="b1.pdf")])
led4.record("compute", 3, [_cell(42.0)])
n_before = len(led4.entries)
dropped = led4.drop_foreign_spans(ACTIVE)
check("drop_foreign_spans drops exactly 1 entry", dropped == 1, f"dropped {dropped}")
check("drop_foreign_spans leaves the active span + the cell intact",
      len(led4.entries) == n_before - 1
      and all((e.payload or {}).get("collection_id") != FOREIGN for e in led4.entries),
      f"{len(led4.entries)} left")
check("dropped foreign span no longer appears in to_sources()",
      all(s.get("collection_id") != FOREIGN for s in led4.to_sources()))

# ── 5. a clean single-vault ledger is a no-op ─────────────────────────────────────────
led5 = EvidenceLedger()
led5.record("search_vault", 1, [_span(ACTIVE), _span(ACTIVE)])
check("clean single-vault ledger: foreign_spans == 0", len(led5.foreign_spans(ACTIVE)) == 0)
check("clean single-vault ledger: drop is a no-op", led5.drop_foreign_spans(ACTIVE) == 0)

# ── 6. no active collection_id ⇒ nothing provably foreign (don't false-flag) ───────────
led6 = EvidenceLedger()
led6.record("search_vault", 1, [_span(FOREIGN)])
check("no active vault ⇒ no leak claim (can't compare)", led6.foreign_spans(None) == [])


# ── 7. the LOOP drops the leak before sources + emits a vault_leak gate event ──────────
# Drive run_agent with a scripted model + a registry stub that injects a foreign-vault span,
# proving the invariant fires on the real hot path (not just the ledger unit).
from src.components.agent_core.loop import run_agent  # noqa: E402
from src.components.agent_core.budgets import Budget  # noqa: E402
from src.components.agent_core.model import ModelResponse, ToolCall  # noqa: E402
from src.components.agent_core.registry import RunScope  # noqa: E402


class _ScriptedModel:
    """Step 1: call search_vault. Step 2: answer with no tools (ends the run)."""
    def __init__(self):
        self._calls = 0

    def stream(self, messages, tool_schemas):
        # No streaming deltas in the test; just yield the final response.
        yield ("done", self._respond())

    def invoke(self, messages, tool_schemas):
        return self._respond()

    def _respond(self):
        self._calls += 1
        if self._calls == 1:
            return ModelResponse(
                text="searching",
                tool_calls=[ToolCall(id="t1", name="search_vault", args={"query": "x"})],
                usage={"in": 10, "out": 5},
            )
        return ModelResponse(text="Done.", tool_calls=[], usage={"in": 5, "out": 2})


class _LeakRegistry:
    """A registry whose search_vault returns one ACTIVE + one FOREIGN span (the leak)."""
    def schemas(self, mode, tools=None, include_knowledge=False):
        return [{"name": "search_vault"}]

    def execute(self, call, scope):
        return {"ok": True, "summary": "2 spans",
                "provenance": [_span(ACTIVE, doc="a.pdf"), _span(FOREIGN, doc="b.pdf")]}


def _passthrough_gate(draft, ledger):
    from src.components.agent_core.loop import GateOutcome
    return GateOutcome(passed=True, failures=[])


scope = RunScope(collection_id=ACTIVE)
budget = Budget(mode="standard", model="scripted", max_steps=8, wall_clock_s=0, token_budget=0)
events = list(run_agent("q", model=_ScriptedModel(), scope=scope, budget=budget,
                        registry=_LeakRegistry(), gate_fn=_passthrough_gate))

vault_leak_evs = [e for e in events if e.get("type") == "gate" and e.get("name") == "vault_leak"]
sources_ev = next((e for e in events if e.get("type") == "sources"), {"sources": []})
check("loop emits a vault_leak gate event when a foreign span is cited",
      len(vault_leak_evs) == 1 and vault_leak_evs[0].get("pass") is False,
      f"events: {[e.get('name') for e in events if e.get('type')=='gate']}")
check("loop's rendered sources contain NO foreign-vault span",
      all(s.get("collection_id") != FOREIGN for s in sources_ev.get("sources", [])),
      f"sources: {sources_ev.get('sources')}")
check("loop's rendered sources DO keep the active-vault span",
      any(s.get("collection_id") == ACTIVE for s in sources_ev.get("sources", [])))


print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
