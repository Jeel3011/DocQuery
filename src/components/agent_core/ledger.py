"""The run-scoped evidence ledger (AGENT_CORE_PLAN §3.2, §3.7).

Every tool result's provenance (CellRef dicts from kernel/grounding, chunk spans from
search/read) accumulates here. The output gates (§3.4, A3) check the model's draft
AGAINST THIS LEDGER — a number or claim with no ledger entry is ungated and gets
redacted. This is `executive/workspace.py`'s Binding idea, generalized to any tool.

Pure bookkeeping — no intelligence, never raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LedgerEntry:
    """One piece of evidence the agent gathered, traceable to a source."""
    tool: str                          # which tool produced it
    step: int                          # loop step it came from
    kind: str                          # "cell" | "span"
    payload: Dict[str, Any]            # the serialized CellRef/span dict
    value: Optional[float] = None      # numeric value if this is a cell (for figure-tracing)

    def trace(self) -> str:
        return self.payload.get("trace") or self.payload.get("snippet", "") or str(self.payload)


@dataclass
class EvidenceLedger:
    """All evidence for one run. The gates query it; the UI's `sources` event reads it."""
    entries: List[LedgerEntry] = field(default_factory=list)

    def record(self, tool: str, step: int, provenance: List[Dict[str, Any]]) -> int:
        """Append every provenance dict from one tool result. Returns count added."""
        added = 0
        for p in provenance or []:
            if not isinstance(p, dict):
                continue
            self.entries.append(
                LedgerEntry(
                    tool=tool,
                    step=step,
                    kind=p.get("kind", "span"),
                    payload=p,
                    value=p.get("value"),
                )
            )
            added += 1
        return added

    def cells(self) -> List[LedgerEntry]:
        return [e for e in self.entries if e.kind == "cell"]

    def values(self) -> List[float]:
        """Every numeric value the agent actually read from a cell — the set a figure
        in the draft must trace into (gate #1, A3)."""
        return [e.value for e in self.cells() if e.value is not None]

    def is_empty(self) -> bool:
        return not self.entries

    def to_sources(self) -> List[Dict[str, Any]]:
        """The `sources` SSE payload (§3.6): the ledger's payloads, de-duplicated."""
        seen = set()
        out: List[Dict[str, Any]] = []
        for e in self.entries:
            if e.kind == "param":
                # params (thresholds, derived compute results) are gate evidence, not
                # document sources — they have no doc/page to show in the sources panel
                continue
            key = (e.payload.get("doc"), e.payload.get("page"),
                   e.payload.get("label"), e.payload.get("period"),
                   e.payload.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            out.append(e.payload)
        return out
