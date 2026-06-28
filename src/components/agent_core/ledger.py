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

    def foreign_spans(self, active_collection_id: Optional[str]) -> List[LedgerEntry]:
        """F-F (tool_hard.md): the cross-vault-leak invariant. Return every span entry whose
        `collection_id` is set AND differs from the active vault's — a chunk that leaked in
        from another vault/firm. The active vault is the run's `scope.collection_id`; vault→
        firm isolation is enforced upstream (F1/F2), so a foreign collection_id IS the leak
        signal at span granularity.

        Spans carry `collection_id` from ingest via `_envelope.span_to_dict` (F1b — no
        re-ingest needed). A span with NO collection_id is NOT counted as a leak (it predates
        the F1b stamping or is a non-vault source like the KB); only a span stamped with a
        DIFFERENT vault is a leak. Cells/params (kernel/grounding outputs) have no vault id and
        are never leaks. Pure read; never raises.

        A non-empty result is a HARD invariant violation: the answer path must NEVER cite a
        chunk from outside the active vault. The loop drops these before rendering `sources`
        and flags the run, so a leak can neither be shown nor silently cited."""
        if not active_collection_id:
            return []   # no active vault to compare against → nothing provably foreign
        out: List[LedgerEntry] = []
        for e in self.entries:
            if e.kind != "span":
                continue
            cid = (e.payload or {}).get("collection_id")
            if cid is not None and str(cid) != str(active_collection_id):
                out.append(e)
        return out

    def drop_foreign_spans(self, active_collection_id: Optional[str]) -> int:
        """Remove any cross-vault-leaked span (see `foreign_spans`) from the ledger so it can
        be neither rendered in `sources` nor used as gate evidence. Returns the count dropped."""
        foreign = self.foreign_spans(active_collection_id)
        if not foreign:
            return 0
        foreign_ids = {id(e) for e in foreign}
        self.entries = [e for e in self.entries if id(e) not in foreign_ids]
        return len(foreign)

    def partial_answer(self) -> str:
        """Render the verified figures gathered so far into a readable, CITED answer.

        Used when the run ends WITHOUT the model writing a final answer (budget hit,
        model error). The verified cells/computations are already in the ledger — it is
        a bug to discard them and show "I ran out of budget" with nothing (observed
        2026-06-11: the model had computed Google 14.8% + Amazon 14.9%, then the budget
        cut it off before it could write them, so the user saw an empty abstain). This
        builds the answer deterministically from the ledger so a long run NEVER returns
        empty-handed when it verified something. No model call — zero cost, zero latency."""
        seen = set()
        lines: List[str] = []
        for e in self.entries:
            if e.kind not in ("cell", "result"):
                continue
            p = e.payload or {}
            val = p.get("display") or p.get("raw") or (
                f"{e.value:,.0f}" if isinstance(e.value, (int, float)) else None)
            if val is None:
                continue
            label = (p.get("label") or p.get("formula") or "").strip()
            period = str(p.get("period") or "").strip()
            doc = p.get("doc") or ""
            page = p.get("page")
            key = (label, period, doc, str(val))
            if key in seen or not label:
                continue
            seen.add(key)
            cite = f" [{doc}{(' p.' + str(int(float(page)))) if page not in (None, '') else ''}]" if doc else ""
            per = f" ({period})" if period else ""
            lines.append(f"- **{label}**{per}: {val}{cite}")
        if not lines:
            return ""
        return ("Here are the figures I was able to verify before stopping (I'm only "
                "stating what traces to the source tables):\n\n" + "\n".join(lines[:20]))

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
