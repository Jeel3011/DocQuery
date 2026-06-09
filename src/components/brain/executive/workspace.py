"""Workspace (working memory) — BRAIN_REASONING_PLAN §5.3.

The hippocampal variable binding that's missing today: as the executor walks the plan
DAG, each resolved sub-goal writes a `Binding` here — "the pivot year" stops being a word
inside one LLM call and becomes a first-class, **provenance-carrying object** (value +
the exact `CellRef`s it came from + a confidence + a human-readable derivation). A later
goal that references `$pivot_year` reads it back from the workspace, so the bridge year is
BOUND once and reused, never re-guessed (the §2 confident-wrong failure mode).

Pure data + lookup; no LLM, no retrieval. The executor owns all writes (single trusted
writer — the same anti-poisoning posture §5.8 wants for long-term memory).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..analyst import CellRef


@dataclass
class Binding:
    """One resolved variable in working memory (§5.3).

    `value` is the resolved thing — a period string ("2022"), an entity ("AWS"), or a
    number. `cells` is the provenance: the exact source cells it traces to (empty only
    for a degraded/narrative binding). `derivation` is the audit string for the Trust UI.
    `abstained` marks a goal that could not be resolved (the executor stops the branch and
    the final answer abstains, rather than inventing a value).

    The `op`/`threshold`/`candidates` fields carry the SELECTION trail forward for the
    self-monitoring organ (§5.5): the kernel op that resolved a pivot, its threshold, and
    EVERY cell it scanned (the completeness trail). The reasoning verifier re-checks the
    predicate against these deterministically — they're empty for non-selection bindings.
    """
    name: str
    value: Any
    cells: List[CellRef] = field(default_factory=list)
    confidence: float = 0.0
    derivation: str = ""
    abstained: bool = False
    reason: str = ""              # why it abstained (when abstained)
    op: str = ""                  # the kernel op that produced this (selection bindings)
    threshold: Optional[float] = None     # the predicate threshold (first_exceeds/last_below)
    candidates: List[CellRef] = field(default_factory=list)  # every cell the op scanned

    @property
    def ok(self) -> bool:
        return not self.abstained and self.value is not None


class Workspace:
    """A run-scoped scratchpad of Bindings, keyed by variable name (§5.3 working memory)."""

    def __init__(self) -> None:
        self._bindings: Dict[str, Binding] = {}

    def write(self, b: Binding) -> None:
        self._bindings[b.name] = b

    def write_alias(self, key: str, b: Binding) -> None:
        """Index an existing binding under an additional key (e.g. its goal id, so
        `plan.final` resolves while `$var` still resolves by binding name). Does not
        duplicate provenance — `all_cells()` dedups identical cells."""
        self._bindings[key] = b

    def get(self, name: str) -> Optional[Binding]:
        return self._bindings.get(name)

    def value_of(self, name: str) -> Any:
        b = self._bindings.get(name)
        return b.value if (b and b.ok) else None

    def resolve_ref(self, ref: Any) -> Any:
        """Substitute a `$var` reference with the bound value from working memory.

        A plain (non-`$`) value passes through unchanged. A `$name` with no resolved
        binding returns None (the executor treats that as an unmet dependency → abstain on
        that goal, never a guess). This is the substitution step the §5.3 executor runs
        before routing each goal.
        """
        if isinstance(ref, str) and ref.startswith("$"):
            return self.value_of(ref[1:])
        return ref

    def all_cells(self) -> List[CellRef]:
        """Every source cell across all non-abstained bindings — the full provenance
        trail for the final answer (Trust UI / numeric verifier). De-duplicated: a binding
        indexed under both its name and its goal id must contribute its cells only once."""
        out: List[CellRef] = []
        seen: set = set()
        for b in self._bindings.values():
            if not b.ok:
                continue
            if id(b) in seen:          # same Binding object under two keys → count once
                continue
            seen.add(id(b))
            out.extend(b.cells)
        return out
