"""Run tracer — durable, structured visibility into every agent run (§I1, §3.7).

The 2026-06-11 bugs were found by LUCK (reading the API stdout log by hand). That is
not robustness. This module persists EVERY loop event to a per-run JSONL journal AND
computes a `health` summary that auto-flags the exact loose-end signals we hit:

  - a tool that ALWAYS returns 0 results / always fails (the dead `search_vault(table)`)
  - the SAME failed tool call repeated (the model stuck, not self-healing)
  - budget exhaustion (ran out before answering)
  - a redaction/withhold (a correct answer suppressed, or a real abstain)

So the next loose end shows up in the journal + health flags, not after five UI runs.
Pure bookkeeping — never raises into the loop (a tracer failure must never break a run).

Journal path: <TRACE_DIR>/run-<id>.jsonl  (TRACE_DIR = AGENT_TRACE_DIR env or /tmp/
docquery_traces). One JSON object per line; the last line is the `health` summary.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DIR = "/tmp/docquery_traces"


def _trace_dir() -> str:
    d = os.environ.get("AGENT_TRACE_DIR", _DEFAULT_DIR)
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:  # noqa: BLE001 — tracing is best-effort
        return _DEFAULT_DIR
    return d


@dataclass
class RunTracer:
    """Accumulates every event of one run, persists a JSONL journal, computes health.

    The loop calls `record(event)` on each §3.6 event it yields. `finish()` writes the
    journal + returns the health summary (also yielded as a `trace_health` event so the
    UI/route can surface it). Designed to be wrapped around the loop generator so NO loop
    code changes except one `tracer.record(ev)` per yielded event.
    """
    run_id: str
    question: str = ""
    mode: str = "standard"
    events: List[Dict[str, Any]] = field(default_factory=list)

    # rolling signals (so health is O(events), no second pass)
    _tool_calls: Counter = field(default_factory=Counter)
    _tool_zero: Counter = field(default_factory=Counter)       # ok-but-0-provenance
    _tool_fail: Counter = field(default_factory=Counter)       # ok=false
    _call_sigs: Counter = field(default_factory=Counter)       # (name,args) repeats
    _gate_fails: Counter = field(default_factory=Counter)
    _last_args: Dict[str, str] = field(default_factory=dict)

    def record(self, ev: Dict[str, Any]) -> None:
        try:
            self.events.append(ev)
            t = ev.get("type")
            if t == "tool_call":
                name = ev.get("name", "?")
                self._tool_calls[name] += 1
                sig = f"{name}:{ev.get('args_summary', '')}"
                self._call_sigs[sig] += 1
            elif t == "tool_result":
                name = ev.get("name", "?")
                if not ev.get("ok"):
                    self._tool_fail[name] += 1
                elif (ev.get("n_provenance") or 0) == 0:
                    self._tool_zero[name] += 1
            elif t == "gate" and ev.get("pass") is False:
                self._gate_fails[ev.get("name", "?")] += 1
        except Exception:  # noqa: BLE001 — never break the run
            pass

    def health(self) -> Dict[str, Any]:
        """Auto-flag the loose-end signals. Empty `flags` = a clean run."""
        flags: List[str] = []
        # A tool that NEVER produced provenance across all its calls = likely-dead wiring
        # (the BUG-F signature: search_vault table always 0).
        for name, calls in self._tool_calls.items():
            zeros = self._tool_zero.get(name, 0)
            fails = self._tool_fail.get(name, 0)
            if calls >= 2 and (zeros + fails) == calls:
                flags.append(f"tool '{name}' returned NOTHING on all {calls} calls "
                             f"(zero={zeros} fail={fails}) — possible dead wiring")
        # The SAME call repeated ≥3× = the model is stuck, not self-healing.
        for sig, n in self._call_sigs.items():
            if n >= 3:
                flags.append(f"repeated identical call ×{n}: {sig[:120]} — model not self-healing")
        if self._gate_fails:
            flags.append(f"gate failures: {dict(self._gate_fails)}")
        meta = next((e for e in reversed(self.events) if e.get("type") == "meta"), {})
        if meta.get("abstained"):
            flags.append("run ABSTAINED (no verified answer shipped)")
        if any(e.get("type") == "gate" and e.get("name") == "budget" for e in self.events):
            flags.append("BUDGET EXHAUSTED before finishing")
        return {
            "run_id": self.run_id,
            "question": self.question[:200],
            "mode": self.mode,
            "n_events": len(self.events),
            "steps": meta.get("steps"),
            "tokens": meta.get("tokens"),
            "abstained": meta.get("abstained"),
            "tool_calls": dict(self._tool_calls),
            "tool_zero_results": dict(self._tool_zero),
            "tool_failures": dict(self._tool_fail),
            "gate_failures": dict(self._gate_fails),
            "flags": flags,
            "clean": not flags,
        }

    def finish(self) -> Dict[str, Any]:
        """Persist the JSONL journal (events + final health line) and return health."""
        h = self.health()
        try:
            path = os.path.join(_trace_dir(), f"run-{self.run_id}.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"type": "run_meta", "run_id": self.run_id,
                                    "question": self.question, "mode": self.mode},
                                   default=str) + "\n")
                for ev in self.events:
                    f.write(json.dumps(ev, default=str) + "\n")
                f.write(json.dumps({"type": "health", **h}, default=str) + "\n")
            h["journal"] = path
            if h["flags"]:
                logger.warning("[agentcore.health] run %s FLAGGED: %s", self.run_id, h["flags"])
            else:
                logger.info("[agentcore.health] run %s clean (%d events, %s steps)",
                            self.run_id, h["n_events"], h.get("steps"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[agentcore.tracer] journal write failed: %s", exc)
        return h
