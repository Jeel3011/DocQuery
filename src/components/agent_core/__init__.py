"""Agent Core (AGENT_CORE_PLAN.md Phase A).

A frontier model calls our proven deterministic machinery as TOOLS, behind
non-bypassable output gates. This package is ADDITIVE: until the loop/route land
(A2/A4) and `USE_AGENT_CORE` is turned on, nothing here is on any live path — the
existing Brain/Spine behaviour is byte-identical (the plan's prime directive).

A1 delivers the tool adapters (`tools/`): thin wrappers over existing code (retrieval,
grid loading, grounding, the kernel) emitting a uniform JSON envelope with provenance,
that NEVER raise. No new intelligence lives in a tool — the moat (extraction/kernel/
grounding/verifiers) is wrapped, not reimplemented.

A2 delivers the loop: `loop.run_agent` (the model IS the orchestrator), `registry`
(schemas + dispatch), `ledger` (run-scoped evidence), `budgets` (per-mode hard
ceilings), `model` (Anthropic-primary, with a ScriptedModel for offline gates), and the
`prompt` skeleton. The output gates are a passthrough STUB here — A3 fills them in.
"""
