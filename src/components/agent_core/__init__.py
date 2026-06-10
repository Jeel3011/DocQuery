"""Agent Core (AGENT_CORE_PLAN.md Phase A).

A frontier model calls our proven deterministic machinery as TOOLS, behind
non-bypassable output gates. This package is ADDITIVE: until the loop/route land
(A2/A4) and `USE_AGENT_CORE` is turned on, nothing here is on any live path — the
existing Brain/Spine behaviour is byte-identical (the plan's prime directive).

A1 delivers the tool adapters only (`tools/`): thin wrappers over existing code
(retrieval, grid loading, grounding, the kernel) emitting a uniform JSON envelope
with provenance, that NEVER raise. No new intelligence lives in a tool — the moat
(extraction/kernel/grounding/verifiers) is wrapped, not reimplemented.
"""
