"""Orchestrator model layer (AGENT_CORE_PLAN §3.2 loop, §3.1 model policy).

A hand-rolled native tool-use binding, Anthropic-primary (the orchestrator decision in
§3.2). The loop never touches a vendor SDK directly — it talks to this uniform
interface, so:
  - the live path is `AnthropicModel` (Claude Opus 4.8, native tool-use API);
  - the OpenAI fallback sits behind a thin message-shape adapter (Phase A fallback);
  - the OFFLINE GATE injects a `ScriptedModel` (a deterministic tool-call script), so
    `eval/test_agent_loop.py` runs with ZERO API spend and no network.

Uniform shapes the loop consumes:

    ToolCall(id, name, args: dict)
    ModelResponse(text: str|None, tool_calls: list[ToolCall], usage: {in,out})

`anthropic` is imported LAZILY inside `AnthropicModel.invoke` — importing this module
costs nothing and needs no API key (the gates don't).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    args: Dict[str, Any]


@dataclass
class ModelResponse:
    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=lambda: {"in": 0, "out": 0})

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class BaseModel:
    """Interface the loop depends on. `invoke` MUST NOT raise on a vendor error in a
    way the loop can't handle — the loop wraps invoke in its own try/retry/degrade."""

    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> ModelResponse:  # noqa: D401
        raise NotImplementedError

    def stream(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
        """Stream a response. Yields ("delta", text) tuples as text is generated, then a
        final ("done", ModelResponse). The DEFAULT implementation calls invoke() and emits
        the whole text as one delta — so a model without real streaming still satisfies the
        contract (used by ScriptedModel + the offline gate). Live models override this with
        the vendor's token stream so the loop can surface text as it is written (the UX fix:
        no 13s freeze, the draft appears live)."""
        resp = self.invoke(messages, tools)
        if resp.text:
            yield ("delta", resp.text)
        yield ("done", resp)


# ── Live Anthropic (Claude) — native tool use ───────────────────────────────────

class AnthropicModel(BaseModel):
    """Claude via the native Messages tool-use API (Opus 4.8 by config).

    Translates our uniform message list (already in Anthropic's role/content shape —
    the loop builds it that way) and the registry's JSON tool schemas into a
    `messages.create` call, then normalizes the response back to ModelResponse.
    """

    def __init__(self, model: str, api_key: str, *, max_tokens: int = 4096,
                 system: Optional[str] = None, temperature: float = 0.0):
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.system = system
        self.temperature = temperature

    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> ModelResponse:
        import anthropic  # lazy — no import/key cost unless a live call happens

        client = anthropic.Anthropic(api_key=self.api_key)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if self.system:
            kwargs["system"] = self.system
        if tools:
            # Registry schemas are {name, description, input_schema} — already the
            # Anthropic tool shape, so they pass straight through.
            kwargs["tools"] = tools

        resp = client.messages.create(**kwargs)

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in resp.content or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                tool_calls.append(ToolCall(
                    id=getattr(block, "id", ""),
                    name=getattr(block, "name", ""),
                    args=getattr(block, "input", {}) or {},
                ))
        usage = getattr(resp, "usage", None)
        return ModelResponse(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            usage={
                "in": getattr(usage, "input_tokens", 0) if usage else 0,
                "out": getattr(usage, "output_tokens", 0) if usage else 0,
            },
        )

    def stream(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
        """Native Claude streaming. Yields ("delta", text) per text chunk, accumulates the
        final message (text + tool_use + usage) into a ModelResponse, then ("done", resp).
        Falls back to the base (invoke-once) path on any streaming error so a run never dies
        on a stream hiccup."""
        import anthropic  # lazy

        client = anthropic.Anthropic(api_key=self.api_key)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if self.system:
            kwargs["system"] = self.system
        if tools:
            kwargs["tools"] = tools

        try:
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:        # incremental text deltas
                    if text:
                        yield ("delta", text)
                final = stream.get_final_message()
        except Exception:  # noqa: BLE001 — never die on a stream error; fall back to invoke
            yield from super().stream(messages, tools)
            return

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in final.content or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                tool_calls.append(ToolCall(
                    id=getattr(block, "id", ""),
                    name=getattr(block, "name", ""),
                    args=getattr(block, "input", {}) or {},
                ))
        usage = getattr(final, "usage", None)
        yield ("done", ModelResponse(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            usage={
                "in": getattr(usage, "input_tokens", 0) if usage else 0,
                "out": getattr(usage, "output_tokens", 0) if usage else 0,
            },
        ))


# ── OpenAI (native tool use) — the dev-now vendor, behind a shape adapter ───────

class OpenAIModel(BaseModel):
    """OpenAI Chat Completions with native tool use (GPT-4o / GPT-5-class).

    The loop builds messages in the ANTHROPIC content-block shape; this adapter
    converts them to OpenAI's flatter shape (system param, assistant `tool_calls`,
    `role:"tool"` results) on the way in, and normalizes the response back to our
    uniform ModelResponse on the way out. So the loop stays vendor-neutral and we run
    on Jeel's OpenAI key now; flipping the config model id back to `claude-*` at
    production swaps in AnthropicModel with no loop change (§3.1 multi-vendor).
    """

    def __init__(self, model: str, api_key: str, *, max_tokens: int = 4096,
                 system: Optional[str] = None, temperature: float = 0.0):
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.system = system
        self.temperature = temperature

    # -- shape adapters (Anthropic-style ⇄ OpenAI-style) --------------------------

    @staticmethod
    def _is_reasoning_model(model_id: str) -> bool:
        """GPT-5 line and o-series use max_completion_tokens + default-only temperature.
        The 4.x chat models use the legacy max_tokens + accept temperature=0."""
        mid = (model_id or "").lower()
        return mid.startswith("gpt-5") or mid.startswith("o1") or mid.startswith("o3") \
            or mid.startswith("o4")

    @staticmethod
    def _tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Registry schemas {name, description, input_schema} → OpenAI function tools."""
        out = []
        for t in tools or []:
            out.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return out

    def _messages_to_openai(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert the loop's Anthropic-shaped messages to OpenAI chat messages."""
        import json as _json

        out: List[Dict[str, Any]] = []
        if self.system:
            out.append({"role": "system", "content": self.system})
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            # Plain string content (the initial user question / repair note).
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            # Block-list content (assistant text+tool_use, or user tool_result).
            if role == "assistant":
                text_parts, tool_calls = [], []
                for b in content or []:
                    if b.get("type") == "text":
                        text_parts.append(b.get("text", ""))
                    elif b.get("type") == "tool_use":
                        tool_calls.append({
                            "id": b.get("id", ""),
                            "type": "function",
                            "function": {"name": b.get("name", ""),
                                         "arguments": _json.dumps(b.get("input", {}))},
                        })
                msg: Dict[str, Any] = {"role": "assistant",
                                       "content": "".join(text_parts) or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                out.append(msg)
            elif role == "user":
                # A user turn carrying tool_result blocks → one OpenAI tool message each.
                emitted = False
                for b in content or []:
                    if b.get("type") == "tool_result":
                        out.append({"role": "tool",
                                    "tool_call_id": b.get("tool_use_id", ""),
                                    "content": b.get("content", "")})
                        emitted = True
                if not emitted:
                    # plain user block content
                    txt = "".join(b.get("text", "") for b in content or []
                                  if b.get("type") == "text")
                    out.append({"role": "user", "content": txt})
        return out

    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> ModelResponse:
        import json as _json
        from openai import OpenAI  # lazy

        client = OpenAI(api_key=self.api_key)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self._messages_to_openai(messages),
        }
        oai_tools = self._tools_to_openai(tools)
        if oai_tools:
            kwargs["tools"] = oai_tools

        # GPT-5 / o-series reasoning models renamed `max_tokens`→`max_completion_tokens`
        # and only accept the default temperature (1). The 4.x chat models use the old
        # `max_tokens` and accept temperature=0. Branch on the id so one adapter serves
        # both — and so flipping AGENT_MODEL_* needs no code change.
        if self._is_reasoning_model(self.model):
            kwargs["max_completion_tokens"] = self.max_tokens
            # omit temperature → use the model default (it rejects custom values)
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = self.temperature

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message

        tool_calls: List[ToolCall] = []
        for tc in (getattr(choice, "tool_calls", None) or []):
            try:
                args = _json.loads(tc.function.arguments or "{}")
            except Exception:  # noqa: BLE001 — malformed args → empty; tool will error cleanly
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))

        usage = getattr(resp, "usage", None)
        return ModelResponse(
            text=(choice.content or None),
            tool_calls=tool_calls,
            usage={
                "in": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "out": getattr(usage, "completion_tokens", 0) if usage else 0,
            },
        )

    def stream(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
        """OpenAI streaming. Yields ("delta", text) per content chunk; accumulates tool_call
        fragments (name/arguments arrive split across chunks) and usage, then ("done", resp).
        Falls back to invoke-once on any stream error."""
        import json as _json
        from openai import OpenAI  # lazy

        client = OpenAI(api_key=self.api_key)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self._messages_to_openai(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        oai_tools = self._tools_to_openai(tools)
        if oai_tools:
            kwargs["tools"] = oai_tools
        if self._is_reasoning_model(self.model):
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = self.temperature

        text_parts: List[str] = []
        # tool calls arrive as fragments keyed by index: {idx: {"id","name","args_str"}}
        tc_acc: Dict[int, Dict[str, str]] = {}
        usage_in = usage_out = 0
        try:
            for chunk in client.chat.completions.create(**kwargs):
                ch_usage = getattr(chunk, "usage", None)
                if ch_usage:
                    usage_in = getattr(ch_usage, "prompt_tokens", 0) or usage_in
                    usage_out = getattr(ch_usage, "completion_tokens", 0) or usage_out
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    text_parts.append(piece)
                    yield ("delta", piece)
                for tc in (getattr(delta, "tool_calls", None) or []):
                    i = tc.index
                    slot = tc_acc.setdefault(i, {"id": "", "name": "", "args": ""})
                    if getattr(tc, "id", None):
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn:
                        if getattr(fn, "name", None):
                            slot["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            slot["args"] += fn.arguments
        except Exception:  # noqa: BLE001 — never die on a stream error; fall back to invoke
            yield from super().stream(messages, tools)
            return

        tool_calls: List[ToolCall] = []
        for i in sorted(tc_acc):
            slot = tc_acc[i]
            try:
                args = _json.loads(slot["args"] or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            tool_calls.append(ToolCall(id=slot["id"], name=slot["name"], args=args))

        yield ("done", ModelResponse(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            usage={"in": usage_in, "out": usage_out},
        ))


# ── Offline scripted model (the gate's mocked orchestrator) ─────────────────────

class ScriptedModel(BaseModel):
    """A deterministic, network-free model for the offline loop gate.

    `script` is a list of ModelResponse (or callables `(messages) -> ModelResponse`)
    consumed one per `invoke`. This lets `eval/test_agent_loop.py` drive an exact
    tool-call sequence (e.g. the AWS bridge: argmax→2022, then lookup→513,983, then a
    final answer) with no API spend. When the script is exhausted it returns a final
    empty-text answer so the loop terminates cleanly.
    """

    def __init__(self, script: List[Any]):
        self._script = list(script)
        self._i = 0
        self.calls: List[Dict[str, Any]] = []  # record of (messages, tools) for assertions

    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> ModelResponse:
        self.calls.append({"messages": messages, "n_tools": len(tools or [])})
        if self._i >= len(self._script):
            return ModelResponse(text="(no more scripted steps)", tool_calls=[])
        item = self._script[self._i]
        self._i += 1
        if callable(item):
            return item(messages)
        return item


def build_model(mode: str, budget, config, *, system: Optional[str] = None) -> BaseModel:
    """Construct the live model for a run from config, routing by model-id prefix.

    Multi-vendor (§3.1): `claude*` → AnthropicModel (the production target),
    anything else (gpt*/o*) → OpenAIModel (the dev-now vendor on Jeel's paid key).
    The loop is vendor-neutral; switching vendors = changing `AGENT_MODEL_*` env vars,
    not code. Raises a clear error if the relevant vendor key is missing — the loop
    catches it and degrades (§3.2). Tests inject a ScriptedModel directly instead.
    """
    model_id = budget.model

    if model_id.startswith("claude"):
        key = getattr(config, "ANTHROPIC_API_KEY", "") or ""
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set — cannot run a live agent loop on Claude. "
                "Set it, point AGENT_MODEL_* at an OpenAI id, or inject a model in tests."
            )
        return AnthropicModel(model_id, key, system=system)

    # OpenAI (gpt-*, o-*, or any non-claude id) — the dev-now path.
    key = getattr(config, "OPENAI_API_KEY", "") or ""
    if not key:
        raise RuntimeError(
            f"OPENAI_API_KEY is not set — cannot run a live agent loop on {model_id!r}."
        )
    return OpenAIModel(model_id, key, system=system)
