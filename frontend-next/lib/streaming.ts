// lib/streaming.ts
// SSE streaming client for POST /api/v1/query/stream
// Uses fetch + ReadableStream (not EventSource — POST requests need fetch).
// Parses the exact SSE format emitted by generate_stream() in generation.py.

import { SourceInfo } from "./api";
import { API_BASE } from "./config";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface StreamEvent {
  type:
    | "sources" | "token" | "error" | "done" | "status" | "meta" | "sub_queries" | "web_search"
    // Brain (map-reduce) step events — emitted by /query/brain/stream
    | "brain_start" | "brain_analyst" | "brain_map" | "brain_verify" | "brain_reduce" | "brain_meta"
    // Agent-core loop events (§3.6) — emitted by /query/agentcore/stream
    | "agent_step" | "agent_thought" | "tool_call" | "tool_result" | "gate" | "artifact"
    // Live generation preview (the UX fix): incremental text as the model writes it. The
    // final `token` event carries the gated/authoritative text that REPLACES the preview.
    | "token_delta";
  // ── Agent-core event fields (§3.6) ──
  n?: number;                 // agent_step (step number)
  text?: string;              // agent_thought
  name?: string;              // tool_call / tool_result / gate
  args_summary?: string;      // tool_call
  ok?: boolean;               // tool_result
  summary?: string;           // tool_result
  n_provenance?: number;      // tool_result
  pass?: boolean;             // gate
  detail?: string;            // gate
  degrade?: boolean;          // meta (agent-core degrade signal)
  mode?: string;              // meta (standard | deep)
  steps?: number;             // meta
  tokens?: number;            // meta
  n_evidence?: number;        // meta
  content?: string;        // for type='token'
  sources?: SourceInfo[];  // for type='sources'
  message?: string;        // for type='error'
  fallback?: boolean;      // for type='meta' — circuit breaker degraded mode
  cache_hit?: boolean;     // for type='meta' — semantic cache hit
  similarity?: number;     // for type='meta' — cache similarity score
  queries?: string[];      // for type='sub_queries' — decomposed sub-queries
  results_count?: number;  // for type='web_search'
  // ── Brain event fields ──
  docs_routed?: number;       // brain_start
  figures?: number;           // brain_analyst (count of deterministically computed figures)
  filename?: string;          // brain_map
  claims?: number;            // brain_map (claims extracted from this doc)
  relevant?: boolean;         // brain_map
  progress?: string;          // brain_map (e.g. "3/12")
  claims_total?: number;      // brain_verify
  claims_verified?: number;   // brain_verify
  docs_relevant?: number;     // brain_reduce
  groundedness?: number;      // brain_reduce (0-1: fraction of answer sentences entailed by verified claims)
  unsupported?: number;       // brain_reduce (count of answer sentences not entailed)
  confidence?: number;        // brain_meta (0-1)
  abstained?: boolean;        // brain_meta
  coverage?: BrainCoverage;   // brain_meta
}

export interface BrainCoverage {
  docs_routed: number;
  docs_read: number;
  docs_relevant: number;
  docs_failed: number;
}

export interface StreamCallbacks {
  onSources: (sources: SourceInfo[]) => void;
  onToken: (token: string) => void;
  onDone: () => void;
  onError: (message: string) => void;
  onFallback?: () => void; // optional — called when backend is in degraded mode
}

export interface StreamQueryRequest {
  question: string;
  filename_filter?: string | null;
  page_filter?: number | null;
  conversation_id?: string | null;
  collection_id?: string | null;  // Phase 1
  multi_hop?: boolean | null;     // Phase 4.5: force the sequential multi-hop loop on the agent endpoint
  mode?: string | null;           // A4: agent-core mode ("standard" | "deep" | "draft") for /query/agentcore/stream
  // G6.1: draft mode extras — doc_type constrains the deliverable genre; instructions
  // are the user's free-text brief. Both travel as-is to the backend's QueryRequest.
  doc_type?: string | null;
  instructions?: string | null;
  // G3 Step E: active vault filter set (doc_type / fiscal_year). EXPLICIT in the request
  // (never a stale store) → becomes the retriever's CONJUNCTIVE metadata_filter on the
  // backend (narrows the vault scope, never replaces it).
  filters?: Record<string, unknown> | null;
  // G8.7: knowledge-source chips — which authorities the agent may use this run. Subset of
  // {"vault","statutes","caselaw"}. A source omitted is GATED server-side (tool stripped or
  // instrument-type filtered) so the agent cannot cite it. Absent ⇒ all enabled.
  sources?: string[] | null;
}

// ─── Main streaming function ───────────────────────────────────────────────────

export async function streamQuery(
  token: string,
  body: StreamQueryRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE}/api/v1/query/stream`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      callbacks.onDone();
      return;
    }
    callbacks.onError("Network error — could not connect to server.");
    return;
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error");
    callbacks.onError(`Server error ${response.status}: ${text}`);
    return;
  }

  if (!response.body) {
    callbacks.onError("No response body — streaming not supported.");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // Decode chunk and accumulate in buffer
      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by \n\n — only process complete lines
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? ""; // keep incomplete last line in buffer

      for (const line of lines) {
        const trimmed = line.trim();

        // Skip empty lines and non-data lines (SSE comments, event: lines)
        if (!trimmed || !trimmed.startsWith("data: ")) continue;

        const dataStr = trimmed.slice(6); // strip "data: "

        // Stream terminator
        if (dataStr === "[DONE]") {
          callbacks.onDone();
          return;
        }

        try {
          const event = JSON.parse(dataStr) as StreamEvent;

          switch (event.type) {
            case "sources":
              callbacks.onSources(event.sources ?? []);
              break;
            case "token":
              if (event.content) callbacks.onToken(event.content);
              break;
            case "error":
              callbacks.onError(event.message ?? "Stream error");
              break;
            case "meta":
              // Degraded mode — circuit breaker fallback was triggered
              if (event.fallback && callbacks.onFallback) {
                callbacks.onFallback();
              }
              break;
            case "status":
              // Informational only — UI handles loading state separately
              break;
          }
        } catch {
          // Malformed JSON line — skip silently
          console.warn("[streaming] Malformed SSE line:", dataStr);
        }
      }
    }
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      callbacks.onDone();
    } else {
      callbacks.onError("Stream interrupted. Please try again.");
    }
  } finally {
    reader.releaseLock();
  }
}


// ─── Agentic streaming (Phase 6) ──────────────────────────────────────────────

export interface AgenticStreamCallbacks extends StreamCallbacks {
  onSubQueries?: (queries: string[]) => void;
  onWebSearch?: (resultsCount: number) => void;
}

export async function streamAgenticQuery(
  token: string,
  body: StreamQueryRequest,
  callbacks: AgenticStreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE}/api/v1/query/agent/stream`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      callbacks.onDone();
      return;
    }
    callbacks.onError("Network error — could not connect to server.");
    return;
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error");
    callbacks.onError(`Server error ${response.status}: ${text}`);
    return;
  }

  if (!response.body) {
    callbacks.onError("No response body — streaming not supported.");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data: ")) continue;
        const dataStr = trimmed.slice(6);

        if (dataStr === "[DONE]") {
          callbacks.onDone();
          return;
        }

        try {
          const event = JSON.parse(dataStr) as StreamEvent;

          switch (event.type) {
            case "sub_queries":
              if (callbacks.onSubQueries && event.queries) {
                callbacks.onSubQueries(event.queries);
              }
              break;
            case "web_search":
              if (callbacks.onWebSearch) {
                callbacks.onWebSearch(event.results_count ?? 0);
              }
              break;
            case "sources":
              callbacks.onSources(event.sources ?? []);
              break;
            case "token":
              if (event.content) callbacks.onToken(event.content);
              break;
            case "error":
              callbacks.onError(event.message ?? "Stream error");
              break;
            case "meta":
              if (event.fallback && callbacks.onFallback) {
                callbacks.onFallback();
              }
              break;
          }
        } catch {
          console.warn("[agentic-streaming] Malformed SSE line:", dataStr);
        }
      }
    }
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      callbacks.onDone();
    } else {
      callbacks.onError("Stream interrupted. Please try again.");
    }
  } finally {
    reader.releaseLock();
  }
}


// ─── Brain (map-reduce synthesis) streaming — Phase 4 ─────────────────────────
// POST /api/v1/query/brain/stream. Requires collection_id and USE_BRAIN=true on
// the backend. Emits the standard sources/token events PLUS brain_* step events
// that drive the live ThinkingStream and the TrustBar coverage/confidence.

export interface BrainStreamCallbacks extends StreamCallbacks {
  onBrainStart?: (docsRouted: number) => void;
  onBrainAnalyst?: (figures: number) => void;
  onBrainMap?: (ev: { filename?: string; claims?: number; relevant?: boolean; progress?: string }) => void;
  onBrainVerify?: (claimsTotal: number, claimsVerified: number) => void;
  onBrainReduce?: (docsRelevant: number, groundedness?: number, unsupported?: number) => void;
  onBrainMeta?: (meta: { confidence: number; abstained: boolean; coverage?: BrainCoverage }) => void;
}

export async function streamBrainQuery(
  token: string,
  body: StreamQueryRequest,
  callbacks: BrainStreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE}/api/v1/query/brain/stream`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      callbacks.onDone();
      return;
    }
    callbacks.onError("Network error — could not connect to server.");
    return;
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error");
    // 403 = USE_BRAIN disabled, 400 = collection_id missing — surface clearly.
    callbacks.onError(`Server error ${response.status}: ${text}`);
    return;
  }

  if (!response.body) {
    callbacks.onError("No response body — streaming not supported.");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data: ")) continue;
        const dataStr = trimmed.slice(6);

        if (dataStr === "[DONE]") {
          callbacks.onDone();
          return;
        }

        try {
          const event = JSON.parse(dataStr) as StreamEvent;

          switch (event.type) {
            case "brain_start":
              callbacks.onBrainStart?.(event.docs_routed ?? 0);
              break;
            case "brain_analyst":
              callbacks.onBrainAnalyst?.(event.figures ?? 0);
              break;
            case "brain_map":
              callbacks.onBrainMap?.({
                filename: event.filename,
                claims: event.claims,
                relevant: event.relevant,
                progress: event.progress,
              });
              break;
            case "brain_verify":
              callbacks.onBrainVerify?.(event.claims_total ?? 0, event.claims_verified ?? 0);
              break;
            case "brain_reduce":
              callbacks.onBrainReduce?.(event.docs_relevant ?? 0, event.groundedness, event.unsupported);
              break;
            case "brain_meta":
              callbacks.onBrainMeta?.({
                confidence: event.confidence ?? 0,
                abstained: event.abstained ?? false,
                coverage: event.coverage,
              });
              break;
            case "sources":
              callbacks.onSources(event.sources ?? []);
              break;
            case "token":
              if (event.content) callbacks.onToken(event.content);
              break;
            case "error":
              callbacks.onError(event.message ?? "Stream error");
              break;
          }
        } catch {
          console.warn("[brain-streaming] Malformed SSE line:", dataStr);
        }
      }
    }
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      callbacks.onDone();
    } else {
      callbacks.onError("Stream interrupted. Please try again.");
    }
  } finally {
    reader.releaseLock();
  }
}


// ─── Agent-core (frontier-model tool loop) streaming — Phase A (A4) ───────────
// POST /api/v1/query/agentcore/stream. Requires collection_id and USE_AGENT_CORE=true
// on the backend (flag off ⇒ 404). The model IS the orchestrator; this emits the §3.6
// loop events (agent_step / agent_thought / tool_call / tool_result / gate / artifact)
// that drive the same ThinkingStream timeline as Brain — now showing the model's tool
// calls and the non-bypassable output gates, plus the standard sources/token answer.

export interface AgentCoreStreamCallbacks extends StreamCallbacks {
  onAgentStep?: (n: number) => void;
  onAgentThought?: (text: string) => void;
  onToolCall?: (name: string, argsSummary?: string) => void;
  onToolResult?: (ev: { name: string; ok: boolean; summary?: string; nProvenance?: number }) => void;
  onGate?: (ev: { name: string; pass: boolean; detail?: string }) => void;
  onAgentMeta?: (meta: { mode?: string; steps?: number; tokens?: number; abstained?: boolean; degrade?: boolean }) => void;
  // Live generation preview chunk (the UX fix). Append to the bubble as it streams; the
  // final onToken() carries the authoritative gated text that REPLACES the accumulated preview.
  onTokenDelta?: (chunk: string) => void;
}

export async function streamAgentCoreQuery(
  token: string,
  body: StreamQueryRequest,
  callbacks: AgentCoreStreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE}/api/v1/query/agentcore/stream`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      callbacks.onDone();
      return;
    }
    callbacks.onError("Network error — could not connect to server.");
    return;
  }

  if (!response.ok) {
    // 404 = USE_AGENT_CORE disabled on the backend. The Agent is the default smart mode,
    // but until the (paid) flag is turned on the route does not exist — so degrade
    // SILENTLY to the standard retrieve→generate path instead of erroring at the user.
    // This keeps the app working today; flip USE_AGENT_CORE=true to get the real loop.
    if (response.status === 404) {
      await streamQuery(token, body, callbacks, signal);
      return;
    }
    const text = await response.text().catch(() => "Unknown error");
    // 400 = collection_id missing — surface clearly.
    callbacks.onError(`Server error ${response.status}: ${text}`);
    return;
  }

  if (!response.body) {
    callbacks.onError("No response body — streaming not supported.");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data: ")) continue;
        const dataStr = trimmed.slice(6);

        if (dataStr === "[DONE]") {
          callbacks.onDone();
          return;
        }

        try {
          const event = JSON.parse(dataStr) as StreamEvent;

          switch (event.type) {
            case "agent_step":
              callbacks.onAgentStep?.(event.n ?? 0);
              break;
            case "agent_thought":
              callbacks.onAgentThought?.(event.text ?? "");
              break;
            case "tool_call":
              callbacks.onToolCall?.(event.name ?? "tool", event.args_summary);
              break;
            case "tool_result":
              callbacks.onToolResult?.({
                name: event.name ?? "tool",
                ok: event.ok ?? false,
                summary: event.summary,
                nProvenance: event.n_provenance,
              });
              break;
            case "gate":
              callbacks.onGate?.({
                name: event.name ?? "gate",
                pass: event.pass ?? false,
                detail: event.detail,
              });
              break;
            case "sources":
              callbacks.onSources(event.sources ?? []);
              break;
            case "token_delta":
              // Live preview as the model writes — appended incrementally. Falls back to
              // onToken (a plain append) if the consumer didn't wire the delta handler.
              if (event.content) (callbacks.onTokenDelta ?? callbacks.onToken)(event.content);
              break;
            case "token":
              if (event.content) callbacks.onToken(event.content);
              break;
            case "meta":
              callbacks.onAgentMeta?.({
                mode: event.mode,
                steps: event.steps,
                tokens: event.tokens,
                abstained: event.abstained,
                degrade: event.degrade,
              });
              if (event.degrade && callbacks.onFallback) callbacks.onFallback();
              break;
            case "error":
              callbacks.onError(event.message ?? "Stream error");
              break;
          }
        } catch {
          console.warn("[agentcore-streaming] Malformed SSE line:", dataStr);
        }
      }
    }
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      callbacks.onDone();
    } else {
      callbacks.onError("Stream interrupted. Please try again.");
    }
  } finally {
    reader.releaseLock();
  }
}


// ─── Review grid (Phase B2) — POST /api/v1/review-grid/stream ───────────────────
// Streams grid_start → cell* → grid_done. Self-contained (its own types) so it does
// not touch the shared StreamEvent union above.

export interface GridCellEvent {
  doc_id: string;
  doc_name?: string;
  column_key: string;
  status: "found" | "missing" | "abstain" | "error";
  value?: string | null;
  quote?: string | null;
  risk: "standard" | "non_standard" | "missing" | "none";
  note?: string | null;
  // WHY an abstain cell abstained, distinguishably (G4): "unparsed" = the agent answered
  // but in a shape we couldn't read (our bug class); "no_evidence" = it couldn't ground an
  // answer; "ambiguous" = conflicting/declined. Lets the UI explain "Unclear" instead of
  // conflating a parser bug with a genuine not-found.
  abstain_reason?: "unparsed" | "no_evidence" | "ambiguous" | null;
  provenance?: Array<Record<string, unknown>>;
  verified?: boolean;
}

export interface GridStart {
  title: string;
  rows: number;
  columns: number;
  cells: number;
  doc_names?: string[];
  column_labels?: string[];
}

export interface GridDone {
  coverage?: Record<string, number>;
  error?: string;
}

export interface ReviewGridColumnSpec {
  key: string;
  label: string;
  prompt: string;
  kind?: string;
  risk_rubric?: string | null;
}

export interface ReviewGridRequest {
  title?: string;
  collection_id: string;
  doc_ids?: string[];
  columns: ReviewGridColumnSpec[];
  // G3 Step E: same semantics as StreamQueryRequest.filters — narrows the reviewed row
  // set (doc_type / fiscal_year), null-safe on the backend.
  filters?: Record<string, unknown> | null;
}

export interface ReviewGridCallbacks {
  onStart?: (s: GridStart) => void;
  onCell: (c: GridCellEvent) => void;
  onDone: (d: GridDone) => void;
  onError: (message: string) => void;
}

export async function streamReviewGrid(
  token: string,
  body: ReviewGridRequest,
  callbacks: ReviewGridCallbacks,
  signal?: AbortSignal
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/v1/review-grid/stream`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    callbacks.onError("Network error — could not connect to server.");
    return;
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error");
    if (response.status === 404) {
      callbacks.onError("Review grid is unavailable (agent core is off).");
      return;
    }
    callbacks.onError(`Server error ${response.status}: ${text}`);
    return;
  }
  if (!response.body) {
    callbacks.onError("No response body — streaming not supported.");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data: ")) continue;
        const dataStr = trimmed.slice(6);
        if (dataStr === "[DONE]") return;
        try {
          const ev = JSON.parse(dataStr) as Record<string, unknown>;
          switch (ev.type) {
            case "grid_start":
              callbacks.onStart?.(ev as unknown as GridStart);
              break;
            case "cell":
              callbacks.onCell(ev as unknown as GridCellEvent);
              break;
            case "grid_done":
              callbacks.onDone(ev as unknown as GridDone);
              break;
          }
        } catch {
          console.warn("[review-grid] malformed SSE line:", dataStr);
        }
      }
    }
  } catch (err) {
    if ((err as Error).name !== "AbortError") {
      callbacks.onError("Stream interrupted. Please try again.");
    }
  } finally {
    reader.releaseLock();
  }
}

// ─── Workflows (Phase G7) — POST /api/v1/workflows/{id}/run ─────────────────────
// A workflow run reuses the EXISTING engine, so it reuses the EXISTING event shapes:
//  · GRID  (Review) → grid_start → cell* → grid_done (streamWorkflowRun, below)
//  · REPORT (Draft) / OUTPUT (Output) → the agent-core token/sources/step/gate events
//    (streamWorkflowReport, below) — a cited memo or freeform deliverable, lands in the
//    answer/artifact surface. No new renderer; both reuse what Ask/Review already render.

export interface WorkflowRunBody {
  collection_id: string;
  params?: Record<string, unknown>;
  doc_ids?: string[];
  filters?: Record<string, unknown> | null;
  conversation_id?: string | null;
}

export async function streamWorkflowRun(
  token: string,
  templateId: string,
  body: WorkflowRunBody,
  callbacks: ReviewGridCallbacks,   // grid-shape templates emit the review-grid events
  signal?: AbortSignal
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/v1/workflows/${encodeURIComponent(templateId)}/run`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    callbacks.onError("Network error — could not connect to server.");
    return;
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error");
    if (response.status === 404) {
      callbacks.onError("Workflows are unavailable (agent core is off).");
      return;
    }
    callbacks.onError(`Server error ${response.status}: ${text}`);
    return;
  }
  if (!response.body) {
    callbacks.onError("No response body — streaming not supported.");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data: ")) continue;
        const dataStr = trimmed.slice(6);
        if (dataStr === "[DONE]") return;
        try {
          const ev = JSON.parse(dataStr) as Record<string, unknown>;
          switch (ev.type) {
            case "grid_start":
              callbacks.onStart?.(ev as unknown as GridStart);
              break;
            case "cell":
              callbacks.onCell(ev as unknown as GridCellEvent);
              break;
            case "grid_done":
              callbacks.onDone(ev as unknown as GridDone);
              break;
          }
        } catch {
          console.warn("[workflow] malformed SSE line:", dataStr);
        }
      }
    }
  } catch (err) {
    if ((err as Error).name !== "AbortError") {
      callbacks.onError("Stream interrupted. Please try again.");
    }
  } finally {
    reader.releaseLock();
  }
}

// REPORT (Draft) / OUTPUT (Output) workflows stream the agent-core events. This helper
// turns the raw loop events into a LIVE ACTIVITY TIMELINE (the same shape ThinkingStream
// renders on the Ask screen): each tool_call → an active step ("Searching the vault"),
// each tool_result → that step completes with a detail/chips, the gate → "Verifying",
// then the deliverable streams in. The run drawer shows the agent WORKING, not a spinner.
export interface WorkflowStep {
  id: string;
  kind: "think" | "search" | "read" | "compute" | "verify";
  label: string;        // "Searching the vault", "Reading goog-2023", "Verifying citations"
  detail?: string;      // the args/result summary
  status: "active" | "done" | "failed";
}
export interface WorkflowReportCallbacks {
  onStep?: (step: WorkflowStep) => void; // a structured live step (new or status update)
  onToken: (chunk: string) => void;      // the deliverable streams in
  onSources?: (sources: SourceInfo[]) => void;
  onGate?: (passed: boolean, detail: string) => void;
  onDone: () => void;
  onError: (message: string) => void;
}

// Friendly label + kind for a tool name (the agent's verbs, Harvey-style).
function _toolStep(name: string): { kind: WorkflowStep["kind"]; label: string } {
  switch (name) {
    case "search_vault": return { kind: "search", label: "Searching the vault" };
    case "read_document": return { kind: "read", label: "Reading the document" };
    case "table_lookup": return { kind: "compute", label: "Looking up a figure" };
    case "compute": return { kind: "compute", label: "Computing from the source" };
    case "list_metrics": return { kind: "compute", label: "Listing the document's metrics" };
    case "survey_collection": return { kind: "search", label: "Surveying the vault" };
    default: return { kind: "think", label: `Using ${name}` };
  }
}

export async function streamWorkflowReport(
  token: string,
  templateId: string,
  body: WorkflowRunBody,
  callbacks: WorkflowReportCallbacks,
  signal?: AbortSignal
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/v1/workflows/${encodeURIComponent(templateId)}/run`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    callbacks.onError("Network error — could not connect to server.");
    return;
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error");
    if (response.status === 404) { callbacks.onError("Workflows are unavailable (agent core is off)."); return; }
    callbacks.onError(`Server error ${response.status}: ${text}`);
    return;
  }
  if (!response.body) { callbacks.onError("No response body — streaming not supported."); return; }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let stepN = 0;  // increments per agent_step so a tool_call/tool_result share a step id

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data: ")) continue;
        const dataStr = trimmed.slice(6);
        if (dataStr === "[DONE]") { callbacks.onDone(); return; }
        try {
          const ev = JSON.parse(dataStr) as Record<string, unknown>;
          // one step id per agent step so a tool_call → its tool_result update the SAME row
          const sid = `s${stepN}`;
          switch (ev.type) {
            case "token":
              callbacks.onToken(String(ev.content ?? ""));
              break;
            case "sources":
              callbacks.onSources?.(((ev.sources as SourceInfo[]) ?? []));
              break;
            case "agent_step":
              stepN += 1;
              break;
            case "tool_call": {
              const t = _toolStep(String(ev.name ?? ""));
              callbacks.onStep?.({
                id: `${sid}-${String(ev.name)}`, kind: t.kind, label: t.label,
                detail: String(ev.args_summary ?? "") || undefined, status: "active",
              });
              break;
            }
            case "tool_result": {
              const t = _toolStep(String(ev.name ?? ""));
              const np = Number(ev.n_provenance ?? 0);
              callbacks.onStep?.({
                id: `${sid}-${String(ev.name)}`, kind: t.kind, label: t.label,
                detail: (String(ev.summary ?? "") || (np ? `${np} source${np === 1 ? "" : "s"}` : undefined)),
                status: ev.ok === false ? "failed" : "done",
              });
              break;
            }
            case "gate":
              if (ev.name === "output") {
                callbacks.onStep?.({
                  id: "gate", kind: "verify", label: "Verifying citations",
                  detail: String(ev.detail ?? ""), status: ev.pass ? "done" : "failed",
                });
                callbacks.onGate?.(!!ev.pass, String(ev.detail ?? ""));
              }
              break;
          }
        } catch {
          console.warn("[workflow-report] malformed SSE line:", dataStr);
        }
      }
    }
    callbacks.onDone();
  } catch (err) {
    if ((err as Error).name !== "AbortError") {
      callbacks.onError("Stream interrupted. Please try again.");
    }
  } finally {
    reader.releaseLock();
  }
}

// ─── Redline (G6.3) — clause-by-clause review of one document vs a playbook OR a ──────
// catalog doc type (LEGAL_TASK_CATALOG §2.1). The backend streams one `finding` per clause
// topic, then a `redline_done` summary. Each finding is grounded (a quoted clause span) or
// a flagged MISSING/ABSTAIN — never a silent gap. Mirrors streamWorkflowRun's SSE plumbing.
export interface RedlineFindingEvent {
  type: "finding";
  clause_topic: string;
  status: "deviation" | "conforming" | "missing" | "abstain";
  target_quote: string | null;
  deviation: string | null;
  suggested_edit: string | null;
  rationale: string | null;
  playbook_standard: string;
  grounded: boolean;
}
export interface RedlineDoneEvent {
  type: "redline_done";
  deviations: number;
  conforming: number;
  missing: number;
  abstained: number;
}
export interface RedlineRunBody {
  collection_id: string;
  doc_id: string;
  // Pass EITHER a catalog doc_type (the §2.1 link — derives clause topics from the doc
  // type's structure + the firm's stored playbook) OR explicit playbook_rows.
  doc_type?: string | null;
  playbook_rows?: Array<{ clause_topic: string; standard_position: string; fallback_position?: string | null }>;
  title?: string | null;
}
export interface RedlineCallbacks {
  onFinding: (f: RedlineFindingEvent) => void;
  onDone: (d: RedlineDoneEvent) => void;
  onError: (message: string) => void;
}

export async function streamRedline(
  token: string,
  body: RedlineRunBody,
  callbacks: RedlineCallbacks,
  signal?: AbortSignal
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/v1/redline/stream`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    callbacks.onError("Network error — could not connect to server.");
    return;
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error");
    if (response.status === 404) {
      callbacks.onError("Redline is unavailable (agent core is off).");
      return;
    }
    callbacks.onError(`Server error ${response.status}: ${text}`);
    return;
  }
  if (!response.body) {
    callbacks.onError("No response body — streaming not supported.");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data: ")) continue;
        const dataStr = trimmed.slice(6);
        if (dataStr === "[DONE]") return;
        try {
          const ev = JSON.parse(dataStr) as Record<string, unknown>;
          if (ev.type === "finding") callbacks.onFinding(ev as unknown as RedlineFindingEvent);
          else if (ev.type === "redline_done") callbacks.onDone(ev as unknown as RedlineDoneEvent);
          else if (ev.type === "error") callbacks.onError(String(ev.message ?? "Redline error"));
        } catch {
          console.warn("[redline] malformed SSE line:", dataStr);
        }
      }
    }
  } catch (err) {
    if ((err as Error).name !== "AbortError") {
      callbacks.onError("Stream interrupted. Please try again.");
    }
  } finally {
    reader.releaseLock();
  }
}
