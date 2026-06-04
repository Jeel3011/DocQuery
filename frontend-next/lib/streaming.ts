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
    | "brain_start" | "brain_map" | "brain_verify" | "brain_reduce" | "brain_meta";
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
