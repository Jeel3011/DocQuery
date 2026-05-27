// lib/streaming.ts
// SSE streaming client for POST /api/v1/query/stream
// Uses fetch + ReadableStream (not EventSource — POST requests need fetch).
// Parses the exact SSE format emitted by generate_stream() in generation.py.

import { SourceInfo } from "./api";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface StreamEvent {
  type: "sources" | "token" | "error" | "done" | "status" | "meta" | "sub_queries";
  content?: string;       // for type='token'
  sources?: SourceInfo[]; // for type='sources'
  message?: string;       // for type='error'
  fallback?: boolean;     // for type='meta' — circuit breaker degraded mode
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
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const event = JSON.parse(dataStr) as StreamEvent & { queries?: string[] };

          switch (event.type) {
            case "sub_queries":
              if (callbacks.onSubQueries && event.queries) {
                callbacks.onSubQueries(event.queries);
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
