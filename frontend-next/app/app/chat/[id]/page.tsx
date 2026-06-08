"use client";

// app/app/chat/[id]/page.tsx
// Active conversation — SSE streaming, inline source citations, message history.
// Supports both Standard and Agentic (Deep) query modes.
// Reads ?q= query param from suggestion buttons to auto-submit on mount.

import { useEffect, useRef, useState, useCallback, Suspense } from "react";
import { useParams, useSearchParams } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { ChatInput } from "@/components/chat/ChatInput";
import { ArtifactPanel, detectArtifact, Artifact } from "@/components/chat/ArtifactPanel";
import { ThinkingStreamFixture, ThinkingStream, ThinkingStep } from "@/components/chat/ThinkingStream";
import { SkeletonMessage } from "@/components/ui/Skeleton";
import { MOCK_ANSWER_META, AnswerMeta, ConfidenceLevel } from "@/components/chat/TrustBar";
import { useAuthStore } from "@/stores/auth.store";
import { getMessages, MessageResponse, SourceInfo, exportConversation, compareDocuments, ComparisonResult, DocumentResponse, listDocuments } from "@/lib/api";
import { streamQuery, streamAgenticQuery, streamBrainQuery } from "@/lib/streaming";
import { useCollectionStore } from "@/stores/collection.store";
import { toast } from "sonner";
import { Search, Download, FolderOpen, GitCompare, Globe, X, ChevronRight } from "lucide-react";

interface LocalMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceInfo[];
  isStreaming?: boolean;
  isFallback?: boolean;
  subQueries?: string[];
  webSearchUsed?: boolean;
  // Brain (map-reduce synthesis) — real thinking stream + trust meta
  isBrain?: boolean;
  thinkingSteps?: ThinkingStep[];
  thinkingTotalMs?: number;
  answerMeta?: AnswerMeta;
}

export default function ChatPage() {
  return (
    <Suspense fallback={<div className="flex-1" />}>
      <ChatPageInner />
    </Suspense>
  );
}

function ChatPageInner() {
  const { id: convId } = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const { token, user } = useAuthStore();
  const { activeCollectionId } = useCollectionStore();

  const [messages, setMessages] = useState<LocalMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [loading, setLoading] = useState(true);
  const [agenticMode, setAgenticMode] = useState(false);
  const [brainMode, setBrainMode] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const [showCompare, setShowCompare] = useState(false);
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [compareDocA, setCompareDocA] = useState("");
  const [compareDocB, setCompareDocB] = useState("");
  const [compareFocus, setCompareFocus] = useState("");
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareResult, setCompareResult] = useState<ComparisonResult | null>(null);
  const [artifact, setArtifact] = useState<Artifact | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const streamingIdRef = useRef<string | null>(null);
  const autoSubmittedRef = useRef(false);
  const exportRef = useRef<HTMLDivElement>(null);
  // Stable refs to avoid re-creating handleSubmit on every state change
  const isStreamingRef = useRef(false);
  const agenticModeRef = useRef(false);
  const brainModeRef = useRef(false);
  const activeCollectionIdRef = useRef<string | null>(null);

  // Keep refs in sync with state
  useEffect(() => { isStreamingRef.current = isStreaming; }, [isStreaming]);
  useEffect(() => { agenticModeRef.current = agenticMode; }, [agenticMode]);
  useEffect(() => { brainModeRef.current = brainMode; }, [brainMode]);
  useEffect(() => { activeCollectionIdRef.current = activeCollectionId; }, [activeCollectionId]);

  // Close export dropdown on outside click
  useEffect(() => {
    if (!showExport) return;
    function handleClick(e: MouseEvent) {
      if (exportRef.current && !exportRef.current.contains(e.target as Node)) {
        setShowExport(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [showExport]);

  const userInitials = user?.email?.slice(0, 2).toUpperCase() ?? "U";

  // ── Stream handler (stable reference — uses refs, not state) ──────────────
  const handleSubmit = useCallback(
    async (question: string) => {
      if (!token || isStreamingRef.current) return;

      abortRef.current?.abort();
      abortRef.current = new AbortController();

      const userMsgId = crypto.randomUUID();
      const assistantMsgId = crypto.randomUUID();
      streamingIdRef.current = assistantMsgId;

      setMessages((prev) => [
        ...prev,
        { id: userMsgId, role: "user", content: question },
        { id: assistantMsgId, role: "assistant", content: "", isStreaming: true },
      ]);
      setIsStreaming(true);

      let isFallback = false;

      const callbacks = {
        onSources: (sources: SourceInfo[]) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId ? { ...m, sources } : m
            )
          );
        },
        onToken: (tok: string) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, content: m.content + tok }
                : m
            )
          );
        },
        onFallback: () => {
          isFallback = true;
        },
        onDone: () => {
          setMessages((prev) => {
            const updated = prev.map((m) =>
              m.id === assistantMsgId ? { ...m, isStreaming: false, isFallback } : m
            );
            // Auto-detect artifact from the completed assistant message
            const completed = updated.find((m) => m.id === assistantMsgId);
            if (completed) {
              const detected = detectArtifact(completed.content);
              if (detected) setArtifact(detected);
            }
            return updated;
          });
          setIsStreaming(false);
        },
        onError: (msg: string) => {
          toast.error(msg);
          setMessages((prev) => prev.filter((m) => m.id !== assistantMsgId));
          setIsStreaming(false);
        },
      };

      const collId = activeCollectionIdRef.current;

      if (brainModeRef.current) {
        // Brain (map-reduce synthesis) requires a collection scope.
        if (!collId) {
          toast.error("Select a collection to use Brain synthesis");
          setMessages((prev) => prev.filter((m) => m.id !== assistantMsgId && m.id !== userMsgId));
          setIsStreaming(false);
          return;
        }

        // Live thinking-step state machine, mirrored into the message.
        const brainStart = Date.now();
        let relevantCount = 0;
        let steps: ThinkingStep[] = [
          { id: "route", label: "Routing", detail: "Selecting relevant documents", status: "active" },
          { id: "read", label: "Reading documents", status: "pending" },
          { id: "verify", label: "Verifying claims", status: "pending" },
          { id: "ground", label: "Checking answer against evidence", status: "pending" },
          { id: "synth", label: "Writing answer", status: "pending" },
        ];
        const pushSteps = () => {
          const snapshot = steps.map((s) => ({ ...s }));
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantMsgId ? { ...m, thinkingSteps: snapshot } : m))
          );
        };
        const setStep = (id: string, patch: Partial<ThinkingStep>) => {
          steps = steps.map((s) => (s.id === id ? { ...s, ...patch } : s));
          pushSteps();
        };

        setMessages((prev) =>
          prev.map((m) => (m.id === assistantMsgId ? { ...m, isBrain: true } : m))
        );
        pushSteps();

        await streamBrainQuery(
          token,
          { question, conversation_id: convId, collection_id: collId },
          {
            ...callbacks,
            onDone: () => {
              // Finalize any still-running steps and stamp total time.
              steps = steps.map((s) =>
                s.status === "active" || s.status === "pending" ? { ...s, status: "done" as const } : s
              );
              const total = Date.now() - brainStart;
              const snapshot = steps.map((s) => ({ ...s }));
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId ? { ...m, thinkingSteps: snapshot, thinkingTotalMs: total } : m
                )
              );
              callbacks.onDone();
            },
            onBrainStart: (docsRouted) => {
              setStep("route", { status: "done", detail: `Selected ${docsRouted} document${docsRouted !== 1 ? "s" : ""}` });
              setStep("read", { status: "active", detail: `0 / ${docsRouted}` });
            },
            onBrainAnalyst: (figures) => {
              // The deterministic Analyst (§4b) already computed these figures from
              // source table cells before synthesis. Surface it as its own done step
              // (inserted after Routing) so the user sees the Analyst woke up and did
              // real, traced arithmetic — not the LLM guessing numbers. Only fires
              // when ≥1 figure was computed, so a non-numeric question never shows it.
              if (figures <= 0) return;
              if (!steps.some((s) => s.id === "analyst")) {
                const routeIdx = steps.findIndex((s) => s.id === "route");
                const analystStep: ThinkingStep = {
                  id: "analyst",
                  label: "Computing figures",
                  detail: `Analyst computed ${figures} figure${figures !== 1 ? "s" : ""} from source tables`,
                  status: "done",
                };
                steps = [
                  ...steps.slice(0, routeIdx + 1),
                  analystStep,
                  ...steps.slice(routeIdx + 1),
                ];
                pushSteps();
              }
            },
            onBrainMap: (ev) => {
              if (ev.relevant) relevantCount += 1;
              setStep("read", { status: "active", detail: `${ev.progress ?? ""} · ${relevantCount} relevant` });
            },
            onBrainVerify: (total, verified) => {
              setStep("read", { status: "done" });
              setStep("verify", { status: "active", detail: `Verified ${verified} of ${total} claim${total !== 1 ? "s" : ""}` });
            },
            onBrainReduce: (docsRelevant, groundedness, unsupported) => {
              setStep("verify", { status: "done" });
              // The §4a.3-step-2 answer-entailment check has already run by the time
              // brain_reduce fires. Surface its result as its own trust step.
              if (groundedness !== undefined) {
                const pct = Math.round(groundedness * 100);
                const allGrounded = !unsupported;
                setStep("ground", {
                  status: allGrounded ? "done" : "failed",
                  detail: allGrounded
                    ? `All sentences grounded (${pct}%)`
                    : `${unsupported} sentence${unsupported !== 1 ? "s" : ""} flagged · ${pct}% grounded`,
                });
              } else {
                setStep("ground", { status: "done" });
              }
              setStep("synth", { status: "active", detail: `Merging ${docsRelevant} source${docsRelevant !== 1 ? "s" : ""}` });
            },
            onBrainMeta: ({ confidence, abstained, coverage }) => {
              setStep("synth", { status: "done" });
              const level: ConfidenceLevel =
                abstained || confidence < 0.45 ? "low" : confidence >= 0.75 ? "high" : "medium";
              const meta: AnswerMeta = {
                confidence: level,
                consulted: coverage?.docs_relevant,
                total: coverage?.docs_routed,
                claimTypes: ["fact"],
              };
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantMsgId ? { ...m, answerMeta: meta } : m))
              );
            },
          },
          abortRef.current.signal
        );
      } else if (agenticModeRef.current) {
        await streamAgenticQuery(
          token,
          // Phase 4.5: the ⚡ agentic toggle now drives the sequential multi-hop loop
          // (retrieve → reason → informed follow-up). multi_hop=true overrides the
          // server default per-request; drop the flag to fall back to parallel decompose.
          { question, conversation_id: convId, collection_id: collId, multi_hop: true },
          {
            ...callbacks,
            onSubQueries: (queries) => {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId ? { ...m, subQueries: queries } : m
                )
              );
            },
            onWebSearch: () => {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId ? { ...m, webSearchUsed: true } : m
                )
              );
            },
          },
          abortRef.current.signal
        );
      } else {
        await streamQuery(
          token,
          { question, conversation_id: convId, collection_id: collId },
          callbacks,
          abortRef.current.signal
        );
      }
    },
    [token, convId] // Stable deps only — no isStreaming, agenticMode, brainMode, activeCollectionId
  );

  // ── Load documents for the comparison picker (C2: lazy, not on every mount) ──
  // Fetched once, on demand — when the user hovers/opens Compare — instead of
  // eagerly on chat load where most users never open the modal.
  const documentsLoadedRef = useRef(false);
  const loadDocuments = useCallback(() => {
    if (!token || documentsLoadedRef.current) return;
    documentsLoadedRef.current = true;
    listDocuments(token)
      .then((docs) => setDocuments(docs.filter((d) => d.status === "ready")))
      .catch(() => {
        documentsLoadedRef.current = false; // allow retry on next interaction
      });
  }, [token]);

  async function handleCompare() {
    if (!token || !compareDocA || !compareDocB) return;
    if (compareDocA === compareDocB) {
      toast.error("Select two different documents");
      return;
    }
    setCompareLoading(true);
    setCompareResult(null);
    try {
      const result = await compareDocuments(token, compareDocA, compareDocB, compareFocus || undefined);
      setCompareResult(result);
    } catch {
      toast.error("Comparison failed");
    } finally {
      setCompareLoading(false);
    }
  }

  // ── Load message history ──────────────────────────────────────────────────
  useEffect(() => {
    if (!token || !convId) return;
    setLoading(true);
    setMessages([]);
    autoSubmittedRef.current = false;

    getMessages(token, convId)
      .then((msgs) => {
        setMessages(
          msgs.map((m: MessageResponse) => ({
            id: m.id ?? crypto.randomUUID(),
            role: m.role,
            content: m.content,
            // Normalise sources: assign index-based source_id when DB record lacks one
            // (messages saved via /conversations/{id}/messages store sources without source_id)
            sources: m.sources
              ? m.sources.map((s, i) => ({
                  ...s,
                  source_id: s.source_id ?? i + 1,
                }))
              : undefined,
          }))
        );
        return msgs;
      })
      .then((msgs) => {
        // Auto-submit if ?q= param exists and conversation is empty (from suggestion buttons)
        const q = searchParams.get("q");
        if (q && msgs.length === 0 && !autoSubmittedRef.current) {
          autoSubmittedRef.current = true;
          // Small delay to let state settle
          setTimeout(() => handleSubmit(q), 150);
        }
      })
      .catch(() => toast.error("Failed to load messages"))
      .finally(() => setLoading(false));
  }, [convId, token]); // eslint-disable-line react-hooks/exhaustive-deps
  // ^ Intentionally excluding searchParams and handleSubmit to prevent re-fire loops

  // ── Auto-scroll to bottom ─────────────────────────────────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "instant" });
  }, [messages]);

  function handleCancel() {
    abortRef.current?.abort();
    setMessages((prev) =>
      prev.map((m) =>
        m.id === streamingIdRef.current
          ? { ...m, isStreaming: false, content: m.content || "(cancelled)" }
          : m
      )
    );
    setIsStreaming(false);
  }

  async function handleExport(format: "md" | "pdf") {
    if (!token || !convId) return;
    setShowExport(false);
    try {
      const blob = await exportConversation(token, convId, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `conversation.${format}`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`Exported as ${format.toUpperCase()}`);
    } catch {
      toast.error("Export failed");
    }
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Main chat column ── */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
      {/* Top bar — collection badge + export */}
      <div
        className="flex items-center justify-between px-4 py-1.5 border-b flex-shrink-0"
        style={{
          background: "var(--glass-bg-strong)",
          backdropFilter: "blur(16px)",
          WebkitBackdropFilter: "blur(16px)",
          borderColor: "var(--glass-border)",
        }}
      >
        {/* Active collection indicator */}
        {activeCollectionId ? (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[var(--bg-hover)] text-[10px] text-[var(--accent)]">
            <FolderOpen size={11} />
            <span className="font-medium">Scoped to collection</span>
          </div>
        ) : (
          <div className="flex items-center gap-1.5 px-2.5 py-1 text-[10px] text-[var(--text-muted)]">
            All documents
          </div>
        )}

        {/* Right-side actions */}
        <div className="flex items-center gap-1">
          {/* Compare Documents — docs are loaded lazily (C2): prefetch on hover,
              ensure loaded on click. */}
          <button
            onMouseEnter={loadDocuments}
            onClick={() => { loadDocuments(); setShowCompare(true); setCompareResult(null); }}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-[color,background-color]"
          >
            <GitCompare size={11} />
            Compare
          </button>

          {/* Export */}
          {messages.length > 0 && !loading && (
            <div className="relative" ref={exportRef}>
              <button
                onClick={() => setShowExport(!showExport)}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-[color,background-color]"
              >
                <Download size={11} />
                Export
              </button>
              {showExport && (
                <div className="absolute right-0 top-full mt-1 card p-1 min-w-[120px] z-50 shadow-lg">
                  <button
                    onClick={() => handleExport("md")}
                    className="w-full text-left px-3 py-1.5 text-xs text-[var(--text-primary)] hover:bg-[var(--bg-hover)] rounded-md transition-colors"
                  >
                    📝 Markdown
                  </button>
                  <button
                    onClick={() => handleExport("pdf")}
                    className="w-full text-left px-3 py-1.5 text-xs text-[var(--text-primary)] hover:bg-[var(--bg-hover)] rounded-md transition-colors"
                  >
                    📄 PDF
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Messages — scrollable */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin">
        {loading ? (
          <div className="py-4 space-y-1">
            <SkeletonMessage />
            <SkeletonMessage />
            <SkeletonMessage />
          </div>
        ) : (
          <div className="py-4">
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <div key={msg.id}>
                  {/* Brain thinking stream — real, driven by SSE step events */}
                  {msg.role === "assistant" && msg.isBrain && msg.thinkingSteps && msg.thinkingSteps.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="max-w-3xl mx-auto px-4 md:px-8 mb-3"
                    >
                      <div className="card-dotted p-4">
                        <ThinkingStream
                          steps={msg.thinkingSteps}
                          totalMs={msg.thinkingTotalMs}
                          collapsed={!msg.isStreaming}
                        />
                      </div>
                    </motion.div>
                  )}
                  {/* Agentic thinking stream — mock fixture while streaming */}
                  {msg.role === "assistant" && msg.isStreaming && agenticMode && !msg.isBrain && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="max-w-3xl mx-auto px-4 md:px-8 mb-3"
                    >
                      <div className="card-dotted p-4">
                        <ThinkingStreamFixture />
                      </div>
                    </motion.div>
                  )}
                  {/* Sub-queries indicator for agentic mode (after streaming) */}
                  {msg.subQueries && msg.subQueries.length > 0 && msg.role === "assistant" && !msg.isStreaming && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="max-w-3xl mx-auto px-4 md:px-8 mb-2"
                    >
                      <div className="card-dotted p-3 text-xs">
                        <div className="flex items-center gap-2 text-[var(--text-muted)] font-medium mb-2">
                          <Search size={12} />
                          Decomposed into {msg.subQueries.length} sub-queries
                        </div>
                        <ul className="space-y-1 pl-5">
                          {msg.subQueries.map((sq, i) => (
                            <li key={i} className="text-[var(--text-secondary)] list-disc">
                              {sq}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </motion.div>
                  )}
                  {/* Web search fallback indicator */}
                  {msg.webSearchUsed && msg.role === "assistant" && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="max-w-3xl mx-auto px-4 md:px-8 mb-2"
                    >
                      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[var(--bg-hover)] text-[10px] text-[var(--text-muted)]">
                        <Globe size={11} className="text-[var(--accent)]" />
                        <span>No relevant content found in documents — searched the web for context</span>
                      </div>
                    </motion.div>
                  )}
                  <ChatMessage
                    role={msg.role}
                    content={msg.content}
                    sources={msg.sources}
                    isStreaming={msg.isStreaming}
                    isFallback={msg.isFallback}
                    userInitials={userInitials}
                    showTrust={!msg.isStreaming && msg.role === "assistant" && !!msg.sources?.length}
                    answerMeta={msg.answerMeta ?? MOCK_ANSWER_META}
                  />
                </div>
              ))}
            </AnimatePresence>
            <div ref={bottomRef} className="h-4" />
          </div>
        )}
      </div>

      {/* Input — sticky at bottom */}
      <ChatInput
        onSubmit={handleSubmit}
        onCancel={handleCancel}
        isStreaming={isStreaming}
        agenticMode={agenticMode}
        onToggleAgentic={() => { setAgenticMode((v) => !v); setBrainMode(false); }}
        brainMode={brainMode}
        onToggleBrain={() => { setBrainMode((v) => !v); setAgenticMode(false); }}
      />

      {/* Compare Documents Modal */}
      {showCompare && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="card w-full max-w-lg mx-4 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-medium text-[var(--text-primary)]">
                <GitCompare size={14} />
                Compare Documents
              </div>
              <button
                onClick={() => { setShowCompare(false); setCompareResult(null); }}
                className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
              >
                <X size={14} />
              </button>
            </div>

            {!compareResult ? (
              <>
                {documents.length < 2 && (
                  <p className="text-[11px] text-[var(--text-muted)]">
                    You need at least two processed documents to compare.
                  </p>
                )}
                <div className="space-y-3">
                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1 block">Document A</label>
                    <select
                      value={compareDocA}
                      onChange={(e) => setCompareDocA(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-[var(--bg-hover)] border border-[var(--border)] text-xs text-[var(--text-primary)] focus:outline-none"
                    >
                      <option value="">Select a document…</option>
                      {documents.map((d) => (
                        <option key={d.id} value={d.id}>{d.filename}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1 block">Document B</label>
                    <select
                      value={compareDocB}
                      onChange={(e) => setCompareDocB(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-[var(--bg-hover)] border border-[var(--border)] text-xs text-[var(--text-primary)] focus:outline-none"
                    >
                      <option value="">Select a document…</option>
                      {documents.map((d) => (
                        <option key={d.id} value={d.id}>{d.filename}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1 block">Focus area <span className="normal-case">(optional)</span></label>
                    <input
                      type="text"
                      placeholder="e.g. pricing, methodology, legal terms"
                      value={compareFocus}
                      onChange={(e) => setCompareFocus(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-[var(--bg-hover)] border border-[var(--border)] text-xs text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none"
                    />
                  </div>
                </div>
                <button
                  onClick={handleCompare}
                  disabled={!compareDocA || !compareDocB || compareLoading}
                  className="w-full py-2 rounded-lg bg-[var(--accent)] text-white text-xs font-medium disabled:opacity-40 hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
                >
                  {compareLoading ? (
                    <span className="animate-pulse">Comparing…</span>
                  ) : (
                    <>Compare <ChevronRight size={12} /></>
                  )}
                </button>
              </>
            ) : (
              <div className="space-y-4 max-h-[60vh] overflow-y-auto scrollbar-thin">
                <div className="flex items-center gap-2 text-[10px] text-[var(--text-muted)]">
                  <span className="font-medium text-[var(--text-primary)]">{compareResult.document_a}</span>
                  <span>vs</span>
                  <span className="font-medium text-[var(--text-primary)]">{compareResult.document_b}</span>
                  {compareResult.focus_area && <span>· {compareResult.focus_area}</span>}
                </div>

                <div>
                  <div className="text-[10px] uppercase tracking-wide text-[var(--text-muted)] mb-2">Similarities</div>
                  <ul className="space-y-1.5">
                    {compareResult.similarities.map((s, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-[var(--text-secondary)]">
                        <span className="mt-0.5 text-green-500 flex-shrink-0">●</span>
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <div className="text-[10px] uppercase tracking-wide text-[var(--text-muted)] mb-2">Differences</div>
                  <ul className="space-y-1.5">
                    {compareResult.differences.map((d, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-[var(--text-secondary)]">
                        <span className="mt-0.5 text-orange-400 flex-shrink-0">●</span>
                        {d}
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="pt-2 border-t border-[var(--border)]">
                  <div className="text-[10px] uppercase tracking-wide text-[var(--text-muted)] mb-1">Summary</div>
                  <p className="text-xs text-[var(--text-secondary)]">{compareResult.summary}</p>
                </div>

                <button
                  onClick={() => setCompareResult(null)}
                  className="w-full py-1.5 rounded-lg border border-[var(--border)] text-xs text-[var(--text-muted)] hover:bg-[var(--bg-hover)] transition-colors"
                >
                  Compare again
                </button>
              </div>
            )}
          </div>
        </div>
      )}
      </div>{/* end main chat column */}

      {/* ── Artifact panel (right, slide-in) ── */}
      <ArtifactPanel artifact={artifact} onClose={() => setArtifact(null)} />
    </div>
  );
}
