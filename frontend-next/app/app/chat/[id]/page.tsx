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
import { useAuthStore } from "@/stores/auth.store";
import { getMessages, MessageResponse, SourceInfo, exportConversation, compareMultiDocuments, MultiComparisonResult, DocumentResponse, listDocuments } from "@/lib/api";
import { streamQuery, streamAgenticQuery } from "@/lib/streaming";
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
  const [showExport, setShowExport] = useState(false);
  const [showCompare, setShowCompare] = useState(false);
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [compareDocIds, setCompareDocIds] = useState<string[]>([]);
  const [compareFocus, setCompareFocus] = useState("");
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareResult, setCompareResult] = useState<MultiComparisonResult | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const streamingIdRef = useRef<string | null>(null);
  const autoSubmittedRef = useRef(false);
  const exportRef = useRef<HTMLDivElement>(null);
  // Stable refs to avoid re-creating handleSubmit on every state change
  const isStreamingRef = useRef(false);
  const agenticModeRef = useRef(false);
  const activeCollectionIdRef = useRef<string | null>(null);

  // Keep refs in sync with state
  useEffect(() => { isStreamingRef.current = isStreaming; }, [isStreaming]);
  useEffect(() => { agenticModeRef.current = agenticMode; }, [agenticMode]);
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
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, isStreaming: false, isFallback }
                : m
            )
          );
          setIsStreaming(false);
        },
        onError: (msg: string) => {
          toast.error(msg);
          setMessages((prev) => prev.filter((m) => m.id !== assistantMsgId));
          setIsStreaming(false);
        },
      };

      const collId = activeCollectionIdRef.current;

      if (agenticModeRef.current) {
        await streamAgenticQuery(
          token,
          { question, conversation_id: convId, collection_id: collId },
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
    [token, convId] // Stable deps only — no isStreaming, agenticMode, activeCollectionId
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

  async function handleCompareMulti() {
    if (!token || compareDocIds.length < 2) return;
    setCompareLoading(true);
    setCompareResult(null);
    try {
      const result = await compareMultiDocuments(token, compareDocIds, compareFocus || undefined);
      setCompareResult(result);
    } catch {
      toast.error("Comparison failed");
    } finally {
      setCompareLoading(false);
    }
  }

  function toggleCompareDoc(id: string) {
    setCompareDocIds((prev) =>
      prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id]
    );
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
            sources: m.sources ?? undefined,
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
    <div className="flex flex-col h-full overflow-hidden">
      {/* Top bar — collection badge + export */}
      <div className="flex items-center justify-between px-4 py-1.5 border-b border-[var(--border)] bg-[var(--bg-surface)] flex-shrink-0">
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
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-all"
          >
            <GitCompare size={11} />
            Compare
          </button>

          {/* Export */}
          {messages.length > 0 && !loading && (
            <div className="relative" ref={exportRef}>
              <button
                onClick={() => setShowExport(!showExport)}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-all"
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
          <div className="max-w-3xl mx-auto px-4 md:px-8 py-6 space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex gap-3">
                <div className="w-7 h-7 rounded-lg skeleton flex-shrink-0" />
                <div className="flex-1 space-y-2">
                  <div className="h-3 skeleton rounded w-24" />
                  <div className="glass rounded-2xl p-4 space-y-2">
                    <div className="h-3 skeleton rounded w-3/4" />
                    <div className="h-3 skeleton rounded w-1/2" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="py-4">
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <div key={msg.id}>
                  {/* Sub-queries indicator for agentic mode */}
                  {msg.subQueries && msg.subQueries.length > 0 && msg.role === "assistant" && (
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
        onToggleAgentic={() => setAgenticMode((v) => !v)}
      />

      {/* Compare Documents Modal */}
      {showCompare && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="card w-full max-w-xl mx-4 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-medium text-[var(--text-primary)]">
                <GitCompare size={14} />
                Compare Documents
              </div>
              <button
                onClick={() => { setShowCompare(false); setCompareResult(null); setCompareDocIds([]); }}
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
                    <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1 block">
                      Select documents <span className="normal-case">({compareDocIds.length} selected — need ≥ 2)</span>
                    </label>
                    <div className="max-h-48 overflow-y-auto space-y-1 rounded-lg border border-[var(--border)] bg-[var(--bg-hover)] p-2">
                      {documents.map((d) => (
                        <label key={d.id} className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-[var(--bg-card)] cursor-pointer">
                          <input
                            type="checkbox"
                            checked={compareDocIds.includes(d.id)}
                            onChange={() => toggleCompareDoc(d.id)}
                            className="accent-[var(--accent)]"
                          />
                          <span className="text-xs text-[var(--text-primary)] truncate">{d.filename}</span>
                        </label>
                      ))}
                    </div>
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
                  onClick={handleCompareMulti}
                  disabled={compareDocIds.length < 2 || compareLoading}
                  className="w-full py-2 rounded-lg bg-[var(--accent)] text-white text-xs font-medium disabled:opacity-40 hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
                >
                  {compareLoading ? (
                    <span className="animate-pulse">Comparing…</span>
                  ) : (
                    <>Compare {compareDocIds.length >= 2 ? `${compareDocIds.length} docs` : ""} <ChevronRight size={12} /></>
                  )}
                </button>
              </>
            ) : (
              <div className="space-y-4 max-h-[65vh] overflow-y-auto scrollbar-thin">
                <div className="flex flex-wrap items-center gap-1.5 text-[10px] text-[var(--text-muted)]">
                  {compareResult.documents.map((fname, i) => (
                    <span key={i} className="font-medium text-[var(--text-primary)] bg-[var(--bg-hover)] px-1.5 py-0.5 rounded">{fname}</span>
                  ))}
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

                {compareResult.per_document.length > 0 && (
                  <div>
                    <div className="text-[10px] uppercase tracking-wide text-[var(--text-muted)] mb-2">Per-document summaries</div>
                    <div className="space-y-3">
                      {compareResult.per_document.map((doc, i) => (
                        <div key={i} className="bg-[var(--bg-hover)] rounded-lg p-3 space-y-1.5">
                          <div className="text-xs font-medium text-[var(--text-primary)]">{doc.filename}</div>
                          <p className="text-[11px] text-[var(--text-secondary)]">{doc.summary}</p>
                          {doc.key_points.length > 0 && (
                            <ul className="space-y-0.5">
                              {doc.key_points.map((kp, j) => (
                                <li key={j} className="text-[11px] text-[var(--text-muted)] flex items-start gap-1.5">
                                  <span className="flex-shrink-0 mt-0.5">·</span>{kp}
                                </li>
                              ))}
                            </ul>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {compareResult.matrix.length > 0 && (
                  <div>
                    <div className="text-[10px] uppercase tracking-wide text-[var(--text-muted)] mb-2">Comparison matrix</div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-[11px] border-collapse">
                        <thead>
                          <tr>
                            <th className="text-left py-1 pr-3 text-[var(--text-muted)] font-medium border-b border-[var(--border)]">Dimension</th>
                            {compareResult.documents.map((fname, i) => (
                              <th key={i} className="text-left py-1 px-2 text-[var(--text-muted)] font-medium border-b border-[var(--border)] truncate max-w-[120px]">{fname}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {compareResult.matrix.map((row, i) => (
                            <tr key={i} className="border-b border-[var(--border)] last:border-0">
                              <td className="py-1.5 pr-3 text-[var(--text-primary)] font-medium">{row.dimension}</td>
                              {compareResult.documents.map((fname, j) => (
                                <td key={j} className="py-1.5 px-2 text-[var(--text-secondary)]">{row.values[fname] ?? "—"}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

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
    </div>
  );
}
