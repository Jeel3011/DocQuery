"use client";

// app/app/chat/[id]/page.tsx
// Active conversation — SSE streaming, inline source citations, message history.
// Supports both Standard and Agentic (Deep) query modes.
// Reads ?q= query param from suggestion buttons to auto-submit on mount.

import { useEffect, useRef, useState, useCallback } from "react";
import { useParams, useSearchParams } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { ChatInput } from "@/components/chat/ChatInput";
import { useAuthStore } from "@/stores/auth.store";
import { getMessages, MessageResponse, SourceInfo, exportConversation } from "@/lib/api";
import { streamQuery, streamAgenticQuery } from "@/lib/streaming";
import { useCollectionStore } from "@/stores/collection.store";
import { toast } from "sonner";
import { Search, Download, FolderOpen } from "lucide-react";

interface LocalMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceInfo[];
  isStreaming?: boolean;
  isFallback?: boolean;
  subQueries?: string[];
}

export default function ChatPage() {
  const { id: convId } = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const { token, user } = useAuthStore();
  const { activeCollectionId } = useCollectionStore();

  const [messages, setMessages] = useState<LocalMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [loading, setLoading] = useState(true);
  const [agenticMode, setAgenticMode] = useState(false);
  const [showExport, setShowExport] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const streamingIdRef = useRef<string | null>(null);
  const autoSubmittedRef = useRef(false);
  // Stable refs to avoid re-creating handleSubmit on every state change
  const isStreamingRef = useRef(false);
  const agenticModeRef = useRef(false);
  const activeCollectionIdRef = useRef<string | null>(null);

  // Keep refs in sync with state
  useEffect(() => { isStreamingRef.current = isStreaming; }, [isStreaming]);
  useEffect(() => { agenticModeRef.current = agenticMode; }, [agenticMode]);
  useEffect(() => { activeCollectionIdRef.current = activeCollectionId; }, [activeCollectionId]);

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

        {/* Export */}
        {messages.length > 0 && !loading && (
          <div className="relative">
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
    </div>
  );
}
