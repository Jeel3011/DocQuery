"use client";

// app/app/chat/[id]/page.tsx
// Active conversation — SSE streaming, inline source citations, message history.
// Per UX_MASTERPLAN Section 7: sources are inline below the message, not a side panel.

import { useEffect, useRef, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import { AnimatePresence } from "framer-motion";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { ChatInput } from "@/components/chat/ChatInput";
import { useAuthStore } from "@/stores/auth.store";
import { getMessages, MessageResponse, SourceInfo } from "@/lib/api";
import { streamQuery } from "@/lib/streaming";
import { toast } from "sonner";

interface LocalMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceInfo[];
  isStreaming?: boolean;
  isFallback?: boolean;
}

export default function ChatPage() {
  const { id: convId } = useParams<{ id: string }>();
  const { token, user } = useAuthStore();

  const [messages, setMessages] = useState<LocalMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [loading, setLoading] = useState(true);

  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const streamingIdRef = useRef<string | null>(null);

  const userInitials = user?.email?.slice(0, 2).toUpperCase() ?? "U";

  // ── Load message history ──────────────────────────────────────────────────
  useEffect(() => {
    if (!token || !convId) return;
    setLoading(true);
    setMessages([]);

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
      })
      .catch(() => toast.error("Failed to load messages"))
      .finally(() => setLoading(false));
  }, [convId, token]);

  // ── Auto-scroll to bottom ─────────────────────────────────────────────────
  useEffect(() => {
    // Instant scroll, not smooth — per UX_MASTERPLAN Section 11
    bottomRef.current?.scrollIntoView({ behavior: "instant" });
  }, [messages]);

  // ── Stream handler ────────────────────────────────────────────────────────
  const handleSubmit = useCallback(
    async (question: string) => {
      if (!token || isStreaming) return;

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

      await streamQuery(
        token,
        { question, conversation_id: convId },
        {
          onSources: (sources) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId ? { ...m, sources } : m
              )
            );
          },
          onToken: (tok) => {
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
          onError: (msg) => {
            toast.error(msg);
            setMessages((prev) => prev.filter((m) => m.id !== assistantMsgId));
            setIsStreaming(false);
          },
        },
        abortRef.current.signal
      );
    },
    [token, convId, isStreaming]
  );

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

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Messages — scrollable */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin">
        {loading ? (
          // Skeleton loading per UX_MASTERPLAN Section 11
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
                <ChatMessage
                  key={msg.id}
                  role={msg.role}
                  content={msg.content}
                  sources={msg.sources}
                  isStreaming={msg.isStreaming}
                  isFallback={msg.isFallback}
                  userInitials={userInitials}
                />
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
      />
    </div>
  );
}
