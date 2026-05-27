"use client";

// app/app/chat/page.tsx — Monochrome empty state with dot grid background

import { motion } from "framer-motion";
import { useAuthStore } from "@/stores/auth.store";
import { useRouter } from "next/navigation";
import { createConversation, listDocuments, DocumentResponse } from "@/lib/api";
import { useCollectionStore } from "@/stores/collection.store";
import { toast } from "sonner";
import { useState, useEffect } from "react";
import { ChatInput } from "@/components/chat/ChatInput";

export default function ChatIndexPage() {
  const { user, token } = useAuthStore();
  const router = useRouter();
  const { activeCollectionId } = useCollectionStore();
  const [creating, setCreating] = useState(false);
  const [docs, setDocs] = useState<DocumentResponse[]>([]);
  const name = user?.email?.split("@")[0] ?? "there";

  // Load docs to build context-aware suggestions
  useEffect(() => {
    if (!token) return;
    listDocuments(token).then(setDocs).catch(() => {});
  }, [token]);

  // Build context-aware suggestions based on what the user has uploaded
  const firstDoc = docs.find((d) => d.status === "ready");
  const suggestions = firstDoc
    ? [
        `Summarize "${firstDoc.filename}"`,
        "What are the key findings?",
        "List all tables and figures",
        "Compare the main topics across my documents",
      ]
    : [
        "What can DocQuery help me with?",
        "How do I upload documents?",
        "What file types are supported?",
        "How does document Q&A work?",
      ];

  async function ask(q: string) {
    if (!token || creating) return;
    setCreating(true);
    try {
      const c = await createConversation(token, q.slice(0, 50));
      router.push(`/app/chat/${c.id}?q=${encodeURIComponent(q)}`);
    } catch { toast.error("Failed to create conversation"); }
    finally { setCreating(false); }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 flex items-center justify-center px-6 dot-grid">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}
          className="text-center max-w-md">

          {/* Logo — black square with white D */}
          <div className="w-12 h-12 rounded-xl bg-[var(--accent)] flex items-center justify-center mx-auto mb-5 shadow-lg">
            <span className="text-lg font-bold text-white">D</span>
          </div>

          <h1 className="text-xl font-semibold text-[var(--text-primary)] mb-1">DocQuery</h1>
          <p className="text-[var(--text-secondary)] text-sm mb-8">Ask questions about your documents, {name}.</p>

          {/* Active collection badge */}
          {activeCollectionId && (
            <div className="inline-flex items-center gap-1.5 px-3 py-1 mb-4 rounded-full bg-[var(--bg-hover)] text-[10px] text-[var(--accent)] font-medium">
              ◆ Querying active collection
            </div>
          )}

          {/* Suggestion chips — dotted border cards */}
          <div className="grid grid-cols-2 gap-2 mb-6">
            {suggestions.map((s, i) => (
              <motion.button key={i}
                initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.05 + i * 0.04 }}
                whileHover={{ y: -2 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => ask(s)}
                disabled={creating}
                className="card-dotted text-left p-3.5 text-xs text-[var(--text-secondary)]
                  hover:text-[var(--text-primary)] transition-all disabled:opacity-50">
                {s}
              </motion.button>
            ))}
          </div>

          {!firstDoc && (
            <p className="text-[var(--text-muted)] text-xs">← Upload documents in the sidebar to start</p>
          )}
        </motion.div>
      </div>

      <ChatInput onSubmit={ask} isStreaming={creating} placeholder="Ask a question about your documents…" />
    </div>
  );
}
