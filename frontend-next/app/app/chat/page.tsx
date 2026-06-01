"use client";

import { motion } from "framer-motion";
import { useAuthStore } from "@/stores/auth.store";
import { useRouter } from "next/navigation";
import { createConversation, listDocuments, DocumentResponse } from "@/lib/api";
import { useCollectionStore } from "@/stores/collection.store";
import { toast } from "sonner";
import { useState, useEffect } from "react";
import { ChatInput } from "@/components/chat/ChatInput";
import { FolderOpen } from "lucide-react";

export default function ChatIndexPage() {
  const { user, token } = useAuthStore();
  const router = useRouter();
  const { activeCollectionId } = useCollectionStore();
  const [creating, setCreating] = useState(false);
  const [docs, setDocs] = useState<DocumentResponse[]>([]);
  const name = user?.email?.split("@")[0] ?? "there";

  useEffect(() => {
    if (!token) return;
    listDocuments(token).then(setDocs).catch(() => {});
  }, [token]);

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
      <div className="flex-1 flex items-center justify-center px-6 dot-grid overflow-hidden">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
          className="text-center max-w-md w-full"
        >
          {/* Wordmark */}
          <div className="w-11 h-11 rounded-xl bg-[var(--accent)] flex items-center justify-center mx-auto mb-5 shadow-md">
            <span className="text-base font-bold text-white tracking-tight">D</span>
          </div>

          <h1 className="text-xl font-semibold text-[var(--text-primary)] mb-1 tracking-tight">
            DocQuery
          </h1>
          <p className="text-[var(--text-muted)] text-sm mb-6">
            Ask questions about your documents, {name}.
          </p>

          {/* Active collection badge */}
          {activeCollectionId && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="inline-flex items-center gap-1.5 px-3 py-1 mb-5 rounded-full bg-[var(--bg-hover)] text-[10px] text-[var(--accent)] font-medium border border-[var(--border)]"
            >
              <FolderOpen size={10} />
              Querying active collection
            </motion.div>
          )}

          {/* Suggestion chips */}
          <div className="grid grid-cols-2 gap-2 mb-5">
            {suggestions.map((s, i) => (
              <motion.button
                key={i}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.06 + i * 0.05, ease: [0.16, 1, 0.3, 1] }}
                whileHover={{ y: -2, transition: { duration: 0.12 } }}
                whileTap={{ scale: 0.97 }}
                onClick={() => ask(s)}
                disabled={creating}
                className="card-dotted text-left p-3.5 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
              >
                {s}
              </motion.button>
            ))}
          </div>

          {!firstDoc && (
            <p className="text-[var(--text-muted)] text-xs">
              ← Upload documents in the sidebar to get started
            </p>
          )}
        </motion.div>
      </div>

      <ChatInput onSubmit={ask} isStreaming={creating} placeholder="Ask a question about your documents…" />
    </div>
  );
}
