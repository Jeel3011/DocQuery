"use client";

import { motion } from "framer-motion";
import { useAuthStore } from "@/stores/auth.store";
import { useRouter } from "next/navigation";
import { createConversation, listDocuments, DocumentResponse } from "@/lib/api";
import { useCollectionStore } from "@/stores/collection.store";
import { toast } from "sonner";
import { useState, useEffect } from "react";
import { ChatInput } from "@/components/chat/ChatInput";
import { FolderOpen, ArrowRight, FileText, Brain, Zap } from "lucide-react";
import { useProfileStore } from "@/stores/profile.store";
import { NamePrompt } from "@/components/app/NamePrompt";
import { getMe } from "@/lib/api";

const ease = [0.16, 1, 0.3, 1] as const;

export default function ChatIndexPage() {
  const { user, token } = useAuthStore();
  const router = useRouter();
  const { activeCollectionId } = useCollectionStore();
  const { preferredName, hydrateFromServer } = useProfileStore();
  const [creating, setCreating] = useState(false);
  const [docs, setDocs] = useState<DocumentResponse[]>([]);
  const fallbackName = user?.email?.split("@")[0] ?? "there";
  const name = preferredName ?? fallbackName;

  useEffect(() => {
    if (!token) return;
    listDocuments(token).then(setDocs).catch(() => {});
    // Reconcile the preferred name with the server (source of truth).
    getMe(token).then((m) => hydrateFromServer(m.preferred_name)).catch(() => {});
  }, [token, hydrateFromServer]);

  const firstDoc = docs.find((d) => d.status === "ready");
  const suggestions = firstDoc
    ? [
        { q: `Summarize "${firstDoc.filename}"`, icon: FileText },
        { q: "What are the key findings across my documents?", icon: Brain },
        { q: "List all tables and figures", icon: FileText },
        { q: "Compare the main topics across my documents", icon: Brain },
      ]
    : [
        { q: "What can DocQuery help me with?", icon: Brain },
        { q: "How do I upload documents?", icon: FileText },
        { q: "What file types are supported?", icon: FileText },
        { q: "How does document Q&A work?", icon: Zap },
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
    <div className="flex flex-col h-full" style={{ background: "var(--canvas)" }}>
      {/* First-run: ask what the assistant should call the user */}
      <NamePrompt fallback={fallbackName} />

      {/* Dot grid background */}
      <div className="absolute inset-0 dot-grid opacity-[0.12] pointer-events-none" />

      <div className="flex-1 flex items-center justify-center px-6 overflow-hidden relative">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.32, ease }}
          className="w-full max-w-2xl"
        >
          {/* Logo mark */}
          <div className="flex flex-col items-center text-center mb-10">
            <div
              className="w-14 h-14 rounded-2xl flex items-center justify-center mb-5 shadow-md"
              style={{
                background: "var(--ink)",
                boxShadow: "var(--shadow-lg)",
              }}
            >
              <span
                className="text-xl font-semibold"
                style={{ color: "var(--on-ink)", fontFamily: "Fraunces, Georgia, serif", letterSpacing: "-0.02em" }}
              >
                D
              </span>
            </div>

            <h1
              className="mb-2"
              style={{
                fontFamily: "Fraunces, Georgia, serif",
                fontSize: "28px",
                fontWeight: 400,
                letterSpacing: "-0.025em",
                color: "var(--ink)",
                lineHeight: 1.1,
              }}
            >
              Good day, {name}.
            </h1>
            <p className="text-[15px]" style={{ color: "var(--ink-3)" }}>
              What would you like to know about your documents?
            </p>

            {/* Active collection badge */}
            {activeCollectionId && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 mt-4 rounded-full text-[11px] font-medium"
                style={{
                  background: "var(--accent-soft)",
                  border: "1px solid var(--line-2)",
                  color: "var(--accent-taupe)",
                }}
              >
                <FolderOpen size={11} />
                Querying active collection
              </motion.div>
            )}
          </div>

          {/* Suggestion cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
            {suggestions.map(({ q, icon: Icon }, i) => (
              <motion.button
                key={i}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.08 + i * 0.05, ease }}
                whileHover={{ y: -2, transition: { duration: 0.12 } }}
                whileTap={{ scale: 0.98 }}
                onClick={() => ask(q)}
                disabled={creating}
                className="group text-left p-4 rounded-2xl flex items-start gap-3 transition-all disabled:opacity-50"
                style={{
                  background: "var(--surface)",
                  border: "1px solid var(--line)",
                  boxShadow: "var(--shadow-sm)",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.boxShadow = "var(--shadow-md)";
                  (e.currentTarget as HTMLElement).style.borderColor = "var(--line-2)";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.boxShadow = "var(--shadow-sm)";
                  (e.currentTarget as HTMLElement).style.borderColor = "var(--line)";
                }}
              >
                <div
                  className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5"
                  style={{
                    background: "var(--surface-3)",
                    border: "1px solid var(--line)",
                    color: "var(--ink-3)",
                  }}
                >
                  <Icon size={13} strokeWidth={1.6} />
                </div>
                <div className="flex-1 min-w-0">
                  <p
                    className="text-[13px] font-medium leading-snug"
                    style={{ color: "var(--ink-2)" }}
                  >
                    {q}
                  </p>
                </div>
                <ArrowRight
                  size={14}
                  className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity mt-1"
                  style={{ color: "var(--ink-3)" }}
                />
              </motion.button>
            ))}
          </div>

          {!firstDoc && (
            <p
              className="text-center text-[12px]"
              style={{ color: "var(--ink-3)" }}
            >
              ← Upload documents in the sidebar to get started
            </p>
          )}
        </motion.div>
      </div>

      <ChatInput onSubmit={ask} isStreaming={creating} placeholder="Ask a question about your documents…" />
    </div>
  );
}
