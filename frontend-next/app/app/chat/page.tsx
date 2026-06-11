"use client";

import { motion } from "framer-motion";
import { useAuthStore } from "@/stores/auth.store";
import { useRouter } from "next/navigation";
import { createConversation, listDocuments, DocumentResponse, listCollections } from "@/lib/api";
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
  const [vaultName, setVaultName] = useState<string | null>(null);
  // Mode chosen on the landing — threaded into the new conversation so picking "Agent"
  // here actually starts the conversation in agent mode (was: always fast).
  const [agentCoreMode, setAgentCoreMode] = useState(false);
  const [brainMode, setBrainMode] = useState(false);
  const [agenticMode, setAgenticMode] = useState(false);
  // First name: leading alphabetic run so "jeel15thummar" → "jeel" (not "jeel15thummar").
  const fallbackName = (user?.email?.split("@")[0].match(/^[a-zA-Z]+/) ?? ["there"])[0];
  const name = preferredName ?? fallbackName;

  useEffect(() => {
    if (!token) return;
    listDocuments(token).then(setDocs).catch(() => {});
    // Reconcile the preferred name with the server (source of truth).
    getMe(token).then((m) => hydrateFromServer(m.preferred_name)).catch(() => {});
  }, [token, hydrateFromServer]);

  // Resolve the active collection's name for the composer's vault chip.
  useEffect(() => {
    if (!token || !activeCollectionId) { setVaultName(null); return; }
    let cancelled = false;
    listCollections(token)
      .then((cols) => { if (!cancelled) setVaultName(cols.find((c) => c.id === activeCollectionId)?.name ?? null); })
      .catch(() => { if (!cancelled) setVaultName(null); });
    return () => { cancelled = true; };
  }, [token, activeCollectionId]);

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
      // Carry the chosen mode into the conversation so it opens in the right mode.
      const mode = agentCoreMode ? "agent" : brainMode ? "brain" : agenticMode ? "deep" : "";
      const qp = `q=${encodeURIComponent(q)}${mode ? `&mode=${mode}` : ""}`;
      router.push(`/app/chat/${c.id}?${qp}`);
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
          {/* Logo mark — a living grey orb (breathing float + glow) */}
          <div className="flex flex-col items-center text-center mb-9">
            <div className="relative mb-6">
              <motion.div
                aria-hidden
                className="absolute rounded-full"
                style={{ inset: -12, background: "radial-gradient(circle, rgba(0,0,0,0.06), transparent 70%)" }}
                animate={{ scale: [1, 1.12, 1], opacity: [0.5, 0.85, 0.5] }}
                transition={{ duration: 4.5, repeat: Infinity, ease: "easeInOut" }}
              />
              <motion.div
                className="w-[60px] h-[60px] rounded-full relative overflow-hidden"
                style={{
                  background: "radial-gradient(circle at 33% 25%, #FFFFFF 0%, #EAEAEA 38%, #B8B8B8 70%, #8E8E8E 100%)",
                  boxShadow: "0 16px 34px -10px rgba(0,0,0,0.38), 0 3px 8px -3px rgba(0,0,0,0.18), inset 0 2px 6px rgba(255,255,255,0.92), inset 0 -10px 18px -6px rgba(0,0,0,0.24)",
                }}
                animate={{ y: [0, -6, 0], scale: [1, 1.03, 1] }}
                transition={{ duration: 5.5, repeat: Infinity, ease: "easeInOut" }}
              >
                <span className="absolute rounded-full" style={{ inset: "14% 42% 54% 20%", background: "radial-gradient(circle, rgba(255,255,255,0.95), transparent 70%)", filter: "blur(2px)" }} />
                <motion.span
                  className="absolute inset-0 rounded-full"
                  style={{ background: "linear-gradient(120deg, transparent 35%, rgba(255,255,255,0.45) 50%, transparent 65%)" }}
                  animate={{ x: ["-60%", "60%"], opacity: [0, 0.7, 0] }}
                  transition={{ duration: 6, repeat: Infinity, ease: "easeInOut", repeatDelay: 1.5 }}
                />
              </motion.div>
            </div>

            <h1
              className="mb-2.5"
              style={{
                fontFamily: "Fraunces, Georgia, serif",
                fontSize: "40px",
                fontWeight: 500,
                letterSpacing: "-0.03em",
                color: "var(--ink)",
                lineHeight: 1.05,
              }}
            >
              Good day, {name}.
            </h1>
            <p className="text-[16px] max-w-md" style={{ color: "var(--ink-3)" }}>
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

      <ChatInput
        onSubmit={ask}
        isStreaming={creating}
        placeholder="Ask a question about your documents…"
        vaultName={vaultName}
        agentCoreMode={agentCoreMode}
        onToggleAgentCore={() => { setAgentCoreMode((v) => !v); setBrainMode(false); setAgenticMode(false); }}
        brainMode={brainMode}
        onToggleBrain={() => { setBrainMode((v) => !v); setAgentCoreMode(false); setAgenticMode(false); }}
        agenticMode={agenticMode}
        onToggleAgentic={() => { setAgenticMode((v) => !v); setAgentCoreMode(false); setBrainMode(false); }}
      />
    </div>
  );
}
