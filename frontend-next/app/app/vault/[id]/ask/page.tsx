"use client";

// /app/vault/[id]/ask — Ask landing, re-homed under the vault (G2 Step D).
//
// A composer landing scoped to the vault. Most Ask sessions start from the vault
// workspace (Step C) HERO composer, which routes straight into a conversation; this
// page is the addressable /ask entry (deep-link, "New question" affordance). Submitting
// creates a conversation and routes to /app/vault/[id]/ask/[cid], which carries the
// real streaming timeline.
//
// Scope (§9 risk #1): the vault [id] segment is authoritative — we read it from the
// route, never the store. VaultScopeSync mirrors it into the store for the composer chip.

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Search, GitCompare, BarChart3, TrendingUp, Scale, Coins, ArrowUpRight } from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/auth.store";
import { createConversation, listCollections, getCollectionDocuments } from "@/lib/api";
import { isFinanceVault } from "@/lib/docType";
import { ChatInput } from "@/components/chat/ChatInput";

const ease = [0.23, 1, 0.32, 1] as const;

// Doc-type-aware suggestions: DocQuery is law-first, so LEGAL is the default — shown for a
// legal vault, a mixed vault, or any vault whose docs aren't classified yet. A vault that
// leans FINANCIAL (dominant doc_type = financial_filing) gets the finance set instead.
const LEGAL_SUGGESTIONS = [
  { icon: Search, title: "Key terms", q: "What are the governing law, term, and termination notice period in this contract? Quote each clause." },
  { icon: GitCompare, title: "Indemnity & liability", q: "Is there an indemnity cap and a limitation of liability? Quote the exact caps, or say if liability is uncapped." },
  { icon: Scale, title: "Risk flags", q: "Flag any non-standard or unusual clauses in this contract that a reviewer should look at." },
  { icon: GitCompare, title: "Dispute resolution", q: "How are disputes resolved — arbitration or courts? What is the seat/venue and which rules apply?" },
];

const FINANCE_SUGGESTIONS = [
  { icon: TrendingUp, title: "Revenue & growth", q: "What was total revenue in the latest year and how did it change year-over-year? Show the figures with their source." },
  { icon: BarChart3, title: "Profitability", q: "What were operating income and net income, and what are the operating and net margins? Compute each from the source tables." },
  { icon: Coins, title: "Balance-sheet strength", q: "What are total assets, total liabilities, and cash & equivalents at the latest period end? Cite each figure." },
  { icon: GitCompare, title: "Compare periods", q: "Compare the key metrics (revenue, net income, total assets) across the years available, and note the trend." },
];

export default function VaultAskLandingPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const { token } = useAuthStore();
  const vaultId = params.id;

  const [vaultName, setVaultName] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  // Suggestion set keyed off the vault's dominant doc-type (legal default, law-first).
  const [suggestions, setSuggestions] = useState(LEGAL_SUGGESTIONS);

  useEffect(() => {
    if (!token || !vaultId) return;
    let cancelled = false;
    listCollections(token)
      .then((cols) => { if (!cancelled) setVaultName(cols.find((c) => c.id === vaultId)?.name ?? null); })
      .catch(() => { if (!cancelled) setVaultName(null); });
    // Pick the finance set only when the vault leans financial; otherwise keep legal.
    getCollectionDocuments(token, vaultId)
      .then((docs) => { if (!cancelled && isFinanceVault(docs)) setSuggestions(FINANCE_SUGGESTIONS); })
      .catch(() => { /* keep the legal default on any error */ });
    return () => { cancelled = true; };
  }, [token, vaultId]);

  async function ask(q: string) {
    if (!token || creating || !q.trim()) return;
    setCreating(true);
    try {
      const c = await createConversation(token, q.slice(0, 50));
      // Route into the conversation scoped to THIS vault; ?q= auto-submits on mount.
      router.push(`/app/vault/${vaultId}/ask/${c.id}?q=${encodeURIComponent(q)}`);
    } catch {
      toast.error("Failed to start conversation");
      setCreating(false);
    }
  }

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin relative" style={{ background: "var(--canvas)" }}>

      <div className="relative h-full flex flex-col items-center justify-center w-full max-w-3xl mx-auto px-4 md:px-8 py-10 text-left">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease }}
          className="w-full"
        >
          <h2 className="text-[30px] md:text-[36px] leading-[1.08] font-bold tracking-[-0.035em]" style={{ color: "var(--ink)" }}>
            What would you like to know?
          </h2>
          <p className="text-[14px] mt-3 max-w-lg leading-relaxed" style={{ color: "var(--text-muted)" }}>
            {vaultName
              ? <>Scoped to <span className="font-medium" style={{ color: "var(--text-secondary)" }}>{vaultName}</span> · pick a prompt or ask your own. Every clause and figure is grounded in a cited source — or honestly withheld.</>
              : "Pick a prompt or ask your own. Every clause and figure is grounded in a cited source — or honestly withheld."}
          </p>

          <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
            {suggestions.map((s, i) => (
              <motion.button
                key={s.title}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.06 + i * 0.05, ease }}
                onClick={() => ask(s.q)}
                disabled={creating}
                className="group relative text-left rounded-2xl p-3.5 transition-all duration-200 hover:-translate-y-0.5 overflow-hidden disabled:opacity-50"
                style={{
                  background: "linear-gradient(180deg, #FFFFFF, #F6F6F6)",
                  border: "1px solid rgba(0,0,0,0.10)",
                  boxShadow: "0 6px 20px -10px rgba(0,0,0,0.16), inset 0 1px 0 rgba(255,255,255,0.95)",
                }}
              >
                <div className="flex items-center gap-2.5 mb-1.5">
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center text-[var(--text-secondary)] group-hover:text-white group-hover:bg-[var(--accent)] transition-colors"
                    style={{ background: "var(--surface-3)", border: "1px solid var(--line)" }}
                  >
                    <s.icon size={15} />
                  </div>
                  <span className="text-[14px] font-semibold tracking-[-0.01em]" style={{ color: "var(--text-primary)" }}>{s.title}</span>
                </div>
                <p className="text-[12.5px] leading-snug group-hover:text-[var(--text-secondary)] transition-colors pr-5" style={{ color: "var(--text-muted)" }}>
                  {s.q}
                </p>
                <ArrowUpRight
                  size={15}
                  className="absolute bottom-3 right-3 text-[var(--text-muted)] opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200"
                />
              </motion.button>
            ))}
          </div>
        </motion.div>
      </div>

      <div className="sticky bottom-0">
        <ChatInput
          onSubmit={ask}
          isStreaming={creating}
          placeholder="Ask about a clause, term, or risk across this vault…"
          vaultName={vaultName}
        />
      </div>
    </div>
  );
}
