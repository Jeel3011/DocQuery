"use client";

// /app/vault/[id]/deep — Deep Analysis landing (G5). The third Vault mode: a cited,
// multi-section report across the WHOLE vault, authored by the SAME agent engine through
// the SAME non-bypassable citation gate — every section's every claim cites a real
// span/cell or is VISIBLY withheld.
//
// This landing mirrors the Ask landing's shape (suggested prompts → create a conversation
// → route into the streaming report view at /deep/[cid]?q=…). It deliberately reuses the
// SAME composer + conversation engine; deep only sets analysisMode="deep" downstream
// (see deep/[cid]/page.tsx), which makes the backend use the Deep Analysis prompt +
// per-section gate. No new design system — composition of built primitives (plan §3/§6).
//
// Scope (§9 risk #1): the vault [id] is authoritative — read from the route, never the
// store. VaultScopeSync mirrors it into the store for the top-bar chip.

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Telescope, ShieldCheck, GitCompare, ListChecks, TrendingUp, BarChart3, Coins, ArrowUpRight } from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/auth.store";
import { createConversation, listCollections, getCollectionDocuments } from "@/lib/api";
import { isFinanceVault } from "@/lib/docType";
import { ChatInput } from "@/components/chat/ChatInput";

const ease = [0.23, 1, 0.32, 1] as const;

// Deep-analysis starter prompts — whole-vault, multi-section reports. Each asks for a
// STRUCTURED, cited report, which the deep prompt turns into `## sections` the per-section
// gate then binds. Law-first by default; a finance-leaning vault leads with finance reports.
const LEGAL_SUGGESTIONS = [
  {
    icon: ListChecks,
    title: "Full contract review report",
    q: "Produce a comprehensive review of every contract in this vault. Cover parties & term, commercial terms, liability & indemnity, termination, and any red-flag clauses — with each finding quoted and cited.",
  },
  {
    icon: GitCompare,
    title: "Cross-document synthesis",
    q: "Analyze the whole vault and report where the documents agree, conflict, or leave gaps on the key obligations and terms. Cite each document for every point.",
  },
  {
    icon: ShieldCheck,
    title: "Risk & red-flag memo",
    q: "Write a risk memo across this vault: identify every non-standard, unusual, or high-risk clause a reviewer should escalate, grouped by theme, each traced to its source clause.",
  },
  {
    icon: TrendingUp,
    title: "Financial position report",
    q: "Produce a multi-section report on the financial position across the filings in this vault — revenue & growth, profitability, and balance-sheet strength — with every figure computed from the source tables and cited.",
  },
];

const FINANCE_SUGGESTIONS = [
  {
    icon: BarChart3,
    title: "Full financial review",
    q: "Produce a comprehensive financial review across the filings in this vault — revenue & growth, profitability & margins, balance-sheet strength, and cash flow — with every figure computed from the source tables and cited.",
  },
  {
    icon: GitCompare,
    title: "Cross-filing comparison",
    q: "Compare the filings in this vault across the key metrics (revenue, net income, total assets, margins) for every period available, and report the trends. Cite each figure.",
  },
  {
    icon: Coins,
    title: "Liquidity & leverage memo",
    q: "Write a memo on liquidity and leverage across these filings: cash & equivalents, total debt, current ratio, and debt-to-equity where derivable — each figure traced to a source cell or honestly withheld.",
  },
  {
    icon: ShieldCheck,
    title: "Risk-factor synthesis",
    q: "Synthesize the risk factors disclosed across the filings in this vault, grouped by theme, and note where issuers diverge. Cite each source.",
  },
];

export default function VaultDeepAnalysisPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const { token } = useAuthStore();
  const vaultId = params.id;

  const [vaultName, setVaultName] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  // Report-prompt set keyed off the vault's dominant doc-type (legal default, law-first).
  const [suggestions, setSuggestions] = useState(LEGAL_SUGGESTIONS);

  useEffect(() => {
    if (!token || !vaultId) return;
    let cancelled = false;
    listCollections(token)
      .then((cols) => { if (!cancelled) setVaultName(cols.find((c) => c.id === vaultId)?.name ?? null); })
      .catch(() => { if (!cancelled) setVaultName(null); });
    getCollectionDocuments(token, vaultId)
      .then((docs) => { if (!cancelled && isFinanceVault(docs)) setSuggestions(FINANCE_SUGGESTIONS); })
      .catch(() => { /* keep the legal default on any error */ });
    return () => { cancelled = true; };
  }, [token, vaultId]);

  async function analyze(q: string) {
    if (!token || creating || !q.trim()) return;
    setCreating(true);
    try {
      const c = await createConversation(token, q.slice(0, 50));
      // Route into the deep report view scoped to THIS vault; ?q= auto-submits on mount.
      router.push(`/app/vault/${vaultId}/deep/${c.id}?q=${encodeURIComponent(q)}`);
    } catch {
      toast.error("Failed to start analysis");
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
          <div className="flex items-center gap-2.5 mb-3">
            <span
              className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
            >
              <Telescope size={17} />
            </span>
            <span className="text-[11px] font-semibold uppercase tracking-[0.12em]" style={{ color: "var(--text-muted)" }}>
              Deep Analysis
            </span>
          </div>

          <h2 className="text-[30px] md:text-[36px] leading-[1.08] font-bold tracking-[-0.035em]" style={{ color: "var(--ink)" }}>
            A cited report across the vault.
          </h2>
          <p className="text-[14px] mt-3 max-w-xl leading-relaxed" style={{ color: "var(--text-muted)" }}>
            {vaultName
              ? <>The agent surveys every document in <span className="font-medium" style={{ color: "var(--text-secondary)" }}>{vaultName}</span>, drills into the cells and clauses it cites, and assembles a multi-section report. Every claim is grounded in a cited source — or honestly withheld, section by section.</>
              : "The agent surveys every document in this vault, drills into the cells and clauses it cites, and assembles a multi-section report. Every claim is grounded in a cited source — or honestly withheld, section by section."}
          </p>

          <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
            {suggestions.map((s, i) => (
              <motion.button
                key={s.title}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.06 + i * 0.05, ease }}
                onClick={() => analyze(s.q)}
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

          <p className="text-[11.5px] mt-4 leading-relaxed" style={{ color: "var(--text-muted)" }}>
            A deep analysis runs longer than Ask — the agent reads broadly before it writes. You&rsquo;ll watch each step live.
          </p>
        </motion.div>
      </div>

      <div className="sticky bottom-0">
        <ChatInput
          onSubmit={analyze}
          isStreaming={creating}
          placeholder="Describe the report you want across this vault…"
          vaultName={vaultName}
        />
      </div>
    </div>
  );
}
