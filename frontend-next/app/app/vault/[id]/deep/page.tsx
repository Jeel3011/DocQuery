"use client";

// /app/vault/[id]/deep — Deep Analysis (G2 Step G).
//
// Scope-discipline note (plan §G, §0): Deep Analysis is a LATER phase (G5). G2 only
// lays the route + the mode plumbing so the three vault actions (Ask · Review · Deep
// Analysis) all have a home and the workspace action row can route here. The real
// long-running analysis + artifact authoring (the ArtifactPanel becomes the output
// surface) ships in G5/G6 — we deliberately do NOT build it now (no model-burn, plan
// §6). This is an honest "Coming in G5" placeholder that frames what the mode will do,
// styled in the same ArtifactPanel-document shell so it reads as a real workspace tab.
//
// Scope (§9 risk #1): the vault [id] is authoritative — read from the route, never the
// store. VaultScopeSync mirrors it into the store for the top-bar chip.

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { ArrowLeft, Telescope, FileText, GitCompare, ListChecks, Sparkles } from "lucide-react";
import { useAuthStore } from "@/stores/auth.store";
import { listCollections } from "@/lib/api";

const ease = [0.23, 1, 0.32, 1] as const;

// What Deep Analysis will do once G5 lands — shown as a preview so the placeholder
// communicates intent, not just "coming soon".
const PREVIEW = [
  {
    icon: ListChecks,
    title: "Multi-step reasoning over the whole vault",
    body: "Decompose a complex question into sub-goals, run them across every document, and assemble a reasoned, cited answer — not a single retrieval pass.",
  },
  {
    icon: GitCompare,
    title: "Cross-document synthesis",
    body: "Reconcile terms, figures, and obligations across the set; surface where documents agree, conflict, or leave a gap — each claim traced to its source.",
  },
  {
    icon: FileText,
    title: "Authored artifacts",
    body: "Produce a structured memo or comparison you can read, export, and trust — every line grounded in a cited clause or cell, or honestly withheld.",
  },
];

export default function VaultDeepAnalysisPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const { token } = useAuthStore();
  const vaultId = params.id;

  const [vaultName, setVaultName] = useState<string | null>(null);

  useEffect(() => {
    if (!token || !vaultId) return;
    let cancelled = false;
    listCollections(token)
      .then((cols) => { if (!cancelled) setVaultName(cols.find((c) => c.id === vaultId)?.name ?? null); })
      .catch(() => { if (!cancelled) setVaultName(null); });
    return () => { cancelled = true; };
  }, [token, vaultId]);

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin relative" style={{ background: "var(--canvas)" }}>

      <div className="relative max-w-3xl mx-auto px-6 py-8">
        {/* Back to the vault workspace */}
        <button
          onClick={() => router.push(`/app/vault/${encodeURIComponent(vaultId)}`)}
          className="inline-flex items-center gap-1.5 text-[12px] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors mb-6"
        >
          <ArrowLeft size={13} /> Back to vault
        </button>

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease }}
          className="flex items-start gap-3 mb-3"
        >
          <span
            className="w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0"
            style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
          >
            <Telescope size={19} />
          </span>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1
                style={{ fontFamily: "Fraunces, Georgia, serif", fontSize: 27, fontWeight: 500, letterSpacing: "-0.025em", color: "var(--ink)" }}
              >
                Deep Analysis
              </h1>
              <span
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wide"
                style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--text-muted)" }}
              >
                <Sparkles size={10} /> Coming in G5
              </span>
            </div>
            <p className="text-[12px] text-[var(--text-muted)] mt-0.5">
              {vaultName
                ? <>Long-form, multi-step reasoning over <span className="font-medium" style={{ color: "var(--text-secondary)" }}>{vaultName}</span>.</>
                : "Long-form, multi-step reasoning over this vault."}
            </p>
          </div>
        </motion.div>

        {/* Document-shell card (the ArtifactPanel surface this mode will author into) */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.08, ease }}
          className="rounded-2xl overflow-hidden mt-5"
          style={{ background: "var(--surface)", border: "1px solid var(--line)" }}
        >
          <div
            className="px-5 py-3 border-b flex items-center gap-2"
            style={{ borderColor: "var(--line)" }}
          >
            <FileText size={13} className="text-[var(--text-muted)]" />
            <span className="text-[12px] font-medium text-[var(--text-secondary)]">Analysis workspace</span>
          </div>

          <div className="px-5 py-6 space-y-5">
            {PREVIEW.map((p, i) => (
              <motion.div
                key={p.title}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.16 + i * 0.06, ease }}
                className="flex items-start gap-3"
              >
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-[var(--text-secondary)]"
                  style={{ background: "var(--surface-3)", border: "1px solid var(--line)" }}
                >
                  <p.icon size={15} />
                </div>
                <div className="min-w-0">
                  <p className="text-[13.5px] font-semibold tracking-[-0.01em]" style={{ color: "var(--text-primary)" }}>
                    {p.title}
                  </p>
                  <p className="text-[12.5px] leading-relaxed mt-0.5" style={{ color: "var(--text-muted)" }}>
                    {p.body}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>

          <div
            className="px-5 py-3.5 border-t flex items-center justify-between"
            style={{ borderColor: "var(--line)", background: "var(--surface-2)" }}
          >
            <span className="text-[11.5px] text-[var(--text-muted)]">
              Ask and Review are live now — start there.
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => router.push(`/app/vault/${encodeURIComponent(vaultId)}/ask`)}
                className="px-3 py-1.5 rounded-lg text-[12px] font-medium card hover:shadow-[var(--shadow-md)] transition-shadow"
              >
                Ask a question
              </button>
              <button
                onClick={() => router.push(`/app/vault/${encodeURIComponent(vaultId)}/review`)}
                className="px-3 py-1.5 rounded-lg text-[12px] font-medium card hover:shadow-[var(--shadow-md)] transition-shadow"
              >
                Review table
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
