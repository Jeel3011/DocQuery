"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { TrendingUp, Scale } from "lucide-react";

const AUDIENCES = [
  {
    id: "finance",
    label: "Finance",
    icon: TrendingUp,
    headline: "Analyse contracts & filings in seconds",
    query: "Summarise the key financial covenants in the credit agreement and flag any material adverse change clauses.",
    answer: `The credit agreement contains **three key covenants** [1]:

- **Net Leverage Ratio** must remain below 4.5× at each quarter-end (§12.3)
- **Interest Coverage Ratio** must exceed 2.0× on a trailing 12-month basis (§12.4)
- **Minimum Liquidity** of ₹50 Cr cash or equivalents at all times (§12.5)

**Material Adverse Change clause** (§18.1): triggered if any event causes a ≥20% decline in consolidated EBITDA vs prior year or results in litigation exceeding ₹25 Cr in aggregate exposure [2].`,
    sources: ["Credit Agreement v4.2.pdf · §12.3", "Credit Agreement v4.2.pdf · §18.1"],
  },
  {
    id: "legal",
    label: "Legal",
    icon: Scale,
    headline: "Review agreements & flag risk instantly",
    query: "What are the indemnification obligations and liability caps across these MSAs?",
    answer: `Across the three MSAs reviewed, indemnification obligations vary significantly [1]:

- **MSA-A** (Acme Corp): mutual indemnification, liability capped at **1× annual fees** paid in prior 12 months (Clause 9.2)
- **MSA-B** (Beta Ltd): one-sided indemnification favoring vendor; cap set at **₹10 Lakhs** regardless of contract value — **flag for negotiation** [2]
- **MSA-C** (Gamma Inc): no aggregate cap for IP infringement claims — **unlimited exposure risk** (Clause 11.4)

**Recommendation:** Standardise to mutual indemnification with a 2× annual fee cap.`,
    sources: ["MSA-Acme-v3.pdf · Clause 9.2", "MSA-Beta-v1.pdf · Clause 7.1", "MSA-Gamma-v2.pdf · Clause 11.4"],
  },
];

export function AudienceTabs() {
  const [active, setActive] = useState("finance");
  const current = AUDIENCES.find((a) => a.id === active)!;

  return (
    <section className="py-20 px-6 lg:px-12 max-w-5xl mx-auto relative z-10">
      <div className="text-center mb-10">
        <h2 className="text-3xl font-bold text-[var(--text-primary)] mb-3 tracking-tight">
          Built for precision work
        </h2>
        <p className="text-[var(--text-secondary)] max-w-lg mx-auto">
          Finance and legal professionals need answers they can cite and defend. DocQuery delivers exactly that.
        </p>
      </div>

      {/* Tab selector */}
      <div className="flex justify-center mb-8">
        <div className="inline-flex rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] p-1 gap-1">
          {AUDIENCES.map((a) => {
            const Icon = a.icon;
            return (
              <button
                key={a.id}
                onClick={() => setActive(a.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  active === a.id
                    ? "bg-[var(--accent)] text-white shadow-sm"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
                }`}
              >
                <Icon size={14} />
                {a.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Demo panel */}
      <AnimatePresence mode="wait">
        <motion.div
          key={active}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
          className="card shadow-md overflow-hidden"
        >
          <div className="px-6 py-4 border-b border-[var(--border)] bg-[var(--bg-base)]">
            <p className="text-[10px] font-medium text-[var(--text-muted)] uppercase tracking-widest mb-2">Sample query</p>
            <p className="text-sm text-[var(--text-secondary)] italic">&ldquo;{current.query}&rdquo;</p>
          </div>
          <div className="p-6">
            <p className="text-[10px] font-medium text-[var(--text-muted)] uppercase tracking-widest mb-3">DocQuery answer</p>
            <div className="text-sm text-[var(--text-primary)] leading-relaxed space-y-1 whitespace-pre-wrap">
              {current.answer.split("\n").map((line, i) => (
                <p key={i} className={line.startsWith("**") ? "font-semibold" : ""}>
                  {line.replace(/\*\*/g, "").replace(/\[\d+\]/g, "").trim()}
                </p>
              ))}
            </div>
            <div className="mt-4 pt-3 border-t border-dashed border-[var(--border-dotted)] flex flex-wrap gap-2">
              {current.sources.map((s, i) => (
                <span key={i} className="inline-flex items-center gap-1.5 px-2 py-1 text-[10px] text-[var(--text-secondary)] bg-[var(--bg-hover)] border border-[var(--border)] rounded-lg">
                  <span className="w-3.5 h-3.5 rounded bg-[var(--accent)] text-white text-[8px] font-bold flex items-center justify-center">{i + 1}</span>
                  {s}
                </span>
              ))}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </section>
  );
}
