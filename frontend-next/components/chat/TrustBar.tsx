"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Info, BookOpen } from "lucide-react";

export type ConfidenceLevel = "high" | "medium" | "low";
export type ClaimType = "fact" | "argument" | "strategy";

export interface AnswerMeta {
  confidence?: ConfidenceLevel;
  consulted?: number;
  total?: number;
  claimTypes?: ClaimType[];
}

// Fixture: replace with real SSE fields when Brain ships them
export const MOCK_ANSWER_META: AnswerMeta = {
  confidence: "high",
  consulted: 14,
  total: 34,
  claimTypes: ["fact", "argument"],
};

const confidenceConfig: Record<ConfidenceLevel, { label: string; bar: string; text: string }> = {
  high: { label: "High confidence", bar: "bg-[var(--status-ready)]", text: "text-[var(--status-ready)]" },
  medium: { label: "Medium confidence", bar: "bg-[var(--status-processing)]", text: "text-[var(--status-processing)]" },
  low: { label: "Low confidence", bar: "bg-[var(--status-failed)]", text: "text-[var(--status-failed)]" },
};

const barWidth: Record<ConfidenceLevel, string> = {
  high: "w-[85%]",
  medium: "w-[55%]",
  low: "w-[25%]",
};

const claimLabels: Record<ClaimType, { label: string; cls: string }> = {
  fact: { label: "FACT", cls: "label-fact" },
  argument: { label: "ARGUMENT", cls: "label-argument" },
  strategy: { label: "STRATEGY", cls: "label-strategy" },
};

interface TrustBarProps {
  meta: AnswerMeta;
}

export function TrustBar({ meta }: TrustBarProps) {
  const [showCoveragePopover, setShowCoveragePopover] = useState(false);

  const conf = meta.confidence ? confidenceConfig[meta.confidence] : null;

  if (!conf && !meta.consulted) return null;

  return (
    <div className="flex flex-wrap items-center gap-3 mt-2 pt-2 border-t border-dashed border-[var(--border-dotted)]">
      {/* Confidence bar */}
      {conf && (
        <div className="flex items-center gap-2">
          <div className="w-16 h-1 rounded-full bg-[var(--bg-active)] overflow-hidden">
            <motion.div
              className={`h-full rounded-full ${conf.bar}`}
              initial={{ width: 0 }}
              animate={{ width: "var(--bar-w)" }}
              style={{ "--bar-w": barWidth[meta.confidence!] } as React.CSSProperties}
              transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1], delay: 0.1 }}
            />
          </div>
          <span className={`text-[10px] font-medium ${conf.text}`}>{conf.label}</span>
          <div className="group relative">
            <Info size={10} className="text-[var(--text-muted)] cursor-help" />
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 w-48 p-2 bg-[var(--text-primary)] text-white text-[10px] rounded-lg leading-snug opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity z-[var(--z-dropdown)] whitespace-normal">
              Based on source coverage and answer consistency. High = strong multi-source agreement.
            </div>
          </div>
        </div>
      )}

      {/* Coverage badge */}
      {meta.consulted != null && meta.total != null && (
        <div className="relative">
          <button
            onClick={() => setShowCoveragePopover((v) => !v)}
            className="flex items-center gap-1 text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
          >
            <BookOpen size={10} />
            <span>
              {meta.consulted} / {meta.total} docs consulted
            </span>
          </button>
          {showCoveragePopover && (
            <div className="absolute bottom-full left-0 mb-2 w-52 card p-3 z-[var(--z-dropdown)] shadow-lg">
              <p className="text-[10px] font-medium text-[var(--text-primary)] mb-1.5">Coverage breakdown</p>
              <div className="space-y-1 text-[10px] text-[var(--text-secondary)]">
                <div className="flex justify-between">
                  <span className="text-[var(--status-ready)]">Read</span>
                  <span>{meta.consulted} docs</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-muted)]">Skipped (no match)</span>
                  <span>{(meta.total ?? 0) - (meta.consulted ?? 0)} docs</span>
                </div>
              </div>
              <p className="text-[9px] text-[var(--text-muted)] mt-2 leading-snug">
                Only docs with relevant embeddings were read. The rest were skipped to reduce latency.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Claim-type labels */}
      {meta.claimTypes && meta.claimTypes.length > 0 && (
        <div className="flex items-center gap-1.5">
          {meta.claimTypes.map((t) => (
            <span key={t} className={claimLabels[t].cls}>
              {claimLabels[t].label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
