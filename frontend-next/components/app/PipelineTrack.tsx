"use client";

// PipelineTrack — a document's journey through OUR trust pipeline, rendered as a single
// horizontal track that visibly walks the four real ingest stages (G2 Step C, the NEW
// upload UX). This is the differentiator: dead "processing…" time becomes a visible
// verification pipeline — the trust signal Harvey's generic spinner lacks.
//
// The four nodes map to the worker's REAL progress bands (src/worker/tasks.py), so the
// dot's position is truth, never theater:
//   Parse  10–30%  ·  Chunk 30–50%  ·  Embed 50–100%  ·  Ready 100% (status=ready)
// A failed doc lights the stage it died on in risk-red. Fidelity (the dot that lands at
// Ready) stays neutral until Step F surfaces real per-doc fidelity — we never fake green.

import { motion } from "framer-motion";
import { FileText, AlertCircle } from "lucide-react";
import { FidelityDot, type Fidelity } from "./DocMeta";
import type { DocumentResponse } from "@/lib/api";

const STAGES = [
  { key: "parse", label: "Parse", floor: 10 },
  { key: "chunk", label: "Chunk", floor: 30 },
  { key: "embed", label: "Embed", floor: 50 },
  { key: "ready", label: "Ready", floor: 100 },
] as const;

// Which node is "current" for a given progress %. <30 → parse, <50 → chunk, <100 → embed,
// 100 → ready. Returns the index into STAGES.
function activeIndex(pct: number): number {
  if (pct >= 100) return 3;
  if (pct >= 50) return 2;
  if (pct >= 30) return 1;
  return 0;
}

export function PipelineTrack({ doc, fidelity = null }: { doc: DocumentResponse; fidelity?: Fidelity }) {
  const pct = doc.status === "ready" ? 100 : doc.processing_progress ?? 10;
  const failed = doc.status === "failed";
  const active = activeIndex(pct);
  const ready = doc.status === "ready";

  return (
    <div
      className="px-4 py-3 rounded-xl"
      style={{ background: "var(--surface)", border: "1px solid var(--line)" }}
    >
      {/* File name + a trailing fidelity dot once ready */}
      <div className="flex items-center gap-2.5 mb-3">
        <FileText size={13} className="text-[var(--text-muted)] flex-shrink-0" />
        <span className="flex-1 truncate text-[13px] text-[var(--text-primary)]">{doc.filename}</span>
        {ready && <FidelityDot fidelity={fidelity} withLabel />}
        {failed && (
          <span className="inline-flex items-center gap-1 text-[12px]" style={{ color: "var(--status-failed)" }}>
            <AlertCircle size={12} /> failed
          </span>
        )}
      </div>

      {/* The track: nodes joined by segments. Filled segments + passed nodes = ink;
          the current node pulses; pending = hairline grey. */}
      <div className="flex items-center">
        {STAGES.map((s, i) => {
          const passed = i < active || ready;
          const isCurrent = i === active && !ready && !failed;
          const isFailNode = failed && i === active;
          const nodeColor = isFailNode
            ? "var(--status-failed)"
            : passed || ready
            ? "var(--ink)"
            : "var(--line-2)";

          return (
            <div key={s.key} className="flex items-center flex-1 last:flex-none">
              {/* Node */}
              <div className="flex flex-col items-center gap-1.5 flex-shrink-0">
                <span className="relative flex items-center justify-center" style={{ width: 14, height: 14 }}>
                  {isCurrent && (
                    <motion.span
                      className="absolute inset-0 rounded-full"
                      style={{ background: "var(--ink)", opacity: 0.18 }}
                      animate={{ scale: [1, 1.9, 1], opacity: [0.22, 0, 0.22] }}
                      transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
                    />
                  )}
                  <span
                    className="rounded-full"
                    style={{
                      width: isCurrent ? 9 : 7,
                      height: isCurrent ? 9 : 7,
                      background: nodeColor,
                      boxShadow: isCurrent ? "0 0 0 3px rgba(14,14,14,0.10)" : "none",
                      transition: "all 180ms",
                    }}
                  />
                </span>
                <span
                  className="text-[10px] font-medium"
                  style={{ color: passed || isCurrent || isFailNode ? "var(--ink-2)" : "var(--text-muted)" }}
                >
                  {s.label}
                </span>
              </div>

              {/* Connector segment (not after the last node) */}
              {i < STAGES.length - 1 && (
                <div className="flex-1 h-px mx-1.5 relative -mt-4" style={{ background: "var(--line)" }}>
                  <motion.div
                    className="absolute inset-y-0 left-0"
                    style={{ background: failed ? "var(--status-failed)" : "var(--ink)" }}
                    initial={false}
                    animate={{ width: i < active || ready ? "100%" : "0%" }}
                    transition={{ duration: 0.5, ease: [0.23, 1, 0.32, 1] }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
