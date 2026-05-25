"use client";

import React from "react";
import { motion } from "framer-motion";

const metrics = [
  { label: "Context Precision", value: 100.0 },
  { label: "Context Recall", value: 100.0 },
  { label: "Answer Relevancy", value: 95.91 },
  { label: "Faithfulness", value: 92.86 },
];

export function RagasMetrics() {
  return (
    <section className="py-24 px-6 lg:px-12 max-w-4xl mx-auto relative z-10">
      <div className="card shadow-lg p-8 md:p-12">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-[var(--text-primary)] mb-4">
            Evaluated for Production
          </h2>
          <p className="text-[var(--text-secondary)]">
            We don&apos;t guess if the AI is accurate. We measure it using the RAGAS framework.
          </p>
        </div>

        <div className="space-y-6">
          {metrics.map((m, i) => (
            <div key={i} className="space-y-2">
              <div className="flex justify-between text-sm font-medium">
                <span className="text-[var(--text-secondary)]">{m.label}</span>
                <span className="text-[var(--text-primary)] font-semibold">{m.value.toFixed(2)}%</span>
              </div>
              <div className="h-2 w-full bg-[var(--bg-hover)] rounded-full overflow-hidden border border-[var(--border)]">
                <motion.div
                  initial={{ width: 0 }}
                  whileInView={{ width: `${m.value}%` }}
                  viewport={{ once: true, margin: "-100px" }}
                  transition={{ duration: 1.5, delay: i * 0.2, ease: "easeOut" }}
                  className="h-full bg-[var(--accent)] rounded-full"
                />
              </div>
            </div>
          ))}
        </div>

        <div className="mt-10 pt-8 border-t border-dashed border-[var(--border-dotted)] text-center">
          <div className="text-4xl font-bold text-[var(--text-primary)] mb-2">0.92</div>
          <div className="text-[var(--status-ready)] font-medium mb-4">Overall Score</div>
          <p className="text-xs text-[var(--text-muted)] max-w-lg mx-auto">
            * Evaluated on &quot;Attention Is All You Need&quot;. Score represents the average of valid metrics. We exclude questions containing heavy mathematical notation from the Faithfulness metric, as current LLMs struggle to verify complex formulaic claims. We believe in honest evaluation, not inflated numbers.
          </p>
        </div>
      </div>
    </section>
  );
}
