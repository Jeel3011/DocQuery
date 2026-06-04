"use client";

import React from "react";
import { motion } from "framer-motion";
import { Activity, ArrowRight } from "lucide-react";
import { Reveal } from "./Reveal";

const metrics = [
  { label: "Context Precision", value: 100.0, note: "No irrelevant context returned" },
  { label: "Context Recall", value: 100.0, note: "No relevant context missed" },
  { label: "Answer Relevancy", value: 95.91, note: "Answer addresses the question" },
  { label: "Faithfulness", value: 92.86, note: "Claims grounded in retrieved context" },
];

export function RagasMetrics() {
  return (
    <section
      id="accuracy"
      className="relative z-10 scroll-mt-24"
      style={{
        paddingTop: "clamp(80px, 10vw, 140px)",
        paddingBottom: "clamp(80px, 10vw, 140px)",
        background: "var(--surface-2)",
        borderTop: "1px solid var(--line)",
        borderBottom: "1px solid var(--line)",
      }}
    >
      <div className="section-container">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-start">

          {/* Left: editorial copy */}
          <Reveal>
            <p className="eyebrow mb-4">Evaluated accuracy</p>
            <h2
              className="font-display font-light mb-6"
              style={{
                fontSize: "clamp(36px, 5vw, 58px)",
                lineHeight: "1.05",
                letterSpacing: "-0.03em",
                color: "var(--ink)",
              }}
            >
              We measure it.
              <br />
              We don't claim it.
            </h2>
            <p
              className="text-[17px] leading-relaxed mb-8"
              style={{ color: "var(--ink-2)", letterSpacing: "-0.01em" }}
            >
              DocQuery is evaluated using the rigorous RAGAS framework on a real research paper. Every number below is reproduced — not cherry-picked.
            </p>

            {/* Big score */}
            <div
              className="inline-flex items-end gap-4 p-6 rounded-2xl mb-8"
              style={{
                background: "var(--surface)",
                border: "1px solid var(--line)",
                boxShadow: "var(--shadow-md)",
              }}
            >
              <div>
                <span
                  className="font-display block"
                  style={{
                    fontSize: "72px",
                    fontWeight: 300,
                    letterSpacing: "-0.04em",
                    lineHeight: 1,
                    color: "var(--ink)",
                    fontFeatureSettings: '"tnum","ss01"',
                  }}
                >
                  0.92
                </span>
                <p
                  className="text-[13px] font-semibold mt-1"
                  style={{ color: "var(--status-ready)", letterSpacing: "-0.01em" }}
                >
                  Overall RAGAS score
                </p>
              </div>
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center mb-1"
                style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--status-ready)" }}
              >
                <Activity size={18} strokeWidth={1.6} />
              </div>
            </div>

            <p
              className="text-[12px] leading-relaxed"
              style={{ color: "var(--ink-3)" }}
            >
              * Evaluated on "Attention Is All You Need." Questions with heavy mathematical notation excluded from Faithfulness — we believe in honest evaluation, not inflated numbers.
            </p>
          </Reveal>

          {/* Right: metric bars */}
          <Reveal delay={0.1}>
            <div className="flex flex-col gap-6">
              {metrics.map((m, i) => (
                <div key={i}>
                  <div className="flex items-end justify-between mb-2">
                    <div>
                      <p
                        className="text-[14px] font-semibold"
                        style={{ color: "var(--ink)", letterSpacing: "-0.01em" }}
                      >
                        {m.label}
                      </p>
                      <p className="text-[12px] mt-0.5" style={{ color: "var(--ink-3)" }}>
                        {m.note}
                      </p>
                    </div>
                    <span
                      className="font-display text-[22px] tabular-nums"
                      style={{ color: "var(--ink)", letterSpacing: "-0.03em", fontWeight: 300, lineHeight: 1 }}
                    >
                      {m.value.toFixed(2)}%
                    </span>
                  </div>

                  {/* Track */}
                  <div
                    className="h-2 w-full rounded-full overflow-hidden"
                    style={{
                      background: "var(--surface-3)",
                      border: "1px solid var(--line)",
                    }}
                  >
                    <motion.div
                      initial={{ width: 0 }}
                      whileInView={{ width: `${m.value}%` }}
                      viewport={{ once: true, margin: "-80px" }}
                      transition={{ duration: 1.4, delay: i * 0.12, ease: "easeOut" }}
                      className="h-full rounded-full"
                      style={{
                        background: `linear-gradient(90deg, var(--accent-taupe) 0%, var(--ink) ${Math.min(m.value, 100)}%)`,
                      }}
                    />
                  </div>
                </div>
              ))}

              {/* Framework link */}
              <a
                href="https://docs.ragas.io"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-[13px] font-medium pt-2"
                style={{ color: "var(--accent-taupe)" }}
              >
                Learn about the RAGAS framework
                <ArrowRight size={13} />
              </a>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
