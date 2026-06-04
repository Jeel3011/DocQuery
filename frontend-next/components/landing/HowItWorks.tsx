"use client";

import React from "react";
import { motion } from "framer-motion";
import { Upload, MessageSquareText, BookOpenCheck, Sparkles } from "lucide-react";
import { Reveal } from "./Reveal";

const steps = [
  {
    num: "01",
    icon: Upload,
    title: "Upload your corpus",
    description: "Drop PDFs, Word docs, spreadsheets, or presentations. Async Celery workers parse, chunk, and index every page — tables extracted as structured HTML, not flattened text — and the collection is queryable in seconds.",
    note: "Supports PDF, DOCX, PPTX, XLSX, TXT · Up to 10 MB per file",
    stats: [
      { value: "5", label: "file formats" },
      { value: "~3s", label: "to index a 30-pg PDF" },
      { value: "100+", label: "docs per collection" },
    ],
  },
  {
    num: "02",
    icon: MessageSquareText,
    title: "Ask in plain language",
    description: "No Boolean syntax, no keyword guessing. The Brain routes the query, fans out across the relevant documents in parallel, and fuses dense vector recall with BM25 keyword precision — then reranks with a cross-encoder so the best passages rise to the top.",
    note: "Hybrid dense + sparse retrieval · RRF fusion · cross-encoder reranking",
    stats: [
      { value: "< 2s", label: "first token (p50)" },
      { value: "2", label: "retrieval signals fused" },
      { value: "RRF", label: "+ MiniLM rerank" },
    ],
  },
  {
    num: "03",
    icon: BookOpenCheck,
    title: "Get a cited, verifiable answer",
    description: "Answers stream token-by-token with numbered citations linking to the exact passage in the exact document. Tabular questions return interactive, sortable tables. A self-review pass checks every claim against its source before you ever see it.",
    note: "Inline citation chips · sortable tables · self-review · RAGAS evaluated",
    stats: [
      { value: "0.92", label: "RAGAS overall" },
      { value: "1.00", label: "context precision" },
      { value: "100%", label: "claims source-linked" },
    ],
  },
];

export function HowItWorks() {
  return (
    <section
      id="how-it-works"
      className="relative z-10"
      style={{
        paddingTop: "clamp(80px, 10vw, 140px)",
        paddingBottom: "clamp(80px, 10vw, 140px)",
        background: "var(--surface-2)",
        borderTop: "1px solid var(--line)",
        borderBottom: "1px solid var(--line)",
      }}
    >
      <div className="section-container">
        {/* Header */}
        <Reveal className="mb-20 max-w-2xl">
          <p className="eyebrow mb-4">How it works</p>
          <h2
            className="font-display font-light mb-5"
            style={{
              fontSize: "clamp(36px, 5vw, 58px)",
              lineHeight: "1.05",
              letterSpacing: "-0.03em",
              color: "var(--ink)",
            }}
          >
            Three steps from
            <br />
            upload to insight.
          </h2>
          <p
            className="text-[17px] leading-relaxed"
            style={{ color: "var(--ink-2)", letterSpacing: "-0.01em" }}
          >
            No configuration. No prompt engineering. No learning curve.
          </p>
        </Reveal>

        {/* Steps */}
        <div className="relative">
          {/* Vertical connector (desktop) */}
          <div
            className="hidden lg:block absolute top-8 bottom-8 left-[27px]"
            style={{ width: "1px", background: "linear-gradient(to bottom, var(--line) 0%, var(--line-2) 50%, var(--line) 100%)" }}
          />

          <div className="flex flex-col gap-0">
            {steps.map((step, i) => (
              <Reveal key={i} delay={i * 0.1}>
                <div className="relative flex gap-8 lg:gap-14 pb-14 last:pb-0">

                  {/* Number + Icon circle */}
                  <div className="relative flex flex-col items-center shrink-0">
                    <div
                      className="relative z-10 w-[56px] h-[56px] rounded-2xl flex items-center justify-center shrink-0"
                      style={{
                        background: "var(--surface)",
                        border: "1px solid var(--line-2)",
                        boxShadow: "var(--shadow-md)",
                        color: "var(--ink)",
                      }}
                    >
                      <step.icon size={22} strokeWidth={1.6} />
                    </div>
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0 pt-2">
                    {/* Step number */}
                    <span className="eyebrow mb-3 block">{step.num}</span>

                    <h3
                      className="font-semibold text-[22px] mb-4 leading-tight"
                      style={{ color: "var(--ink)", letterSpacing: "-0.02em" }}
                    >
                      {step.title}
                    </h3>

                    <p
                      className="text-[16px] leading-relaxed mb-6 max-w-xl"
                      style={{ color: "var(--ink-2)" }}
                    >
                      {step.description}
                    </p>

                    {/* Data stats row */}
                    <div className="flex flex-wrap gap-3 mb-5">
                      {step.stats.map((s, si) => (
                        <div
                          key={si}
                          className="flex flex-col gap-0.5 px-4 py-2.5 rounded-xl"
                          style={{
                            background: "var(--surface)",
                            border: "1px solid var(--line)",
                            boxShadow: "var(--shadow-sm)",
                          }}
                        >
                          <span
                            className="font-display"
                            style={{
                              fontSize: "22px",
                              fontWeight: 300,
                              lineHeight: 1,
                              letterSpacing: "-0.03em",
                              color: "var(--ink)",
                              fontFeatureSettings: '"tnum","ss01"',
                            }}
                          >
                            {s.value}
                          </span>
                          <span className="text-[11px]" style={{ color: "var(--ink-3)" }}>
                            {s.label}
                          </span>
                        </div>
                      ))}
                    </div>

                    {/* Technical footnote */}
                    <div
                      className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-[12px] font-medium"
                      style={{
                        background: "var(--surface)",
                        border: "1px solid var(--line)",
                        color: "var(--ink-3)",
                        boxShadow: "var(--shadow-sm)",
                      }}
                    >
                      <Sparkles size={11} style={{ color: "var(--ink-2)" }} />
                      {step.note}
                    </div>
                  </div>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
