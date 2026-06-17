"use client";

import React from "react";
import { Table, Cpu, Activity, Zap, Layers, Workflow } from "lucide-react";
import { Reveal } from "./Reveal";

type Feature = {
  icon: React.ElementType;
  title: string;
  description: string;
  detail: string;
};

/* Hero row: the two things that matter most — the Brain orchestration and
   table-aware ingestion. These get a wide 2-up feature row at the top. */
const heroFeatures: Feature[] = [
  {
    icon: Workflow,
    title: "Multi-agent Brain",
    description:
      "For large collections a map-reduce orchestration routes the query, reads documents in parallel, synthesises the findings, then verifies every claim against the source — at any scale.",
    detail: "Router → Map (parallel readers) → Reduce (synthesiser) → Verifier. Live SSE step events throughout, so you watch it reason.",
  },
  {
    icon: Table,
    title: "Table-aware ingestion",
    description:
      "Finance and legal docs live in tables. DocQuery extracts them as structured HTML — not flattened text — so covenant schedules, cap tables, and rate grids stay queryable.",
    detail: "Unstructured.io hi_res strategy · tables → structured HTML · images → base64 · title-based semantic chunking. PDF, DOCX, PPTX, XLSX, TXT.",
  },
];

/* Supporting row: the rest of the system architecture, balanced 4-up. */
const gridFeatures: Feature[] = [
  {
    icon: Layers,
    title: "Hybrid retrieval",
    description: "Dense vectors + BM25 keyword matching, fused via Reciprocal Rank Fusion then cross-encoder reranked.",
    detail: "Pinecone (text-embedding-3-small) + BM25 + RRF + ms-marco-MiniLM-L-6-v2.",
  },
  {
    icon: Activity,
    title: "RAGAS-evaluated",
    description: "0.92 overall on the RAGAS framework. Context Precision & Recall both 1.00 — verified, not claimed.",
    detail: "Faithfulness 0.93 · Answer Relevancy 0.96. Math-heavy questions excluded honestly.",
  },
  {
    icon: Cpu,
    title: "Async workers",
    description: "Parsing never blocks the API. Priority-routed Celery queues with DLQ inspection and auto-scaling.",
    detail: "Redis broker · fast/default/heavy queues · spot-instance Fargate.",
  },
  {
    icon: Zap,
    title: "SSE streaming",
    description: "Tokens stream the instant they're generated. Circuit-breaker falls back to source passages if the LLM is down.",
    detail: "Server-Sent Events · inline citation chips · graceful degradation.",
  },
];

const ease = [0.23, 1, 0.32, 1] as const;

function FeatureCard({
  icon: Icon,
  title,
  description,
  detail,
  delay,
  large = false,
}: Feature & { delay: number; large?: boolean }) {
  return (
    <Reveal delay={delay} className="h-full">
      <div
        className="h-full flex flex-col gap-5 rounded-[20px]"
        style={{
          background: "var(--surface)",
          border: "1px solid var(--line)",
          boxShadow: "var(--shadow-md)",
          padding: large ? "36px" : "28px",
        }}
      >
        <div
          className="rounded-xl flex items-center justify-center shrink-0"
          style={{
            width: large ? 48 : 42,
            height: large ? 48 : 42,
            background: "var(--ink)",
            color: "var(--on-ink)",
          }}
        >
          <Icon size={large ? 22 : 19} strokeWidth={1.7} />
        </div>

        <div className="flex flex-col gap-3 flex-1">
          <h3
            className="font-semibold leading-snug"
            style={{ color: "var(--ink)", letterSpacing: "-0.02em", fontSize: large ? "20px" : "16px" }}
          >
            {title}
          </h3>
          <p
            className="leading-relaxed"
            style={{ color: "var(--ink-2)", fontSize: large ? "15px" : "13.5px" }}
          >
            {description}
          </p>
          <p
            className="text-[12.5px] leading-relaxed mt-auto pt-4"
            style={{ color: "var(--ink-3)", borderTop: "1px solid var(--line)" }}
          >
            {detail}
          </p>
        </div>
      </div>
    </Reveal>
  );
}

export function FeaturesBento() {
  return (
    <section
      id="product"
      className="relative z-10 scroll-mt-24"
      style={{ paddingTop: "clamp(80px, 10vw, 140px)", paddingBottom: "clamp(80px, 10vw, 140px)" }}
    >
      <div className="section-container">
        {/* Header */}
        <Reveal className="text-center mb-16">
          <p className="eyebrow mb-4">Architecture</p>
          <h2
            className="font-display font-light mb-5"
            style={{
              fontSize: "clamp(36px, 5vw, 58px)",
              lineHeight: "1.05",
              letterSpacing: "-0.03em",
              color: "var(--ink)",
            }}
          >
            A Brain on top of
            <br />
            production-grade retrieval.
          </h2>
          <p
            className="max-w-xl mx-auto text-[17px] leading-relaxed"
            style={{ color: "var(--ink-2)", letterSpacing: "-0.01em" }}
          >
            Map-reduce orchestration, table-aware ingestion, and verified citations — the same patterns enterprise AI teams build in-house, out of the box.
          </p>
        </Reveal>

        {/* Hero row — Brain + table handling, the two focal points */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-5">
          {heroFeatures.map((f, i) => (
            <FeatureCard key={i} {...f} delay={i * 0.06} large />
          ))}
        </div>

        {/* Supporting row — balanced 4-up, no dead whitespace */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {gridFeatures.map((f, i) => (
            <FeatureCard key={i} {...f} delay={0.12 + i * 0.06} />
          ))}
        </div>
      </div>
    </section>
  );
}
