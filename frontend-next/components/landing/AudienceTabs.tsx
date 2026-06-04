"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { TrendingUp, Scale, FlaskConical, Shield, ChevronRight } from "lucide-react";
import { Reveal } from "./Reveal";

type TableBlock = { headers: string[]; rows: string[][] };
type Audience = {
  id: string;
  label: string;
  icon: React.ElementType;
  headline: string;
  body: string;
  query: string;
  table?: TableBlock;
  answer: string;
  sources: string[];
};

const AUDIENCES: Audience[] = [
  {
    id: "finance",
    label: "Finance & Accounting",
    icon: TrendingUp,
    headline: "Revenue concentration & variance, segment by segment",
    body: "Customer concentration, segment variance to plan, margin walk — pulled from the MD&A and the segment schedules and returned as a table you can paste straight into a board memo.",
    query:
      "Break down FY revenue by segment, show variance to plan and YoY, and flag any single-customer concentration above 10% of total revenue.",
    table: {
      headers: ["Segment", "FY rev", "vs Plan", "YoY", "Top-cust %"],
      rows: [
        ["Enterprise SaaS", "₹212 Cr", "+6%", "+19%", "8%"],
        ["Services", "₹64 Cr", "−11%", "+3%", "22%"],
        ["Hardware resale", "₹28 Cr", "−4%", "−7%", "31%"],
      ],
    },
    answer: `Enterprise SaaS beat plan (+6%) and drove all the YoY growth (+19%). Services missed plan by 11% on slower onboarding. Two concentration risks need a footnote: Services derives 22% of revenue from one customer and Hardware resale 31% — both above the 10% disclosure threshold and worth a risk-factor note. Hardware is shrinking on both axes; consider whether it still earns the working capital.`,
    sources: [
      "Annual Report FY24.pdf · MD&A p.34 — segment results",
      "Annual Report FY24.pdf · Note 27 — customer concentration",
      "Board Plan FY24.xlsx · Sheet 'Plan vs Actual'",
    ],
  },
  {
    id: "legal",
    label: "Legal & Compliance",
    icon: Scale,
    headline: "Renewal & termination exposure across the contract book",
    body: "Auto-renewal traps, notice windows, termination-for-convenience rights and price-escalation caps — aligned across every active contract so nothing renews silently against you.",
    query:
      "Across all active vendor contracts, list the auto-renewal terms and notice windows, and flag anything renewing in the next 60 days or with an uncapped price escalator.",
    table: {
      headers: ["Contract", "Auto-renew", "Notice", "Next renewal", "Escalator"],
      rows: [
        ["CloudHost Inc", "12 mo", "90 days", "in 42 days", "CPI-linked"],
        ["DataPipe Ltd", "24 mo", "30 days", "in 8 mo", "Uncapped"],
        ["SecureAuth", "Month-to-month", "30 days", "rolling", "Fixed"],
      ],
    },
    answer: `One urgent item: the CloudHost contract auto-renews for 12 months in 42 days and its 90-day notice window has already closed — to stop it you needed to act 48 days ago, so you're locked for another term. DataPipe's escalator is uncapped, exposing you to unbounded price rises at the 24-month renewal; negotiate a CPI or fixed cap before then. SecureAuth is low-risk and cancellable any month.`,
    sources: [
      "CloudHost-MSA.pdf · §14.1 — term & auto-renewal",
      "DataPipe-Agreement.pdf · §6.3 — price escalation",
      "Contract Register.xlsx · 'Active' tab — notice windows",
    ],
  },
  {
    id: "research",
    label: "Research & Strategy",
    icon: FlaskConical,
    headline: "Compare methodologies across the literature",
    body: "Methodology comparison, dataset and metric reconciliation, reproducibility gaps — lined up across papers so you can see which result is actually comparable to which, with section-level citations.",
    query:
      "Compare the evaluation methodology of these three retrieval papers — datasets, metrics, and whether the results are directly comparable.",
    table: {
      headers: ["Paper", "Dataset", "Primary metric", "Comparable?"],
      rows: [
        ["DPR (2020)", "NaturalQ, TriviaQA", "Top-20 retrieval acc.", "Baseline"],
        ["ColBERT (2020)", "MS MARCO", "MRR@10", "No — diff. metric"],
        ["RRF fusion (2009)", "TREC ad-hoc", "MAP", "No — diff. task"],
      ],
    },
    answer: `The three are not directly comparable as published. DPR reports top-20 retrieval accuracy on open-QA datasets; ColBERT reports MRR@10 on passage ranking; the RRF paper reports MAP on classic TREC ad-hoc retrieval. To compare them you'd need to re-run all three on one dataset with one metric — most fairly MS MARCO with both MRR@10 and Recall@k. The headline numbers in each abstract are measuring different things.`,
    sources: [
      "DPR.pdf · §4.1 — evaluation setup",
      "ColBERT.pdf · §4 — MS MARCO results",
      "Reciprocal-Rank-Fusion.pdf · §3 — TREC evaluation",
    ],
  },
];

const ease = [0.16, 1, 0.3, 1] as const;

function ResultTable({ table }: { table: TableBlock }) {
  return (
    <div
      className="rounded-xl overflow-hidden mb-4"
      style={{ border: "1px solid var(--line)", boxShadow: "var(--shadow-sm)" }}
    >
      <table className="w-full border-collapse text-[12px]">
        <thead>
          <tr style={{ background: "var(--ink)" }}>
            {table.headers.map((h, i) => (
              <th
                key={i}
                className="text-left font-semibold px-3 py-2"
                style={{ color: "var(--on-ink)", letterSpacing: "-0.01em" }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, ri) => (
            <tr key={ri} style={{ background: ri % 2 ? "var(--surface-2)" : "var(--surface)" }}>
              {row.map((cell, ci) => {
                const lc = cell.toLowerCase();
                const isPass = cell === "PASS";
                const isRisk =
                  cell === "AT RISK" || cell === "FAIL" ||
                  lc.includes("uncapped") || lc.includes("flat") ||
                  lc.startsWith("in ") || lc.startsWith("no —") ||
                  /^[2-9]\d%$/.test(cell);  // single-customer concentration ≥ 20%
                return (
                  <td
                    key={ci}
                    className="px-3 py-2"
                    style={{
                      borderTop: "1px solid var(--line)",
                      color: isPass ? "var(--ink)" : isRisk ? "var(--status-failed)" : "var(--ink-2)",
                      fontWeight: ci === 0 || isPass || isRisk ? 600 : 400,
                    }}
                  >
                    {isPass && <span className="mr-1">✓</span>}
                    {isRisk && cell === "AT RISK" && <span className="mr-1">⚠</span>}
                    {cell}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AnswerBlock({ answer }: { answer: string }) {
  const lines = answer.split("\n").filter(Boolean);
  return (
    <div className="space-y-2">
      {lines.map((line, i) => {
        const isBullet = line.startsWith("•") || line.startsWith("**•") || line.trim().startsWith("•");
        const isWarning = line.includes("⚠") || line.includes("Flag") || line.includes("unlimited");
        const isSynthesis = line.startsWith("**Synthesis") || line.startsWith("**Recommendation");
        const clean = line
          .replace(/\*\*/g, "")
          .replace(/\[\d+\]/g, "")
          .trim();
        if (!clean) return null;
        return (
          <p
            key={i}
            className={`text-[14px] leading-relaxed ${isBullet ? "pl-2" : ""}`}
            style={{
              color: isWarning ? "var(--status-failed)" : isSynthesis ? "var(--ink)" : "var(--ink-2)",
              fontWeight: isSynthesis ? 600 : 400,
            }}
          >
            {clean}
          </p>
        );
      })}
    </div>
  );
}

export function AudienceTabs() {
  const [active, setActive] = useState("finance");
  const current = AUDIENCES.find((a) => a.id === active)!;

  return (
    <section
      className="relative z-10"
      style={{ paddingTop: "clamp(80px, 10vw, 140px)", paddingBottom: "clamp(80px, 10vw, 140px)" }}
    >
      <div className="section-container">
        {/* Header */}
        <Reveal className="mb-14">
          <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6">
            <div>
              <p className="eyebrow mb-4">Use cases</p>
              <h2
                className="font-display font-light"
                style={{
                  fontSize: "clamp(36px, 5vw, 58px)",
                  lineHeight: "1.05",
                  letterSpacing: "-0.03em",
                  color: "var(--ink)",
                }}
              >
                Built for
                <br />
                precision work.
              </h2>
            </div>
            <p
              className="max-w-sm text-[16px] leading-relaxed lg:text-right"
              style={{ color: "var(--ink-2)" }}
            >
              Finance, legal, and research professionals need answers they can cite and defend. DocQuery delivers exactly that.
            </p>
          </div>
        </Reveal>

        {/* Tab selector */}
        <Reveal>
          <div
            className="inline-flex rounded-2xl p-1.5 gap-1 mb-10"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--line)",
              boxShadow: "var(--shadow-sm)",
            }}
          >
            {AUDIENCES.map((a) => {
              const Icon = a.icon;
              const isActive = active === a.id;
              return (
                <button
                  key={a.id}
                  onClick={() => setActive(a.id)}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-semibold transition-all"
                  style={{
                    background: isActive ? "var(--ink)" : "transparent",
                    color: isActive ? "var(--on-ink)" : "var(--ink-2)",
                    boxShadow: isActive ? "var(--shadow-sm)" : "none",
                  }}
                >
                  <Icon size={13} />
                  {a.label.split(" ")[0]}
                </button>
              );
            })}
          </div>
        </Reveal>

        {/* Demo panel */}
        <AnimatePresence mode="wait">
          <motion.div
            key={active}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.22, ease }}
            className="rounded-[24px] overflow-hidden"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--line)",
              boxShadow: "var(--shadow-lg)",
            }}
          >
            {/* Panel header */}
            <div
              className="flex flex-col lg:flex-row lg:items-center gap-4 px-8 py-6"
              style={{
                background: "var(--surface-2)",
                borderBottom: "1px solid var(--line)",
              }}
            >
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
                style={{
                  background: "var(--accent-soft)",
                  border: "1px solid var(--line-2)",
                  color: "var(--accent-taupe)",
                }}
              >
                <current.icon size={18} strokeWidth={1.6} />
              </div>
              <div>
                <h3
                  className="font-semibold text-[16px] leading-tight"
                  style={{ color: "var(--ink)", letterSpacing: "-0.01em" }}
                >
                  {current.headline}
                </h3>
                <p className="text-[13px] mt-0.5" style={{ color: "var(--ink-3)" }}>
                  {current.body}
                </p>
              </div>
            </div>

            {/* Query + answer */}
            <div className="grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x" style={{ borderColor: "var(--line)" }}>
              {/* Query side */}
              <div className="p-8 flex flex-col gap-4">
                <p className="eyebrow">Sample query</p>
                <div
                  className="flex-1 p-5 rounded-2xl text-[14px] leading-relaxed italic"
                  style={{
                    background: "var(--surface-3)",
                    border: "1px solid var(--line)",
                    color: "var(--ink-2)",
                  }}
                >
                  "{current.query}"
                </div>
              </div>

              {/* Answer side */}
              <div className="p-8 flex flex-col gap-4">
                <div className="flex items-center gap-2">
                  <p className="eyebrow">DocQuery answer</p>
                  <span
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium"
                    style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-3)" }}
                    title="Tables render with the same sortable AnswerTable component used in the live chat."
                  >
                    <span className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--ink)" }} />
                    Rendered as in-app
                  </span>
                </div>

                <div
                  className="flex-1 p-5 rounded-2xl"
                  style={{
                    background: "var(--surface)",
                    border: "1px solid var(--line)",
                    boxShadow: "var(--shadow-sm)",
                  }}
                >
                  {current.table && <ResultTable table={current.table} />}
                  <AnswerBlock answer={current.answer} />
                </div>

                {/* Citations */}
                <div className="flex flex-wrap gap-2 pt-1">
                  <Shield size={12} style={{ color: "var(--status-ready)", marginTop: 3 }} />
                  {current.sources.map((s, i) => (
                    <div
                      key={i}
                      className="inline-flex items-center gap-2 px-2.5 py-1 rounded-lg text-[11px] font-medium"
                      style={{
                        background: "var(--surface-3)",
                        border: "1px solid var(--line)",
                        color: "var(--ink-2)",
                      }}
                    >
                      <span
                        className="w-4 h-4 rounded flex items-center justify-center text-[9px] font-bold shrink-0"
                        style={{ background: "var(--ink)", color: "var(--on-ink)" }}
                      >
                        {i + 1}
                      </span>
                      {s}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  );
}
