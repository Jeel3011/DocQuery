"use client";

import { motion } from "framer-motion";
import { Table2, PenLine, Sparkles, Play, Search, Shield } from "lucide-react";
import { Reveal } from "@/components/landing/Reveal";
import { PlatformPageShell, PlatformHero, PlatformCTA, ease } from "@/components/landing/PlatformPageShell";

const OUTPUT_META = {
  Review: { icon: Table2, label: "Review" },
  Draft: { icon: PenLine, label: "Draft" },
  Output: { icon: Sparkles, label: "Output" },
};

const PRACTICE_AREAS: { area: string; cards: { title: string; description: string; type: keyof typeof OUTPUT_META; steps: number }[] }[] = [
  {
    area: "Diligence",
    cards: [
      { title: "Change-of-control sweep", description: "Flag every change-of-control clause across the vault, with risk assessment.", type: "Review", steps: 3 },
      { title: "Indemnity comparison", description: "Head-to-head indemnity caps and carve-outs across all contracts.", type: "Review", steps: 4 },
    ],
  },
  {
    area: "Drafting",
    cards: [
      { title: "Redline summary memo", description: "A cited memo summarizing every redline against the standard template.", type: "Draft", steps: 5 },
    ],
  },
  {
    area: "Finance",
    cards: [
      { title: "Revenue bridge", description: "Reconcile period-over-period revenue across filings, cell by cell.", type: "Output", steps: 4 },
    ],
  },
];

const FEATURES = [
  {
    title: "Output routed by shape, not by template",
    description: "A workflow declares its shape — grid, report, or output — and the result renders into the matching surface: a live cell table for Review, a streamed cited deliverable for Draft and Output. One generic runner, not N bespoke screens.",
  },
  {
    title: "Forms generated from a schema, not hand-built",
    description: "Each workflow's parameters render generically from its params_schema — document multiselect, text, or textarea. Add a workflow by declaring its schema; the run drawer just works.",
  },
  {
    title: "The same live agent timeline as Ask",
    description: "Draft and Output workflows show the identical tool-by-tool transcript as the Agent — search, read, verify — so a multi-step workflow is never a black box between click and result.",
  },
  {
    title: "Grouped by practice area, filterable by output type",
    description: "Workflows surface grouped under the practice area they serve, with a quick filter across Review / Draft / Output and a search box — built to scale past a handful of playbooks.",
  },
];

function WorkflowsDemo() {
  return (
    <Reveal delay={0.2}>
      <div
        className="rounded-[22px] overflow-hidden mt-4"
        style={{
          background: "rgba(255,255,255,0.72)",
          backdropFilter: "blur(22px) saturate(1.0)",
          WebkitBackdropFilter: "blur(22px) saturate(1.0)",
          border: "1px solid rgba(255,255,255,0.80)",
          boxShadow: "var(--shadow-xl), inset 0 1px 0 rgba(255,255,255,0.90)",
        }}
      >
        <div
          className="flex items-center gap-3 px-5 py-3.5"
          style={{
            background: "linear-gradient(180deg, rgba(255,255,255,0.90) 0%, rgba(255,255,255,0.55) 100%)",
            borderBottom: "1px solid var(--glass-border)",
          }}
        >
          <div className="flex gap-1.5">
            {["#C4C4C4", "#A8A8A8", "#8C8C8C"].map((c, i) => (
              <div key={i} className="w-3 h-3 rounded-full opacity-90" style={{ background: c }} />
            ))}
          </div>
          <div className="flex-1 mx-6 flex items-center justify-center gap-2 px-4 py-1.5 rounded-lg" style={{ background: "rgba(244,244,244,0.70)", border: "1px solid var(--line)" }}>
            <Shield size={10} className="opacity-40" />
            <span className="text-[11px] font-medium" style={{ color: "var(--ink-3)" }}>Workflows — Acquisition Diligence</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg" style={{ background: "rgba(244,244,244,0.60)", border: "1px solid var(--line)" }}>
            <Search size={11} className="opacity-40" />
            <span className="text-[11px]" style={{ color: "var(--ink-3)" }}>Search workflows</span>
          </div>
        </div>

        <div className="p-6 space-y-7">
          {PRACTICE_AREAS.map((group, gi) => (
            <div key={group.area}>
              <Reveal delay={0.1 + gi * 0.05}>
                <h3 className="text-[11px] font-semibold tracking-wide mb-3" style={{ color: "var(--ink-2)" }}>{group.area}</h3>
              </Reveal>
              <div className="grid sm:grid-cols-2 gap-3">
                {group.cards.map((c, ci) => {
                  const meta = OUTPUT_META[c.type];
                  return (
                    <motion.div
                      key={c.title}
                      initial={{ opacity: 0, y: 8 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: 0.15 + gi * 0.05 + ci * 0.04, duration: 0.4, ease }}
                      className="rounded-xl p-4 group cursor-default"
                      style={{ background: "var(--surface-2)", border: "1px solid var(--line)" }}
                    >
                      <div className="flex items-start justify-between mb-1.5">
                        <span className="text-[13.5px] font-semibold leading-snug pr-2" style={{ color: "var(--ink)" }}>{c.title}</span>
                        <span className="inline-flex items-center gap-1 text-[11px] opacity-0 group-hover:opacity-100 transition-opacity shrink-0" style={{ color: "var(--ink)" }}>
                          <Play size={11} /> Run
                        </span>
                      </div>
                      <p className="text-[12px] leading-relaxed mb-3" style={{ color: "var(--ink-3)" }}>{c.description}</p>
                      <div className="flex items-center gap-1.5 text-[11px]" style={{ color: "var(--ink-3)" }}>
                        <meta.icon size={12} />
                        <span>{meta.label}</span>
                        <span>·</span>
                        <span>{c.steps} steps</span>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Reveal>
  );
}

export default function WorkflowsPlatformPage() {
  return (
    <PlatformPageShell>
      <PlatformHero
        breadcrumb="Platform Overview / Workflows"
        title={<>Repeatable playbooks, <span style={{ color: "var(--accent-taupe)" }}>cited</span> end to end.</>}
        description="Diligence sweeps, redline memos, revenue bridges — packaged as one-click workflows across a vault. Every output is cited or flagged, the same contract the Agent and Review grid hold."
      />

      <section id="how-it-works" className="relative z-10" style={{ paddingBottom: "clamp(60px, 8vw, 100px)" }}>
        <div className="section-container">
          <WorkflowsDemo />
        </div>
      </section>

      <section className="relative z-10" style={{ paddingBottom: "clamp(80px, 10vw, 130px)" }}>
        <div className="section-container">
          <Reveal className="text-center mb-12">
            <p className="eyebrow mb-4">How it&apos;s built</p>
            <h2 className="font-display font-light" style={{ fontSize: "clamp(30px, 4vw, 46px)", lineHeight: "1.08", letterSpacing: "-0.03em", color: "var(--ink)" }}>
              One runner. Three output shapes.
            </h2>
          </Reveal>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {FEATURES.map((f, i) => (
              <Reveal key={f.title} delay={i * 0.05}>
                <div className="h-full rounded-[20px] p-7" style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-md)" }}>
                  <h3 className="font-semibold mb-2.5" style={{ color: "var(--ink)", fontSize: "16px", letterSpacing: "-0.01em" }}>{f.title}</h3>
                  <p className="leading-relaxed" style={{ color: "var(--ink-2)", fontSize: "14px" }}>{f.description}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <PlatformCTA />
    </PlatformPageShell>
  );
}
