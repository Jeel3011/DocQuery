"use client";

import { motion } from "framer-motion";
import { Search, BookOpen, Calculator, CheckCircle2, ShieldAlert, FileText, Shield, Sparkles, Zap } from "lucide-react";
import { Reveal } from "@/components/landing/Reveal";
import { PlatformPageShell, PlatformHero, PlatformCTA, ease } from "@/components/landing/PlatformPageShell";

const TOOLS = [
  { icon: Search, name: "search_vault", detail: "Hybrid dense + BM25 retrieval, RRF-fused and reranked." },
  { icon: BookOpen, name: "read_document", detail: "Pulls the exact passage or clause, in full." },
  { icon: FileText, name: "table_lookup", detail: "Resolves a line item from a source grid cell." },
  { icon: Calculator, name: "compute", detail: "Deterministic arithmetic over cited cells — never free-text math." },
  { icon: CheckCircle2, name: "verify_numbers", detail: "Cross-checks every figure against its source before it ships." },
];

const STEPS = [
  { label: "Searching documents for change-of-control provisions", status: "done" as const },
  { label: "Reading Section 8.2 — Sezzle MSA", status: "done" as const },
  { label: "Verifying every figure against sources", status: "active" as const },
];

const FEATURES = [
  {
    title: "Every figure traces or is withheld",
    description: "Non-bypassable output gates bind the draft to an evidence ledger. If a number can't be traced to a cell or a quote can't be traced to a clause, it's withheld — never guessed, never hedged into a soft answer.",
  },
  {
    title: "A visible tool timeline, not a spinner",
    description: "Watch the agent search, read, compute, and verify in real time. Each tool call is its own step with the document it touched — the same transparency you'd expect from a junior associate's working notes.",
  },
  {
    title: "Inline citations you can open",
    description: "Every claim in the answer carries a citation chip. Click it to see the grounding quote and the exact page or cell it came from — the answer is never more confident than its evidence.",
  },
  {
    title: "Scoped to a vault, a filter, or everything",
    description: "Ask one document, a filtered subset (by type or fiscal year), or the whole vault. The same agent, the same gates, at any scale.",
  },
];

function AgentDemo() {
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
            <span className="text-[11px] font-medium" style={{ color: "var(--ink-3)" }}>Ask — Acquisition Diligence</span>
          </div>
          <div className="flex items-center gap-1.5" style={{ color: "var(--ink-3)" }}>
            {[Sparkles, BookOpen, Zap].map((Icon, i) => (
              <div key={i} className="w-6 h-6 flex items-center justify-center rounded-md" style={{ background: "rgba(244,244,244,0.60)" }}>
                <Icon size={11} />
              </div>
            ))}
          </div>
        </div>

        <div className="p-6">
          {/* User question */}
          <Reveal delay={0.05}>
            <div className="flex justify-end mb-4">
              <div className="max-w-[80%] px-4 py-2.5 rounded-2xl text-[13.5px]" style={{ background: "var(--ink)", color: "var(--on-ink)" }}>
                What are the change-of-control provisions in the Sezzle MSA?
              </div>
            </div>
          </Reveal>

          {/* Tool timeline */}
          <div className="mb-4 space-y-2.5">
            {STEPS.map((s, i) => (
              <motion.div
                key={s.label}
                initial={{ opacity: 0, x: -8 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.15 + i * 0.12, duration: 0.4, ease }}
                className="flex items-center gap-2.5 text-[12px]"
              >
                <span
                  className="w-4 h-4 rounded-full flex items-center justify-center shrink-0"
                  style={{ background: s.status === "done" ? "var(--fidelity-good)" : "var(--surface-3)", border: s.status === "active" ? "1.5px solid var(--ink-3)" : "none" }}
                >
                  {s.status === "done" && <CheckCircle2 size={11} style={{ color: "var(--on-ink)" }} />}
                </span>
                <span style={{ color: s.status === "active" ? "var(--ink)" : "var(--ink-3)" }}>{s.label}</span>
              </motion.div>
            ))}
          </div>

          {/* Answer with citations */}
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.55, duration: 0.5, ease }}
            className="rounded-xl p-4 text-[13.5px] leading-relaxed"
            style={{ background: "var(--surface-2)", border: "1px solid var(--line)", color: "var(--ink)" }}
          >
            Section 8.2 grants either party the right to terminate within 30 days of a change of control
            <span className="inline-flex items-center justify-center w-4 h-4 rounded-full text-[10px] font-semibold mx-1" style={{ background: "var(--ink)", color: "var(--on-ink)" }}>1</span>
            , with no termination fee if notice is given in writing
            <span className="inline-flex items-center justify-center w-4 h-4 rounded-full text-[10px] font-semibold mx-1" style={{ background: "var(--ink)", color: "var(--on-ink)" }}>2</span>
            .
            <div className="mt-3 pt-3 flex items-center gap-1.5 text-[11px]" style={{ borderTop: "1px solid var(--line)", color: "var(--ink-3)" }}>
              <ShieldAlert size={11} style={{ color: "var(--fidelity-good)" }} /> 2 of 2 claims traced to source
            </div>
          </motion.div>
        </div>
      </div>
    </Reveal>
  );
}

export default function AgentPlatformPage() {
  return (
    <PlatformPageShell>
      <PlatformHero
        breadcrumb="Platform Overview / Agent"
        title={<>An agent that would rather <span style={{ color: "var(--accent-taupe)" }}>abstain</span> than guess.</>}
        description="A tool-using agent loops through search, read, compute, and verify — bound to your documents by output gates that can't be talked around. Every figure traces to a clause or cell, or it doesn't ship."
      />

      <section id="how-it-works" className="relative z-10" style={{ paddingBottom: "clamp(60px, 8vw, 100px)" }}>
        <div className="section-container">
          <AgentDemo />
        </div>
      </section>

      <section className="relative z-10" style={{ paddingBottom: "clamp(60px, 8vw, 100px)" }}>
        <div className="section-container">
          <Reveal className="text-center mb-12">
            <p className="eyebrow mb-4">The tool loop</p>
            <h2 className="font-display font-light" style={{ fontSize: "clamp(30px, 4vw, 46px)", lineHeight: "1.08", letterSpacing: "-0.03em", color: "var(--ink)" }}>
              Five verified tools. No free-text math.
            </h2>
          </Reveal>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
            {TOOLS.map((t, i) => (
              <Reveal key={t.name} delay={i * 0.05}>
                <div className="h-full rounded-2xl p-5 flex flex-col gap-3" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
                  <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: "var(--ink)", color: "var(--on-ink)" }}>
                    <t.icon size={17} strokeWidth={1.7} />
                  </div>
                  <code className="text-[12.5px] font-semibold" style={{ color: "var(--ink)", fontFamily: "var(--font-mono, monospace)" }}>{t.name}</code>
                  <p className="text-[12px] leading-relaxed" style={{ color: "var(--ink-3)" }}>{t.detail}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className="relative z-10" style={{ paddingBottom: "clamp(80px, 10vw, 130px)" }}>
        <div className="section-container">
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
