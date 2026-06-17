"use client";

import { Shield, BookOpen, Lock, Server, CheckCircle2, ArrowRight, Upload, Search, FileCheck, KeyRound, EyeOff, ScrollText } from "lucide-react";
import { Reveal } from "./Reveal";
import Link from "next/link";

const TRUST_ITEMS = [
  {
    icon: Lock,
    title: "No training on your data",
    desc: "Your documents never leave to train any model. Every query is inference-only. Your IP stays yours.",
    detail: "Zero data retention requested on every upstream LLM call.",
  },
  {
    icon: BookOpen,
    title: "Every answer is cited",
    desc: "Citations link to exact passages in your corpus. No hallucinated facts, no unverifiable claims.",
    detail: "A verifier entails each claim against its source before you see it.",
  },
  {
    icon: Shield,
    title: "Audit trail built in",
    desc: "Every query, reasoning step, and source is logged for compliance, review, and export.",
    detail: "Brain coverage ledger records docs routed / read / relevant per run.",
  },
  {
    icon: Server,
    title: "Deploy on your infrastructure",
    desc: "Run on your own AWS, GCP, or on-prem environment. Full data residency guaranteed.",
    detail: "Containerised — VPC-isolated, no egress beyond the LLM endpoint you choose.",
  },
];

/* How a document is handled, end to end — the data-flow users actually care about. */
const DATA_FLOW = [
  {
    icon: Upload,
    title: "On upload",
    desc: "Parsed in your environment, encrypted at rest. Prompt-injection patterns in document text are detected and neutralised before any LLM call.",
  },
  {
    icon: Search,
    title: "On query",
    desc: "Retrieval runs over your vectors only. Context is sanitised, inference-only, with zero-retention requested upstream. Nothing is used for training.",
  },
  {
    icon: FileCheck,
    title: "On answer",
    desc: "Each claim is entailment-checked against its source; unsupported claims are dropped. The query, sources, and reasoning steps are written to an exportable audit log.",
  },
];

/* Concrete posture — grouped, not a flat chip list. */
const POSTURE = [
  {
    icon: KeyRound,
    heading: "Access & isolation",
    points: ["Per-collection access controls", "Row-level security on every table", "JWT auth, scoped tokens", "No cross-tenant data paths"],
  },
  {
    icon: EyeOff,
    heading: "Data handling",
    points: ["Inference-only LLM calls", "Zero-retention upstream", "Prompt-injection sanitisation", "No data leaves your VPC"],
  },
  {
    icon: ScrollText,
    heading: "Compliance posture",
    points: ["SOC 2 Type II in progress", "GDPR & India DPDP aware", "Exportable audit logs", "Configurable data retention"],
  },
];

export function TrustStrip() {
  return (
    <>
      {/* ── Trust grid ── */}
      <section
        id="security"
        className="relative z-10 scroll-mt-24"
        style={{
          paddingTop: "clamp(80px, 10vw, 140px)",
          paddingBottom: "clamp(80px, 10vw, 140px)",
        }}
      >
        <div className="section-container">
          <Reveal className="mb-14 max-w-2xl">
            <p className="eyebrow mb-4">Security & trust</p>
            <h2
              className="font-display font-light mb-5"
              style={{
                fontSize: "clamp(36px, 5vw, 58px)",
                lineHeight: "1.05",
                letterSpacing: "-0.03em",
                color: "var(--ink)",
              }}
            >
              Built for
              <br />
              regulated industries.
            </h2>
            <p className="text-[17px] leading-relaxed" style={{ color: "var(--ink-2)", letterSpacing: "-0.01em" }}>
              Finance, legal, and research teams handle privileged material. DocQuery is engineered so that trust is structural — not a setting you have to remember to turn on.
            </p>
          </Reveal>

          <Reveal className="mb-12">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
              {TRUST_ITEMS.map((item, i) => {
                const Icon = item.icon;
                return (
                  <div
                    key={i}
                    className="h-full flex flex-col gap-4 rounded-[20px] p-7"
                    style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-sm)" }}
                  >
                    <div
                      className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
                      style={{ background: "var(--ink)", color: "var(--on-ink)" }}
                    >
                      <Icon size={17} strokeWidth={1.7} />
                    </div>
                    <h3 className="font-semibold text-[15px]" style={{ color: "var(--ink)", letterSpacing: "-0.01em" }}>
                      {item.title}
                    </h3>
                    <p className="text-[13px] leading-relaxed" style={{ color: "var(--ink-2)" }}>
                      {item.desc}
                    </p>
                    <p className="text-[12px] leading-relaxed mt-auto pt-3" style={{ color: "var(--ink-3)", borderTop: "1px solid var(--line)" }}>
                      {item.detail}
                    </p>
                  </div>
                );
              })}
            </div>
          </Reveal>

          {/* ── Data-flow: how a document is handled, end to end ── */}
          <Reveal className="mb-12">
            <div
              className="rounded-[24px] overflow-hidden"
              style={{ background: "var(--surface-2)", border: "1px solid var(--line)" }}
            >
              <div className="px-7 py-5" style={{ borderBottom: "1px solid var(--line)" }}>
                <p className="eyebrow">How your data is handled</p>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x" style={{ borderColor: "var(--line)" }}>
                {DATA_FLOW.map((step, i) => {
                  const Icon = step.icon;
                  return (
                    <div key={i} className="p-7 flex flex-col gap-3">
                      <div className="flex items-center gap-2.5">
                        <div
                          className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
                          style={{ background: "var(--ink)", color: "var(--on-ink)" }}
                        >
                          <Icon size={15} strokeWidth={1.8} />
                        </div>
                        <span className="font-semibold text-[14px]" style={{ color: "var(--ink)", letterSpacing: "-0.01em" }}>
                          {step.title}
                        </span>
                      </div>
                      <p className="text-[13px] leading-relaxed" style={{ color: "var(--ink-2)" }}>
                        {step.desc}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          </Reveal>

          {/* ── Posture: access / handling / compliance ── */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {POSTURE.map((group, i) => {
              const Icon = group.icon;
              return (
                <Reveal key={i} delay={i * 0.07}>
                  <div
                    className="h-full p-7 rounded-[20px]"
                    style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-sm)" }}
                  >
                    <div className="flex items-center gap-2.5 mb-5">
                      <div
                        className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
                        style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
                      >
                        <Icon size={16} strokeWidth={1.7} />
                      </div>
                      <span className="font-semibold text-[14px]" style={{ color: "var(--ink)", letterSpacing: "-0.01em" }}>
                        {group.heading}
                      </span>
                    </div>
                    <ul className="flex flex-col gap-2.5">
                      {group.points.map((p) => (
                        <li key={p} className="flex items-start gap-2 text-[13px]" style={{ color: "var(--ink-2)" }}>
                          <CheckCircle2 size={14} style={{ color: "var(--ink-3)", marginTop: 2 }} className="shrink-0" />
                          {p}
                        </li>
                      ))}
                    </ul>
                  </div>
                </Reveal>
              );
            })}
          </div>
        </div>
      </section>

      {/* ── Big CTA ── */}
      <section
        className="relative z-10 overflow-hidden"
        style={{
          paddingTop: "clamp(80px, 10vw, 140px)",
          paddingBottom: "clamp(80px, 10vw, 140px)",
          background: "var(--ink)",
        }}
      >
        {/* Subtle glow */}
        <div
          aria-hidden="true"
          className="absolute inset-0 pointer-events-none"
          style={{
            background: "radial-gradient(ellipse 80% 60% at 50% 100%, rgba(255,255,255,0.10) 0%, transparent 70%)",
          }}
        />

        <div className="section-container relative">
          <Reveal>
            <div className="flex flex-col items-center text-center gap-8">
              <p
                className="eyebrow"
                style={{ color: "rgba(250,250,250,0.50)" }}
              >
                Get started
              </p>

              <h2
                className="font-display font-light max-w-3xl"
                style={{
                  fontSize: "clamp(40px, 6vw, 72px)",
                  lineHeight: "1.03",
                  letterSpacing: "-0.03em",
                  color: "var(--on-ink)",
                  textWrap: "balance",
                } as React.CSSProperties}
              >
                Your documents deserve
                <br />
                better than keyword search.
              </h2>

              <p
                className="max-w-lg text-[18px] leading-relaxed"
                style={{ color: "rgba(250,250,250,0.65)", letterSpacing: "-0.01em" }}
              >
                Stop hunting through folders. Ask DocQuery — and get a cited, verifiable answer in under two seconds.
              </p>

              <div className="flex flex-col sm:flex-row items-center gap-3 mt-2">
                <Link
                  href="/login"
                  className="inline-flex items-center gap-2 px-8 py-4 rounded-[14px] font-semibold text-[16px]"
                  style={{
                    background: "var(--on-ink)",
                    color: "var(--ink)",
                    boxShadow: "0 4px 20px -4px rgba(0,0,0,0.30)",
                    letterSpacing: "-0.01em",
                  }}
                >
                  Start for free
                  <ArrowRight size={18} strokeWidth={2.2} />
                </Link>
                <a
                  href="#how-it-works"
                  className="px-8 py-4 rounded-[14px] font-medium text-[16px] border"
                  style={{
                    color: "rgba(250,250,250,0.75)",
                    borderColor: "rgba(250,250,250,0.18)",
                    letterSpacing: "-0.01em",
                  }}
                >
                  See how it works
                </a>
              </div>

              <p
                className="text-[13px]"
                style={{ color: "rgba(250,250,250,0.40)" }}
              >
                No credit card required · Free tier available
              </p>
            </div>
          </Reveal>
        </div>
      </section>
    </>
  );
}
