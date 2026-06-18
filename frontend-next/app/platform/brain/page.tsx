"use client";

import React, { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search, Layers, GitMerge, ShieldCheck,
  FileText, Check, Brain, ChevronRight,
  CheckCircle2, ShieldAlert, Zap,
} from "lucide-react";
import { Reveal } from "@/components/landing/Reveal";
import { PlatformPageShell, PlatformHero, PlatformCTA, ease } from "@/components/landing/PlatformPageShell";

/* ─────────────────────────────────────────────────────────────────────────── */
/* Pipeline stages — the four phases of every Brain run                        */
/* ─────────────────────────────────────────────────────────────────────────── */

const STAGES = [
  {
    id: "route",
    icon: Search,
    label: "Route",
    headline: "Route the query",
    detail:
      "The Brain reads the question and decides which documents are relevant, which retrieval strategy to use (dense / BM25 / hybrid), and whether the question needs a single-hop lookup or a multi-hop synthesis across documents.",
    badge: "Sub-100 ms",
  },
  {
    id: "read",
    icon: Layers,
    label: "Read",
    headline: "Read in parallel",
    detail:
      "Relevant documents are sent to parallel reader agents — each searching, extracting, and grounding independently. Tables come back as structured grids; clauses come back as typed spans. No cross-contamination between readers.",
    badge: "Map phase",
  },
  {
    id: "synthesise",
    icon: GitMerge,
    label: "Synthesise",
    headline: "Synthesise the findings",
    detail:
      "A reducer merges the reader outputs into a single coherent answer, resolving conflicts by source authority (filed document beats secondary reference) and reconciling any numeric discrepancies explicitly in the output.",
    badge: "Reduce phase",
  },
  {
    id: "verify",
    icon: ShieldCheck,
    label: "Verify",
    headline: "Verify every claim",
    detail:
      "A non-bypassable output gate entails each claim against its grounding evidence. Claims that can't be fully traced are withheld. The answer you see is everything that survived — nothing that didn't.",
    badge: "Gate · 0 wrong shipped",
  },
] as const;

type StageId = (typeof STAGES)[number]["id"];

/* ─────────────────────────────────────────────────────────────────────────── */
/* Animated demo                                                                */
/* ─────────────────────────────────────────────────────────────────────────── */

const DOCS = [
  { name: "Credit Agreement.pdf", meta: "48 p" },
  { name: "Earnings Q4.pdf", meta: "32 p" },
  { name: "Audit Notes.pdf", meta: "12 p" },
];

const ANSWER =
  "All three covenants hold for Q4. Net leverage 4.1× (≤ 4.5× threshold), Interest Coverage 2.4× (≥ 2.0×), Min. Liquidity ₹68 Cr (≥ ₹50 Cr). The MAC clause is 2 pts from breach — monitor Q1.";

const TABLE = {
  headers: ["Covenant", "Threshold", "Actual", "Status"],
  rows: [
    ["Net Leverage", "≤ 4.5×", "4.1×", "PASS"],
    ["Interest Coverage", "≥ 2.0×", "2.4×", "PASS"],
    ["Min. Liquidity", "≥ ₹50 Cr", "₹68 Cr", "PASS"],
  ],
};

const CITES = [
  { doc: "Credit Agreement.pdf", loc: "§12.3–12.5" },
  { doc: "Earnings Q4.pdf", loc: "p. 9 table" },
  { doc: "Credit Agreement.pdf", loc: "§18.1" },
];

type Phase = "idle" | "route" | "read" | "synthesise" | "verify" | "done";

const PHASE_ORDER: Phase[] = ["idle", "route", "read", "synthesise", "verify", "done"];
const PHASE_DURATIONS: Record<Phase, number> = {
  idle: 1200,
  route: 1400,
  read: 2200,
  synthesise: 1600,
  verify: 1400,
  done: 3800,
};

function BrainDemo() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [readDoc, setReadDoc] = useState(-1);
  const [ansChars, setAnsChars] = useState(0);
  const [showTable, setShowTable] = useState(false);
  const [showCites, setShowCites] = useState(false);
  const cancelRef = useRef(false);

  const activeStageIdx =
    phase === "route" ? 0
    : phase === "read" ? 1
    : phase === "synthesise" ? 2
    : phase === "verify" ? 3
    : phase === "done" ? 4
    : -1;

  useEffect(() => {
    cancelRef.current = false;
    const wait = (ms: number) =>
      new Promise<void>((res) => setTimeout(() => { if (!cancelRef.current) res(); }, ms));

    async function run() {
      while (!cancelRef.current) {
        // reset
        setPhase("idle");
        setReadDoc(-1);
        setAnsChars(0);
        setShowTable(false);
        setShowCites(false);
        await wait(PHASE_DURATIONS.idle);

        // route
        setPhase("route");
        await wait(PHASE_DURATIONS.route);

        // read — scan docs one by one
        setPhase("read");
        for (let i = 0; i < DOCS.length && !cancelRef.current; i++) {
          setReadDoc(i);
          await wait(PHASE_DURATIONS.read / DOCS.length);
        }
        setReadDoc(DOCS.length);
        await wait(300);

        // synthesise
        setPhase("synthesise");
        setShowTable(true);
        await wait(PHASE_DURATIONS.synthesise);

        // verify + stream answer
        setPhase("verify");
        for (let i = 1; i <= ANSWER.length && !cancelRef.current; i++) {
          setAnsChars(i);
          await wait(14);
        }
        setShowCites(true);
        await wait(600);

        setPhase("done");
        await wait(PHASE_DURATIONS.done);
      }
    }

    run();
    return () => { cancelRef.current = true; };
  }, []);

  const isReading = phase === "read";
  const isAnswering = phase === "verify" || phase === "done";

  return (
    <div
      className="rounded-[22px] overflow-hidden"
      style={{
        background: "rgba(255,255,255,0.72)",
        backdropFilter: "blur(22px) saturate(1.0)",
        WebkitBackdropFilter: "blur(22px) saturate(1.0)",
        border: "1px solid rgba(255,255,255,0.80)",
        boxShadow: "var(--shadow-xl), inset 0 1px 0 rgba(255,255,255,0.90)",
      }}
    >
      {/* Browser chrome */}
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
        <div
          className="flex-1 mx-6 flex items-center justify-center gap-2 px-4 py-1.5 rounded-lg"
          style={{ background: "rgba(244,244,244,0.70)", border: "1px solid var(--line)" }}
        >
          <Brain size={10} className="opacity-40" />
          <span className="text-[11px] font-medium" style={{ color: "var(--ink-3)" }}>
            DocQuery Brain — Q4 Board Pack
          </span>
          <div
            className="w-1.5 h-1.5 rounded-full ml-1"
            style={{
              background: phase === "done" ? "var(--fidelity-good)" : "var(--ink-3)",
              opacity: 0.85,
            }}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[220px_1fr]" style={{ minHeight: 420 }}>

        {/* ── Left sidebar: pipeline stages ── */}
        <div
          className="flex flex-col p-4 gap-1"
          style={{ borderRight: "1px solid var(--glass-border)", background: "rgba(248,248,248,0.55)" }}
        >
          <p className="eyebrow px-2 mb-2">Pipeline</p>

          {STAGES.map((s, i) => {
            const done = activeStageIdx > i;
            const active = activeStageIdx === i;
            const Icon = s.icon;
            return (
              <div
                key={s.id}
                className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl transition-all"
                style={{
                  background: active ? "rgba(255,255,255,0.90)" : "transparent",
                  border: active ? "1px solid var(--line)" : "1px solid transparent",
                  boxShadow: active ? "var(--shadow-sm)" : "none",
                }}
              >
                <div
                  className="w-6 h-6 rounded-lg flex items-center justify-center shrink-0 transition-all"
                  style={{
                    background: done ? "var(--ink)" : active ? "var(--ink)" : "var(--surface-3)",
                    color: done || active ? "var(--on-ink)" : "var(--ink-3)",
                    border: "1px solid rgba(0,0,0,0.08)",
                  }}
                >
                  {done ? <Check size={11} /> : <Icon size={11} className={active ? "animate-pulse" : ""} />}
                </div>
                <span
                  className="text-[12px] font-medium"
                  style={{ color: active ? "var(--ink)" : done ? "var(--ink-2)" : "var(--ink-3)" }}
                >
                  {s.label}
                </span>
                {active && (
                  <span className="ml-auto flex gap-0.5">
                    {[0, 1, 2].map((d) => (
                      <motion.span
                        key={d}
                        className="w-1 h-1 rounded-full"
                        style={{ background: "var(--ink)" }}
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 1.2, repeat: Infinity, delay: d * 0.2 }}
                      />
                    ))}
                  </span>
                )}
              </div>
            );
          })}

          {/* Doc list */}
          <div className="mt-4 pt-3" style={{ borderTop: "1px solid var(--line)" }}>
            <p className="eyebrow px-2 mb-2">Documents</p>
            {DOCS.map((d, i) => {
              const scanning = isReading && readDoc === i;
              const done = (isReading && readDoc > i) || phase === "synthesise" || phase === "verify" || phase === "done";
              return (
                <div
                  key={d.name}
                  className="flex items-center gap-2 px-2 py-1.5 rounded-lg overflow-hidden relative"
                  style={{
                    border: scanning ? "1px solid var(--ink)" : "1px solid transparent",
                    background: scanning ? "rgba(255,255,255,0.80)" : "transparent",
                    transition: "border-color 200ms ease, background 200ms ease",
                  }}
                >
                  {scanning && (
                    <motion.div
                      className="absolute inset-y-0 left-0 w-1/2"
                      style={{ background: "linear-gradient(90deg, transparent, rgba(0,0,0,0.06), transparent)" }}
                      initial={{ x: "-100%" }}
                      animate={{ x: "200%" }}
                      transition={{ duration: 0.9, ease: "linear", repeat: Infinity }}
                    />
                  )}
                  <FileText size={10} style={{ color: "var(--ink-3)" }} className="shrink-0" />
                  <span className="text-[10px] truncate flex-1" style={{ color: scanning ? "var(--ink)" : "var(--ink-3)" }}>
                    {d.name}
                  </span>
                  {done && <Check size={9} style={{ color: "var(--ink)" }} className="shrink-0" />}
                </div>
              );
            })}
          </div>
        </div>

        {/* ── Right: chat area ── */}
        <div className="flex flex-col p-5 gap-3">

          {/* Question bubble */}
          <div className="flex justify-end">
            <div
              className="max-w-[85%] px-3.5 py-2.5 rounded-2xl rounded-br-sm text-[13px] leading-snug"
              style={{
                background: "linear-gradient(165deg, var(--ink-2), var(--ink))",
                color: "var(--on-ink)",
                boxShadow: "0 6px 16px -6px rgba(0,0,0,0.40), inset 0 1px 0 rgba(255,255,255,0.12)",
              }}
            >
              Do all three financial covenants still hold after the Q4 EBITDA drop, and is the MAC clause at risk?
            </div>
          </div>

          {/* Route phase */}
          <AnimatePresence>
            {(phase === "route") && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ ease }}
                className="flex items-center gap-2 px-3 py-2 rounded-xl text-[12px] max-w-[80%]"
                style={{
                  background: "var(--surface-2)",
                  border: "1px solid var(--line)",
                  color: "var(--ink-3)",
                }}
              >
                <Zap size={11} style={{ color: "var(--ink-2)" }} className="animate-pulse" />
                Routing query across 3 documents — financial covenant check, multi-hop
              </motion.div>
            )}
          </AnimatePresence>

          {/* Read phase */}
          <AnimatePresence>
            {isReading && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ ease }}
                className="flex items-center gap-2 px-3 py-2 rounded-xl text-[12px] max-w-[90%]"
                style={{
                  background: "var(--surface-2)",
                  border: "1px solid var(--line)",
                  color: "var(--ink-3)",
                }}
              >
                <Layers size={11} style={{ color: "var(--ink-2)" }} className="animate-pulse" />
                Reading {readDoc < DOCS.length ? `${DOCS[readDoc]?.name}` : "all documents"} in parallel…
              </motion.div>
            )}
          </AnimatePresence>

          {/* Synthesise: table appears */}
          <AnimatePresence>
            {(phase === "synthesise" || phase === "verify" || phase === "done") && showTable && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35, ease }}
                className="rounded-xl overflow-hidden"
                style={{ border: "1px solid var(--line)", boxShadow: "var(--shadow-sm)" }}
              >
                <table className="w-full border-collapse text-[11px]">
                  <thead>
                    <tr style={{ background: "var(--ink)" }}>
                      {TABLE.headers.map((h, i) => (
                        <th key={i} className="text-left font-semibold px-2.5 py-1.5"
                          style={{ color: "var(--on-ink)", letterSpacing: "-0.01em" }}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {TABLE.rows.map((row, ri) => (
                      <motion.tr
                        key={ri}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.1 * ri }}
                        style={{ background: ri % 2 ? "var(--surface-2)" : "var(--surface)" }}
                      >
                        {row.map((cell, ci) => {
                          const isPass = cell === "PASS";
                          return (
                            <td key={ci} className="px-2.5 py-1.5"
                              style={{
                                borderTop: "1px solid var(--line)",
                                color: isPass ? "var(--fidelity-good)" : "var(--ink-2)",
                                fontWeight: ci === 0 || isPass ? 600 : 400,
                              }}>
                              {isPass ? "✓ PASS" : cell}
                            </td>
                          );
                        })}
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Verify: streamed answer */}
          <AnimatePresence>
            {isAnswering && (
              <motion.div
                initial={{ opacity: 0, x: -10, scale: 0.98 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                transition={{ duration: 0.25, ease }}
                className="flex flex-col gap-2 max-w-[94%]"
              >
                <div
                  className="px-3.5 py-2.5 rounded-2xl rounded-tl-sm text-[13px] leading-relaxed"
                  style={{
                    background: "linear-gradient(165deg, var(--surface), var(--surface-2))",
                    border: "1px solid var(--line)",
                    boxShadow: "0 6px 16px -6px rgba(0,0,0,0.10), inset 0 1px 0 rgba(255,255,255,0.90)",
                    color: "var(--ink-2)",
                  }}
                >
                  {ANSWER.slice(0, ansChars)}
                  {ansChars < ANSWER.length && (
                    <span
                      className="inline-block w-[2px] h-[12px] ml-0.5 translate-y-[2px]"
                      style={{ background: "var(--ink)", animation: "cursor-blink 0.7s ease-in-out infinite" }}
                    />
                  )}
                </div>

                {/* Citations */}
                <AnimatePresence>
                  {showCites && (
                    <motion.div
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ ease }}
                      className="flex flex-wrap gap-1.5 items-center"
                    >
                      <ShieldAlert size={10} style={{ color: "var(--fidelity-good)" }} />
                      {CITES.map((c, i) => (
                        <span
                          key={i}
                          className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[9px] font-medium"
                          style={{ background: "var(--surface)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
                        >
                          <span
                            className="w-3 h-3 rounded-[3px] text-[7px] font-bold flex items-center justify-center"
                            style={{ background: "var(--ink)", color: "var(--on-ink)" }}
                          >
                            {i + 1}
                          </span>
                          {c.doc} · {c.loc}
                        </span>
                      ))}
                      <span className="text-[9px]" style={{ color: "var(--fidelity-good)" }}>
                        3/3 claims traced
                      </span>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Idle state */}
          {phase === "idle" && (
            <div className="flex-1 flex items-center justify-center">
              <div className="flex flex-col items-center gap-3 opacity-40">
                <Brain size={28} strokeWidth={1.4} style={{ color: "var(--ink-3)" }} />
                <span className="text-[12px]" style={{ color: "var(--ink-3)" }}>
                  Brain initialising…
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Stage deep-dive cards                                                        */
/* ─────────────────────────────────────────────────────────────────────────── */

function StageCards() {
  const [active, setActive] = useState<StageId>("route");
  const current = STAGES.find((s) => s.id === active)!;
  const Icon = current.icon;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-6">
      {/* Stage list */}
      <div className="flex flex-col gap-1">
        {STAGES.map((s) => {
          const SIcon = s.icon;
          const isActive = active === s.id;
          return (
            <button
              key={s.id}
              onClick={() => setActive(s.id)}
              className="flex items-center gap-3 px-4 py-3.5 rounded-xl text-left transition-all"
              style={{
                background: isActive ? "var(--ink)" : "transparent",
                border: isActive ? "1px solid var(--ink)" : "1px solid var(--line)",
                color: isActive ? "var(--on-ink)" : "var(--ink-2)",
              }}
            >
              <SIcon size={15} strokeWidth={1.7} />
              <span className="font-semibold text-[14px]" style={{ letterSpacing: "-0.01em" }}>
                {s.label}
              </span>
              <ChevronRight
                size={14}
                className="ml-auto transition-opacity"
                style={{ opacity: isActive ? 0.6 : 0.25 }}
              />
            </button>
          );
        })}
      </div>

      {/* Detail pane */}
      <AnimatePresence mode="wait">
        <motion.div
          key={active}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.22, ease }}
          className="rounded-[20px] p-8 flex flex-col gap-5"
          style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-md)" }}
        >
          <div className="flex items-center gap-3">
            <div
              className="w-11 h-11 rounded-xl flex items-center justify-center shrink-0"
              style={{ background: "var(--ink)", color: "var(--on-ink)" }}
            >
              <Icon size={20} strokeWidth={1.6} />
            </div>
            <div>
              <p className="font-semibold text-[17px]" style={{ color: "var(--ink)", letterSpacing: "-0.02em" }}>
                {current.headline}
              </p>
              <span
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium mt-0.5"
                style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-3)" }}
              >
                {current.badge}
              </span>
            </div>
          </div>
          <p className="text-[15px] leading-relaxed" style={{ color: "var(--ink-2)" }}>
            {current.detail}
          </p>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Feature cards                                                                */
/* ─────────────────────────────────────────────────────────────────────────── */

const FEATURES = [
  {
    title: "Map-reduce at any scale",
    description:
      "The router fans out to as many parallel readers as there are relevant documents. A single-hop question on one file and a 50-document synthesis run through the same pipeline — the Brain scales, you don't reconfigure anything.",
  },
  {
    title: "Non-bypassable output gates",
    description:
      "The verify stage is not a prompt instruction. It's a structural gate: every claim in the answer is matched against the evidence ledger produced by the reader agents. If the ledger doesn't support the claim, the claim is dropped — not softened.",
  },
  {
    title: "Abstain over guess",
    description:
      "When the evidence is insufficient, the Brain says so explicitly — with the specific gap identified. A partial answer with an honest abstain note is always more useful than a confident hallucination.",
  },
  {
    title: "Live SSE step events",
    description:
      "Every pipeline transition fires a Server-Sent Event — so the UI shows Route → Read → Synthesise → Verify as it happens, not a spinner followed by a wall of text. You watch the Brain reason.",
  },
  {
    title: "Finance and legal grounded differently",
    description:
      "Financial answers are grounded to extracted grid cells (row × column × source page). Legal answers are grounded to typed clause spans. The same Brain, different grounding registries — because numbers and clauses aren't the same kind of evidence.",
  },
  {
    title: "Cross-document conflict resolution",
    description:
      "When two documents disagree on a figure, the reduce phase surfaces the conflict explicitly and resolves by source authority — the filed document wins over the deck, the signed agreement wins over the draft.",
  },
];

/* ─────────────────────────────────────────────────────────────────────────── */
/* Page                                                                         */
/* ─────────────────────────────────────────────────────────────────────────── */

export default function BrainPlatformPage() {
  return (
    <PlatformPageShell>
      <PlatformHero
        breadcrumb="Platform Overview / Brain"
        title={
          <>
            Route. Read.{" "}
            <span style={{ color: "var(--accent-taupe)" }}>Verify.</span>
            <br />
            Never guess.
          </>
        }
        description="The Brain is a map-reduce orchestration layer. It routes every query, fans out across relevant documents in parallel, synthesises the findings, then runs a non-bypassable output gate before anything reaches you."
      />

      {/* Live demo */}
      <section className="relative z-10" style={{ paddingBottom: "clamp(60px, 8vw, 100px)" }}>
        <div className="section-container">
          <Reveal delay={0.1}>
            <BrainDemo />
          </Reveal>
        </div>
      </section>

      {/* Pipeline stage deep-dive */}
      <section
        className="relative z-10"
        style={{
          paddingTop: "clamp(60px, 8vw, 100px)",
          paddingBottom: "clamp(60px, 8vw, 100px)",
          background: "var(--surface-2)",
          borderTop: "1px solid var(--line)",
          borderBottom: "1px solid var(--line)",
        }}
      >
        <div className="section-container">
          <Reveal className="mb-12">
            <p className="eyebrow mb-4">The pipeline</p>
            <h2
              className="font-display font-light"
              style={{ fontSize: "clamp(30px, 4vw, 46px)", lineHeight: "1.08", letterSpacing: "-0.03em", color: "var(--ink)" }}
            >
              Four stages. Every run.
            </h2>
          </Reveal>
          <Reveal delay={0.08}>
            <StageCards />
          </Reveal>
        </div>
      </section>

      {/* Feature grid */}
      <section className="relative z-10" style={{ paddingTop: "clamp(60px, 8vw, 100px)", paddingBottom: "clamp(80px, 10vw, 130px)" }}>
        <div className="section-container">
          <Reveal className="text-center mb-14">
            <p className="eyebrow mb-4">Why it&apos;s different</p>
            <h2
              className="font-display font-light"
              style={{ fontSize: "clamp(30px, 4vw, 46px)", lineHeight: "1.08", letterSpacing: "-0.03em", color: "var(--ink)" }}
            >
              Designed to be wrong less.
            </h2>
          </Reveal>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {FEATURES.map((f, i) => (
              <Reveal key={f.title} delay={i * 0.04}>
                <div
                  className="h-full rounded-[20px] p-7"
                  style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-md)" }}
                >
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center mb-4"
                    style={{ background: "var(--surface-3)", border: "1px solid var(--line)" }}
                  >
                    <CheckCircle2 size={15} style={{ color: "var(--ink-2)" }} strokeWidth={1.7} />
                  </div>
                  <h3
                    className="font-semibold mb-2.5"
                    style={{ color: "var(--ink)", fontSize: "15px", letterSpacing: "-0.01em" }}
                  >
                    {f.title}
                  </h3>
                  <p className="leading-relaxed" style={{ color: "var(--ink-2)", fontSize: "13.5px" }}>
                    {f.description}
                  </p>
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
