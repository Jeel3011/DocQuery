"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import { FileText, Brain, Shield, Check, Search, Layers, GitMerge } from "lucide-react";

/* ── Brain-story scenarios ──
   Each scenario walks: collection of docs → a hard cross-doc question →
   the Brain visibly reasoning (route → read each doc → synthesize → verify)
   → a synthesized answer → a tighter follow-up cross-question → a crisp
   follow-up answer. Finance leads and renders a real table, because finance
   users live in tables. Then legal (hard clause reasoning), then research
   (cross-paper synthesis). The left sidebar tracks the active collection. */

type Doc = { name: string; meta: string };
type ReasonStep = { icon: React.ElementType; label: string };
type TableBlock = { headers: string[]; rows: string[][] };
type Scenario = {
  kind: "finance" | "legal" | "research";
  collection: string;
  collectionMeta: string;        // e.g. "3 docs"
  docs: Doc[];
  question: string;
  steps: ReasonStep[];
  table?: TableBlock;
  answer: string;          // streamed after the table
  flag?: string;           // optional risk line, rendered red
  cites: { doc: string; loc: string }[];
  followQ: string;         // tighter cross-question after the answer
  followA: string;         // crisp follow-up answer
};

/* Sidebar collections — the active one is whichever scenario is playing. */
const COLLECTIONS = [
  { kind: "finance", name: "Q4 Board Pack", meta: "3 docs" },
  { kind: "legal", name: "Vendor MSAs", meta: "3 docs" },
  { kind: "research", name: "Transformer Papers", meta: "3 docs" },
] as const;

const SCENARIOS: Scenario[] = [
  {
    kind: "finance",
    collection: "Q4 Board Pack",
    collectionMeta: "3 docs",
    docs: [
      { name: "Credit Agreement.pdf", meta: "48p" },
      { name: "Earnings Q4.pdf", meta: "32p" },
      { name: "Audit Notes.pdf", meta: "12p" },
    ],
    question:
      "Do all three financial covenants still hold after the Q4 EBITDA drop, and is the MAC clause at risk?",
    steps: [
      { icon: Search, label: "Locating covenant schedule + Q4 actuals" },
      { icon: Layers, label: "Reading §12.3–12.5 across 3 documents" },
      { icon: GitMerge, label: "Reconciling thresholds vs reported figures" },
    ],
    table: {
      headers: ["Covenant", "Threshold", "Actual", "Status"],
      rows: [
        ["Net Leverage", "≤ 4.5×", "4.1×", "PASS"],
        ["Interest Coverage", "≥ 2.0×", "2.4×", "PASS"],
        ["Min. Liquidity", "≥ ₹50 Cr", "₹68 Cr", "PASS"],
      ],
    },
    answer:
      "All three covenants hold for Q4. But the MAC clause needs watching:",
    flag:
      "§18.1 triggers at ≥20% EBITDA decline YoY. Q4 fell 18% — only 2 pts from breach. Monitor Q1.",
    cites: [
      { doc: "Credit Agreement.pdf", loc: "§12.3–12.5" },
      { doc: "Earnings Q4.pdf", loc: "p.9 table" },
      { doc: "Credit Agreement.pdf", loc: "§18.1" },
    ],
    followQ: "If EBITDA falls another 5%, which covenant breaks first?",
    followA: "Leverage. A further 5% EBITDA drop pushes net leverage to ~4.7× — breaching §12.3 before the MAC clause trips. The 0.40× headroom is the binding constraint.",
  },
  {
    kind: "legal",
    collection: "Vendor MSAs",
    collectionMeta: "3 docs",
    docs: [
      { name: "MSA-Acme.pdf", meta: "22p" },
      { name: "MSA-Beta.pdf", meta: "18p" },
      { name: "MSA-Gamma.pdf", meta: "26p" },
    ],
    question:
      "Compare indemnification caps across all three MSAs and flag any unlimited liability exposure.",
    steps: [
      { icon: Search, label: "Extracting liability + indemnity clauses" },
      { icon: Layers, label: "Aligning caps across 3 agreements" },
      { icon: GitMerge, label: "Scoring exposure relative to deal size" },
    ],
    table: {
      headers: ["Agreement", "Indemnity", "Liability cap"],
      rows: [
        ["MSA-Acme", "Mutual", "1× annual fees"],
        ["MSA-Beta", "Vendor-only", "₹10 L flat"],
        ["MSA-Gamma", "Mutual", "Uncapped (IP)"],
      ],
    },
    answer:
      "Caps diverge sharply. Two items need negotiation:",
    flag:
      "MSA-Gamma §11.4 leaves IP-infringement liability uncapped — unlimited exposure. MSA-Beta's flat ₹10 L cap is disproportionate to a ₹4 Cr deal.",
    cites: [
      { doc: "MSA-Acme.pdf", loc: "Cl. 9.2" },
      { doc: "MSA-Beta.pdf", loc: "Cl. 7.1" },
      { doc: "MSA-Gamma.pdf", loc: "Cl. 11.4" },
    ],
    followQ: "Which single clause is our largest unhedged risk?",
    followA: "MSA-Gamma §11.4 — uncapped IP-infringement liability with no aggregate ceiling. It's the only clause with theoretically unbounded exposure; cap it at 2× fees first.",
  },
  {
    kind: "research",
    collection: "Transformer Papers",
    collectionMeta: "3 docs",
    docs: [
      { name: "Attention.pdf", meta: "12p" },
      { name: "BERT.pdf", meta: "16p" },
      { name: "GPT-3.pdf", meta: "40p" },
    ],
    question:
      "How did the role of attention change from the original Transformer through BERT to GPT-3?",
    steps: [
      { icon: Search, label: "Finding attention sections in each paper" },
      { icon: Layers, label: "Tracing the mechanism across 3 papers" },
      { icon: GitMerge, label: "Synthesising the evolution narrative" },
    ],
    answer:
      "Attention went from a parallelism trick to the backbone of scale: the Transformer made self-attention fully parallel (§3.2), BERT made it bidirectional for pre-training (§3), and GPT-3 showed the same mechanism scales to 175B params and few-shot learning (§2.1).",
    cites: [
      { doc: "Attention.pdf", loc: "§3.2" },
      { doc: "BERT.pdf", loc: "§3" },
      { doc: "GPT-3.pdf", loc: "§2.1" },
    ],
    followQ: "So what was the single inflection point?",
    followA: "Removing recurrence. Once self-attention made sequence modelling fully parallel (Transformer §3.2), scale became cheap — and everything after BERT and GPT-3 is a consequence of that one architectural choice.",
  },
];

type Phase = "collect" | "ask" | "reason" | "answer" | "followAsk" | "followAnswer";

export function HeroChatPreview() {
  const rm = useReducedMotion();
  const [si, setSi] = useState(0);
  const [phase, setPhase] = useState<Phase>("collect");
  const [typed, setTyped] = useState("");
  const [stepIdx, setStepIdx] = useState(-1);    // which reasoning step is active
  const [readDoc, setReadDoc] = useState(-1);    // which doc the Brain is scanning
  const [showTable, setShowTable] = useState(false);
  const [ansChars, setAnsChars] = useState(0);
  const [showFlag, setShowFlag] = useState(false);
  const [showCites, setShowCites] = useState(false);
  const [followTyped, setFollowTyped] = useState("");
  const [followChars, setFollowChars] = useState(0);
  const cancel = useRef(false);

  const S = SCENARIOS[si];

  useEffect(() => {
    cancel.current = false;
    const wait = (ms: number) =>
      new Promise<void>((res) => setTimeout(() => !cancel.current && res(), rm ? Math.min(ms, 180) : ms));

    async function run() {
      while (!cancel.current) {
        for (let s = 0; s < SCENARIOS.length && !cancel.current; s++) {
          const sc = SCENARIOS[s];
          setSi(s);
          // 1) collection appears
          setPhase("collect");
          setTyped(""); setStepIdx(-1); setReadDoc(-1);
          setShowTable(false); setAnsChars(0); setShowFlag(false); setShowCites(false);
          setFollowTyped(""); setFollowChars(0);
          await wait(1400);
          if (cancel.current) return;

          // 2) question types
          setPhase("ask");
          for (let i = 1; i <= sc.question.length && !cancel.current; i++) {
            setTyped(sc.question.slice(0, i));
            await wait(rm ? 0 : 18);
          }
          await wait(450);
          if (cancel.current) return;

          // 3) Brain reasons — steps light up, docs scanned in parallel
          setPhase("reason");
          for (let st = 0; st < sc.steps.length && !cancel.current; st++) {
            setStepIdx(st);
            setReadDoc(st < sc.docs.length ? st : sc.docs.length - 1);
            await wait(820);
          }
          setReadDoc(sc.docs.length); // all read
          setStepIdx(sc.steps.length);
          await wait(500);
          if (cancel.current) return;

          // 4) answer: table first (if any), then streamed text, then flag, then cites
          setPhase("answer");
          if (sc.table) {
            setShowTable(true);
            await wait(700);
          }
          for (let i = 1; i <= sc.answer.length && !cancel.current; i++) {
            setAnsChars(i);
            await wait(rm ? 0 : 13);
          }
          if (sc.flag) { setShowFlag(true); await wait(400); }
          setShowCites(true);
          await wait(1600);
          if (cancel.current) return;

          // 5) tighter follow-up cross-question types
          setPhase("followAsk");
          for (let i = 1; i <= sc.followQ.length && !cancel.current; i++) {
            setFollowTyped(sc.followQ.slice(0, i));
            await wait(rm ? 0 : 20);
          }
          await wait(420);
          if (cancel.current) return;

          // 6) crisp follow-up answer streams
          setPhase("followAnswer");
          for (let i = 1; i <= sc.followA.length && !cancel.current; i++) {
            setFollowChars(i);
            await wait(rm ? 0 : 13);
          }
          await wait(3600);
        }
      }
    }
    run();
    return () => { cancel.current = true; };
  }, [rm]);

  const ease = [0.23, 1, 0.32, 1] as const;
  const reasoning = phase === "reason";

  return (
    <div className="flex" style={{ minHeight: "480px" }}>
      {/* ── Mini sidebar — tracks the active collection per scenario ── */}
      <div
        className="hidden md:flex flex-col gap-2 p-4 shrink-0"
        style={{ width: "200px", borderRight: "1px solid var(--glass-border)", background: "rgba(244,244,244,0.50)" }}
      >
        <p className="eyebrow px-2 mb-1">Collections</p>
        {COLLECTIONS.map((col, i) => {
          const activeCol = i === si;
          return (
            <div
              key={col.name}
              className="flex items-center gap-2 px-3 py-2 rounded-xl"
              style={{
                background: activeCol ? "rgba(255,255,255,0.85)" : "transparent",
                border: activeCol ? "1px solid var(--line)" : "1px solid transparent",
                boxShadow: activeCol ? "var(--shadow-sm)" : "none",
                transition: "background 260ms ease, border-color 260ms ease, box-shadow 260ms ease",
              }}
            >
              <FileText size={11} style={{ color: activeCol ? "var(--ink)" : "var(--ink-3)" }} />
              <div className="min-w-0">
                <p className="text-[11px] font-medium truncate"
                  style={{ color: activeCol ? "var(--ink)" : "var(--ink-2)" }}>
                  {col.name}
                </p>
                <p className="text-[10px]" style={{ color: "var(--ink-3)" }}>{col.meta}</p>
              </div>
            </div>
          );
        })}

        {/* Capability pills */}
        <div className="mt-auto pt-4 flex flex-col gap-1.5">
          {[
            { icon: Brain, label: "Brain synthesis" },
            { icon: Shield, label: "Cited answers" },
            { icon: Search, label: "Sub-2s retrieval" },
          ].map(({ icon: Icon, label }) => (
            <div key={label} className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg">
              <Icon size={10} style={{ color: "var(--ink-3)" }} />
              <span className="text-[10px] font-medium" style={{ color: "var(--ink-3)" }}>{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Chat column ── */}
      <div className="flex-1 min-w-0 flex flex-col min-h-[440px]" style={{ color: "var(--ink)" }}>
      {/* ── Brain status strip ── */}
      <div className="px-4 pt-3.5 pb-2 flex items-center gap-2">
        <div
          className="w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{
            background: reasoning ? "var(--ink)" : "linear-gradient(165deg,#FFFFFF,#F0F0F0)",
            color: reasoning ? "#fff" : "var(--ink-2)",
            border: "1px solid rgba(0,0,0,0.08)",
            boxShadow: "var(--skeu-raised)",
            transition: "background 200ms ease, color 200ms ease",
          }}
        >
          <Brain size={13} className={reasoning ? "animate-pulse" : ""} />
        </div>
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="text-[11px] font-semibold truncate" style={{ color: "var(--ink)" }}>{S.collection}</span>
          <span className="text-[10px]" style={{ color: "var(--ink-3)" }}>·</span>
          <span className="text-[10px]" style={{ color: "var(--ink-3)" }}>{S.docs.length} documents</span>
        </div>
        <AnimatePresence>
          {reasoning && (
            <motion.span
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="ml-auto text-[10px] font-medium flex items-center gap-1"
              style={{ color: "var(--ink-2)" }}
            >
              <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: "var(--ink)" }} />
              Brain working…
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {/* ── Document strip — the collection the Brain reads across ── */}
      <div className="px-4 pb-2 flex gap-1.5">
        {S.docs.map((d, i) => {
          const scanning = reasoning && readDoc === i;
          const done = (reasoning && readDoc > i) || phase === "answer" || phase === "followAsk" || phase === "followAnswer";
          return (
            <motion.div
              key={`${si}-${d.name}`}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * i, ease }}
              className="relative flex-1 rounded-lg px-2 py-1.5 overflow-hidden"
              style={{
                background: "linear-gradient(165deg,#FFFFFF,#FAFAFA)",
                border: scanning ? "1px solid var(--ink)" : "1px solid var(--line)",
                boxShadow: scanning
                  ? "0 4px 14px -6px rgba(0,0,0,0.28)"
                  : "0 2px 6px -4px rgba(0,0,0,0.14), inset 0 1px 0 rgba(255,255,255,0.9)",
                transition: "border-color 200ms ease, box-shadow 200ms ease",
              }}
            >
              {scanning && !rm && (
                <motion.div
                  className="absolute inset-y-0 left-0 w-1/2"
                  style={{ background: "linear-gradient(90deg, transparent, rgba(0,0,0,0.07), transparent)" }}
                  initial={{ x: "-100%" }}
                  animate={{ x: "200%" }}
                  transition={{ duration: 0.8, ease: "linear", repeat: Infinity }}
                />
              )}
              <div className="flex items-center gap-1.5 relative">
                <FileText size={10} style={{ color: "var(--ink-3)" }} className="flex-shrink-0" />
                <span className="text-[9px] font-medium truncate" style={{ color: "var(--ink)" }}>{d.name}</span>
                {done && <Check size={10} style={{ color: "var(--ink)" }} className="ml-auto flex-shrink-0" />}
              </div>
              <span className="text-[8px] relative" style={{ color: "var(--ink-3)" }}>{d.meta} · indexed</span>
            </motion.div>
          );
        })}
      </div>

      <div className="flex-1 px-4 py-1 flex flex-col gap-2.5 overflow-hidden">
        {/* ── Question bubble (right) ── */}
        <AnimatePresence>
          {phase !== "collect" && (
            <motion.div
              key={`q-${si}`}
              initial={{ opacity: 0, x: 14, scale: 0.97 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              transition={{ duration: 0.25, ease }}
              className="flex justify-end"
            >
              <div
                className="max-w-[86%] px-3.5 py-2 rounded-2xl rounded-br-sm text-[13px] leading-snug"
                style={{
                  background: "linear-gradient(165deg, var(--ink-2), var(--ink))",
                  color: "var(--on-ink)",
                  boxShadow: "0 6px 16px -6px rgba(0,0,0,0.40), inset 0 1px 0 rgba(255,255,255,0.12)",
                }}
              >
                {phase === "ask" ? (
                  <>
                    {typed}
                    <span className="inline-block w-[2px] h-[12px] bg-white/70 ml-px translate-y-[2px]"
                      style={{ animation: "cursor-blink 0.7s ease-in-out infinite" }} />
                  </>
                ) : S.question}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Reasoning trace (Brain working) ── */}
        <AnimatePresence>
          {phase === "reason" && (
            <motion.div
              key={`r-${si}`}
              initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              transition={{ ease }}
              className="flex flex-col gap-1 max-w-[88%] px-3 py-2.5 rounded-2xl rounded-tl-sm"
              style={{
                background: "linear-gradient(165deg, var(--surface), var(--surface-2))",
                border: "1px solid var(--line)",
                boxShadow: "0 6px 16px -6px rgba(0,0,0,0.12), inset 0 1px 0 rgba(255,255,255,0.9)",
              }}
            >
              {S.steps.map((st, i) => {
                const active = stepIdx === i;
                const complete = stepIdx > i;
                const Icon = st.icon;
                return (
                  <div key={i} className="flex items-center gap-2">
                    <div
                      className="w-4 h-4 rounded-md flex items-center justify-center flex-shrink-0"
                      style={{
                        background: complete ? "var(--ink)" : active ? "var(--ink)" : "var(--surface-3)",
                        color: complete || active ? "#fff" : "var(--ink-3)",
                        border: "1px solid rgba(0,0,0,0.08)",
                        transition: "background 200ms ease, color 200ms ease",
                      }}
                    >
                      {complete ? <Check size={9} /> : <Icon size={9} className={active ? "animate-pulse" : ""} />}
                    </div>
                    <span
                      className="text-[11px] leading-tight"
                      style={{
                        color: active ? "var(--ink)" : complete ? "var(--ink-2)" : "var(--ink-3)",
                        fontWeight: active ? 600 : 400,
                      }}
                    >
                      {st.label}
                    </span>
                  </div>
                );
              })}
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Answer (left) — persists through the follow-up exchange ── */}
        <AnimatePresence>
          {(phase === "answer" || phase === "followAsk" || phase === "followAnswer") && (
            <motion.div
              key={`a-${si}`}
              initial={{ opacity: 0, x: -12, scale: 0.97 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              transition={{ duration: 0.25, ease }}
              className="flex flex-col gap-2 max-w-[94%]"
            >
              {/* Result table (finance/legal) */}
              {S.table && showTable && (
                <motion.div
                  initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ ease }}
                  className="rounded-xl overflow-hidden"
                  style={{ border: "1px solid var(--line)", boxShadow: "var(--shadow-sm)" }}
                >
                  <table className="w-full border-collapse text-[10.5px]">
                    <thead>
                      <tr style={{ background: "var(--ink)" }}>
                        {S.table.headers.map((h, i) => (
                          <th key={i} className="text-left font-semibold px-2.5 py-1.5"
                            style={{ color: "var(--on-ink)", letterSpacing: "-0.01em" }}>
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {S.table.rows.map((row, ri) => (
                        <motion.tr
                          key={ri}
                          initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                          transition={{ delay: 0.12 * ri }}
                          style={{ background: ri % 2 ? "var(--surface-2)" : "var(--surface)" }}
                        >
                          {row.map((cell, ci) => {
                            const isPass = cell === "PASS";
                            const isRisk = cell === "AT RISK" || cell === "FAIL"
                              || cell.toLowerCase().includes("uncapped") || cell.includes("flat");
                            return (
                              <td key={ci} className="px-2.5 py-1.5"
                                style={{
                                  borderTop: "1px solid var(--line)",
                                  color: isPass ? "var(--ink)" : isRisk ? "var(--status-failed)" : "var(--ink-2)",
                                  fontWeight: ci === 0 || isPass || isRisk ? 600 : 400,
                                }}>
                                {isPass && <span className="mr-1">✓</span>}
                                {isRisk && cell === "AT RISK" && <span className="mr-1">⚠</span>}
                                {cell}
                              </td>
                            );
                          })}
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </motion.div>
              )}

              {/* Streamed answer text */}
              <div
                className="px-3.5 py-2.5 rounded-2xl rounded-tl-sm text-[13px] leading-relaxed"
                style={{
                  background: "linear-gradient(165deg, var(--surface), var(--surface-2))",
                  border: "1px solid var(--line)",
                  boxShadow: "0 6px 16px -6px rgba(0,0,0,0.12), inset 0 1px 0 rgba(255,255,255,0.90)",
                  color: "var(--ink-2)",
                }}
              >
                {S.answer.slice(0, ansChars)}
                {ansChars < S.answer.length && (
                  <span className="inline-block w-[2px] h-[12px] ml-0.5 translate-y-[2px]"
                    style={{ background: "var(--ink)", animation: "cursor-blink 0.7s ease-in-out infinite" }} />
                )}

                {/* Risk flag */}
                {S.flag && showFlag && ansChars >= S.answer.length && (
                  <motion.div
                    initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
                    className="mt-2 flex gap-1.5 text-[12px] leading-snug"
                    style={{ color: "var(--status-failed)" }}
                  >
                    <span className="flex-shrink-0">⚠</span>
                    <span>{S.flag}</span>
                  </motion.div>
                )}
              </div>

              {/* multi-source citations */}
              <AnimatePresence>
                {showCites && (
                  <motion.div
                    initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
                    transition={{ ease }}
                    className="flex flex-wrap gap-1.5 items-center"
                  >
                    <Shield size={10} style={{ color: "var(--ink)" }} />
                    {S.cites.map((c, i) => (
                      <span
                        key={i}
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[9px] font-medium"
                        style={{ background: "var(--surface)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
                      >
                        <span className="w-3 h-3 rounded-[3px] text-[7px] font-bold flex items-center justify-center"
                          style={{ background: "var(--ink)", color: "var(--on-ink)" }}>{i + 1}</span>
                        {c.doc} · {c.loc}
                      </span>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Follow-up cross-question (right) ── */}
        <AnimatePresence>
          {(phase === "followAsk" || phase === "followAnswer") && (
            <motion.div
              key={`fq-${si}`}
              initial={{ opacity: 0, x: 14, scale: 0.97 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              transition={{ duration: 0.25, ease }}
              className="flex justify-end"
            >
              <div
                className="max-w-[82%] px-3.5 py-2 rounded-2xl rounded-br-sm text-[13px] leading-snug"
                style={{
                  background: "linear-gradient(165deg, var(--ink-2), var(--ink))",
                  color: "var(--on-ink)",
                  boxShadow: "0 6px 16px -6px rgba(0,0,0,0.40), inset 0 1px 0 rgba(255,255,255,0.12)",
                }}
              >
                {phase === "followAsk" ? (
                  <>
                    {followTyped}
                    <span className="inline-block w-[2px] h-[12px] bg-white/70 ml-px translate-y-[2px]"
                      style={{ animation: "cursor-blink 0.7s ease-in-out infinite" }} />
                  </>
                ) : S.followQ}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Follow-up answer (left) ── */}
        <AnimatePresence>
          {phase === "followAnswer" && (
            <motion.div
              key={`fa-${si}`}
              initial={{ opacity: 0, x: -12, scale: 0.97 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              transition={{ duration: 0.25, ease }}
              className="px-3.5 py-2.5 rounded-2xl rounded-tl-sm text-[13px] leading-relaxed max-w-[90%]"
              style={{
                background: "linear-gradient(165deg, var(--surface), var(--surface-2))",
                border: "1px solid var(--line)",
                boxShadow: "0 6px 16px -6px rgba(0,0,0,0.12), inset 0 1px 0 rgba(255,255,255,0.90)",
                color: "var(--ink-2)",
              }}
            >
              {S.followA.slice(0, followChars)}
              {followChars < S.followA.length && (
                <span className="inline-block w-[2px] h-[12px] ml-0.5 translate-y-[2px]"
                  style={{ background: "var(--ink)", animation: "cursor-blink 0.7s ease-in-out infinite" }} />
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      </div>
    </div>
  );
}
