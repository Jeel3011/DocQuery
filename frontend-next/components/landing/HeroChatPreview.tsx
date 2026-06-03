"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import { FileText, Brain, Shield, Check } from "lucide-react";

/* ── Brain-story scenarios ──
   Each scenario walks: collection of docs → cross-doc question → Brain reading
   across all docs → synthesized answer with multi-source citations. */
type Doc = { name: string; meta: string };
type Scenario = {
  collection: string;
  docs: Doc[];
  question: string;
  answer: string;
  cites: { doc: string; loc: string }[];
};

const SCENARIOS: Scenario[] = [
  {
    collection: "Q4 Board Pack",
    docs: [
      { name: "Earnings.pdf", meta: "32p" },
      { name: "Credit Agreement.pdf", meta: "48p" },
      { name: "Audit Notes.pdf", meta: "12p" },
    ],
    question: "Do our covenants still hold given the Q4 EBITDA drop?",
    answer:
      "Net leverage rose to 4.1× — still inside the 4.5× covenant (§12.3). But the 18% EBITDA decline is within 2pts of triggering the Material Adverse Change clause (§18.1).",
    cites: [
      { doc: "Credit Agreement.pdf", loc: "§12.3" },
      { doc: "Earnings.pdf", loc: "p.9" },
      { doc: "Credit Agreement.pdf", loc: "§18.1" },
    ],
  },
  {
    collection: "ML Research",
    docs: [
      { name: "Attention.pdf", meta: "12p" },
      { name: "BERT.pdf", meta: "16p" },
      { name: "GPT-3.pdf", meta: "40p" },
    ],
    question: "How did scaling change the role of attention across these papers?",
    answer:
      "Attention introduced parallelizable self-attention; BERT showed bidirectional pre-training; GPT-3 demonstrated the same mechanism scales to few-shot learning at 175B parameters.",
    cites: [
      { doc: "Attention.pdf", loc: "§3.2" },
      { doc: "BERT.pdf", loc: "§3" },
      { doc: "GPT-3.pdf", loc: "§2" },
    ],
  },
];

type Phase = "collect" | "ask" | "read" | "answer";

export function HeroChatPreview() {
  const rm = useReducedMotion();
  const [si, setSi] = useState(0);
  const [phase, setPhase] = useState<Phase>("collect");
  const [typed, setTyped] = useState("");
  const [readDoc, setReadDoc] = useState(-1); // which doc the Brain is scanning
  const [ansChars, setAnsChars] = useState(0);
  const [showCites, setShowCites] = useState(false);
  const cancel = useRef(false);

  const S = SCENARIOS[si];

  useEffect(() => {
    cancel.current = false;
    const wait = (ms: number) =>
      new Promise<void>((res) => setTimeout(() => !cancel.current && res(), rm ? Math.min(ms, 200) : ms));

    async function run() {
      while (!cancel.current) {
        for (let s = 0; s < SCENARIOS.length && !cancel.current; s++) {
          const sc = SCENARIOS[s];
          setSi(s);
          // 1) collection appears
          setPhase("collect");
          setTyped("");
          setReadDoc(-1);
          setAnsChars(0);
          setShowCites(false);
          await wait(1600);
          if (cancel.current) return;

          // 2) question types
          setPhase("ask");
          for (let i = 1; i <= sc.question.length && !cancel.current; i++) {
            setTyped(sc.question.slice(0, i));
            await wait(rm ? 0 : 26);
          }
          await wait(500);
          if (cancel.current) return;

          // 3) Brain reads across each doc
          setPhase("read");
          for (let d = 0; d < sc.docs.length && !cancel.current; d++) {
            setReadDoc(d);
            await wait(640);
          }
          setReadDoc(sc.docs.length); // all read
          await wait(500);
          if (cancel.current) return;

          // 4) synthesized answer streams
          setPhase("answer");
          for (let i = 1; i <= sc.answer.length && !cancel.current; i++) {
            setAnsChars(i);
            await wait(rm ? 0 : 14);
          }
          setShowCites(true);
          await wait(4200);
        }
      }
    }
    run();
    return () => { cancel.current = true; };
  }, [rm]);

  const ease = [0.23, 1, 0.32, 1] as const;

  return (
    <div className="flex flex-col text-[var(--text-primary)] min-h-[420px]">
      {/* ── Brain status strip ── */}
      <div className="px-4 pt-3.5 pb-2 flex items-center gap-2">
        <div
          className="w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{
            background: phase === "read" ? "var(--accent)" : "linear-gradient(165deg,#FFFFFF,#F1EEE9)",
            color: phase === "read" ? "#fff" : "var(--text-secondary)",
            border: "1px solid rgba(0,0,0,0.06)",
            boxShadow: "var(--skeu-raised)",
            transition: "background 200ms ease, color 200ms ease",
          }}
        >
          <Brain size={13} className={phase === "read" ? "animate-pulse" : ""} />
        </div>
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="text-[11px] font-semibold text-[var(--text-primary)] truncate">{S.collection}</span>
          <span className="text-[10px] text-[var(--text-muted)]">·</span>
          <span className="text-[10px] text-[var(--text-muted)]">{S.docs.length} documents</span>
        </div>
        <AnimatePresence>
          {phase === "read" && (
            <motion.span
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="ml-auto text-[10px] font-medium text-[var(--text-secondary)] flex items-center gap-1"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
              synthesizing…
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {/* ── Document strip — the collection the Brain reads across ── */}
      <div className="px-4 pb-2 flex gap-1.5">
        {S.docs.map((d, i) => {
          const reading = phase === "read" && readDoc === i;
          const done = (phase === "read" && readDoc > i) || phase === "answer";
          return (
            <motion.div
              key={`${si}-${d.name}`}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.12 * i, ease }}
              className="relative flex-1 rounded-lg px-2 py-1.5 overflow-hidden"
              style={{
                background: "linear-gradient(165deg,#FFFFFF,#FBFAF7)",
                border: reading ? "1px solid var(--accent)" : "1px solid rgba(0,0,0,0.07)",
                boxShadow: reading
                  ? "0 4px 14px -6px rgba(0,0,0,0.25)"
                  : "0 2px 6px -4px rgba(40,30,20,0.14), inset 0 1px 0 rgba(255,255,255,0.9)",
                transition: "border-color 200ms ease, box-shadow 200ms ease",
              }}
            >
              {/* scan sweep while reading */}
              {reading && !rm && (
                <motion.div
                  className="absolute inset-y-0 left-0 w-1/2"
                  style={{ background: "linear-gradient(90deg, transparent, rgba(0,0,0,0.06), transparent)" }}
                  initial={{ x: "-100%" }}
                  animate={{ x: "200%" }}
                  transition={{ duration: 0.64, ease: "linear", repeat: Infinity }}
                />
              )}
              <div className="flex items-center gap-1.5 relative">
                <FileText size={10} className="text-[var(--text-muted)] flex-shrink-0" />
                <span className="text-[9px] font-medium text-[var(--text-primary)] truncate">{d.name}</span>
                {done && <Check size={10} className="text-[var(--status-ready)] ml-auto flex-shrink-0" />}
              </div>
              <span className="text-[8px] text-[var(--text-muted)] relative">{d.meta} · indexed</span>
            </motion.div>
          );
        })}
      </div>

      <div className="flex-1 px-4 py-1 flex flex-col gap-2.5">
        {/* ── Question bubble (right) ── */}
        <AnimatePresence>
          {(phase === "ask" || phase === "read" || phase === "answer") && (
            <motion.div
              key={`q-${si}`}
              initial={{ opacity: 0, x: 14, scale: 0.97 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              transition={{ duration: 0.25, ease }}
              className="flex justify-end"
            >
              <div
                className="max-w-[82%] px-3.5 py-2 rounded-2xl rounded-br-sm text-[13px] leading-snug"
                style={{
                  background: "linear-gradient(165deg,#2A2A2A,#0A0A0A)",
                  color: "#fff",
                  boxShadow: "0 6px 16px -6px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.16)",
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

        {/* ── Answer (left) ── */}
        <AnimatePresence>
          {phase === "answer" && (
            <motion.div
              key={`a-${si}`}
              initial={{ opacity: 0, x: -12, scale: 0.97 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              transition={{ duration: 0.25, ease }}
              className="flex flex-col gap-1.5 max-w-[92%]"
            >
              <div
                className="px-3.5 py-2.5 rounded-2xl rounded-tl-sm text-[13px] leading-relaxed"
                style={{
                  background: "linear-gradient(165deg,#FFFFFF,#FAF9F6)",
                  border: "1px solid rgba(0,0,0,0.06)",
                  boxShadow: "0 6px 16px -6px rgba(40,30,20,0.14), inset 0 1px 0 rgba(255,255,255,0.9)",
                }}
              >
                {S.answer.slice(0, ansChars)}
                {ansChars < S.answer.length && (
                  <span className="inline-block w-[2px] h-[12px] bg-[var(--accent)] ml-0.5 translate-y-[2px]"
                    style={{ animation: "cursor-blink 0.7s ease-in-out infinite" }} />
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
                    <Shield size={10} className="text-[var(--status-ready)]" />
                    {S.cites.map((c, i) => (
                      <span
                        key={i}
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[9px] font-medium text-[var(--text-secondary)]"
                        style={{ background: "rgba(255,255,255,0.85)", border: "1px solid rgba(0,0,0,0.07)" }}
                      >
                        <span className="w-3 h-3 rounded-[3px] bg-[var(--accent)] text-white text-[7px] font-bold flex items-center justify-center">{i + 1}</span>
                        {c.doc} · {c.loc}
                      </span>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
