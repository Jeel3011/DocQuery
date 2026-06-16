"use client";

import React, { useEffect, useRef } from "react";
import Link from "next/link";
import { motion, useReducedMotion } from "framer-motion";
import { ArrowRight, Sparkles, Shield, Zap, Brain, BookOpen } from "lucide-react";
import { HeroChatPreview } from "./HeroChatPreview";
import { Threads } from "@/components/ui/Threads";

const STATS = [
  { value: "0.96", label: "Answer relevancy", sub: "RAGAS eval" },
  { value: "1.00", label: "Context precision", sub: "RAGAS eval" },
  { value: "< 2s", label: "First answer token", sub: "p50 latency" },
  { value: "100+", label: "Docs per collection", sub: "concurrent" },
];

export function HeroSection() {
  const rm = useReducedMotion();
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (rm) return;
    const el = containerRef.current;
    if (!el) return;
    function onMove(e: MouseEvent) {
      const rect = el!.getBoundingClientRect();
      const cx = (e.clientX - rect.left) / rect.width - 0.5;
      const cy = (e.clientY - rect.top) / rect.height - 0.5;
      el!.style.setProperty("--px", String(cx));
      el!.style.setProperty("--py", String(cy));
    }
    el.addEventListener("mousemove", onMove, { passive: true });
    return () => el.removeEventListener("mousemove", onMove);
  }, [rm]);

  const ease = [0.23, 1, 0.32, 1] as const;

  return (
    <section
      ref={containerRef}
      className="relative flex flex-col items-center justify-center overflow-hidden"
      style={{
        paddingTop: "clamp(120px, 14vw, 180px)",
        paddingBottom: "clamp(80px, 10vw, 140px)",
        "--px": "0",
        "--py": "0",
      } as React.CSSProperties}
    >
      {/* ── Threads — ink-line texture, anchored to the very top of the hero so it's
          behind the eyebrow + headline first, fading out before the chat preview ── */}
      {!rm && (
        <div
          className="absolute left-0 right-0 top-0 z-0 pointer-events-none"
          style={{
            height: "min(120vh, 980px)",
            opacity: 0.5,
            maskImage: "linear-gradient(180deg, black 0%, black 60%, transparent 100%)",
            WebkitMaskImage: "linear-gradient(180deg, black 0%, black 60%, transparent 100%)",
          }}
          aria-hidden="true"
        >
          <Threads color={[0.07, 0.07, 0.07]} amplitude={1.4} distance={0.3} enableMouseInteraction />
        </div>
      )}

      {/* ── Centre-aligned copy block ── */}
      <div className="section-container w-full relative z-10">
        <div className="flex flex-col items-center text-center gap-8">

          {/* Eyebrow chip */}
          <motion.div
            initial={{ opacity: 0, y: rm ? 0 : 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease }}
            className="tag"
          >
            <Brain size={12} />
            Built for finance & legal · Every answer traced to a cell or clause
          </motion.div>

          {/* Headline — Fraunces display editorial */}
          <motion.div
            initial={{ opacity: 0, y: rm ? 0 : 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.05, ease }}
            className="flex flex-col items-center gap-0"
          >
            <h1
              className="font-display font-light text-center"
              style={{
                fontSize: "clamp(52px, 7.5vw, 100px)",
                lineHeight: "1.02",
                letterSpacing: "-0.03em",
                color: "var(--ink)",
                textWrap: "balance",
              } as React.CSSProperties}
            >
              Built for
            </h1>
            <h1
              className="font-display font-light text-center"
              style={{
                fontSize: "clamp(52px, 7.5vw, 100px)",
                lineHeight: "1.02",
                letterSpacing: "-0.03em",
                color: "var(--ink)",
                textWrap: "balance",
              } as React.CSSProperties}
            >
              <span style={{ color: "var(--accent-taupe)" }}>precision</span> work.
            </h1>
          </motion.div>

          {/* Body copy */}
          <motion.p
            initial={{ opacity: 0, y: rm ? 0 : 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, delay: 0.12, ease }}
            className="max-w-[600px]"
            style={{
              fontSize: "clamp(17px, 2vw, 20px)",
              lineHeight: "1.65",
              color: "var(--ink-2)",
              letterSpacing: "-0.01em",
            }}
          >
            Upload contracts, filings, board packs. DocQuery traces every answer back to its clause or cell — and abstains rather than guesses when it can't.
          </motion.p>

          {/* CTAs */}
          <motion.div
            initial={{ opacity: 0, y: rm ? 0 : 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2, ease }}
            className="flex flex-col sm:flex-row items-center gap-3"
          >
            <Link href="/login" className="btn-cta">
              Start for free
              <ArrowRight size={18} strokeWidth={2.2} />
            </Link>
            <a href="#how-it-works" className="btn-ghost-lg">
              See how it works
            </a>
          </motion.div>

          {/* Trust micro-copy */}
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.35, duration: 0.4 }}
            className="text-[13px]"
            style={{ color: "var(--ink-3)" }}
          >
            No credit card required · Free tier available · SOC 2 in progress
          </motion.p>

          {/* Stat grid */}
          <motion.div
            initial={{ opacity: 0, y: rm ? 0 : 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, delay: 0.28, ease }}
            className="w-full max-w-3xl grid grid-cols-2 md:grid-cols-4 mt-4"
            style={{
              borderTop: "1px solid var(--line)",
              borderLeft: "1px solid var(--line)",
              borderRadius: "20px",
              overflow: "hidden",
              background: "rgba(255,255,255,0.55)",
              backdropFilter: "blur(16px)",
              WebkitBackdropFilter: "blur(16px)",
              boxShadow: "var(--shadow-md)",
            }}
          >
            {STATS.map((s, i) => (
              <div
                key={i}
                className="flex flex-col items-center text-center py-7 px-5 gap-1.5"
                style={{
                  borderRight: "1px solid var(--line)",
                  borderBottom: "1px solid var(--line)",
                }}
              >
                <span
                  className="font-display"
                  style={{
                    fontSize: "clamp(26px, 3.5vw, 36px)",
                    fontWeight: 300,
                    letterSpacing: "-0.03em",
                    lineHeight: 1,
                    color: "var(--ink)",
                    fontFeatureSettings: '"tnum","ss01"',
                  }}
                >
                  {s.value}
                </span>
                <span
                  className="text-[13px] font-semibold leading-tight"
                  style={{ color: "var(--ink-2)", letterSpacing: "-0.01em" }}
                >
                  {s.label}
                </span>
                <span
                  className="eyebrow"
                  style={{ fontSize: "10px" }}
                >
                  {s.sub}
                </span>
              </div>
            ))}
          </motion.div>
        </div>

        {/* ── Chat preview window ── */}
        <motion.div
          initial={{ opacity: 0, y: rm ? 0 : 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.72, delay: 0.38, ease }}
          className="mt-16 w-full max-w-4xl mx-auto"
        >
          {/* Browser chrome */}
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
            {/* Title bar */}
            <div
              className="flex items-center gap-3 px-5 py-3.5"
              style={{
                background: "linear-gradient(180deg, rgba(255,255,255,0.90) 0%, rgba(255,255,255,0.55) 100%)",
                borderBottom: "1px solid var(--glass-border)",
              }}
            >
              <div className="flex gap-1.5">
                {["#C4C4C4","#A8A8A8","#8C8C8C"].map((c,i) => (
                  <div key={i} className="w-3 h-3 rounded-full opacity-90" style={{ background: c, boxShadow: `inset 0 1px 0 rgba(255,255,255,0.4)` }} />
                ))}
              </div>
              <div
                className="flex-1 mx-6 flex items-center justify-center gap-2 px-4 py-1.5 rounded-lg"
                style={{ background: "rgba(244,244,244,0.70)", border: "1px solid var(--line)" }}
              >
                <Shield size={10} className="opacity-40" />
                <span className="text-[11px] font-medium" style={{ color: "var(--ink-3)" }}>
                  DocQuery — live demo
                </span>
                <div className="w-1.5 h-1.5 rounded-full ml-1" style={{ background: "var(--status-ready)", opacity: 0.85 }} />
              </div>
              <div className="flex items-center gap-1.5" style={{ color: "var(--ink-3)" }}>
                {[Sparkles, BookOpen, Zap].map((Icon, i) => (
                  <div key={i} className="w-6 h-6 flex items-center justify-center rounded-md" style={{ background: "rgba(244,244,244,0.60)" }}>
                    <Icon size={11} />
                  </div>
                ))}
              </div>
            </div>

            {/* Live chat demo — sidebar + chat, both driven by the active scenario */}
            <HeroChatPreview />
          </div>
        </motion.div>
      </div>
    </section>
  );
}
