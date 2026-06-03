"use client";

import React, { useEffect, useRef } from "react";
import Link from "next/link";
import { motion, useReducedMotion } from "framer-motion";
import { ArrowRight, FileText, Sparkles, Shield, Zap } from "lucide-react";
import { HeroChatPreview } from "./HeroChatPreview";

/* ── Stat pill ── */
function StatPill({
  value,
  label,
  delay,
}: {
  value: string;
  label: string;
  delay: number;
}) {
  const rm = useReducedMotion();
  return (
    <motion.div
      initial={{ opacity: 0, y: rm ? 0 : 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
      className="flex flex-col items-center gap-0.5"
    >
      <span className="text-base font-bold tracking-tight text-[var(--text-primary)] whitespace-nowrap">
        {value}
      </span>
      <span className="text-[11px] text-[var(--text-muted)] whitespace-nowrap">{label}</span>
    </motion.div>
  );
}

export function HeroSection() {
  const rm = useReducedMotion();
  const containerRef = useRef<HTMLDivElement>(null);

  /* Subtle parallax on mouse-move — CSS variable, no state churn */
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

  return (
    <section
      ref={containerRef}
      className="relative min-h-[92vh] flex items-center justify-center pt-24 pb-16 px-6 lg:px-12 overflow-hidden"
      style={{ "--px": "0", "--py": "0" } as React.CSSProperties}
    >
      {/* (Floating doc cards removed — the live demo window now tells the full
          doc-collection story on its own, so extra cards just clutter.) */}

      <div className="max-w-7xl w-full grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-16 items-center relative z-10">

        {/* ── Left: copy + CTAs ── */}
        <motion.div
          initial={{ opacity: 0, y: rm ? 0 : 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, ease: [0.23, 1, 0.32, 1] }}
          className="flex flex-col items-center lg:items-start text-center lg:text-left gap-7"
        >
          {/* Badge */}
          <div className="glass inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-[11px] font-semibold text-[var(--text-secondary)]">
            <Sparkles size={11} className="text-[var(--accent)]" />
            Multi-document reasoning · Brain synthesis
          </div>

          {/* Headline */}
          <h1 className="text-5xl lg:text-6xl xl:text-[68px] font-bold tracking-tight text-[var(--text-primary)] leading-[1.08]"
            style={{ textWrap: "balance" } as React.CSSProperties}
          >
            Ask anything across<br />
            <span className="relative inline-block">
              your documents.
              {/* Subtle underline accent */}
              <svg
                aria-hidden="true"
                className="absolute -bottom-1 left-0 w-full"
                height="4"
                viewBox="0 0 300 4"
                fill="none"
                preserveAspectRatio="none"
              >
                <path
                  d="M0 2 Q75 0 150 2 Q225 4 300 2"
                  stroke="var(--accent)"
                  strokeWidth="2"
                  strokeLinecap="round"
                  opacity="0.25"
                />
              </svg>
            </span>
          </h1>

          {/* Body */}
          <p className="text-lg text-[var(--text-secondary)] max-w-lg leading-relaxed">
            Upload PDFs, contracts, research papers. DocQuery reads across all of them simultaneously — returning cited, verifiable answers in seconds.
          </p>

          {/* Feature pills */}
          <div className="flex flex-wrap gap-2 justify-center lg:justify-start">
            {[
              { icon: Shield, label: "Source citations" },
              { icon: Zap, label: "Sub-second retrieval" },
              { icon: FileText, label: "100+ doc collections" },
            ].map(({ icon: Icon, label }) => (
              <div
                key={label}
                className="glass-sm inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-medium text-[var(--text-secondary)]"
              >
                <Icon size={11} className="text-[var(--text-muted)]" />
                {label}
              </div>
            ))}
          </div>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row items-center gap-3 w-full sm:w-auto mt-2">
            <Link
              href="/login"
              className="btn-primary flex items-center justify-center gap-2 w-full sm:w-auto px-7 py-3.5 text-sm"
            >
              Start for free
              <ArrowRight size={16} />
            </Link>
            <a
              href="#how-it-works"
              className="btn-ghost flex items-center justify-center gap-2 w-full sm:w-auto px-7 py-3.5 text-sm"
              style={{ boxShadow: "var(--skeu-raised)" }}
            >
              See how it works
            </a>
          </div>

          <p className="text-xs text-[var(--text-muted)]">
            No credit card required. Free tier available.
          </p>

          {/* Capability row — honest claims, not fabricated metrics */}
          <div
            className="flex items-center gap-8 pt-4 mt-1 border-t border-[var(--border)]"
          >
            <StatPill value="PDF" label="contracts · papers" delay={0.9} />
            <div className="w-px h-8 bg-[var(--border)]" />
            <StatPill value="Cited" label="every answer" delay={1.0} />
            <div className="w-px h-8 bg-[var(--border)]" />
            <StatPill value="Multi-doc" label="reasoning" delay={1.1} />
          </div>
        </motion.div>

        {/* ── Right: glass chat window ──
            NOTE: no transform on the glass-bearing element. A transformed
            ancestor establishes a containing block that disables
            `backdrop-filter` against the page in Chrome/Safari, so the parallax
            translate lives only on the decorative halo below — the frame itself
            stays untransformed and blurs correctly. */}
        <motion.div
          initial={{ opacity: 0, y: rm ? 0 : 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.65, delay: 0.15, ease: [0.23, 1, 0.32, 1] }}
          className="flex justify-center lg:justify-end"
        >
          <div className="relative w-full max-w-[480px]">
            {/* Glass halo behind window — carries the parallax so the frame can't */}
            <div
              aria-hidden="true"
              className="absolute -inset-6 rounded-3xl"
              style={{
                background:
                  "radial-gradient(ellipse at 60% 40%, rgba(0,0,0,0.04) 0%, transparent 70%)",
                transform: `translate(calc(var(--px, 0) * -10px), calc(var(--py, 0) * -8px))`,
                transition: "transform 0.12s linear",
              }}
            />
            {/* Outer glass frame */}
            <div
              className="relative rounded-2xl overflow-hidden glass-strong"
            >
              {/* Traffic-light title bar — translucent strip over the frosted frame */}
              <div
                className="flex items-center gap-2 px-4 py-3 border-b"
                style={{
                  background: "linear-gradient(180deg, rgba(255,255,255,0.45), rgba(255,255,255,0.15))",
                  borderColor: "var(--glass-border)",
                }}
              >
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-[var(--status-failed)] opacity-70" style={{ boxShadow: "inset 0 1px 0 rgba(255,255,255,0.4)" }} />
                  <div className="w-3 h-3 rounded-full bg-[var(--status-processing)] opacity-70" style={{ boxShadow: "inset 0 1px 0 rgba(255,255,255,0.4)" }} />
                  <div className="w-3 h-3 rounded-full bg-[var(--status-ready)] opacity-70" style={{ boxShadow: "inset 0 1px 0 rgba(255,255,255,0.4)" }} />
                </div>
                <div
                  className="flex-1 text-center text-[10px] font-medium text-[var(--text-muted)] flex items-center justify-center gap-1.5"
                >
                  <div className="w-2 h-2 rounded-full bg-[var(--status-ready)] animate-pulse" />
                  DocQuery — live demo
                </div>
              </div>

              <HeroChatPreview />
            </div>
          </div>
        </motion.div>

      </div>
    </section>
  );
}
