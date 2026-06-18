"use client";

import React, { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowRight } from "lucide-react";

/* ── Use-case terms — DocQuery's practice areas ──────────────────────────── */
const TERMS = [
  "Covenant Review",
  "Contract Analysis",
  "Due Diligence",
  "Earnings Q&A",
  "Clause Extraction",
  "Board Pack Review",
  "Regulatory Filings",
  "MSA Comparison",
  "Cap Table Review",
  "Legal Research",
  "Liability Mapping",
  "Audit Preparation",
];

/* How many items visible in the window (odd number so centre is exact) */
const VISIBLE = 7;
/* Height of one item in px */
const ITEM_H = 80;
/* Auto-advance interval in ms */
const INTERVAL = 1800;

export function UseCaseTicker() {
  const [active, setActive] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* Auto-scroll upward */
  useEffect(() => {
    timerRef.current = setInterval(() => {
      setActive((a) => (a + 1) % TERMS.length);
    }, INTERVAL);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, []);

  /* Pause on hover */
  const pause = () => { if (timerRef.current) clearInterval(timerRef.current); };
  const resume = () => {
    timerRef.current = setInterval(() => {
      setActive((a) => (a + 1) % TERMS.length);
    }, INTERVAL);
  };

  /* Build the visible window: VISIBLE items centred on active */
  const half = Math.floor(VISIBLE / 2);

  /* We duplicate the list so the window never runs out of items */
  const repeated = [...TERMS, ...TERMS, ...TERMS];
  /* Offset into the repeated array so active is always in the middle third */
  const base = TERMS.length + active - half;
  const window = repeated.slice(base, base + VISIBLE);

  return (
    <section
      className="relative z-10 overflow-hidden"
      style={{
        paddingTop: "clamp(80px, 10vw, 130px)",
        paddingBottom: "clamp(80px, 10vw, 130px)",
        background: "var(--surface-2)",
        borderTop: "1px solid var(--line)",
        borderBottom: "1px solid var(--line)",
      }}
    >
      <div className="section-container">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] items-center gap-12">

          {/* ── Left label ── */}
          <div className="flex flex-col gap-5 max-w-xs">
            <p
              className="font-display font-light"
              style={{
                fontSize: "clamp(22px, 2.8vw, 32px)",
                lineHeight: 1.2,
                letterSpacing: "-0.025em",
                color: "var(--ink)",
              }}
            >
              Finance and legal teams use DocQuery for
            </p>
            <Link
              href="/platform"
              className="inline-flex items-center gap-1.5 text-[13px] font-medium transition-opacity hover:opacity-60"
              style={{ color: "var(--ink-3)" }}
            >
              Explore platform
              <ArrowRight size={13} />
            </Link>
          </div>

          {/* ── Ticker ── */}
          <div
            className="relative select-none"
            style={{ height: ITEM_H * VISIBLE, width: "clamp(320px, 48vw, 680px)" }}
            onMouseEnter={pause}
            onMouseLeave={resume}
          >
            {/* Top + bottom fade masks */}
            <div
              className="absolute inset-x-0 top-0 z-10 pointer-events-none"
              style={{
                height: ITEM_H * 2.2,
                background: "linear-gradient(to bottom, var(--surface-2) 0%, transparent 100%)",
              }}
            />
            <div
              className="absolute inset-x-0 bottom-0 z-10 pointer-events-none"
              style={{
                height: ITEM_H * 2.2,
                background: "linear-gradient(to top, var(--surface-2) 0%, transparent 100%)",
              }}
            />

            {/* Items */}
            <div className="absolute inset-0 flex flex-col">
              {window.map((term, i) => {
                const distFromCenter = i - half;
                const isCenter = distFromCenter === 0;
                const abs = Math.abs(distFromCenter);

                /* Opacity: centre=1, ±1=0.28, ±2=0.13, ±3=0.06 */
                const opacity = isCenter ? 1 : abs === 1 ? 0.28 : abs === 2 ? 0.13 : 0.06;

                /* Font weight: centre bold, rest light */
                const weight = isCenter ? 600 : 300;

                /* Font size: centre slightly larger */
                const size = isCenter
                  ? "clamp(36px, 4.5vw, 58px)"
                  : abs === 1
                  ? "clamp(30px, 3.8vw, 50px)"
                  : "clamp(26px, 3.2vw, 42px)";

                return (
                  <div
                    key={`${term}-${i}`}
                    className="flex items-center shrink-0 transition-all"
                    style={{
                      height: ITEM_H,
                      opacity,
                      transition: "opacity 400ms ease",
                    }}
                  >
                    <span
                      className="font-display leading-none"
                      style={{
                        fontSize: size,
                        fontWeight: weight,
                        letterSpacing: "-0.03em",
                        color: "var(--ink)",
                        transition: "font-size 400ms ease, font-weight 400ms ease",
                      }}
                    >
                      {term}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

        </div>
      </div>
    </section>
  );
}
