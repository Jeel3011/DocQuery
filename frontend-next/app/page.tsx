"use client";

import Link from "next/link";
import { HeroSection } from "@/components/landing/HeroSection";
import { BelowFold } from "@/components/landing/BelowFold";
import { motion } from "framer-motion";
import { useState } from "react";

function Nav() {
  const [open, setOpen] = useState(false);

  return (
    <header
      className="fixed top-0 w-full z-[200]"
      style={{
        background: "rgba(244,244,244,0.85)",
        backdropFilter: "blur(20px) saturate(1.0)",
        WebkitBackdropFilter: "blur(20px) saturate(1.0)",
        borderBottom: "1px solid var(--line)",
        boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
      }}
    >
      <div className="section-container">
        <div className="flex items-center justify-between h-[68px]">

          {/* Wordmark */}
          <Link href="/" className="flex items-center gap-2.5 shrink-0">
            <div
              className="w-8 h-8 rounded-[9px] flex items-center justify-center shadow-sm shrink-0"
              style={{ background: "var(--ink)" }}
            >
              <span
                className="font-display font-semibold text-base"
                style={{ color: "var(--on-ink)", letterSpacing: "-0.02em" }}
              >
                D
              </span>
            </div>
            <span
              className="font-display font-semibold text-[19px]"
              style={{ color: "var(--ink)", letterSpacing: "-0.025em" }}
            >
              DocQuery
            </span>
          </Link>

          {/* Desktop nav links */}
          <nav className="hidden lg:flex items-center gap-1">
            {[
              { label: "Product", href: "#product" },
              { label: "Accuracy", href: "#accuracy" },
              { label: "Pricing", href: "#pricing" },
              { label: "Security", href: "#security" },
            ].map(({ label: item, href }) => (
              <a
                key={item}
                href={href}
                className="px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                style={{ color: "var(--ink-2)" }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.background = "var(--accent-soft)";
                  (e.currentTarget as HTMLElement).style.color = "var(--ink)";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.background = "transparent";
                  (e.currentTarget as HTMLElement).style.color = "var(--ink-2)";
                }}
              >
                {item}
              </a>
            ))}
          </nav>

          {/* Desktop CTAs */}
          <div className="hidden lg:flex items-center gap-3">
            <Link
              href="/login"
              className="text-sm font-medium px-4 py-2 rounded-lg transition-colors"
              style={{ color: "var(--ink-2)" }}
            >
              Sign in
            </Link>
            <Link
              href="/login"
              className="btn-cta !py-[10px] !px-5 !text-[14px] !rounded-xl"
            >
              Get started free
            </Link>
          </div>

          {/* Mobile menu button */}
          <button
            className="lg:hidden p-2 rounded-lg"
            style={{ color: "var(--ink-2)" }}
            onClick={() => setOpen(!open)}
            aria-label="Toggle menu"
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              {open ? (
                <path d="M5 5l10 10M15 5l-10 10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
              ) : (
                <path d="M3 6h14M3 10h14M3 14h14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
              )}
            </svg>
          </button>
        </div>

        {/* Mobile dropdown */}
        {open && (
          <div
            className="lg:hidden pb-4 border-t"
            style={{ borderColor: "var(--line)" }}
          >
            <div className="pt-3 space-y-1">
              {[
                { label: "Product", href: "#product" },
                { label: "Accuracy", href: "#accuracy" },
                { label: "Pricing", href: "#pricing" },
                { label: "Security", href: "#security" },
              ].map(({ label: item, href }) => (
                <a
                  key={item}
                  href={href}
                  onClick={() => setOpen(false)}
                  className="block px-3 py-2.5 rounded-lg text-sm font-medium"
                  style={{ color: "var(--ink-2)" }}
                >
                  {item}
                </a>
              ))}
              <div className="pt-3 flex flex-col gap-2">
                <Link href="/login" className="text-sm font-medium text-center py-2.5" style={{ color: "var(--ink-2)" }}>Sign in</Link>
                <Link href="/login" className="btn-cta !text-[14px] !py-3 !rounded-xl text-center">Get started free</Link>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}

export default function RootPage() {
  return (
    <main
      className="relative min-h-screen"
      style={{ background: "var(--canvas)", color: "var(--ink)" }}
    >
      {/* Fixed aurora wash */}
      <div className="aurora aurora-fixed" />
      {/* Faint warm dot grid */}
      <div className="dot-grid fixed inset-0 z-0 opacity-[0.14] pointer-events-none" />

      <Nav />

      <div className="relative z-10">
        <HeroSection />
        <BelowFold />

        <footer
          className="py-16 relative z-10"
          style={{
            borderTop: "1px solid var(--line)",
            background: "var(--surface-2)",
          }}
        >
          <div className="section-container">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-2.5">
                <div
                  className="w-7 h-7 rounded-lg flex items-center justify-center"
                  style={{ background: "var(--ink)" }}
                >
                  <span className="font-display font-semibold text-sm" style={{ color: "var(--on-ink)" }}>D</span>
                </div>
                <span className="font-display font-semibold text-base" style={{ color: "var(--ink)", letterSpacing: "-0.02em" }}>
                  DocQuery
                </span>
              </div>
              <p className="text-sm" style={{ color: "var(--ink-3)" }}>
                © 2026 DocQuery. Precision document intelligence.
              </p>
              <div className="flex gap-6">
                {["Privacy", "Terms", "Security"].map((l) => (
                  <a key={l} href="#" className="text-sm transition-colors" style={{ color: "var(--ink-3)" }}>
                    {l}
                  </a>
                ))}
              </div>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}
