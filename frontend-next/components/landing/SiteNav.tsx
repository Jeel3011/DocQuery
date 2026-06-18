"use client";

import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";
import { ArrowRight, Brain } from "lucide-react";

/* ── Platform mega-menu items ─────────────────────────────────────────────── */
const COL_LEFT = [
  {
    href: "/platform",
    title: "Overview",
    description: "How every pillar works together.",
  },
  {
    href: "/platform/brain",
    title: "Brain",
    description: "Map-reduce orchestration. Route → Read → Synthesise → Verify.",
  },
  {
    href: "/platform/vault",
    title: "Vault",
    description: "Securely store, organise, and verify documents with a live fidelity pipeline.",
  },
  {
    href: "/platform/agent",
    title: "Agent",
    description: "A tool-using agent that cites every figure to a clause or cell — or abstains.",
  },
];

const COL_RIGHT = [
  {
    href: "/platform/workflows",
    title: "Workflows",
    description: "Repeatable review, draft, and output playbooks across a vault.",
  },
  {
    href: "/#how-it-works",
    title: "Finance & Law",
    description: "Revenue concentration, covenant review, clause comparison — by practice area.",
  },
  {
    href: "/#security",
    title: "Security",
    description: "Inference-only LLM calls. No training on your data. Audit logs built in.",
  },
  {
    href: "/#pricing",
    title: "Pricing",
    description: "A genuinely useful free tier. Enterprise sold directly.",
  },
];

/* ── Feature card shown on the right of the mega-menu ────────────────────── */
const FEATURE_CARD = {
  icon: Brain,
  title: "DocQuery Brain",
  body: "Every answer traced to a cell or clause. Verify-or-abstain, never guess.",
  cta: "See the demo",
  href: "/#product",
};

const ease = [0.23, 1, 0.32, 1] as const;

/* ── Fullscreen mega-menu overlay ──────────────────────────────────────────── */
function MegaMenu({ onClose }: { onClose: () => void }) {
  /* Close on Escape */
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  /* Lock body scroll while open */
  useEffect(() => {
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = ""; };
  }, []);

  return (
    <motion.div
      key="mega"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.22, ease }}
      /* covers full viewport below the nav bar (68px) */
      className="fixed left-0 right-0 bottom-0 z-[190]"
      style={{ top: "68px", background: "rgba(10,10,10,0.97)" }}
    >
      {/* Click-away backdrop (the overlay itself acts as the backdrop) */}
      <div className="absolute inset-0" onClick={onClose} />

      {/* Content — sits above the backdrop */}
      <motion.div
        initial={{ opacity: 0, y: -16 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        transition={{ duration: 0.30, ease }}
        className="relative z-10 h-full flex flex-col"
      >
        <div
          className="overflow-y-auto"
          style={{ borderTop: "1px solid rgba(255,255,255,0.08)" }}
        >
          <div className="max-w-7xl mx-auto px-8 lg:px-16 pt-12 pb-16">
            <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr_320px] gap-x-16">

              {/* ── Column 1 ── */}
              <div className="flex flex-col">
                {COL_LEFT.map((item, i) => (
                  <motion.div
                    key={item.href}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.30, delay: 0.04 + i * 0.04, ease }}
                  >
                    <Link
                      href={item.href}
                      onClick={onClose}
                      className="group block py-4"
                      style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}
                    >
                      <p
                        className="font-display mb-1 group-hover:opacity-60 transition-opacity"
                        style={{
                          fontSize: "17px",
                          fontWeight: 500,
                          letterSpacing: "-0.02em",
                          color: "#FAFAFA",
                        }}
                      >
                        {item.title}
                      </p>
                      <p
                        className="text-[13px] leading-snug"
                        style={{ color: "rgba(250,250,250,0.38)" }}
                      >
                        {item.description}
                      </p>
                    </Link>
                  </motion.div>
                ))}
              </div>

              {/* ── Column 2 ── */}
              <div className="flex flex-col">
                {COL_RIGHT.map((item, i) => (
                  <motion.div
                    key={item.href}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.30, delay: 0.06 + i * 0.04, ease }}
                  >
                    <Link
                      href={item.href}
                      onClick={onClose}
                      className="group block py-4"
                      style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}
                    >
                      <p
                        className="font-display mb-1 group-hover:opacity-60 transition-opacity"
                        style={{
                          fontSize: "17px",
                          fontWeight: 500,
                          letterSpacing: "-0.02em",
                          color: "#FAFAFA",
                        }}
                      >
                        {item.title}
                      </p>
                      <p
                        className="text-[13px] leading-snug"
                        style={{ color: "rgba(250,250,250,0.38)" }}
                      >
                        {item.description}
                      </p>
                    </Link>
                  </motion.div>
                ))}
              </div>

              {/* ── Feature card ── */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.34, delay: 0.10, ease }}
                className="lg:pl-8 lg:border-l"
                style={{ borderColor: "rgba(255,255,255,0.08)" }}
              >
                <Link href={FEATURE_CARD.href} onClick={onClose} className="group block">
                  <div
                    className="rounded-2xl p-6 flex flex-col gap-5 transition-colors"
                    style={{
                      background: "rgba(255,255,255,0.05)",
                      border: "1px solid rgba(255,255,255,0.10)",
                    }}
                  >
                    {/* Icon */}
                    <div
                      className="w-10 h-10 rounded-xl flex items-center justify-center"
                      style={{
                        background: "rgba(255,255,255,0.10)",
                        border: "1px solid rgba(255,255,255,0.12)",
                      }}
                    >
                      <FEATURE_CARD.icon
                        size={18}
                        strokeWidth={1.6}
                        style={{ color: "rgba(250,250,250,0.75)" }}
                      />
                    </div>

                    {/* Copy */}
                    <div className="flex flex-col gap-1.5">
                      <p
                        className="font-display"
                        style={{
                          fontSize: "22px",
                          fontWeight: 500,
                          letterSpacing: "-0.02em",
                          color: "#FAFAFA",
                          lineHeight: 1.2,
                        }}
                      >
                        {FEATURE_CARD.title}
                      </p>
                      <p
                        className="text-[13px] leading-relaxed"
                        style={{ color: "rgba(250,250,250,0.45)" }}
                      >
                        {FEATURE_CARD.body}
                      </p>
                    </div>

                    {/* CTA */}
                    <div
                      className="flex items-center gap-1.5 text-[13px] font-medium transition-opacity group-hover:opacity-60 mt-1"
                      style={{ color: "rgba(250,250,250,0.65)" }}
                    >
                      {FEATURE_CARD.cta}
                      <ArrowRight size={13} className="group-hover:translate-x-0.5 transition-transform" />
                    </div>
                  </div>
                </Link>
              </motion.div>

            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}

/* ── SiteNav ────────────────────────────────────────────────────────────────── */
export function SiteNav() {
  const [megaOpen, setMegaOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const closeMega = () => setMegaOpen(false);

  return (
    <>
      <header
        className="fixed top-0 w-full z-[200]"
        style={{
          background: megaOpen
            ? "rgba(10,10,10,0.97)"
            : "rgba(244,244,244,0.85)",
          backdropFilter: megaOpen ? "none" : "blur(20px) saturate(1.0)",
          WebkitBackdropFilter: megaOpen ? "none" : "blur(20px) saturate(1.0)",
          borderBottom: "1px solid " + (megaOpen ? "rgba(255,255,255,0.08)" : "var(--line)"),
          boxShadow: megaOpen ? "none" : "0 1px 3px rgba(0,0,0,0.06)",
          transition: "background 220ms ease, border-color 220ms ease",
        }}
      >
        <div className="max-w-7xl mx-auto px-8 lg:px-16">
          <div className="flex items-center justify-between h-[68px]">

            {/* Wordmark */}
            <Link
              href="/"
              onClick={closeMega}
              className="flex items-center gap-2.5 shrink-0"
            >
              <div
                className="w-8 h-8 rounded-[9px] flex items-center justify-center shadow-sm shrink-0"
                style={{ background: megaOpen ? "#FAFAFA" : "var(--ink)" }}
              >
                <span
                  className="font-display font-semibold text-base"
                  style={{
                    color: megaOpen ? "#0E0E0E" : "var(--on-ink)",
                    letterSpacing: "-0.02em",
                  }}
                >
                  D
                </span>
              </div>
              <span
                className="font-display font-semibold text-[19px]"
                style={{
                  color: megaOpen ? "#FAFAFA" : "var(--ink)",
                  letterSpacing: "-0.025em",
                  transition: "color 220ms ease",
                }}
              >
                DocQuery
              </span>
            </Link>

            {/* Desktop nav */}
            <nav className="hidden lg:flex items-center gap-1">
              {/* Platform trigger */}
              <button
                onClick={() => setMegaOpen((o) => !o)}
                className="px-4 py-2 rounded-lg text-sm font-medium transition-colors inline-flex items-center gap-1.5"
                style={{
                  color: megaOpen ? "rgba(250,250,250,0.85)" : "var(--ink-2)",
                  background: "transparent",
                }}
              >
                Platform
                <svg
                  width="10"
                  height="10"
                  viewBox="0 0 10 10"
                  fill="none"
                  style={{
                    transform: megaOpen ? "rotate(180deg)" : "none",
                    transition: "transform 220ms cubic-bezier(0.23,1,0.32,1)",
                  }}
                >
                  <path
                    d="M2 3.5L5 6.5L8 3.5"
                    stroke="currentColor"
                    strokeWidth="1.4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>

              {/* Other links */}
              {[
                { label: "Product", href: "/#product" },
                { label: "Accuracy", href: "/#accuracy" },
                { label: "Pricing", href: "/#pricing" },
                { label: "Security", href: "/#security" },
              ].map(({ label, href }) => (
                <a
                  key={label}
                  href={href}
                  onClick={closeMega}
                  className="px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                  style={{ color: megaOpen ? "rgba(250,250,250,0.55)" : "var(--ink-2)" }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLElement).style.color = megaOpen
                      ? "rgba(250,250,250,0.85)"
                      : "var(--ink)";
                    if (!megaOpen)
                      (e.currentTarget as HTMLElement).style.background = "var(--accent-soft)";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLElement).style.color = megaOpen
                      ? "rgba(250,250,250,0.55)"
                      : "var(--ink-2)";
                    (e.currentTarget as HTMLElement).style.background = "transparent";
                  }}
                >
                  {label}
                </a>
              ))}
            </nav>

            {/* Desktop CTAs */}
            <div className="hidden lg:flex items-center gap-3">
              <Link
                href="/login"
                onClick={closeMega}
                className="text-sm font-medium px-4 py-2 rounded-lg transition-colors"
                style={{ color: megaOpen ? "rgba(250,250,250,0.60)" : "var(--ink-2)" }}
              >
                Sign in
              </Link>
              <Link
                href="/login"
                onClick={closeMega}
                className="inline-flex items-center gap-1.5 text-[14px] font-semibold px-5 py-[10px] rounded-xl transition-colors"
                style={
                  megaOpen
                    ? {
                        background: "#FAFAFA",
                        color: "#0E0E0E",
                      }
                    : {
                        background: "var(--ink)",
                        color: "var(--on-ink)",
                      }
                }
              >
                Get started free
              </Link>
            </div>

            {/* Mobile hamburger */}
            <button
              className="lg:hidden p-2 rounded-lg"
              style={{ color: megaOpen ? "rgba(250,250,250,0.70)" : "var(--ink-2)" }}
              onClick={() => { setMobileOpen((o) => !o); closeMega(); }}
              aria-label="Toggle menu"
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                {mobileOpen ? (
                  <path d="M5 5l10 10M15 5l-10 10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                ) : (
                  <path d="M3 6h14M3 10h14M3 14h14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                )}
              </svg>
            </button>
          </div>

          {/* Mobile dropdown */}
          {mobileOpen && (
            <div
              className="lg:hidden pb-4 border-t"
              style={{ borderColor: megaOpen ? "rgba(255,255,255,0.08)" : "var(--line)" }}
            >
              <div className="pt-3 space-y-1">
                <div
                  className="px-3 py-1.5 text-[10px] font-semibold tracking-wider uppercase"
                  style={{ color: megaOpen ? "rgba(250,250,250,0.35)" : "var(--ink-3)" }}
                >
                  Platform
                </div>
                {[...COL_LEFT, ...COL_RIGHT].map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    onClick={() => setMobileOpen(false)}
                    className="block px-3 py-2.5 rounded-lg text-sm font-medium"
                    style={{ color: "var(--ink-2)" }}
                  >
                    {item.title}
                  </Link>
                ))}
                <div className="my-1 border-t" style={{ borderColor: "var(--line)" }} />
                {[
                  { label: "Product", href: "/#product" },
                  { label: "Accuracy", href: "/#accuracy" },
                  { label: "Pricing", href: "/#pricing" },
                  { label: "Security", href: "/#security" },
                ].map(({ label, href }) => (
                  <a
                    key={label}
                    href={href}
                    onClick={() => setMobileOpen(false)}
                    className="block px-3 py-2.5 rounded-lg text-sm font-medium"
                    style={{ color: "var(--ink-2)" }}
                  >
                    {label}
                  </a>
                ))}
                <div className="pt-3 flex flex-col gap-2">
                  <Link
                    href="/login"
                    className="text-sm font-medium text-center py-2.5"
                    style={{ color: "var(--ink-2)" }}
                  >
                    Sign in
                  </Link>
                  <Link
                    href="/login"
                    className="text-sm font-semibold text-center py-3 rounded-xl"
                    style={{ background: "var(--ink)", color: "var(--on-ink)" }}
                  >
                    Get started free
                  </Link>
                </div>
              </div>
            </div>
          )}
        </div>
      </header>

      {/* ── Fullscreen mega-menu ── */}
      <AnimatePresence>
        {megaOpen && <MegaMenu onClose={closeMega} />}
      </AnimatePresence>
    </>
  );
}
