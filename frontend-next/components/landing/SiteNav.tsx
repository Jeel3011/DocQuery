"use client";

import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import { Compass, Table2, MessageSquare, Workflow as WorkflowIcon, ArrowRight } from "lucide-react";

const PLATFORM_ITEMS = [
  {
    href: "/platform",
    icon: Compass,
    title: "Overview",
    description: "How Vault, Agent, and Workflows work together across a matter.",
  },
  {
    href: "/platform/vault",
    icon: Table2,
    title: "Vault",
    description: "Upload, classify, and verify documents with a live fidelity pipeline.",
  },
  {
    href: "/platform/agent",
    icon: MessageSquare,
    title: "Agent",
    description: "A tool-using agent that cites every figure to a clause or cell — or abstains.",
  },
  {
    href: "/platform/workflows",
    icon: WorkflowIcon,
    title: "Workflows",
    description: "Repeatable review, draft, and output playbooks across a vault.",
  },
];

const ease = [0.23, 1, 0.32, 1] as const;

function PlatformMenu() {
  const [open, setOpen] = useState(false);
  return (
    <div
      className="relative"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        className="px-4 py-2 rounded-lg text-sm font-medium transition-colors inline-flex items-center gap-1"
        style={{ color: open ? "var(--ink)" : "var(--ink-2)", background: open ? "var(--accent-soft)" : "transparent" }}
      >
        Platform
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform 220ms cubic-bezier(0.23,1,0.32,1)" }}>
          <path d="M2 3.5L5 6.5L8 3.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      <AnimatePresence>
        {open && (
          <>
            {/* Full-bleed backdrop dim, like Harvey's mega-menu scrim */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.25, ease }}
              className="fixed left-0 right-0 top-[68px] bottom-0 z-[150]"
              style={{ background: "rgba(14,14,14,0.18)" }}
            />
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.32, ease }}
              className="fixed left-0 right-0 top-[68px] z-[151]"
            >
              <div
                style={{
                  background: "var(--surface)",
                  borderBottom: "1px solid var(--line)",
                  boxShadow: "var(--shadow-xl)",
                }}
              >
                <div className="section-container py-12">
                  <div className="grid grid-cols-4 gap-10">
                    {PLATFORM_ITEMS.map((item, i) => (
                      <motion.div
                        key={item.href}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.36, delay: 0.04 + i * 0.035, ease }}
                      >
                        <Link href={item.href} className="group block" onClick={() => setOpen(false)}>
                          <div
                            className="w-11 h-11 rounded-xl flex items-center justify-center mb-5 transition-transform duration-300"
                            style={{ background: "var(--ink)", color: "var(--on-ink)" }}
                          >
                            <item.icon size={18} strokeWidth={1.6} />
                          </div>
                          <div className="flex items-center gap-1.5 mb-2">
                            <span
                              className="font-display"
                              style={{ fontSize: "20px", fontWeight: 500, letterSpacing: "-0.02em", color: "var(--ink)" }}
                            >
                              {item.title}
                            </span>
                            <ArrowRight size={13} className="opacity-0 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all" style={{ color: "var(--ink-3)" }} />
                          </div>
                          <p className="text-[13.5px] leading-relaxed" style={{ color: "var(--ink-3)" }}>
                            {item.description}
                          </p>
                        </Link>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

export function SiteNav() {
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

          {/* Desktop nav links — persistent across every page, including /platform/* subpages */}
          <nav className="hidden lg:flex items-center gap-1">
            <PlatformMenu />
            {[
              { label: "Product", href: "/#product" },
              { label: "Accuracy", href: "/#accuracy" },
              { label: "Pricing", href: "/#pricing" },
              { label: "Security", href: "/#security" },
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
              <div className="px-3 py-1.5 eyebrow">Platform</div>
              {PLATFORM_ITEMS.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setOpen(false)}
                  className="flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium"
                  style={{ color: "var(--ink-2)" }}
                >
                  <item.icon size={14} />
                  {item.title}
                </Link>
              ))}
              <div className="my-1 border-t" style={{ borderColor: "var(--line)" }} />
              {[
                { label: "Product", href: "/#product" },
                { label: "Accuracy", href: "/#accuracy" },
                { label: "Pricing", href: "/#pricing" },
                { label: "Security", href: "/#security" },
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
