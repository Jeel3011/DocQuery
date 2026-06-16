"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { Reveal } from "./Reveal";
import { SiteNav } from "./SiteNav";

const ease = [0.23, 1, 0.32, 1] as const;

function Footer() {
  return (
    <footer
      className="py-16 relative z-10"
      style={{ borderTop: "1px solid var(--line)", background: "var(--surface-2)" }}
    >
      <div className="section-container">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: "var(--ink)" }}>
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
  );
}

export function PlatformHero({
  breadcrumb,
  title,
  description,
}: {
  breadcrumb: string;
  title: React.ReactNode;
  description: string;
}) {
  return (
    <section
      className="relative overflow-hidden"
      style={{ paddingTop: "clamp(120px, 14vw, 168px)", paddingBottom: "clamp(56px, 7vw, 88px)" }}
    >
      <div className="section-container relative z-10">
        <Reveal>
          <p className="text-[13px] mb-5" style={{ color: "var(--ink-3)" }}>
            {breadcrumb}
          </p>
        </Reveal>
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-8">
          <Reveal delay={0.05} className="max-w-2xl">
            <h1
              className="font-display font-light"
              style={{
                fontSize: "clamp(40px, 6vw, 68px)",
                lineHeight: "1.04",
                letterSpacing: "-0.03em",
                color: "var(--ink)",
                textWrap: "balance",
              }}
            >
              {title}
            </h1>
          </Reveal>
          <Reveal delay={0.12} className="max-w-sm shrink-0">
            <p
              style={{ fontSize: "16px", lineHeight: "1.6", color: "var(--ink-2)", letterSpacing: "-0.01em" }}
            >
              {description}
            </p>
            <Link href="/login" className="btn-cta mt-5 inline-flex">
              Start for free
              <ArrowRight size={18} strokeWidth={2.2} />
            </Link>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

export function PlatformCTA() {
  return (
    <section className="relative z-10" style={{ paddingTop: "clamp(60px, 8vw, 100px)", paddingBottom: "clamp(80px, 10vw, 130px)" }}>
      <div className="section-container">
        <Reveal>
          <div
            className="rounded-[28px] flex flex-col items-center text-center gap-6 px-8 py-16"
            style={{ background: "var(--ink)", color: "var(--on-ink)" }}
          >
            <h2
              className="font-display font-light"
              style={{ fontSize: "clamp(28px, 4vw, 44px)", letterSpacing: "-0.03em", lineHeight: 1.1 }}
            >
              Bring your own documents.
              <br />
              See what traces, and what doesn&apos;t.
            </h2>
            <p className="max-w-md text-[15px] leading-relaxed opacity-70">
              No credit card required. Upload a contract or filing and watch the fidelity pipeline run live.
            </p>
            <Link href="/login" className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-[15px]" style={{ background: "var(--on-ink)", color: "var(--ink)" }}>
              Start for free
              <ArrowRight size={17} strokeWidth={2.2} />
            </Link>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

export function PlatformPageShell({ children }: { children: React.ReactNode }) {
  return (
    <main className="relative min-h-screen" style={{ background: "var(--canvas)", color: "var(--ink)" }}>
      <div className="aurora aurora-fixed" />
      <div className="dot-grid fixed inset-0 z-0 opacity-[0.14] pointer-events-none" />
      <SiteNav />
      <div className="relative z-10">
        {children}
        <Footer />
      </div>
    </main>
  );
}

export { ease };
