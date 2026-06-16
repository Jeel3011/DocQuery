"use client";

import { HeroSection, HeroProof } from "@/components/landing/HeroSection";
import { BelowFold } from "@/components/landing/BelowFold";
import { SiteNav } from "@/components/landing/SiteNav";

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

      <SiteNav />

      <div className="relative z-10">
        <HeroSection />
        <HeroProof />
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
