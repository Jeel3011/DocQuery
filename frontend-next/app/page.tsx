import Link from "next/link";
import { HeroSection } from "@/components/landing/HeroSection";
import { BelowFold } from "@/components/landing/BelowFold";

export default function RootPage() {
  return (
    <main className="relative min-h-screen text-[var(--text-primary)] selection:bg-[var(--accent-subtle)]">

      {/* Ambient aurora — fixed behind the whole page; the single source of colour.
          z-0 (NOT -1) so it sits above the body bg but below content (z-10). */}
      <div className="aurora aurora-fixed" />
      {/* Faint dot-grid over the aurora, also fixed */}
      <div className="dot-grid fixed inset-0 z-0 opacity-[0.16] pointer-events-none" />

      <div className="relative z-10">
        <header className="absolute top-0 w-full p-6 flex justify-between items-center z-50">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-[var(--accent)] rounded-lg flex items-center justify-center font-bold text-white shadow-sm">
              D
            </div>
            <span className="font-bold text-lg tracking-tight text-[var(--text-primary)]">DocQuery</span>
          </div>
          <Link
            href="/login"
            className="glass-sm text-sm font-medium text-[var(--text-primary)] px-4 py-2 rounded-full hover:bg-[var(--bg-hover)] transition-colors"
          >
            Sign In →
          </Link>
        </header>

        <HeroSection />
        <BelowFold />

        <footer className="py-12 text-center text-[var(--text-muted)] text-sm border-t border-[var(--border-dotted)] border-dashed mt-24">
          <p>© 2026 DocQuery. Built for scale.</p>
        </footer>
      </div>
    </main>
  );
}
