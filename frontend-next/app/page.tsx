import Link from "next/link";
import { HeroSection } from "@/components/landing/HeroSection";
import { BelowFold } from "@/components/landing/BelowFold";

export default function RootPage() {
  return (
    <main className="min-h-screen bg-[var(--bg-base)] dot-grid text-[var(--text-primary)] selection:bg-[var(--accent-subtle)]">
      
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
            className="text-sm font-medium text-[var(--text-primary)] border border-[var(--border)] px-4 py-2 rounded-full bg-[var(--bg-surface)] hover:bg-[var(--bg-hover)] transition-colors"
          >
            Sign In →
          </Link>
        </header>

        <HeroSection />
        <BelowFold />
        
        <footer className="py-12 text-center text-[var(--text-muted)] text-sm border-t border-[var(--border-dotted)] border-dashed mt-24 bg-[var(--bg-surface)]">
          <p>© 2026 DocQuery. Built for scale.</p>
        </footer>
      </div>
    </main>
  );
}
