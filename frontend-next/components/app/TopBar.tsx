"use client";

// TopBar — the slim app-shell header that replaces the document sidebar (G2 Step A).
// Layout:  [ logo · DocQuery ]  [ vault switcher ]  ··· spacer ···  [ ⌘K ]  [ account ]
// No file sidebar. The body below is full-bleed children on the clean white canvas.

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { Settings, LogOut, ChevronDown, Inbox, Building2 } from "lucide-react";
import { CollectionResponse } from "@/lib/api";
import { VaultSwitcher } from "./VaultSwitcher";
import { useFirmStore } from "@/stores/firm.store";
import { ROLE_LABEL } from "@/app/app/settings/firm/_shared";

interface TopBarProps {
  collections: CollectionResponse[];
  activeId: string | null;
  email?: string | null;
  name?: string | null;
  onNewVault?: () => void;
  onLogout: () => void;
}

export function TopBar({ collections, activeId, email, name, onNewVault, onLogout }: TopBarProps) {
  const router = useRouter();
  const [acctOpen, setAcctOpen] = useState(false);
  const acctRef = useRef<HTMLDivElement>(null);
  // Firm-console link renders only for a manage_members holder (the store's server-resolved caps —
  // never a hardcoded role check). The middleware still blocks the route directly (T8); this just
  // avoids dangling a link a user can't use.
  const canManageFirm = useFirmStore((s) => s.caps.has("manage_members"));
  // The user's seat in the firm — surfaced so a paralegal/associate always knows their role
  // (and therefore why some actions are partner-held). Server-resolved (never a hardcoded map).
  const role = useFirmStore((s) => s.role);

  useEffect(() => {
    if (!acctOpen) return;
    function onDown(e: MouseEvent) {
      if (acctRef.current && !acctRef.current.contains(e.target as Node)) setAcctOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setAcctOpen(false);
    }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [acctOpen]);

  function openCommandPalette() {
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true, bubbles: true }));
  }

  const initial = (name || email || "?").trim().charAt(0).toUpperCase();

  return (
    <header
      className="flex items-center gap-3 px-4 h-14 flex-shrink-0 border-b border-[var(--border)] glass-sm"
      style={{ zIndex: "var(--z-sticky)" as unknown as number }}
    >
      {/* Logo → Vault Home */}
      <button
        onClick={() => router.push("/app")}
        className="flex items-center gap-2 flex-shrink-0 rounded-lg pr-1 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
        title="DocQuery — all vaults"
      >
        <div
          className="w-6 h-6 rounded-[7px] flex items-center justify-center font-bold text-xs shadow-sm flex-shrink-0"
          style={{ background: "var(--ink)", color: "var(--on-ink)", fontFamily: "Fraunces, Georgia, serif" }}
        >
          D
        </div>
        <span
          className="text-[15px] font-semibold tracking-tight hidden sm:inline"
          style={{ color: "var(--ink)", fontFamily: "Fraunces, Georgia, serif", letterSpacing: "-0.025em" }}
        >
          DocQuery
        </span>
      </button>

      {/* Divider */}
      <div className="h-5 w-px bg-[var(--border)] flex-shrink-0" aria-hidden />

      {/* Vault switcher */}
      <VaultSwitcher collections={collections} activeId={activeId} onNewVault={onNewVault} />

      <div className="flex-1" />

      {/* ⌘K */}
      <button
        onClick={openCommandPalette}
        title="Command palette (⌘K)"
        className="px-2 py-1 rounded-lg text-[11px] font-mono text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--accent)] hover:text-[var(--text-primary)] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
      >
        ⌘K
      </button>

      {/* Account menu */}
      <div ref={acctRef} className="relative flex-shrink-0">
        <button
          onClick={() => setAcctOpen((v) => !v)}
          className="flex items-center gap-1.5 pl-1 pr-1.5 py-1 rounded-lg hover:bg-[var(--bg-hover)] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
          aria-haspopup="menu"
          aria-expanded={acctOpen}
          title="Account"
        >
          <span
            className="w-7 h-7 rounded-full flex items-center justify-center text-[12px] font-semibold flex-shrink-0"
            style={{ background: "var(--surface-3)", color: "var(--ink-2)", border: "1px solid var(--line)" }}
          >
            {initial}
          </span>
          <ChevronDown size={13} className="text-[var(--text-muted)]" />
        </button>

        <AnimatePresence>
          {acctOpen && (
            <motion.div
              initial={{ opacity: 0, y: -4, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -4, scale: 0.98 }}
              transition={{ duration: 0.12, ease: [0.23, 1, 0.32, 1] }}
              className="absolute right-0 top-full mt-1.5 w-[230px] rounded-xl overflow-hidden bg-[var(--bg-surface)] border border-[var(--border)]"
              style={{ zIndex: "var(--z-dropdown)" as unknown as number, boxShadow: "var(--shadow-lg)" }}
              role="menu"
            >
              <div className="px-3 py-3 border-b border-[var(--border)]">
                <div className="flex items-center justify-between gap-2">
                  {name && <p className="text-[13px] font-medium text-[var(--text-primary)] truncate">{name}</p>}
                  {role && (
                    <span
                      className="flex-shrink-0 px-1.5 py-0.5 rounded-md text-[10px] font-medium tracking-wide uppercase"
                      style={{ background: "var(--surface-3)", color: "var(--ink-2)", border: "1px solid var(--line)" }}
                      title="Your role in this firm"
                    >
                      {ROLE_LABEL[role]}
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-[var(--text-muted)] truncate">{email}</p>
              </div>
              <button
                onClick={() => { setAcctOpen(false); router.push("/app/review-queue"); }}
                className="w-full flex items-center gap-2.5 px-3 py-2.5 text-left text-[13px] transition-colors hover:bg-[var(--bg-hover)] text-[var(--text-secondary)]"
                role="menuitem"
              >
                <Inbox size={14} className="text-[var(--text-muted)]" />
                <span>My review queue</span>
              </button>
              {canManageFirm && (
                <button
                  onClick={() => { setAcctOpen(false); router.push("/app/settings/firm"); }}
                  className="w-full flex items-center gap-2.5 px-3 py-2.5 text-left text-[13px] transition-colors hover:bg-[var(--bg-hover)] text-[var(--text-secondary)]"
                  role="menuitem"
                >
                  <Building2 size={14} className="text-[var(--text-muted)]" />
                  <span>Firm console</span>
                </button>
              )}
              <button
                onClick={() => { setAcctOpen(false); router.push("/app/settings"); }}
                className="w-full flex items-center gap-2.5 px-3 py-2.5 text-left text-[13px] transition-colors hover:bg-[var(--bg-hover)] text-[var(--text-secondary)]"
                role="menuitem"
              >
                <Settings size={14} className="text-[var(--text-muted)]" />
                <span>Settings & analytics</span>
              </button>
              <button
                onClick={() => { setAcctOpen(false); onLogout(); }}
                className="w-full flex items-center gap-2.5 px-3 py-2.5 text-left text-[13px] transition-colors hover:bg-[var(--bg-hover)] text-[var(--text-secondary)] hover:text-[var(--status-failed)] border-t border-[var(--border)]"
                role="menuitem"
              >
                <LogOut size={14} />
                <span>Log out</span>
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </header>
  );
}
