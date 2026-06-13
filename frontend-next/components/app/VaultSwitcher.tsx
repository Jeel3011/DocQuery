"use client";

// VaultSwitcher — the top-bar dropdown that scopes the whole app to one vault
// (= collection). Selecting a vault NAVIGATES to /app/vault/[id]; it does NOT write the
// store directly — the route is the source of truth and VaultScopeSync derives the store
// from it (G2 §9 risk #1). "All vaults" routes back to the Vault Home (/app).

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronsUpDown, Check, FolderOpen, Plus, LayoutGrid } from "lucide-react";
import { CollectionResponse } from "@/lib/api";

interface VaultSwitcherProps {
  collections: CollectionResponse[];
  activeId: string | null;
  onNewVault?: () => void;
}

export function VaultSwitcher({ collections, activeId, onNewVault }: VaultSwitcherProps) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const active = collections.find((c) => c.id === activeId) ?? null;

  // Close on outside click + Escape.
  useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  function pick(id: string | null) {
    setOpen(false);
    router.push(id ? `/app/vault/${id}` : "/app");
  }

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-[13px] font-medium transition-colors hover:bg-[var(--bg-hover)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
        style={{ color: "var(--text-primary)" }}
        aria-haspopup="listbox"
        aria-expanded={open}
        title="Switch vault"
      >
        <FolderOpen size={14} className="text-[var(--text-muted)] flex-shrink-0" />
        <span className="truncate max-w-[180px]">{active ? active.name : "All vaults"}</span>
        <ChevronsUpDown size={13} className="text-[var(--text-muted)] flex-shrink-0" />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98 }}
            transition={{ duration: 0.12, ease: [0.23, 1, 0.32, 1] }}
            className="absolute left-0 top-full mt-1.5 w-[260px] rounded-xl overflow-hidden bg-[var(--bg-surface)] border border-[var(--border)]"
            style={{ zIndex: "var(--z-dropdown)" as unknown as number, boxShadow: "var(--shadow-lg)" }}
            role="listbox"
          >
            {/* All vaults → Vault Home */}
            <button
              onClick={() => pick(null)}
              className="w-full flex items-center gap-2.5 px-3 py-2.5 text-left text-[13px] transition-colors hover:bg-[var(--bg-hover)] border-b border-[var(--border)]"
            >
              <LayoutGrid size={14} className="text-[var(--text-muted)] flex-shrink-0" />
              <span className="flex-1 text-[var(--text-primary)]">All vaults</span>
              {activeId === null && <Check size={14} className="text-[var(--accent)]" />}
            </button>

            {/* Vault list */}
            <div className="max-h-[300px] overflow-y-auto scrollbar-thin py-1">
              {collections.length === 0 ? (
                <p className="px-3 py-3 text-[12px] text-center text-[var(--text-muted)]">No vaults yet.</p>
              ) : (
                collections.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => pick(c.id)}
                    className="w-full flex items-center gap-2.5 px-3 py-2 text-left text-[13px] transition-colors hover:bg-[var(--bg-hover)]"
                    role="option"
                    aria-selected={activeId === c.id}
                  >
                    <FolderOpen
                      size={14}
                      className={`flex-shrink-0 ${activeId === c.id ? "text-[var(--accent)]" : "text-[var(--text-muted)]"}`}
                    />
                    <span className="flex-1 truncate text-[var(--text-primary)]">{c.name}</span>
                    <span className="text-[10px] text-[var(--text-muted)]">{c.document_count ?? 0}</span>
                    {activeId === c.id && <Check size={14} className="text-[var(--accent)]" />}
                  </button>
                ))
              )}
            </div>

            {/* New vault */}
            {onNewVault && (
              <button
                onClick={() => { setOpen(false); onNewVault(); }}
                className="w-full flex items-center gap-2.5 px-3 py-2.5 text-left text-[13px] transition-colors hover:bg-[var(--bg-hover)] border-t border-[var(--border)] text-[var(--text-secondary)]"
              >
                <Plus size={14} className="flex-shrink-0" />
                <span>New vault</span>
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
