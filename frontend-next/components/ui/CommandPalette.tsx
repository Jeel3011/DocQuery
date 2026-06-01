"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Search, Plus, Upload, MessageSquare, FolderOpen, X } from "lucide-react";
import { useRouter } from "next/navigation";

interface Command {
  id: string;
  label: string;
  description?: string;
  icon: React.ReactNode;
  action: () => void;
  keywords?: string[];
}

interface CommandPaletteProps {
  onNewChat?: () => void;
  onUpload?: () => void;
  conversations?: { id: string; title: string }[];
  collections?: { id: string; name: string }[];
}

export function CommandPalette({ onNewChat, onUpload, conversations = [], collections = [] }: CommandPaletteProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(0);
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);

  const close = useCallback(() => { setOpen(false); setQuery(""); setSelected(0); }, []);

  // ⌘K / Ctrl+K
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 50);
  }, [open]);

  const allCommands: Command[] = [
    {
      id: "new-chat",
      label: "New Chat",
      description: "Start a fresh conversation",
      icon: <Plus size={14} />,
      action: () => { close(); onNewChat?.(); },
      keywords: ["new", "chat", "start"],
    },
    {
      id: "upload",
      label: "Upload Documents",
      description: "Add files to your library",
      icon: <Upload size={14} />,
      action: () => { close(); onUpload?.(); },
      keywords: ["upload", "file", "pdf", "document"],
    },
    ...conversations.slice(0, 8).map((c) => ({
      id: `conv-${c.id}`,
      label: c.title || "Untitled",
      description: "Jump to conversation",
      icon: <MessageSquare size={14} />,
      action: () => { close(); router.push(`/app/chat/${c.id}`); },
      keywords: [c.title?.toLowerCase() ?? ""],
    })),
    ...collections.slice(0, 6).map((c) => ({
      id: `coll-${c.id}`,
      label: c.name,
      description: "Open collection",
      icon: <FolderOpen size={14} />,
      action: () => { close(); },
      keywords: [c.name.toLowerCase()],
    })),
  ];

  const filtered = query.trim()
    ? allCommands.filter((cmd) => {
        const q = query.toLowerCase();
        return (
          cmd.label.toLowerCase().includes(q) ||
          cmd.description?.toLowerCase().includes(q) ||
          cmd.keywords?.some((k) => k.includes(q))
        );
      })
    : allCommands;

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") { e.preventDefault(); setSelected((s) => Math.min(s + 1, filtered.length - 1)); }
    if (e.key === "ArrowUp") { e.preventDefault(); setSelected((s) => Math.max(s - 1, 0)); }
    if (e.key === "Enter") { e.preventDefault(); filtered[selected]?.action(); }
    if (e.key === "Escape") close();
  }

  return (
    <>
      <AnimatePresence>
        {open && (
          <>
            <motion.div
              key="backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="fixed inset-0 bg-black/30"
              style={{ zIndex: "var(--z-dialog)" as unknown as number }}
              onClick={close}
            />
            <motion.div
              key="palette"
              initial={{ opacity: 0, scale: 0.97, y: -8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.97, y: -8 }}
              transition={{ type: "spring", stiffness: 400, damping: 32 }}
              className="fixed top-[20vh] left-1/2 -translate-x-1/2 w-full max-w-[520px] mx-4 bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl shadow-xl overflow-hidden"
              style={{ zIndex: "var(--z-dialog)" as unknown as number }}
            >
              {/* Search input */}
              <div className="flex items-center gap-3 px-4 py-3 border-b border-[var(--border)]">
                <Search size={16} className="text-[var(--text-muted)] flex-shrink-0" />
                <input
                  ref={inputRef}
                  value={query}
                  onChange={(e) => { setQuery(e.target.value); setSelected(0); }}
                  onKeyDown={handleKey}
                  placeholder="Search commands, chats, collections…"
                  className="flex-1 bg-transparent text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none"
                  autoComplete="off"
                />
                <button onClick={close} className="p-1 rounded text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors" aria-label="Close">
                  <X size={14} />
                </button>
              </div>

              {/* Results */}
              <div className="py-1.5 max-h-[340px] overflow-y-auto scrollbar-thin">
                {filtered.length === 0 ? (
                  <p className="px-4 py-6 text-xs text-center text-[var(--text-muted)]">No results</p>
                ) : (
                  filtered.map((cmd, i) => (
                    <button
                      key={cmd.id}
                      onClick={cmd.action}
                      onMouseEnter={() => setSelected(i)}
                      className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                        i === selected ? "bg-[var(--bg-hover)]" : ""
                      }`}
                    >
                      <span className={`flex-shrink-0 ${i === selected ? "text-[var(--text-primary)]" : "text-[var(--text-muted)]"}`}>
                        {cmd.icon}
                      </span>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-medium text-[var(--text-primary)] truncate">{cmd.label}</p>
                        {cmd.description && (
                          <p className="text-[10px] text-[var(--text-muted)] truncate">{cmd.description}</p>
                        )}
                      </div>
                    </button>
                  ))
                )}
              </div>

              {/* Footer hint */}
              <div className="px-4 py-2 border-t border-[var(--border)] flex items-center gap-3 text-[10px] text-[var(--text-muted)]">
                <span>↑↓ navigate</span>
                <span>↵ select</span>
                <span>esc close</span>
                <span className="ml-auto">⌘K</span>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
