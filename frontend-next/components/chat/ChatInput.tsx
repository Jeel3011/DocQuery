"use client";

import { useRef, useState, useCallback, KeyboardEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, Square, Loader2, Zap } from "lucide-react";

const SLASH_COMMANDS = [
  { cmd: "/compare", desc: "Compare two documents side by side" },
  { cmd: "/analyze", desc: "Deep agentic analysis of the corpus" },
  { cmd: "/draft", desc: "Draft a document from findings" },
  { cmd: "/summarize", desc: "Summarize all documents" },
];

interface ChatInputProps {
  onSubmit: (message: string) => void;
  onCancel?: () => void;
  isStreaming: boolean;
  placeholder?: string;
  disabled?: boolean;
  agenticMode?: boolean;
  onToggleAgentic?: () => void;
}

export function ChatInput({
  onSubmit,
  onCancel,
  isStreaming,
  placeholder = "Ask a question about your documents…",
  disabled = false,
  agenticMode = false,
  onToggleAgentic,
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const [showSlash, setShowSlash] = useState(false);
  const [slashIdx, setSlashIdx] = useState(0);
  const ref = useRef<HTMLTextAreaElement>(null);

  const slashFiltered = value.startsWith("/")
    ? SLASH_COMMANDS.filter((c) => c.cmd.startsWith(value.toLowerCase()))
    : [];

  const adjustHeight = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  }, []);

  function onChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    const v = e.target.value;
    setValue(v);
    adjustHeight();
    setShowSlash(v.startsWith("/") && slashFiltered.length > 0);
    setSlashIdx(0);
  }

  function applySlash(cmd: string) {
    setValue(cmd + " ");
    setShowSlash(false);
    ref.current?.focus();
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (showSlash && slashFiltered.length > 0) {
      if (e.key === "ArrowDown") { e.preventDefault(); setSlashIdx((i) => Math.min(i + 1, slashFiltered.length - 1)); return; }
      if (e.key === "ArrowUp") { e.preventDefault(); setSlashIdx((i) => Math.max(i - 1, 0)); return; }
      if (e.key === "Tab" || e.key === "Enter") { e.preventDefault(); applySlash(slashFiltered[slashIdx].cmd); return; }
      if (e.key === "Escape") { setShowSlash(false); return; }
    }
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
  }

  function submit() {
    const t = value.trim();
    if (!t || isStreaming || disabled) return;
    onSubmit(t);
    setValue("");
    setShowSlash(false);
    if (ref.current) ref.current.style.height = "auto";
  }

  const canSend = value.trim().length > 0 && !isStreaming && !disabled;

  return (
    <div className="border-t border-[var(--border)] px-4 md:px-8 py-4 bg-[var(--bg-surface)] flex-shrink-0">
      <div className="max-w-3xl mx-auto relative">
        {/* Streaming status */}
        {isStreaming && (
          <div className="flex items-center gap-2 mb-2 text-xs text-[var(--text-muted)]">
            <Loader2 size={12} className="animate-spin" />
            <span>{agenticMode ? "Agentic analysis in progress…" : "Generating response…"}</span>
          </div>
        )}

        {/* Slash-command popover */}
        <AnimatePresence>
          {showSlash && slashFiltered.length > 0 && (
            <motion.div
              key="slash"
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 4 }}
              transition={{ duration: 0.12 }}
              className="absolute bottom-full left-0 right-0 mb-2 card overflow-hidden shadow-lg"
              style={{ zIndex: "var(--z-dropdown)" as unknown as number }}
            >
              {slashFiltered.map((c, i) => (
                <button
                  key={c.cmd}
                  onMouseDown={(e) => { e.preventDefault(); applySlash(c.cmd); }}
                  className={`w-full flex items-center gap-3 px-3 py-2 text-left transition-colors ${
                    i === slashIdx ? "bg-[var(--bg-hover)]" : ""
                  }`}
                >
                  <span className="text-xs font-mono font-semibold text-[var(--text-primary)]">{c.cmd}</span>
                  <span className="text-[11px] text-[var(--text-muted)]">{c.desc}</span>
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Input wrapper */}
        <div
          className={`flex items-end gap-3 card px-4 py-3 transition-all duration-150
            ${value.length > 0 ? "border-[var(--accent)] shadow-[0_0_0_3px_rgba(10,10,10,0.08)]" : ""}`}
        >
          <textarea
            ref={ref}
            value={value}
            onChange={onChange}
            onKeyDown={onKey}
            placeholder={agenticMode ? "Ask a complex question (agentic deep analysis)…" : placeholder}
            disabled={disabled}
            rows={1}
            aria-label="Chat input"
            className="flex-1 bg-transparent text-sm text-[var(--text-primary)] leading-6
              placeholder:text-[var(--text-muted)] resize-none outline-none min-h-[24px]"
            style={{ maxHeight: 140 }}
          />

          {/* Agentic mode toggle */}
          {onToggleAgentic && (
            <button
              onClick={onToggleAgentic}
              title={agenticMode ? "Agentic mode ON" : "Standard mode — click for deep analysis"}
              aria-pressed={agenticMode}
              className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]
                ${agenticMode
                  ? "bg-[var(--accent)] text-white shadow-sm"
                  : "bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
                }`}
            >
              <Zap size={14} className={agenticMode ? "fill-current" : ""} />
            </button>
          )}

          {/* Send / Stop */}
          {isStreaming ? (
            <button
              onClick={onCancel}
              aria-label="Stop generating"
              className="w-8 h-8 rounded-lg bg-red-50 border border-red-200 flex items-center justify-center flex-shrink-0 hover:bg-red-100 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-300"
            >
              <Square size={11} className="text-[var(--status-failed)] fill-current" />
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={!canSend}
              aria-label="Send message"
              className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]
                ${canSend
                  ? "bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] active:scale-95"
                  : "bg-[var(--bg-hover)] text-[var(--text-muted)] cursor-not-allowed"}`}
            >
              <ArrowUp size={14} />
            </button>
          )}
        </div>

        <p className="text-[10px] text-[var(--text-muted)] mt-2 text-center select-none">
          ↵ send · ⇧↵ newline · / slash commands{agenticMode ? " · ⚡ agentic mode" : ""} · always verify sources
        </p>
      </div>
    </div>
  );
}
