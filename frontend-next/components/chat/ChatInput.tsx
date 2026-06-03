"use client";

import { useRef, useState, useCallback, KeyboardEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, Square, Loader2, Zap, Brain } from "lucide-react";

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
  brainMode?: boolean;
  onToggleBrain?: () => void;
}

export function ChatInput({
  onSubmit,
  onCancel,
  isStreaming,
  placeholder = "Ask a question about your documents…",
  disabled = false,
  agenticMode = false,
  onToggleAgentic,
  brainMode = false,
  onToggleBrain,
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
    el.style.height = Math.min(el.scrollHeight, 180) + "px";
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
    <div className="px-4 md:px-8 pb-5 pt-2 flex-shrink-0 bg-transparent">
      <div className="max-w-3xl mx-auto relative">

        {/* Slash-command popover */}
        <AnimatePresence>
          {showSlash && slashFiltered.length > 0 && (
            <motion.div
              key="slash"
              initial={{ opacity: 0, scale: 0.97, y: 4 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.97, y: 4 }}
              transition={{ duration: 0.15, ease: [0.23, 1, 0.32, 1] }}
              className="absolute bottom-full left-0 right-0 mb-2 rounded-xl overflow-hidden"
              style={{
                transformOrigin: "bottom left",
                zIndex: "var(--z-dropdown)" as unknown as number,
                background: "var(--glass-bg-strong)",
                backdropFilter: "blur(16px)",
                WebkitBackdropFilter: "blur(16px)",
                border: "1px solid var(--glass-border)",
                boxShadow: "var(--glass-shadow-lg)",
              }}
            >
              {slashFiltered.map((c, i) => (
                <button
                  key={c.cmd}
                  onMouseDown={(e) => { e.preventDefault(); applySlash(c.cmd); }}
                  className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                    i === slashIdx ? "bg-[rgba(0,0,0,0.04)]" : ""
                  }`}
                >
                  <span className="text-xs font-mono font-semibold text-[var(--text-primary)]">{c.cmd}</span>
                  <span className="text-[11px] text-[var(--text-muted)]">{c.desc}</span>
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main glass input bar — one tall premium surface */}
        <div
          className={`flex flex-col gap-2 rounded-[20px] px-4 pt-3 pb-2.5 transition-[border-color,box-shadow] duration-[160ms] ease-[cubic-bezier(0.23,1,0.32,1)]`}
          style={{
            background: "linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,255,255,0.62))",
            backdropFilter: "blur(24px) saturate(1.6)",
            WebkitBackdropFilter: "blur(24px) saturate(1.6)",
            border: value.length > 0
              ? "1px solid rgba(10,10,10,0.3)"
              : "1px solid var(--glass-border)",
            boxShadow: value.length > 0
              ? "0 16px 48px -12px rgba(40,30,20,0.22), 0 0 0 3px rgba(10,10,10,0.05), inset 0 1px 0 rgba(255,255,255,0.85)"
              : "0 14px 40px -14px rgba(40,30,20,0.20), inset 0 1px 0 rgba(255,255,255,0.8)",
          }}
        >
          {/* Streaming status — integrated thin row inside the bar */}
          {isStreaming && (
            <div className="flex items-center gap-2 text-[11px] text-[var(--text-muted)] pb-1 border-b border-[var(--glass-border)]">
              <Loader2 size={11} className="animate-spin" />
              <span>
                {brainMode
                  ? "Brain synthesizing across documents…"
                  : agenticMode
                  ? "Agentic analysis in progress…"
                  : "Generating response…"}
              </span>
            </div>
          )}

          {/* Input row */}
          <div className="flex items-end gap-3">
          <textarea
            ref={ref}
            value={value}
            onChange={onChange}
            onKeyDown={onKey}
            placeholder={
              brainMode
                ? "Ask a cross-document question (Brain synthesis)…"
                : agenticMode
                ? "Ask a complex question (agentic deep analysis)…"
                : placeholder
            }
            disabled={disabled}
            rows={1}
            aria-label="Chat input"
            className="flex-1 bg-transparent text-[15px] text-[var(--text-primary)] leading-7
              placeholder:text-[var(--text-muted)] resize-none outline-none min-h-[36px] py-1"
            style={{ maxHeight: 180 }}
          />

          {/* Mode toggles */}
          <div className="flex items-center gap-1.5 pb-0.5">
            {onToggleBrain && (
              <button
                onClick={onToggleBrain}
                title={brainMode ? "Brain synthesis ON" : "Enable Brain synthesis"}
                aria-pressed={brainMode}
                className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 transition-[background-color,color,box-shadow,transform] duration-[120ms] ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
                style={brainMode ? {
                  background: "var(--accent)",
                  color: "white",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.2), var(--skeu-inset)",
                } : {
                  background: "var(--glass-bg)",
                  color: "var(--text-muted)",
                  border: "1px solid var(--glass-border)",
                  boxShadow: "var(--skeu-raised)",
                }}
              >
                <Brain size={14} className={brainMode ? "fill-current" : ""} />
              </button>
            )}

            {onToggleAgentic && (
              <button
                onClick={onToggleAgentic}
                title={agenticMode ? "Agentic mode ON" : "Enable agentic mode"}
                aria-pressed={agenticMode}
                className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 transition-[background-color,color,box-shadow,transform] duration-[120ms] ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
                style={agenticMode ? {
                  background: "var(--accent)",
                  color: "white",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.2), var(--skeu-inset)",
                } : {
                  background: "var(--glass-bg)",
                  color: "var(--text-muted)",
                  border: "1px solid var(--glass-border)",
                  boxShadow: "var(--skeu-raised)",
                }}
              >
                <Zap size={14} className={agenticMode ? "fill-current" : ""} />
              </button>
            )}

            {/* Send / Stop */}
            {isStreaming ? (
              <button
                onClick={onCancel}
                aria-label="Stop generating"
                className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 transition-[background-color,transform] duration-[120ms] ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97]"
                style={{
                  background: "rgba(220,38,38,0.08)",
                  border: "1px solid rgba(220,38,38,0.2)",
                  boxShadow: "var(--skeu-raised)",
                  color: "var(--status-failed)",
                }}
              >
                <Square size={11} className="fill-current" />
              </button>
            ) : (
              <button
                onClick={submit}
                disabled={!canSend}
                aria-label="Send message"
                className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 transition-[background-color,color,transform,box-shadow] duration-[120ms] ease-[cubic-bezier(0.23,1,0.32,1)] enabled:active:scale-[0.97] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
                style={canSend ? {
                  background: "var(--accent)",
                  color: "white",
                  boxShadow: "0 2px 10px rgba(0,0,0,0.2), var(--skeu-inset)",
                } : {
                  background: "var(--glass-bg)",
                  color: "var(--text-muted)",
                  border: "1px solid var(--glass-border)",
                  boxShadow: "var(--skeu-raised)",
                  cursor: "not-allowed",
                }}
              >
                <ArrowUp size={14} />
              </button>
            )}
          </div>
          </div>{/* end input row */}
        </div>

        <p className="text-[10px] text-[var(--text-muted)] mt-2 text-center select-none">
          ↵ send · ⇧↵ newline · / commands{brainMode ? " · 🧠 brain" : agenticMode ? " · ⚡ agentic" : ""} · always verify sources
        </p>
      </div>
    </div>
  );
}
