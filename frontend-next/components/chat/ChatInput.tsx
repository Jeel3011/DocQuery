"use client";

// components/chat/ChatInput.tsx — Monochrome
// Supports agentic mode toggle

import { useRef, useState, useCallback, KeyboardEvent } from "react";
import { ArrowUp, Square, Loader2, Zap } from "lucide-react";

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
  const ref = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  }, []);

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); submit(); }
  }

  function submit() {
    const t = value.trim();
    if (!t || isStreaming || disabled) return;
    onSubmit(t);
    setValue("");
    if (ref.current) ref.current.style.height = "auto";
  }

  const canSend = value.trim().length > 0 && !isStreaming && !disabled;

  return (
    <div className="border-t border-[var(--border)] px-4 md:px-8 py-4 bg-[var(--bg-surface)] flex-shrink-0">
      <div className="max-w-3xl mx-auto">
        {isStreaming && (
          <div className="flex items-center gap-2 mb-2 text-xs text-[var(--text-muted)]">
            <Loader2 size={12} className="animate-spin" />
            <span>{agenticMode ? "Agentic analysis in progress…" : "Generating response…"}</span>
          </div>
        )}

        <div className={`flex items-end gap-3 card px-4 py-3 transition-all duration-150
          ${value.length > 0 ? "border-[var(--accent)] shadow-[0_0_0_3px_rgba(10,10,10,0.06)]" : ""}`}>
          <textarea
            ref={ref}
            value={value}
            onChange={(e) => { setValue(e.target.value); adjustHeight(); }}
            onKeyDown={onKey}
            placeholder={agenticMode ? "Ask a complex question (agentic deep analysis)…" : placeholder}
            disabled={disabled}
            rows={1}
            className="flex-1 bg-transparent text-sm text-[var(--text-primary)] leading-6
              placeholder:text-[var(--text-muted)] resize-none outline-none min-h-[24px]"
            style={{ maxHeight: 120 }}
          />

          {/* Agentic mode toggle */}
          {onToggleAgentic && (
            <button
              onClick={onToggleAgentic}
              title={agenticMode ? "Agentic mode ON — deeper analysis with query decomposition" : "Standard mode — click for agentic deep analysis"}
              className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 transition-all duration-150
                ${agenticMode
                  ? "bg-[var(--accent)] text-white shadow-sm"
                  : "bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
                }`}
            >
              <Zap size={14} className={agenticMode ? "fill-current" : ""} />
            </button>
          )}

          {isStreaming ? (
            <button onClick={onCancel}
              className="w-8 h-8 rounded-lg bg-red-50 border border-red-200 flex items-center justify-center flex-shrink-0 hover:bg-red-100 transition-colors">
              <Square size={11} className="text-[var(--status-failed)] fill-current" />
            </button>
          ) : (
            <button onClick={submit} disabled={!canSend}
              className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 transition-all duration-150
                ${canSend
                  ? "bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] active:scale-95"
                  : "bg-[var(--bg-hover)] text-[var(--text-muted)] cursor-not-allowed"}`}>
              <ArrowUp size={14} />
            </button>
          )}
        </div>

        <p className="text-[10px] text-[var(--text-muted)] mt-2 text-center">
          ⌘↵ to send{agenticMode ? " · ⚡ Agentic mode" : ""} · DocQuery can make mistakes — verify sources
        </p>
      </div>
    </div>
  );
}
