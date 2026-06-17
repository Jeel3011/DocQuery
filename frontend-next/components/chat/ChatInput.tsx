"use client";

import { useRef, useState, useCallback, KeyboardEvent, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, Square, Loader2, FolderOpen, ChevronDown, Check } from "lucide-react";

const SLASH_COMMANDS = [
  { cmd: "/clauses", desc: "Extract key clauses across the contracts" },
  { cmd: "/risks", desc: "Flag non-standard or risky clauses" },
  { cmd: "/compare", desc: "Compare two contracts side by side" },
  { cmd: "/summarize", desc: "Summarize a contract or matter" },
];

const SLASH_TEMPLATES: Record<string, string> = {
  "/clauses": "Extract the key clauses (governing law, term & termination, indemnity, "
            + "confidentiality, limitation of liability, dispute resolution) and quote each.",
  "/risks": "Flag any non-standard, one-sided, or unusual clauses a reviewer should look at, "
          + "and quote the clause for each.",
  "/compare": "Compare these two contracts clause-by-clause and highlight where the terms differ: ",
  "/summarize": "Summarize this contract: parties, purpose, key commercial terms, and any red flags.",
};

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
  agentCoreMode?: boolean;
  onToggleAgentCore?: () => void;
  vaultName?: string | null;
  onChooseVault?: () => void;
  centered?: boolean;
  // G8.7: knowledge-source chips. `sources` is the set currently enabled (a subset of
  // {"vault","statutes","caselaw"}); toggling one flips whether the agent may use + cite
  // that authority. Only meaningful in Agent mode (the only mode that calls the tools).
  // Absent ⇒ chips not rendered (no toggle wired) — byte-identical.
  sources?: string[];
  onToggleSource?: (key: string) => void;
}

// The three knowledge sources the agent can draw on. "Web" is intentionally absent —
// agent-core has no web tool, so we don't ship a dead toggle (matches the backend).
const SOURCE_CHIPS: { key: string; label: string }[] = [
  { key: "vault", label: "This vault" },
  { key: "statutes", label: "Indian statutes" },
  { key: "caselaw", label: "Case law" },
];

type ModeKey = "fast" | "agent" | "brain";

const MODES: { key: ModeKey; label: string; desc: string }[] = [
  { key: "fast",  label: "Direct", desc: "Fast direct answer" },
  { key: "agent", label: "Agent",  desc: "Verified tool loop — cites every figure" },
  { key: "brain", label: "Brain",  desc: "Cross-document synthesis" },
];

export function ChatInput({
  onSubmit,
  onCancel,
  isStreaming,
  placeholder = "Ask about a clause, term, or risk…",
  disabled = false,
  agenticMode = false,
  onToggleAgentic,
  brainMode = false,
  onToggleBrain,
  agentCoreMode = false,
  onToggleAgentCore,
  vaultName,
  onChooseVault,
  centered = false,
  sources,
  onToggleSource,
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const [showSlash, setShowSlash] = useState(false);
  const [slashIdx, setSlashIdx] = useState(0);
  const [focused, setFocused] = useState(false);
  const [modeOpen, setModeOpen] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);
  const modeRef = useRef<HTMLDivElement>(null);

  const activeMode: ModeKey = agentCoreMode ? "agent" : brainMode ? "brain" : "fast";
  const activeLabel = MODES.find((m) => m.key === activeMode)?.label ?? "Direct";
  const hasModeToggle = !!onToggleAgentCore || !!onToggleBrain;

  function selectMode(key: ModeKey) {
    setModeOpen(false);
    if (key === "agent" && !agentCoreMode) onToggleAgentCore?.();
    else if (key === "brain" && !brainMode) onToggleBrain?.();
    else if (key === "fast") {
      if (agentCoreMode) onToggleAgentCore?.();
      if (brainMode) onToggleBrain?.();
    }
  }

  useEffect(() => {
    if (!modeOpen) return;
    function onDown(e: MouseEvent) {
      if (modeRef.current && !modeRef.current.contains(e.target as Node)) setModeOpen(false);
    }
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [modeOpen]);

  const slashFiltered = value.startsWith("/")
    ? SLASH_COMMANDS.filter((c) => c.cmd.startsWith(value.toLowerCase()))
    : [];

  const adjustHeight = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }, []);

  function onChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    const v = e.target.value;
    setValue(v);
    adjustHeight();
    setShowSlash(v.startsWith("/") && slashFiltered.length > 0);
    setSlashIdx(0);
  }

  function applySlash(cmd: string) {
    setValue(SLASH_TEMPLATES[cmd] ?? cmd + " ");
    setShowSlash(false);
    ref.current?.focus();
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (showSlash && slashFiltered.length > 0) {
      if (e.key === "ArrowDown") { e.preventDefault(); setSlashIdx((i) => Math.min(i + 1, slashFiltered.length - 1)); return; }
      if (e.key === "ArrowUp")   { e.preventDefault(); setSlashIdx((i) => Math.max(i - 1, 0)); return; }
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
    <div className="relative z-10 px-4 md:px-6 pb-4 pt-2 flex-shrink-0">
      <div className="max-w-3xl mx-auto relative">

        {/* Slash-command popover */}
        <AnimatePresence>
          {showSlash && slashFiltered.length > 0 && (
            <motion.div
              key="slash"
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 4 }}
              transition={{ duration: 0.12 }}
              className="absolute bottom-full left-0 right-0 mb-2 rounded-xl overflow-hidden"
              style={{
                background: "var(--surface)",
                border: "1px solid var(--line)",
                boxShadow: "var(--shadow-lg)",
                zIndex: "var(--z-dropdown)" as unknown as number,
              }}
            >
              {slashFiltered.map((c, i) => (
                <button
                  key={c.cmd}
                  onMouseDown={(e) => { e.preventDefault(); applySlash(c.cmd); }}
                  className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                    i === slashIdx ? "bg-[var(--surface-2)]" : ""
                  }`}
                >
                  <span className="text-xs font-mono font-semibold text-[var(--text-primary)]">{c.cmd}</span>
                  <span className="text-[11px] text-[var(--text-muted)]">{c.desc}</span>
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Slim command-bar ── */}
        <div
          style={{
            background: "#FFFFFF",
            border: focused || value.length > 0
              ? "1px solid rgba(0,0,0,0.18)"
              : "1px solid rgba(0,0,0,0.09)",
            borderRadius: 16,
            boxShadow: focused || value.length > 0
              ? "0 8px 32px -8px rgba(0,0,0,0.14), 0 2px 8px -2px rgba(0,0,0,0.08)"
              : "0 2px 12px -4px rgba(0,0,0,0.08), 0 1px 3px rgba(0,0,0,0.04)",
            transition: "border-color 140ms ease, box-shadow 140ms ease",
          }}
        >
          {/* ── Row 1: vault chip + mode status (streaming indicator lives here) ── */}
          <div
            className="flex items-center gap-2 px-3"
            style={{
              height: 36,
              borderBottom: "1px solid rgba(0,0,0,0.06)",
            }}
          >
            {/* Vault chip */}
            <button
              type="button"
              onClick={onChooseVault}
              disabled={!onChooseVault}
              className="flex items-center gap-1.5 text-[11px] font-medium transition-colors rounded-md px-1.5 py-0.5 disabled:cursor-default enabled:hover:bg-[var(--surface-2)]"
              style={{ color: vaultName ? "var(--text-primary)" : "var(--text-muted)" }}
              title={vaultName ? `Vault: ${vaultName} — click to change` : undefined}
            >
              <FolderOpen size={11} style={{ flexShrink: 0 }} />
              <span className="truncate max-w-[180px]">{vaultName || "No vault"}</span>
            </button>

            {/* G8.7: knowledge-source chips — only in Agent mode (the tool-using path) and
                only when a toggle is wired. A chip off ⇒ the agent cannot use/cite that
                source (gated server-side, not just visually). */}
            {agentCoreMode && onToggleSource && sources && (
              <div className="flex items-center gap-1" role="group" aria-label="Knowledge sources">
                {SOURCE_CHIPS.map((s) => {
                  const on = sources.includes(s.key);
                  return (
                    <button
                      key={s.key}
                      type="button"
                      onClick={() => onToggleSource(s.key)}
                      aria-pressed={on}
                      title={on ? `${s.label} — enabled (click to exclude)` : `${s.label} — excluded (click to enable)`}
                      className="flex items-center gap-1 text-[11px] font-medium rounded-md px-1.5 py-0.5 transition-colors"
                      style={{
                        background: on ? "var(--surface-2)" : "transparent",
                        color: on ? "var(--text-primary)" : "var(--text-muted)",
                        border: on ? "1px solid var(--line)" : "1px solid transparent",
                        textDecoration: on ? "none" : "line-through",
                        opacity: on ? 1 : 0.6,
                      }}
                    >
                      {s.label}
                    </button>
                  );
                })}
              </div>
            )}

            {/* Streaming status — replaces mode label while active */}
            <span className="ml-auto flex items-center gap-1.5 text-[10px] select-none" style={{ color: "var(--text-muted)" }}>
              {isStreaming ? (
                <>
                  <Loader2 size={10} className="animate-spin" />
                  <span>
                    {agentCoreMode ? "Agent working…" : brainMode ? "Brain synthesizing…" : "Generating…"}
                  </span>
                </>
              ) : (
                <span>
                  {agentCoreMode ? "Agent · verified tool loop"
                    : brainMode ? "Brain · cross-doc synthesis"
                    : "Direct · fast answer"}
                </span>
              )}
            </span>
          </div>

          {/* ── Row 2: textarea + mode dropdown + send ── */}
          <div className="flex items-center gap-2 px-3" style={{ minHeight: 44 }}>
            <textarea
              ref={ref}
              value={value}
              onChange={onChange}
              onKeyDown={onKey}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              placeholder={
                agentCoreMode
                  ? "Ask about a clause, term, or risk — the agent quotes and cites every answer…"
                  : brainMode
                  ? "Ask a cross-document question (Brain synthesis)…"
                  : placeholder
              }
              disabled={disabled}
              rows={1}
              aria-label="Chat input"
              className="flex-1 bg-transparent text-[14px] text-[var(--text-primary)] leading-6
                placeholder:text-[var(--text-muted)] resize-none outline-none border-0
                focus:outline-none focus:ring-0 py-3 appearance-none"
              style={{ maxHeight: 160, boxShadow: "none", border: "none" }}
            />

            {/* Mode dropdown — only when toggles are wired */}
            {hasModeToggle && (
              <div ref={modeRef} className="relative flex-shrink-0">
                <button
                  type="button"
                  onClick={() => setModeOpen((v) => !v)}
                  className="flex items-center gap-1 px-2.5 h-7 rounded-lg text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
                  style={{
                    background: "var(--surface-2)",
                    border: "1px solid var(--line)",
                    color: "var(--text-secondary)",
                  }}
                >
                  <span>{activeLabel}</span>
                  <ChevronDown size={10} style={{ opacity: 0.5 }} />
                </button>
                <AnimatePresence>
                  {modeOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: 4, scale: 0.97 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 4, scale: 0.97 }}
                      transition={{ duration: 0.12, ease: [0.23, 1, 0.32, 1] }}
                      className="absolute bottom-full right-0 mb-2 w-52 rounded-xl overflow-hidden"
                      style={{
                        background: "var(--surface)",
                        border: "1px solid var(--line)",
                        boxShadow: "var(--shadow-lg)",
                        zIndex: "var(--z-dropdown)" as unknown as number,
                        transformOrigin: "bottom right",
                      }}
                    >
                      <div className="px-3 py-2 border-b border-[var(--line)]">
                        <p className="text-[10px] font-semibold tracking-widest uppercase text-[var(--text-muted)]">Execution Strategy</p>
                      </div>
                      {MODES.filter((m) =>
                        m.key === "fast" ||
                        (m.key === "agent" && !!onToggleAgentCore) ||
                        (m.key === "brain" && !!onToggleBrain)
                      ).map((m) => (
                        <button
                          key={m.key}
                          onClick={() => selectMode(m.key)}
                          className="w-full flex items-center justify-between gap-3 px-3 py-2.5 text-left hover:bg-[var(--surface-2)] transition-colors"
                        >
                          <div>
                            <p className="text-[13px] font-medium text-[var(--text-primary)]">{m.label}</p>
                            <p className="text-[11px] text-[var(--text-muted)]">{m.desc}</p>
                          </div>
                          {activeMode === m.key && <Check size={13} style={{ color: "var(--ink)", flexShrink: 0 }} />}
                        </button>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}

            {/* Send / Stop */}
            {isStreaming ? (
              <button
                onClick={onCancel}
                aria-label="Stop generating"
                className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 transition-colors active:scale-[0.97]"
                style={{
                  background: "rgba(220,38,38,0.08)",
                  border: "1px solid rgba(220,38,38,0.2)",
                  color: "var(--status-failed)",
                }}
              >
                <Square size={10} className="fill-current" />
              </button>
            ) : (
              <button
                onClick={submit}
                disabled={!canSend}
                aria-label="Send message"
                className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 transition-[background-color,color,transform] duration-[120ms] enabled:active:scale-[0.97] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
                style={canSend ? {
                  background: "var(--accent)",
                  color: "white",
                } : {
                  background: "var(--surface-2)",
                  color: "var(--text-muted)",
                  border: "1px solid var(--line)",
                  cursor: "not-allowed",
                }}
              >
                <ArrowUp size={13} />
              </button>
            )}
          </div>
        </div>

        <p className="text-[10px] text-[var(--text-muted)] mt-1.5 text-center select-none">
          ↵ send · ⇧↵ newline · / commands · always verify sources
        </p>
      </div>
    </div>
  );
}
