"use client";

import { useRef, useState, useCallback, KeyboardEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, Square, Loader2, Brain, Sparkles, FolderOpen } from "lucide-react";

const SLASH_COMMANDS = [
  { cmd: "/clauses", desc: "Extract key clauses across the contracts" },
  { cmd: "/risks", desc: "Flag non-standard or risky clauses" },
  { cmd: "/compare", desc: "Compare two contracts side by side" },
  { cmd: "/summarize", desc: "Summarize a contract or matter" },
];

// Clicking a slash command inserts a ready-to-send question, not the bare token.
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
  vaultName?: string | null;     // active collection name → shown as a "vault chip"
  onChooseVault?: () => void;    // open the vault/collection picker (the chip is a button)
  centered?: boolean;            // empty conversation → lift the composer toward center
}

// The three intelligence modes, surfaced as a labeled segmented control (Harvey-style)
// instead of cryptic bare icons. "Agent" = the verified tool-loop (agentCore); "Brain" =
// cross-doc synthesis; "Fast" = the default direct path (no special toggle).
type ModeKey = "fast" | "agent" | "brain" | "agentic";

export function ChatInput({
  onSubmit,
  onCancel,
  isStreaming,
  placeholder = "Ask about a clause, term, or risk in your contracts…",
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
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const [showSlash, setShowSlash] = useState(false);
  const [slashIdx, setSlashIdx] = useState(0);
  const [focused, setFocused] = useState(false);
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
    // Insert the full question template when one exists; else fall back to the bare token.
    setValue(SLASH_TEMPLATES[cmd] ?? cmd + " ");
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
    <motion.div
      transition={{ duration: 0.55, ease: [0.23, 1, 0.32, 1] }}
      className="relative z-10 px-4 md:px-8 pb-5 pt-2 flex-shrink-0 bg-transparent"
    >
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

        {/* Main composer — a warm 'sheet of paper' with layered depth + focus glow.
            Editorial/warm-paper direction: warm off-white gradient, a hairline warm
            border, a soft stacked shadow (paper lifting off the page), and a faint top
            highlight. On focus/typing it gently lifts + warms its ring. */}
        <motion.div
          animate={{
            scale: focused ? 1.006 : 1,
            y: focused ? -1 : 0,
          }}
          transition={{ duration: 0.35, ease: [0.23, 1, 0.32, 1] }}
          className="flex flex-col gap-2.5 rounded-[22px] px-4 pt-3.5 pb-3"
          style={{
            background: "linear-gradient(180deg, #FFFFFF 0%, #FAFAFA 100%)",
            border: focused || value.length > 0
              ? "1px solid rgba(0,0,0,0.22)"
              : "1px solid rgba(0,0,0,0.08)",
            boxShadow: focused || value.length > 0
              ? "0 24px 60px -18px rgba(0,0,0,0.20), 0 8px 20px -8px rgba(0,0,0,0.10), 0 0 0 4px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.95)"
              : "0 14px 40px -16px rgba(0,0,0,0.14), 0 3px 8px -3px rgba(0,0,0,0.07), inset 0 1px 0 rgba(255,255,255,0.92)",
          }}
        >
          {/* Streaming status — integrated thin row inside the bar */}
          {isStreaming && (
            <div className="flex items-center gap-2 text-[11px] text-[var(--text-muted)] pb-1 border-b border-[var(--glass-border)]">
              <Loader2 size={11} className="animate-spin" />
              <span>
                {agentCoreMode
                  ? "Agent working — searching, reading, computing…"
                  : brainMode
                  ? "Brain synthesizing across documents…"
                  : "Generating response…"}
              </span>
            </div>
          )}

          {/* ── Top row: vault chip (clickable → opens the vault picker) ── */}
          <div className="flex items-center gap-2 pb-1.5 border-b border-[var(--glass-border)]">
            <button
              type="button"
              onClick={onChooseVault}
              disabled={!onChooseVault}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all duration-[140ms] enabled:hover:-translate-y-px enabled:active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] disabled:cursor-default"
              style={{
                background: vaultName
                  ? "linear-gradient(180deg, #F4F4F4, #ECECEC)"
                  : "transparent",
                border: vaultName ? "1px solid rgba(0,0,0,0.14)" : "1px solid rgba(0,0,0,0.10)",
                color: vaultName ? "var(--text-primary)" : "var(--text-muted)",
                boxShadow: vaultName ? "0 1px 2px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.8)" : "none",
              }}
              title={vaultName ? `Vault: ${vaultName} — click to change` : "Choose a vault"}
            >
              <FolderOpen size={12} />
              <span className="truncate max-w-[160px]">{vaultName || "Choose vault"}</span>
            </button>
            <span className="text-[10px] text-[var(--text-muted)] ml-auto select-none">
              {agentCoreMode ? "Agent · verified tool loop"
                : brainMode ? "Brain · cross-doc synthesis"
                : "Fast · direct answer"}
            </span>
          </div>

          {/* Input row */}
          <div className="flex items-end gap-3">
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
            className="flex-1 bg-transparent text-[15px] text-[var(--text-primary)] leading-7
              placeholder:text-[var(--text-muted)] resize-none outline-none border-0 focus:outline-none focus:ring-0 min-h-[36px] py-1 appearance-none"
            style={{ maxHeight: 180, boxShadow: "none", border: "none" }}
          />

          {/* Mode segment — labeled pills (Harvey-style), not cryptic icons */}
          <div
            className="flex items-center gap-0.5 p-0.5 rounded-xl flex-shrink-0"
            style={{ background: "var(--bg-hover)", border: "1px solid var(--glass-border)" }}
          >
            {onToggleAgentCore && (
              <ModePill
                active={agentCoreMode} onClick={onToggleAgentCore}
                icon={<Sparkles size={12} className={agentCoreMode ? "fill-current" : ""} />}
                label="Agent" title="Verified tool loop — cites & computes every figure"
              />
            )}
            {/* Brain (cross-doc synthesis) is the internal Gate-A comparator — surfaced
                only as a hidden dev toggle, not a normal user mode. The agentic/Deep
                (multi-hop) pill was retired with the agent-core pivot (2026-06-12). */}
            {onToggleBrain && (
              <ModePill
                active={brainMode} onClick={onToggleBrain}
                icon={<Brain size={12} className={brainMode ? "fill-current" : ""} />}
                label="Brain" title="Cross-document synthesis (internal comparator)"
              />
            )}
          </div>

          <div className="flex items-center pb-0.5">
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
        </motion.div>

        <p className="text-[10px] text-[var(--text-muted)] mt-2 text-center select-none">
          ↵ send · ⇧↵ newline · / commands{agentCoreMode ? " · ✦ agent" : brainMode ? " · 🧠 brain" : ""} · always verify sources
        </p>
      </div>
    </motion.div>
  );
}

// A labeled mode pill inside the segmented control. Active = filled accent; inactive =
// quiet. Replaces the old bare icon-only toggles so a user can read what each mode does.
function ModePill({
  active, onClick, icon, label, title,
}: {
  active: boolean; onClick: () => void; icon: React.ReactNode; label: string; title: string;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      aria-pressed={active}
      className="relative flex items-center gap-1 px-2.5 h-7 rounded-lg text-[11px] font-medium transition-[color] duration-[180ms] active:scale-[0.96] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
      style={{ color: active ? "#FFFFFF" : "var(--text-muted)" }}
    >
      {/* Shared-layout pill: the active highlight SLIDES between modes (Harvey feel) */}
      {active && (
        <motion.span
          layoutId="mode-pill-active"
          transition={{ type: "spring", stiffness: 480, damping: 36 }}
          className="absolute inset-0 rounded-lg"
          style={{
            background: "linear-gradient(180deg, #2A2A2A, #0E0E0E)",
            boxShadow: "0 2px 6px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.12)",
          }}
        />
      )}
      <span className="relative z-10 flex items-center gap-1">{icon}<span>{label}</span></span>
    </button>
  );
}
