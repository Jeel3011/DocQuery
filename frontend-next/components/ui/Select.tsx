"use client";

// components/ui/Select.tsx — a custom, on-brand dropdown that replaces the native <select>
// (the OS-rendered yellow menu in the firm console looked off-brand). Monochrome, B&W, with the
// design-system tokens. Follows the interface-polish principles:
//   • concentric radius (trigger rounded-lg inside a rounded-2xl dialog)
//   • scale-on-press feedback (active:scale-[0.96] — the skill's canonical press value)
//   • subtle staggered enter + soft exit via framer-motion (bounce 0)
//   • full 40px+ hit areas on trigger + every option
//   • no `transition: all` — only color/transform transitions
//   • useReducedMotion() honored
// Keyboard: Enter/Space/ArrowDown opens; Up/Down moves; Enter selects; Esc closes; click-away closes.

import { useEffect, useRef, useState, useId } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Check, ChevronsUpDown } from "lucide-react";
import { clsx } from "clsx";

export interface SelectOption {
  value: string;
  label: string;
  hint?: string;        // a secondary line (e.g. a role under a name)
  disabled?: boolean;
}

interface SelectProps {
  value: string;
  onChange: (value: string) => void;
  options: SelectOption[];
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  ariaLabel?: string;
}

const EASE = [0.2, 0, 0, 1] as const;

export function Select({
  value, onChange, options, placeholder = "Select…", disabled, className, ariaLabel,
}: SelectProps) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState<number>(-1);   // keyboard-highlighted index
  const rootRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();
  const listId = useId();

  const selected = options.find((o) => o.value === value) ?? null;

  // Click-away + Escape close.
  useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") { setOpen(false); }
    }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // When opening, highlight the current selection (or first enabled option).
  useEffect(() => {
    if (!open) return;
    const cur = options.findIndex((o) => o.value === value);
    setActive(cur >= 0 ? cur : options.findIndex((o) => !o.disabled));
  }, [open, value, options]);

  function move(dir: 1 | -1) {
    setActive((prev) => {
      let i = prev;
      for (let step = 0; step < options.length; step++) {
        i = (i + dir + options.length) % options.length;
        if (!options[i]?.disabled) return i;
      }
      return prev;
    });
  }

  function onTriggerKey(e: React.KeyboardEvent) {
    if (disabled) return;
    if (!open && (e.key === "Enter" || e.key === " " || e.key === "ArrowDown")) {
      e.preventDefault(); setOpen(true); return;
    }
    if (open) {
      if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
      else if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        const opt = options[active];
        if (opt && !opt.disabled) { onChange(opt.value); setOpen(false); }
      }
    }
  }

  return (
    <div ref={rootRef} className={clsx("relative", className)}>
      <button
        type="button"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={ariaLabel}
        onClick={() => !disabled && setOpen((v) => !v)}
        onKeyDown={onTriggerKey}
        className={clsx(
          "w-full min-h-[40px] flex items-center justify-between gap-2 px-3 py-2 rounded-lg text-left text-[13px]",
          "transition-[border-color,background-color,box-shadow,transform] duration-150 active:scale-[0.96]",
          "disabled:cursor-not-allowed disabled:opacity-50",
          "outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-ring)]"
        )}
        style={{
          background: "var(--surface-2)",
          border: `1px solid ${open ? "var(--ink)" : "var(--line-2)"}`,
          color: selected ? "var(--ink)" : "var(--ink-3)",
        }}
      >
        <span className="truncate">{selected ? selected.label : placeholder}</span>
        <ChevronsUpDown size={14} className="shrink-0" style={{ color: "var(--ink-3)" }} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            ref={listRef}
            role="listbox"
            id={listId}
            initial={{ opacity: 0, y: reduce ? 0 : -4, scale: reduce ? 1 : 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: reduce ? 0 : -3, scale: reduce ? 1 : 0.99 }}
            transition={{ duration: reduce ? 0.1 : 0.16, ease: EASE }}
            className="absolute left-0 right-0 top-full mt-1.5 z-[var(--z-dropdown)] max-h-60 overflow-y-auto scrollbar-thin rounded-xl p-1"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--line-2)",
              boxShadow: "var(--shadow-lg)",
            }}
          >
            {options.length === 0 ? (
              <p className="px-3 py-2 text-[12px]" style={{ color: "var(--ink-3)" }}>No options</p>
            ) : (
              options.map((o, i) => {
                const isSel = o.value === value;
                const isActive = i === active;
                return (
                  <button
                    key={o.value}
                    type="button"
                    role="option"
                    aria-selected={isSel}
                    disabled={o.disabled}
                    onMouseEnter={() => !o.disabled && setActive(i)}
                    onClick={() => { if (!o.disabled) { onChange(o.value); setOpen(false); } }}
                    className={clsx(
                      "w-full min-h-[38px] flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left text-[13px]",
                      "transition-[background-color,color] duration-100",
                      "disabled:cursor-not-allowed disabled:opacity-40"
                    )}
                    style={{
                      background: isActive && !o.disabled ? "var(--surface-3)" : "transparent",
                      color: "var(--ink)",
                    }}
                  >
                    <span className="w-4 shrink-0 flex items-center justify-center">
                      {isSel && <Check size={13} style={{ color: "var(--ink)" }} />}
                    </span>
                    <span className="flex-1 min-w-0">
                      <span className="block truncate">{o.label}</span>
                      {o.hint && (
                        <span className="block truncate text-[11px]" style={{ color: "var(--ink-3)" }}>
                          {o.hint}
                        </span>
                      )}
                    </span>
                  </button>
                );
              })
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
