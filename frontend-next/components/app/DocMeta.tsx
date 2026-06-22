"use client";

// DocMeta — shared, doc-metadata presentation atoms used by the vault card (Step B)
// and the files table (Step C): the domain-type chip and the trust fidelity dot.
//
// G2 §9 risk #2/#3: doc_type (domain) and fidelity come from the Step F backend slice
// and are NOT persisted yet. So these degrade GRACEFULLY to a neutral placeholder until
// then — never a fake green. Category (domain) and Type (file format) are DISTINCT and
// must not be conflated (risk #3): DocTypeChip renders the domain class; the files table
// renders the file format in its own column.

import { ShieldCheck, ShieldAlert, FileText, Scale, FileSpreadsheet, Lock, ChevronDown } from "lucide-react";
import { useEffect, useRef, useState } from "react";

// G1d structural classes (src/components/data_ingestion.py::classify_document).
export type DocType = "legal_contract" | "financial_filing" | "mixed" | "generic" | null;
export type Fidelity = "good" | "partial" | null;

const TYPE_META: Record<
  Exclude<DocType, null>,
  { label: string; icon: typeof FileText }
> = {
  legal_contract: { label: "Contract", icon: Scale },
  financial_filing: { label: "Filing", icon: FileSpreadsheet },
  mixed: { label: "Mixed", icon: FileText },
  generic: { label: "Document", icon: FileText },
};

// Domain-class chip. Neutral grey in the B&W system — the type is information, not a
// status, so it carries no semantic color. Null type → a quiet "—" placeholder.
export function DocTypeChip({ type }: { type: DocType }) {
  if (!type) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] text-[var(--text-muted)] border border-dashed border-[var(--border-dotted)]">
        —
      </span>
    );
  }
  const { label, icon: Icon } = TYPE_META[type];
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-medium"
      style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
    >
      <Icon size={11} className="text-[var(--text-muted)]" />
      {label}
    </span>
  );
}

// Fidelity dot — the trust signal Harvey lacks. ONE of the three places color is allowed
// (plan §0.5): green = high-fidelity extraction, amber = partial (needs a human look).
// Null fidelity → a hollow neutral dot, NOT a fake green (we never imply confidence we
// don't have — that's the whole product thesis).
export function FidelityDot({ fidelity, withLabel = false }: { fidelity: Fidelity; withLabel?: boolean }) {
  if (!fidelity) {
    return (
      <span className="inline-flex items-center gap-1.5 text-[var(--text-muted)]" title="Fidelity not yet computed">
        <span className="w-2 h-2 rounded-full border border-[var(--line-2)]" />
        {withLabel && <span className="text-[11px]">—</span>}
      </span>
    );
  }
  const good = fidelity === "good";
  const color = good ? "var(--fidelity-good)" : "var(--fidelity-partial)";
  const Icon = good ? ShieldCheck : ShieldAlert;
  return (
    <span
      className="inline-flex items-center gap-1.5"
      style={{ color }}
      title={good ? "High-fidelity extraction" : "Partial extraction — review recommended"}
    >
      <span className="w-2 h-2 rounded-full" style={{ background: color }} />
      {withLabel && (
        <span className="inline-flex items-center gap-1 text-[11px] font-medium">
          <Icon size={11} />
          {good ? "High" : "Partial"}
        </span>
      )}
    </span>
  );
}

// ── F1e lifecycle (matter status) ──────────────────────────────────────────────
// Matter lifecycle is INFORMATION, not a trust state, so it carries NO semantic color
// (the 5 reserved colors are fidelity/verdict/gate only). It reads as a neutral ink dot +
// label, exactly like DocTypeChip. The status set is lockstep with the backend CHECK
// (schemas.py MATTER_STATUSES). Verb+object-free single nouns (the trust-vocabulary rule).
export type MatterStatus = "active" | "on_hold" | "closed" | "archived" | "legal_hold";

export const MATTER_STATUS_META: Record<MatterStatus, string> = {
  active: "Active",
  on_hold: "On hold",
  closed: "Closed",
  archived: "Archived",
  legal_hold: "Legal hold",
};

const MATTER_STATUS_ORDER: MatterStatus[] = [
  "active", "on_hold", "closed", "archived", "legal_hold",
];

function statusLabel(s: string | null | undefined): string {
  if (!s) return MATTER_STATUS_META.active; // legacy/null reads as Active (the DB default)
  return MATTER_STATUS_META[s as MatterStatus] ?? s;
}

// A solid neutral dot for an active matter, hollow for any settled/held state — a quiet
// visual cue without color. Used both standalone and inside the control.
function LifecycleDot({ status }: { status: string | null | undefined }) {
  const active = !status || status === "active";
  return active ? (
    <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: "var(--ink-2)" }} />
  ) : (
    <span className="w-2 h-2 rounded-full flex-shrink-0 border" style={{ borderColor: "var(--line-2)" }} />
  );
}

// The read-only lifecycle pill (e.g. in a list/switcher) — dot + label, no interaction.
export function LifecycleBadge({ status }: { status: string | null | undefined }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-medium"
      style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
      title={`Matter status: ${statusLabel(status)}`}
    >
      <LifecycleDot status={status} />
      {statusLabel(status)}
    </span>
  );
}

// The interactive lifecycle control for the vault header: a small dropdown that PATCHes the
// vault status. F2 will partner-gate WHO may change it; F1e ships the control. `onChange`
// does the PATCH + optimistic update; `disabled` covers the in-flight + (future) RBAC case.
export function LifecycleControl({
  status,
  onChange,
  disabled = false,
}: {
  status: string | null | undefined;
  onChange: (next: MatterStatus) => void;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const current = (status as MatterStatus) || "active";

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

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => !disabled && setOpen((v) => !v)}
        disabled={disabled}
        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[12px] font-medium transition-colors hover:bg-[var(--bg-hover)] disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
        style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
        aria-haspopup="listbox"
        aria-expanded={open}
        title={`Matter status: ${statusLabel(current)}`}
      >
        <LifecycleDot status={current} />
        {statusLabel(current)}
        <ChevronDown size={12} className="text-[var(--text-muted)]" />
      </button>

      {open && (
        <div
          className="absolute left-0 top-full mt-1.5 w-[160px] rounded-xl overflow-hidden bg-[var(--surface)] border border-[var(--line)] py-1"
          style={{ zIndex: "var(--z-dropdown)" as unknown as number, boxShadow: "var(--shadow-lg)" }}
          role="listbox"
        >
          {MATTER_STATUS_ORDER.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => { setOpen(false); if (s !== current) onChange(s); }}
              className="w-full flex items-center gap-2 px-3 py-2 text-left text-[12px] transition-colors hover:bg-[var(--bg-hover)]"
              role="option"
              aria-selected={s === current}
              style={{ color: s === current ? "var(--ink)" : "var(--ink-2)" }}
            >
              <LifecycleDot status={s} />
              <span className="flex-1">{MATTER_STATUS_META[s]}</span>
              {s === current && <span className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--ink)" }} />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ── F1e privilege lock glyph ────────────────────────────────────────────────────
// A doc marked privileged shows a filled lock; unmarked shows a hollow/outline lock that
// only appears on row hover (so it doesn't clutter a clean export-eligible doc). Privilege
// is a BOUNDARY marker, not a trust verdict, so it stays monochrome (ink, never a reserved
// color). Clicking toggles via PATCH /documents/{id}. F2 will partner-gate the toggle.
export function PrivilegeToggle({
  privileged,
  onToggle,
  disabled = false,
}: {
  privileged: boolean;
  onToggle: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      aria-pressed={privileged}
      title={
        privileged
          ? "Privileged — attorney-client / work-product. Excluded from shared surfaces and watermarked in exports. Click to unmark."
          : "Mark privileged (attorney-client / work-product)"
      }
      className={`p-1.5 rounded-lg transition-[color,opacity] disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] hover:bg-[var(--bg-hover)] ${
        privileged
          ? "text-[var(--ink)] opacity-100"
          : "text-[var(--text-muted)] opacity-0 group-hover:opacity-100 hover:text-[var(--ink)] focus-visible:opacity-100"
      }`}
    >
      <Lock size={14} strokeWidth={privileged ? 2.25 : 1.75} fill={privileged ? "currentColor" : "none"} />
    </button>
  );
}
