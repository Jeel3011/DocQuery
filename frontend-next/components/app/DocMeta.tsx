"use client";

// DocMeta — shared, doc-metadata presentation atoms used by the vault card (Step B)
// and the files table (Step C): the domain-type chip and the trust fidelity dot.
//
// G2 §9 risk #2/#3: doc_type (domain) and fidelity come from the Step F backend slice
// and are NOT persisted yet. So these degrade GRACEFULLY to a neutral placeholder until
// then — never a fake green. Category (domain) and Type (file format) are DISTINCT and
// must not be conflated (risk #3): DocTypeChip renders the domain class; the files table
// renders the file format in its own column.

import { ShieldCheck, ShieldAlert, FileText, Scale, FileSpreadsheet } from "lucide-react";

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
