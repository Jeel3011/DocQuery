// Shared doc-type helpers (G5 Ask polish). The vault's dominant document class drives
// which starter-prompt set a landing shows (legal vs finance). Pure + null-safe so it can
// be used from any landing without duplicating the tally logic.

import type { DocumentResponse } from "./api";

export type DocClass = "legal_contract" | "financial_filing" | "mixed" | "generic";

// The dominant CONCRETE class across a vault's documents, or null when there's no clear
// signal (empty vault, or only mixed/generic/untyped docs). "mixed"/"generic" don't count
// as a domain vote — they're explicitly non-committal — so a vault of unclassified docs
// returns null and the caller falls back to its default (law-first) set.
export function dominantDocType(docs: Pick<DocumentResponse, "doc_type">[]): DocClass | null {
  const tally: Record<string, number> = {};
  for (const d of docs) {
    const t = d.doc_type;
    if (t === "legal_contract" || t === "financial_filing") {
      tally[t] = (tally[t] ?? 0) + 1;
    }
  }
  const ranked = Object.entries(tally).sort((a, b) => b[1] - a[1]);
  return ranked.length ? (ranked[0][0] as DocClass) : null;
}

// True when the vault leans financial (the only case we switch AWAY from the law-first
// default). Legal/unknown both keep the legal set, matching DocQuery's law-first posture.
export function isFinanceVault(docs: Pick<DocumentResponse, "doc_type">[]): boolean {
  return dominantDocType(docs) === "financial_filing";
}
