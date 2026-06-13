"use client";

// Review grid (Phase B2) — the legal contract-review head-to-head surface (India-first).
// Pick a collection of contracts + clause columns, run, and watch each cell fill in live.
// Every FOUND cell quotes its source span (clickable); MISSING means the clause is genuinely
// absent (itself a finding); ABSTAIN / ERROR are honestly flagged. The headline: "every
// clause grounded in a quoted source, or flagged — no silent wrong cell."

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useAuthStore } from "@/stores/auth.store";
import { listCollections, getCollectionDocuments, CollectionResponse, DocumentResponse } from "@/lib/api";
import {
  streamReviewGrid, GridCellEvent, GridStart, GridDone, ReviewGridColumnSpec,
} from "@/lib/streaming";

// ── Legal-clause column presets, tuned for INDIAN contracts. Each column = one term to
//    surface across every contract in the collection. The agent reads each contract's
//    actual text and quotes its source (cite-or-abstain). risk_rubric drives the
//    standard / non-standard flag a reviewer scans for.
const PRESETS: ReviewGridColumnSpec[] = [
  { key: "governing_law", label: "Governing Law & Seat",
    prompt: "Find the governing-law clause and (if an arbitration clause exists) the seat/venue "
          + "of arbitration. Quote the exact jurisdiction and seat.",
    risk_rubric: "standard if Indian law and an Indian seat (e.g. Mumbai/Delhi/Bengaluru); "
               + "non_standard if foreign law or a foreign seat" },
  { key: "term_termination", label: "Term & Termination",
    prompt: "Find the contract term (duration) and the notice period / grounds required to "
          + "terminate. Quote the notice period (e.g. '90 days written notice')." },
  { key: "indemnity", label: "Indemnity",
    prompt: "Find the indemnification clause and any cap/limit on indemnity liability "
          + "(an amount, a multiple of fees, or 'uncapped'). Quote it.",
    risk_rubric: "standard if a clear monetary cap exists; non_standard if indemnity is uncapped/unlimited" },
  { key: "confidentiality", label: "Confidentiality",
    prompt: "Find the confidentiality / non-disclosure clause and its survival period after "
          + "termination. Quote the survival period if stated." },
  { key: "liability_cap", label: "Limitation of Liability",
    prompt: "Find the limitation-of-liability clause and the liability cap amount or formula. Quote it.",
    risk_rubric: "standard if liability is capped; non_standard if liability is unlimited or the clause is absent" },
  { key: "dispute_resolution", label: "Dispute Resolution",
    prompt: "How are disputes resolved — arbitration (which rules/institution, e.g. SIAC, ICA, "
          + "ad-hoc under the Arbitration & Conciliation Act 1996) or courts? Quote the clause." },
];

type CellKey = string; // `${doc_id}::${column_key}`
const ck = (doc: string, col: string): CellKey => `${doc}::${col}`;

const STATUS_STYLE: Record<string, { bg: string; fg: string; label: string }> = {
  found:   { bg: "rgba(16,16,16,0.04)", fg: "var(--text-primary)", label: "" },
  missing: { bg: "rgba(180,140,0,0.10)", fg: "#7a5d00", label: "not found" },
  abstain: { bg: "rgba(0,0,0,0.05)", fg: "var(--text-muted)", label: "unclear" },
  error:   { bg: "rgba(220,38,38,0.08)", fg: "#b3261e", label: "error" },
};

// Props let this page be RE-HOMED under /app/vault/[id]/review (G2 Step E) without
// forking the grid logic into a second, drift-prone copy. When the vault route mounts
// it, `scopedCollectionId` is the route's [id] — the AUTHORITATIVE vault scope (§9
// risk #1: the URL wins). On that path the in-page collection picker is dropped (the
// route IS the scope) and streamReviewGrid binds to the route-derived id. The legacy
// /app/grid route passes nothing and behaves byte-identically (picker + ?collection=).
export interface ReviewGridProps {
  scopedCollectionId?: string; // route-authoritative vault id (vault route only)
}

export default function ReviewGridPage(props: ReviewGridProps = {}) {
  return (
    <Suspense fallback={<div className="flex-1" />}>
      <ReviewGridInner {...props} />
    </Suspense>
  );
}

// Exported so the vault route can render the grid directly with props.
export { ReviewGridPage as ReviewGrid };

function ReviewGridInner({ scopedCollectionId }: ReviewGridProps) {
  const { token } = useAuthStore();
  const searchParams = useSearchParams();
  // Two scope sources: the route-authoritative vault id (vault route, §9 risk #1) takes
  // precedence; otherwise the legacy ?collection=<id> deep-link (used by old /app/grid
  // links). Either wins over the first-collection default below.
  const scopedCollection = scopedCollectionId ?? searchParams.get("collection");
  const routeScoped = !!scopedCollectionId; // vault route: hide the picker
  const [collections, setCollections] = useState<CollectionResponse[]>([]);
  const [collectionId, setCollectionId] = useState<string>(scopedCollection ?? "");
  const [docs, setDocs] = useState<DocumentResponse[]>([]);
  const [columns, setColumns] = useState<ReviewGridColumnSpec[]>(PRESETS.slice(0, 3));

  const [running, setRunning] = useState(false);
  const [start, setStart] = useState<GridStart | null>(null);
  const [cells, setCells] = useState<Record<CellKey, GridCellEvent>>({});
  const [done, setDone] = useState<GridDone | null>(null);
  const [detail, setDetail] = useState<GridCellEvent | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // On the vault route the URL is the source of truth: lock the scope to the route id
  // regardless of what's loaded, so a deep-link / back-button / second tab can never
  // bind the grid to the wrong vault (§9 risk #1). Re-runs if the route id changes.
  useEffect(() => {
    if (routeScoped && scopedCollectionId) setCollectionId(scopedCollectionId);
  }, [routeScoped, scopedCollectionId]);

  // Load collections once (needed for the legacy picker; harmless on the vault route).
  useEffect(() => {
    if (!token) return;
    listCollections(token).then((cs) => {
      setCollections(cs);
      // Legacy route only: honor a ?collection= scoped id if present, else first collection.
      // The vault route is locked by the effect above and must not fall back to first.
      if (!routeScoped && cs.length && !collectionId) {
        const scoped = scopedCollection && cs.some((c) => c.id === scopedCollection) ? scopedCollection : null;
        setCollectionId(scoped ?? cs[0].id);
      }
    }).catch(() => {});
  }, [token]); // eslint-disable-line react-hooks/exhaustive-deps

  // Load the chosen collection's documents (the rows).
  useEffect(() => {
    if (!token || !collectionId) return;
    getCollectionDocuments(token, collectionId).then(setDocs).catch(() => setDocs([]));
  }, [token, collectionId]);

  const cellCount = docs.length * columns.length;
  const tooMany = cellCount > 120;

  const coverage = useMemo(() => {
    const vals = Object.values(cells);
    const verified = vals.filter((c) => c.verified).length;
    const missing = vals.filter((c) => c.status === "missing").length;
    const abstain = vals.filter((c) => c.status === "abstain").length;
    const error = vals.filter((c) => c.status === "error").length;
    return { verified, missing, abstain, error, total: vals.length };
  }, [cells]);

  function toggleColumn(p: ReviewGridColumnSpec) {
    setColumns((cur) =>
      cur.find((c) => c.key === p.key) ? cur.filter((c) => c.key !== p.key) : [...cur, p]
    );
  }

  async function run() {
    if (!token || !collectionId || !columns.length || tooMany || running) return;
    setRunning(true); setCells({}); setDone(null); setErr(null); setStart(null);
    abortRef.current = new AbortController();
    await streamReviewGrid(
      token,
      { title: "Review grid", collection_id: collectionId, columns },
      {
        onStart: (s) => setStart(s),
        onCell: (c) => setCells((prev) => ({ ...prev, [ck(c.doc_id, c.column_key)]: c })),
        onDone: (d) => { setDone(d); if (d.error) setErr(d.error); setRunning(false); },
        onError: (m) => { setErr(m); setRunning(false); },
      },
      abortRef.current.signal
    );
  }

  function cancel() {
    abortRef.current?.abort();
    setRunning(false);
  }

  // ── CSV export (client-side: the streamed cells already live in page state) ────
  function exportCsv() {
    const esc = (v: unknown) => {
      const s = v == null ? "" : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const header = ["Document", ...columns.map((c) => c.label)];
    const lines = [header.map(esc).join(",")];
    for (const d of docs) {
      const row = [d.filename];
      for (const col of columns) {
        const cell = cells[ck(d.id, col.key)];
        if (!cell) { row.push(""); continue; }
        // value for found; otherwise the honest status word — never a blank that hides a flag
        const v = cell.status === "found"
          ? `${cell.value ?? ""}${cell.risk === "non_standard" ? " [non-standard]" : ""}`
          : cell.status === "missing" ? "NOT FOUND"
          : cell.status === "abstain" ? "UNCLEAR"
          : "ERROR";
        row.push(v);
      }
      lines.push(row.map(esc).join(","));
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `review-grid-${collectionId.slice(0, 8)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const hasCells = Object.keys(cells).length > 0;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header / controls */}
      <div className="px-6 py-4 border-b border-[var(--glass-border)] flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <h1 className="text-[17px] font-semibold text-[var(--text-primary)]">Contract Review Grid</h1>
          <span className="text-[11px] text-[var(--text-muted)]">
            Every clause grounded in a quoted source, or flagged — no silent wrong cells.
          </span>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          {/* Collection picker — legacy /app/grid only. On the vault route the [id]
              segment IS the scope (§9 risk #1), so the picker is dropped and we show the
              locked vault name instead. */}
          {routeScoped ? (
            <span className="text-[13px] px-3 py-1.5 rounded-lg border border-[var(--glass-border)] bg-white text-[var(--text-secondary)]">
              {collections.find((c) => c.id === collectionId)?.name ?? "This vault"}
            </span>
          ) : (
            <select
              value={collectionId}
              onChange={(e) => setCollectionId(e.target.value)}
              className="text-[13px] px-3 py-1.5 rounded-lg border border-[var(--glass-border)] bg-white"
            >
              {collections.map((c) => (
                <option key={c.id} value={c.id}>{c.name} ({c.document_count ?? "?"})</option>
              ))}
            </select>
          )}

          {/* Column presets */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {PRESETS.map((p) => {
              const on = !!columns.find((c) => c.key === p.key);
              return (
                <button
                  key={p.key}
                  onClick={() => toggleColumn(p)}
                  className="text-[11px] px-2.5 py-1 rounded-full border transition-colors"
                  style={{
                    background: on ? "linear-gradient(180deg,#2A2A2A,#0E0E0E)" : "transparent",
                    color: on ? "#fff" : "var(--text-muted)",
                    borderColor: on ? "transparent" : "var(--glass-border)",
                  }}
                  title={p.prompt}
                >
                  {p.label}
                </button>
              );
            })}
          </div>

          <div className="ml-auto flex items-center gap-2">
            <span className="text-[11px] text-[var(--text-muted)]">
              {docs.length} docs × {columns.length} cols = <b>{cellCount}</b> cells
              {tooMany && <span className="text-[#b3261e]"> · over 120 limit</span>}
            </span>
            {!running && hasCells && (
              <button onClick={exportCsv}
                className="text-[12px] px-3 py-1.5 rounded-lg border border-[var(--glass-border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]">
                Export CSV
              </button>
            )}
            {running ? (
              <button onClick={cancel}
                className="text-[12px] px-3 py-1.5 rounded-lg border border-[rgba(220,38,38,0.3)] text-[#b3261e]">
                Stop
              </button>
            ) : (
              <button onClick={run} disabled={!collectionId || !columns.length || tooMany}
                className="text-[12px] px-4 py-1.5 rounded-lg text-white disabled:opacity-40"
                style={{ background: "var(--accent)" }}>
                Run grid
              </button>
            )}
          </div>
        </div>

        {/* Coverage headline */}
        {(running || done) && (
          <div className="flex items-center gap-3 text-[11px]">
            <CoverChip label="Verified" n={coverage.verified} color="#0e0e0e" />
            <CoverChip label="Not found" n={coverage.missing} color="#7a5d00" />
            <CoverChip label="Unclear" n={coverage.abstain} color="var(--text-muted)" />
            {coverage.error > 0 && <CoverChip label="Error" n={coverage.error} color="#b3261e" />}
            <span className="text-[var(--text-muted)]">
              {coverage.total}{start ? `/${start.cells}` : ""} cells{running ? " · running…" : " · done"}
            </span>
          </div>
        )}
        {err && <div className="text-[12px] text-[#b3261e]">{err}</div>}
      </div>

      {/* The grid */}
      <div className="flex-1 overflow-auto p-6">
        {docs.length === 0 ? (
          <div className="text-[13px] text-[var(--text-muted)]">
            This collection has no documents. Pick another collection to build a grid.
          </div>
        ) : (
          <table className="border-collapse text-[12px] w-full">
            <thead>
              <tr>
                <th className="sticky left-0 bg-[var(--bg)] text-left px-3 py-2 font-semibold border-b border-[var(--glass-border)] min-w-[180px]">
                  Document
                </th>
                {columns.map((c) => (
                  <th key={c.key}
                    className="text-left px-3 py-2 font-semibold border-b border-[var(--glass-border)] min-w-[200px]">
                    {c.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.id} className="align-top">
                  <td className="sticky left-0 bg-[var(--bg)] px-3 py-2 border-b border-[var(--glass-border)] font-medium text-[var(--text-primary)] truncate max-w-[220px]"
                      title={d.filename}>
                    {d.filename}
                  </td>
                  {columns.map((col) => {
                    const cell = cells[ck(d.id, col.key)];
                    return (
                      <td key={col.key} className="px-1.5 py-1.5 border-b border-[var(--glass-border)]">
                        <GridCellView cell={cell} running={running} onClick={() => cell && setDetail(cell)} />
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Cell detail drawer (the click-to-source) */}
      {detail && <CellDetail cell={detail} onClose={() => setDetail(null)} />}
    </div>
  );
}

function CoverChip({ label, n, color }: { label: string; n: number; color: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full"
      style={{ background: "rgba(0,0,0,0.04)" }}>
      <span className="w-2 h-2 rounded-full" style={{ background: color }} />
      <b style={{ color: "var(--text-primary)" }}>{n}</b>
      <span className="text-[var(--text-muted)]">{label}</span>
    </span>
  );
}

function GridCellView({ cell, running, onClick }:
  { cell?: GridCellEvent; running: boolean; onClick: () => void }) {
  if (!cell) {
    return (
      <div className="h-9 rounded-md flex items-center px-2 text-[var(--text-muted)]"
        style={{ background: running ? "rgba(0,0,0,0.03)" : "transparent" }}>
        {running ? <span className="animate-pulse">…</span> : ""}
      </div>
    );
  }
  const st = STATUS_STYLE[cell.status] ?? STATUS_STYLE.abstain;
  const clickable = cell.status === "found";
  return (
    <button
      onClick={clickable ? onClick : undefined}
      className="w-full text-left rounded-md px-2 py-1.5 min-h-[36px] transition-colors"
      style={{ background: st.bg, cursor: clickable ? "pointer" : "default" }}
      title={cell.note || ""}
    >
      {cell.status === "found" ? (
        <div className="flex flex-col gap-0.5">
          <span className="text-[var(--text-primary)] leading-snug line-clamp-3">{cell.value}</span>
          <span className="inline-flex items-center gap-1 text-[10px]">
            {cell.risk === "non_standard" && <RiskDot color="#b3261e" t="non-standard" />}
            {cell.risk === "standard" && <RiskDot color="#1a7f37" t="standard" />}
            <span className="text-[var(--accent)] underline">source</span>
          </span>
        </div>
      ) : (
        <span style={{ color: st.fg }} className="text-[11px] italic">{st.label}</span>
      )}
    </button>
  );
}

function RiskDot({ color, t }: { color: string; t: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      <span style={{ color }}>{t}</span>
    </span>
  );
}

function CellDetail({ cell, onClose }: { cell: GridCellEvent; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex justify-end" onClick={onClose}>
      <div className="absolute inset-0 bg-black/20" />
      <div className="relative w-[420px] max-w-[90vw] h-full bg-white shadow-xl p-5 overflow-auto"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-[14px] font-semibold">{cell.column_key}</h3>
          <button onClick={onClose} className="text-[var(--text-muted)] text-[13px]">✕</button>
        </div>
        <div className="text-[11px] text-[var(--text-muted)] mb-1">{cell.doc_name}</div>
        {cell.value && (
          <div className="text-[14px] text-[var(--text-primary)] mb-3 font-medium">{cell.value}</div>
        )}
        {cell.quote && (
          <div className="mb-3">
            <div className="text-[10px] uppercase tracking-wide text-[var(--text-muted)] mb-1">Grounding quote</div>
            <blockquote className="text-[12px] italic border-l-2 border-[var(--glass-border)] pl-3 text-[var(--text-secondary)]">
              “{cell.quote}”
            </blockquote>
          </div>
        )}
        <div className="text-[10px] uppercase tracking-wide text-[var(--text-muted)] mb-1">Sources</div>
        <div className="flex flex-col gap-2">
          {(cell.provenance || []).map((p, i) => (
            <div key={i} className="text-[11px] rounded-lg p-2 bg-[rgba(0,0,0,0.03)]">
              <div className="font-medium text-[var(--text-primary)]">
                {(p.doc as string) || "source"} {p.page ? `· p.${p.page}` : ""}
              </div>
              {p.snippet ? <div className="text-[var(--text-muted)] mt-0.5 line-clamp-3">{p.snippet as string}</div> : null}
            </div>
          ))}
          {(!cell.provenance || cell.provenance.length === 0) && (
            <div className="text-[11px] text-[var(--text-muted)]">No source spans recorded.</div>
          )}
        </div>
        {cell.note && (
          <div className="mt-3 text-[11px] text-[var(--text-muted)]">Note: {cell.note}</div>
        )}
      </div>
    </div>
  );
}
