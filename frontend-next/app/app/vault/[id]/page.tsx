"use client";

// /app/vault/[id] — Inside a Vault, the workspace (G2 Step C).
// Harvey's vault-detail shape, adapted (plan §8): top→bottom, NOT a 3-column split.
//   header → action row (Review · Draft) → HERO composer + suggested chips → files table.
// The vault screen IS the ask surface (Harvey's key move): submitting the composer opens
// an Ask conversation scoped to this vault. Ask (Step D) and Review (Step E) are now
// re-homed under this route: the composer opens /app/vault/[id]/ask/[cid] and Review
// opens /app/vault/[id]/review — both pre-scoped to this vault by the route [id].
//
// Scope rule (G2 §9 risk #1): the route [id] is the AUTHORITATIVE vault scope. We read it
// from useParams and pass it explicitly to every collection_id call site here (upload,
// review link, doc fetch) — never reach into the store for it.
//
// doc_type / fidelity come from the Step F backend slice and are now rendered as the
// Category chip + Fidelity dot (real values; null degrades to a neutral placeholder for
// legacy docs ingested before Step F / legal docs that skip the table-fidelity pass).

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft, FolderOpen, Table2, FileEdit, FileText, CheckCircle, AlertCircle,
  RefreshCw, Search, Trash2, Telescope,
} from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/auth.store";
import {
  listCollections, getCollectionDocuments, createConversation, deleteDocument,
  CollectionResponse, DocumentResponse,
} from "@/lib/api";
import { ChatInput } from "@/components/chat/ChatInput";
import { UploadZone } from "@/components/app/UploadZone";
import { PipelineTrack } from "@/components/app/PipelineTrack";
import { DocTypeChip, FidelityDot } from "@/components/app/DocMeta";
import { EmptyState } from "@/components/ui/EmptyState";

const ease = [0.23, 1, 0.32, 1] as const;

// Filter-chip labels for the G1d doc_type classes (mirror DocMeta's TYPE_META so the chip
// reads the same as the Category column). Falls back to the raw key if a new class appears.
const TYPE_LABELS: Record<string, string> = {
  legal_contract: "Contracts",
  financial_filing: "Filings",
  mixed: "Mixed",
  generic: "Documents",
};

// Doc-type-aware suggested prompts. Until Step F surfaces a vault-level type, we default
// to the legal-contract set (DocQuery is law-first); a finance-typed vault swaps these in F.
const CONTRACT_PROMPTS = [
  "Find the change-of-control provisions across these contracts",
  "Summarize governing law, term & termination for each",
  "Flag any non-standard indemnity or liability caps",
  "Which agreements auto-renew, and on what notice?",
];

function timeAgo(d: string | null): string {
  if (!d) return "—";
  const m = Math.floor((Date.now() - new Date(d).getTime()) / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function fmtSize(bytes: number | null): string {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function fileFormat(d: DocumentResponse): string {
  const ext = d.file_type || d.filename.split(".").pop() || "";
  return ext.replace(/^\./, "").toUpperCase() || "—";
}

export default function VaultWorkspacePage() {
  const params = useParams();
  const router = useRouter();
  const { token } = useAuthStore();
  const vaultId = (params?.id as string) ?? "";

  const [vault, setVault] = useState<CollectionResponse | null>(null);
  const [docs, setDocs] = useState<DocumentResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [query, setQuery] = useState("");
  const [lastSync, setLastSync] = useState<number | null>(null);
  const [confirmDel, setConfirmDel] = useState<string | null>(null); // doc id pending confirm
  // G3 Step E: active vault filters (doc_type / fiscal_year). null = no narrowing. These
  // drive BOTH the client-side table view AND retrieval scope (passed to Ask/Review).
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [yearFilter, setYearFilter] = useState<number | null>(null);
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const loadVault = useCallback(async () => {
    if (!token || !vaultId) return;
    try {
      const cols = await listCollections(token);
      setVault(cols.find((c) => c.id === vaultId) ?? null);
    } catch { /* non-fatal — header just shows "Vault" */ }
  }, [token, vaultId]);

  const loadDocs = useCallback(async () => {
    if (!token || !vaultId) return;
    try {
      setDocs(await getCollectionDocuments(token, vaultId));
      setLastSync(Date.now());
    } catch {
      toast.error("Failed to load documents");
    } finally {
      setLoading(false);
    }
  }, [token, vaultId]);

  useEffect(() => {
    loadVault();
    loadDocs();
  }, [loadVault, loadDocs]);

  // Live status poll: while any doc is still processing, refetch every 3s so the row's
  // parse→chunk→embed progress advances and flips to ready/failed (same logic the old
  // sidebar used). Clears itself when nothing is processing.
  useEffect(() => {
    // Poll faster (1.2s) while anything is processing so the pipeline track catches the
    // intermediate Parse→Chunk→Embed bands rather than snapping straight to Ready; the
    // poll stops itself once nothing is processing.
    if (docs.some((d) => d.status === "processing")) {
      pollRef.current = setTimeout(loadDocs, 1200);
    }
    return () => { if (pollRef.current) clearTimeout(pollRef.current); };
  }, [docs, loadDocs]);

  // G3 Step E: the active filter set as the backend's metadata_filter shape
  // ({doc_type, fiscal_year}). null when no filter is active → no narrowing. This is the
  // SINGLE source of truth for both the table view and the retrieval scope, so the two
  // can never disagree.
  const activeFilters: Record<string, string | number> | null =
    typeFilter || yearFilter != null
      ? {
          ...(typeFilter ? { doc_type: typeFilter } : {}),
          ...(yearFilter != null ? { fiscal_year: yearFilter } : {}),
        }
      : null;

  // The filter set travels to Ask/Review EXPLICITLY in the URL (mirror §9 risk #1 — the
  // request carries the scope; no stale global store). Encoded as a compact JSON param the
  // child page reads back and puts on the stream body.
  const filtersParam = activeFilters
    ? `&filters=${encodeURIComponent(JSON.stringify(activeFilters))}`
    : "";

  // Submitting the composer opens an Ask conversation scoped to this vault. Step D
  // re-homed Ask under /app/vault/[id]/ask/[cid] — the route [id] is the authoritative
  // scope (§9 risk #1), so the conversation reads collection_id from the URL, not the
  // store. ?q= auto-submits on mount; &filters= carries the active vault filter (G3 E).
  async function ask(q: string) {
    if (!token || creating || !q.trim()) return;
    setCreating(true);
    try {
      const c = await createConversation(token, q.slice(0, 50));
      router.push(
        `/app/vault/${vaultId}/ask/${c.id}?q=${encodeURIComponent(q)}${filtersParam}`
      );
    } catch {
      toast.error("Failed to start conversation");
      setCreating(false);
    }
  }

  // Delete a document (optimistic; deleteDocument treats 404 as success — the delete-
  // robustness fix). Restores the row on a real failure.
  async function handleDelete(id: string) {
    if (!token) return;
    setConfirmDel(null);
    const prev = docs;
    setDocs((p) => p.filter((d) => d.id !== id));
    setVault((v) => (v ? { ...v, document_count: Math.max(0, (v.document_count ?? 1) - 1) } : v));
    try {
      await deleteDocument(token, id);
      toast.success("Document deleted");
    } catch {
      setDocs(prev);
      toast.error("Failed to delete document");
    }
  }

  // Chip option lists, derived from the REAL doc metadata in this vault (so a chip only
  // appears when at least one doc carries that value — no dead chips).
  const typeOptions = Array.from(
    new Set(docs.map((d) => d.doc_type).filter((t): t is NonNullable<typeof t> => !!t))
  );
  const yearOptions = Array.from(
    new Set(docs.map((d) => d.fiscal_year).filter((y): y is number => y != null))
  ).sort((a, b) => b - a);

  // View filter: filename search AND the active doc_type / fiscal_year chips. Null-safe
  // for fiscal_year — a doc with an UNKNOWN year is KEPT (we never hide a doc on a value
  // we couldn't derive; mirrors the backend's null-safe filter).
  const filtered = docs.filter((d) => {
    if (query.trim() && !d.filename.toLowerCase().includes(query.toLowerCase())) return false;
    if (typeFilter && d.doc_type !== typeFilter) return false;
    if (yearFilter != null && d.fiscal_year != null && d.fiscal_year !== yearFilter) return false;
    return true;
  });
  // Processing docs walk the pipeline tracks (the live verification view); settled docs
  // (ready/failed) live in the table below. Both come from the same polled list.
  const processing = filtered.filter((d) => d.status === "processing");
  const settled = filtered.filter((d) => d.status !== "processing");

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin relative">

      <div className="relative max-w-5xl mx-auto px-6 py-8">
        {/* Back */}
        <button
          onClick={() => router.push("/app")}
          className="inline-flex items-center gap-1.5 text-[12px] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors mb-5"
        >
          <ArrowLeft size={13} /> All vaults
        </button>

        {/* Header */}
        <div className="flex items-center gap-3 mb-8">
          <span
            className="w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0"
            style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
          >
            <FolderOpen size={19} />
          </span>
          <div className="min-w-0">
            <h1
              className="truncate"
              style={{ fontFamily: "Fraunces, Georgia, serif", fontSize: 27, fontWeight: 500, letterSpacing: "-0.025em", color: "var(--ink)" }}
            >
              {loading && !vault ? "…" : vault?.name ?? "Vault"}
            </h1>
            <p className="text-[12px] text-[var(--text-muted)]">
              {docs.length} file{docs.length === 1 ? "" : "s"}
            </p>
          </div>
        </div>

        {/* Action row — Review · Deep Analysis · Draft as siblings above the composer
            (plan §8.1). Ask is the hero composer below; these are the other vault modes.
            Deep Analysis is LIVE (G5); Draft is the G6 stub. */}
        <div className="flex items-center justify-center gap-2.5 mb-4">
          <button
            onClick={() =>
              router.push(
                `/app/vault/${encodeURIComponent(vaultId)}/review${
                  activeFilters ? `?filters=${encodeURIComponent(JSON.stringify(activeFilters))}` : ""
                }`
              )
            }
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-[13px] font-medium card hover:shadow-[var(--shadow-md)] transition-shadow"
          >
            <Table2 size={14} className="text-[var(--text-muted)]" /> Review table
          </button>
          <button
            onClick={() => router.push(`/app/vault/${encodeURIComponent(vaultId)}/deep`)}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-[13px] font-medium card hover:shadow-[var(--shadow-md)] transition-shadow"
          >
            <Telescope size={14} className="text-[var(--text-muted)]" /> Deep Analysis
          </button>
          <button
            onClick={() => router.push(`/app/vault/${encodeURIComponent(vaultId)}/draft`)}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-[13px] font-medium card hover:shadow-[var(--shadow-md)] transition-shadow"
          >
            <FileEdit size={14} className="text-[var(--text-muted)]" /> Draft document
          </button>
        </div>

        {/* HERO composer — the vault screen IS the ask surface (plan §8.1). */}
        <ChatInput
          onSubmit={ask}
          isStreaming={creating}
          placeholder="Ask about a clause, term, or risk across this vault…"
          vaultName={vault?.name ?? null}
          centered
        />

        {/* Suggested-prompt chips (doc-type-aware; legal default until Step F). */}
        <div className="flex flex-wrap items-center justify-center gap-2 -mt-1 mb-10 px-4">
          {CONTRACT_PROMPTS.map((p, i) => (
            <motion.button
              key={p}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 + i * 0.04, ease }}
              onClick={() => ask(p)}
              disabled={creating}
              className="px-3 py-1.5 rounded-full text-[12px] transition-colors disabled:opacity-50"
              style={{ background: "var(--surface)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
            >
              {p}
            </motion.button>
          ))}
        </div>

        {/* ── Files section ── */}
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-[14px] font-semibold" style={{ color: "var(--ink)" }}>Files</h2>
          <div className="flex items-center gap-2.5">
            {lastSync && (
              <span className="text-[11px] text-[var(--text-muted)] hidden sm:inline">
                Last synced {timeAgo(new Date(lastSync).toISOString())}
              </span>
            )}
            <div
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg"
              style={{ background: "var(--surface)", border: "1px solid var(--line)" }}
            >
              <Search size={12} className="text-[var(--text-muted)]" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search files"
                className="bg-transparent outline-none text-[12px] w-28 text-[var(--text-primary)] placeholder:text-[var(--text-muted)]"
              />
            </div>
            <button
              onClick={loadDocs}
              title="Refresh"
              className="p-1.5 rounded-lg hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
            >
              <RefreshCw size={13} />
            </button>
          </div>
        </div>

        {/* G3 Step E — vault filter chips. doc_type / fiscal_year, derived from the real
            doc metadata. Selecting one narrows the table view AND becomes scope.filters on
            the next Ask/Review stream (so "FY2023 only" actually scopes what the agent
            searches). Only shown when there's something to filter (≥2 options). */}
        {(typeOptions.length > 1 || yearOptions.length > 1) && (
          <div className="flex flex-wrap items-center gap-2 mb-3">
            <span className="text-[11px] uppercase tracking-wide text-[var(--text-muted)] mr-1">
              Filter
            </span>
            {typeOptions.length > 1 &&
              typeOptions.map((t) => {
                const active = typeFilter === t;
                return (
                  <button
                    key={t}
                    onClick={() => setTypeFilter(active ? null : t)}
                    className="px-2.5 py-1 rounded-full text-[12px] font-medium transition-colors"
                    style={{
                      background: active ? "var(--ink)" : "var(--surface)",
                      border: "1px solid var(--line)",
                      color: active ? "var(--surface)" : "var(--ink-2)",
                    }}
                  >
                    {TYPE_LABELS[t] ?? t}
                  </button>
                );
              })}
            {yearOptions.length > 1 &&
              yearOptions.map((y) => {
                const active = yearFilter === y;
                return (
                  <button
                    key={y}
                    onClick={() => setYearFilter(active ? null : y)}
                    className="px-2.5 py-1 rounded-full text-[12px] font-medium transition-colors tabular-nums"
                    style={{
                      background: active ? "var(--ink)" : "var(--surface)",
                      border: "1px solid var(--line)",
                      color: active ? "var(--surface)" : "var(--ink-2)",
                    }}
                  >
                    FY{y}
                  </button>
                );
              })}
            {activeFilters && (
              <button
                onClick={() => { setTypeFilter(null); setYearFilter(null); }}
                className="px-2 py-1 rounded-full text-[11px] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
              >
                Clear
              </button>
            )}
          </div>
        )}

        {/* Ingress — slim command bar + full-page drag overlay (the NEW hybrid UX). */}
        <div className="mb-3">
          <UploadZone token={token ?? ""} collectionId={vaultId} onUploaded={loadDocs} />
        </div>

        {/* Live pipeline tracks — processing docs visibly walk Parse→Chunk→Embed→Ready.
            This is our differentiator: the verification pipeline is SEEN, not hidden. */}
        <AnimatePresence>
          {processing.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="space-y-2 mb-3 overflow-hidden"
            >
              {processing.map((d) => (
                <PipelineTrack key={d.id} doc={d} fidelity={null} />
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Settled files table (ready / failed) */}
        <div className="rounded-2xl overflow-hidden mb-5" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
          {loading ? (
            <div className="p-8 text-center text-[13px] text-[var(--text-muted)]">Loading files…</div>
          ) : settled.length === 0 ? (
            processing.length === 0 ? (
              <EmptyState
                icon={<FileText size={18} />}
                title={query ? "No files match" : "No files yet"}
                description={query ? "Try a different search." : "Add documents above — drag them anywhere on this page, or use the bar."}
              />
            ) : null
          ) : (
            <table className="w-full text-left" style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr className="text-[11px] uppercase tracking-wide text-[var(--text-muted)]">
                  <th className="font-medium px-4 py-2.5">Name</th>
                  <th className="font-medium px-3 py-2.5 hidden md:table-cell">Category</th>
                  <th className="font-medium px-3 py-2.5 hidden sm:table-cell">Type</th>
                  <th className="font-medium px-3 py-2.5 hidden lg:table-cell">Fidelity</th>
                  <th className="font-medium px-3 py-2.5 hidden md:table-cell">Modified</th>
                  <th className="font-medium px-3 py-2.5 hidden sm:table-cell">Size</th>
                  <th className="font-medium px-4 py-2.5">Status</th>
                  <th className="font-medium px-3 py-2.5 w-px" aria-label="Actions" />
                </tr>
              </thead>
              <tbody>
                {settled.map((d) => (
                  <tr
                    key={d.id}
                    className="group text-[13px] border-t hover:bg-[var(--bg-hover)] transition-colors"
                    style={{ borderColor: "var(--line)" }}
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2.5 min-w-0">
                        <FileText size={14} className="text-[var(--text-muted)] flex-shrink-0" />
                        <span className="truncate text-[var(--text-primary)] max-w-[280px]">{d.filename}</span>
                      </div>
                    </td>
                    {/* Category = G1d doc_type (domain) — real value (Step F); null → neutral chip */}
                    <td className="px-3 py-3 hidden md:table-cell"><DocTypeChip type={d.doc_type ?? null} /></td>
                    {/* Type = file format (distinct from Category) */}
                    <td className="px-3 py-3 hidden sm:table-cell text-[var(--text-muted)] text-[12px]">{fileFormat(d)}</td>
                    {/* Fidelity = trust dot — real value (Step F); null → hollow neutral dot */}
                    <td className="px-3 py-3 hidden lg:table-cell"><FidelityDot fidelity={d.fidelity ?? null} /></td>
                    <td className="px-3 py-3 hidden md:table-cell text-[var(--text-muted)] text-[12px]">{timeAgo(d.created_at)}</td>
                    <td className="px-3 py-3 hidden sm:table-cell text-[var(--text-muted)] text-[12px]">{fmtSize(d.file_size_bytes)}</td>
                    <td className="px-4 py-3">
                      {d.status === "ready" ? (
                        <span className="inline-flex items-center gap-1.5 text-[12px]" style={{ color: "var(--fidelity-good)" }}>
                          <CheckCircle size={13} /> Ready
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 text-[12px]" style={{ color: "var(--status-failed)" }}>
                          <AlertCircle size={13} /> Failed
                        </span>
                      )}
                    </td>
                    {/* Delete — hover-revealed trash that flips to an inline confirm */}
                    <td className="px-3 py-3 text-right whitespace-nowrap">
                      {confirmDel === d.id ? (
                        <span className="inline-flex items-center gap-1.5">
                          <button
                            onClick={() => handleDelete(d.id)}
                            className="text-[11px] px-2 py-1 rounded-md font-medium"
                            style={{ background: "var(--status-failed)", color: "#fff" }}
                          >
                            Delete
                          </button>
                          <button
                            onClick={() => setConfirmDel(null)}
                            className="text-[11px] px-2 py-1 rounded-md text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                          >
                            Cancel
                          </button>
                        </span>
                      ) : (
                        <button
                          onClick={() => setConfirmDel(d.id)}
                          title="Delete document"
                          aria-label={`Delete ${d.filename}`}
                          className="p-1.5 rounded-lg text-[var(--text-muted)] opacity-0 group-hover:opacity-100 hover:text-[var(--status-failed)] hover:bg-[var(--bg-hover)] transition-[color,opacity] focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
                        >
                          <Trash2 size={14} />
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
