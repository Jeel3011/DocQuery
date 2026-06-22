"use client";

// /app — Vault Home (G2 Step B).
// A card grid of vaults (= collections): name · file count · last activity · a subtle
// type-mix indicator. "+ New Vault" creates one (own dialog); empty state invites the
// first vault. Card click → /app/vault/[id]. Born in the clean B&W system; status color
// appears only on the fidelity-style type dots (meaning, not decoration).
//
// Scope rule (G2 §9 risk #1): this page is OFF a vault route, so it deliberately does NOT
// touch collection.store — the route becomes the source of truth only once you open a
// vault (VaultScopeSync handles that in the shell).

import { useEffect, useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { FolderOpen, FileText, Plus, X, ShieldAlert } from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/auth.store";
import {
  listCollections,
  createCollection,
  getPracticeTemplate,
  scanConflicts,
  CollectionResponse,
  PracticeTemplate,
  ConflictFinding,
  MatterParty,
  MATTER_KINDS,
} from "@/lib/api";
import Folder from "@/components/ui/Folder";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/Dialog";

const ease = [0.23, 1, 0.32, 1] as const;

// F1c: which flagship a matter kind foregrounds → a human label for the "pins" line.
const FLAGSHIP_LABEL: Record<string, string> = {
  review_grid: "Review grid",
  covenant_cockpit: "Covenant cockpit",
  obligation_sentinel: "Obligation sentinel",
  argument_engine: "Argument engine",
  case_file: "Case file",
};
const KB_LABEL: Record<string, string> = {
  vault: "Your documents",
  statutes: "Statutes",
  caselaw: "Case law",
};

function timeAgo(d: string | null): string {
  if (!d) return "";
  const m = Math.floor((Date.now() - new Date(d).getTime()) / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export default function VaultHomePage() {
  const router = useRouter();
  const { token } = useAuthStore();
  const [collections, setCollections] = useState<CollectionResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNew, setShowNew] = useState(false);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  // F1c: matter typing + practice template + conflict scan, all in the create dialog.
  const [matterKind, setMatterKind] = useState<string | null>(null);
  const [parties, setParties] = useState<MatterParty[]>([]);
  const [template, setTemplate] = useState<PracticeTemplate | null>(null);
  const [conflicts, setConflicts] = useState<ConflictFinding[]>([]);
  const conflictTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const load = useCallback(async () => {
    if (!token) return;
    try {
      const cols = await listCollections(token);
      cols.sort(
        (a, b) => new Date(b.updated_at ?? 0).getTime() - new Date(a.updated_at ?? 0).getTime()
      );
      setCollections(cols);
    } catch {
      toast.error("Failed to load vaults");
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    load();
  }, [load]);

  // F1c: load the practice template when a matter kind is picked. $0 on the server (static
  // map); the dialog shows the seed grid columns, KB defaults, and flagship pin it returns.
  useEffect(() => {
    if (!token || !showNew || !matterKind) {
      setTemplate(null);
      return;
    }
    let live = true;
    getPracticeTemplate(token, matterKind)
      .then((t) => { if (live) setTemplate(t); })
      .catch(() => { if (live) setTemplate(null); });
    return () => { live = false; };
  }, [token, showNew, matterKind]);

  // F1c: metadata-only conflict scan, debounced as parties are edited (pre-create, so the
  // banner can show before committing). Only named parties are sent; nothing else.
  useEffect(() => {
    if (conflictTimer.current) clearTimeout(conflictTimer.current);
    const named = parties.filter((p) => p.name.trim());
    if (!token || !showNew || named.length === 0) {
      setConflicts([]);
      return;
    }
    conflictTimer.current = setTimeout(() => {
      scanConflicts(token, named)
        .then((r) => setConflicts(r.conflicts || []))
        .catch(() => setConflicts([]));
    }, 450);
    return () => { if (conflictTimer.current) clearTimeout(conflictTimer.current); };
  }, [token, showNew, parties]);

  function resetDialog() {
    setNewName("");
    setMatterKind(null);
    setParties([]);
    setTemplate(null);
    setConflicts([]);
  }

  async function handleCreate() {
    if (!token || !newName.trim() || creating) return;
    setCreating(true);
    try {
      const named = parties.filter((p) => p.name.trim());
      const c = await createCollection(token, newName.trim(), undefined, {
        matter_kind: matterKind,
        parties: named.length ? named : null,
      });
      // The create response carries the authoritative conflict result; surface adverse hits.
      if (c.has_adverse) {
        toast.warning("Conflict check flagged an adverse party. Review in vault settings.");
      }
      setShowNew(false);
      resetDialog();
      toast.success(`Created vault "${c.name}"`);
      router.push(`/app/vault/${c.id}`);
    } catch {
      toast.error("Failed to create vault");
    } finally {
      setCreating(false);
    }
  }

  const adverseCount = conflicts.filter((c) => c.severity === "adverse").length;

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin relative">

      <div className="relative max-w-5xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="flex items-end justify-between mb-8">
          <div>
            <h1
              style={{
                fontFamily: "Fraunces, Georgia, serif",
                fontSize: 32,
                fontWeight: 500,
                letterSpacing: "-0.03em",
                color: "var(--ink)",
                lineHeight: 1.05,
              }}
            >
              Vaults
            </h1>
            <p className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
              {loading
                ? "Loading…"
                : collections.length === 0
                ? "Create a vault to organize a matter, deal, or document set."
                : `${collections.length} vault${collections.length === 1 ? "" : "s"}`}
            </p>
          </div>
          {!loading && collections.length > 0 && (
            <button className="btn-primary flex items-center gap-1.5" onClick={() => setShowNew(true)}>
              <Plus size={14} /> New vault
            </button>
          )}
        </div>

        {/* Body */}
        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <div
                key={i}
                className="h-[124px] rounded-2xl animate-pulse"
                style={{ background: "var(--surface-3)", border: "1px solid var(--line)" }}
              />
            ))}
          </div>
        ) : collections.length === 0 ? (
          <div
            className="rounded-2xl flex flex-col items-center text-center px-6 py-14"
            style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-sm)" }}
          >
            {/* Animated folder — click to fan it open. Sized with bottom headroom so
                the papers have room to lift without clipping the heading below. */}
            <div className="mb-12 mt-2" style={{ height: 100 }}>
              <Folder color="#0E0E0E" size={1.4} />
            </div>
            <h3
              className="text-[17px] font-semibold"
              style={{ color: "var(--ink)", fontFamily: "Fraunces, Georgia, serif", letterSpacing: "-0.01em" }}
            >
              No vaults yet
            </h3>
            <p className="text-[13px] mt-1.5 max-w-[420px]" style={{ color: "var(--ink-3)" }}>
              A vault holds the documents for one matter — a deal, a contract set, or a filing.
              Create your first to begin.
            </p>
            <button className="btn-primary flex items-center gap-1.5 mt-5" onClick={() => setShowNew(true)}>
              <Plus size={14} /> Create your first vault
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {collections.map((c, i) => (
              <motion.button
                key={c.id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: Math.min(i * 0.04, 0.3), duration: 0.45, ease }}
                whileHover={{ y: -3 }}
                onClick={() => router.push(`/app/vault/${c.id}`)}
                className="group text-left p-5 rounded-2xl flex flex-col gap-4 transition-shadow card hover:shadow-[var(--shadow-lg)]"
                style={{ borderColor: "var(--line)" }}
              >
                <div className="flex items-start justify-between">
                  <span
                    className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                    style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
                  >
                    <FolderOpen size={17} />
                  </span>
                  {/* Type-mix indicator — kept a single neutral "documents present" dot.
                      listCollections returns only document_count, not per-doc doc_type, so a
                      real type-mix would cost an extra getCollectionDocuments per card on the
                      home grid. The real type/fidelity surfacing (Step F) lives in the vault
                      doc table where the data is already fetched; the home dot stays neutral. */}
                  {(c.document_count ?? 0) > 0 && (
                    <span
                      className="w-2 h-2 rounded-full mt-1.5"
                      style={{ background: "var(--line-2)" }}
                      title="Document set"
                    />
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <p
                    className="text-[15px] font-semibold truncate"
                    style={{ color: "var(--ink)", fontFamily: "Fraunces, Georgia, serif", letterSpacing: "-0.01em" }}
                  >
                    {c.name}
                  </p>
                  <p className="text-[12px] text-[var(--text-muted)] flex items-center gap-1.5 mt-1.5">
                    <FileText size={12} />
                    {c.document_count ?? 0} file{(c.document_count ?? 0) === 1 ? "" : "s"}
                    {c.updated_at && <span aria-hidden>·</span>}
                    {c.updated_at && <span>{timeAgo(c.updated_at)}</span>}
                  </p>
                </div>
              </motion.button>
            ))}
          </div>
        )}
      </div>

      {/* New-vault dialog (this page's own entry point — the switcher has its own in the shell).
          F1c: matter-kind picker + parties → the practice template loads (grid columns, KB
          default chips, flagship pin) and a metadata-only conflict scan runs. */}
      <Dialog
        open={showNew}
        onOpenChange={(o) => { setShowNew(o); if (!o) resetDialog(); }}
      >
        <DialogContent maxWidth="520px">
          <DialogHeader>
            <DialogTitle>New vault</DialogTitle>
          </DialogHeader>
          <div className="px-5 py-5 flex-1 min-h-0 overflow-y-auto scrollbar-thin flex flex-col gap-5">
            {/* Vault name */}
            <div>
              <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-2">Vault name</label>
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleCreate();
                  if (e.key === "Escape") setShowNew(false);
                }}
                placeholder="e.g. Acme acquisition contracts"
                autoFocus
                className="w-full px-3 py-2.5 rounded-xl text-[14px] outline-none focus:border-[var(--accent)]"
                style={{ background: "var(--surface-2)", border: "1px solid var(--line)", color: "var(--ink)" }}
              />
            </div>

            {/* Matter-kind picker — the 9 kinds as toggles. Picking one loads the template. */}
            <div>
              <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-2">
                Matter type <span className="text-[var(--ink-3)] font-normal">(optional)</span>
              </label>
              <div className="flex flex-wrap gap-1.5">
                {MATTER_KINDS.map((k) => {
                  const on = matterKind === k.value;
                  return (
                    <button
                      key={k.value}
                      type="button"
                      onClick={() => setMatterKind(on ? null : k.value)}
                      className="px-2.5 py-1 rounded-lg text-[12px] font-medium transition-colors duration-[120ms]"
                      style={{
                        background: on ? "var(--accent)" : "var(--surface-3)",
                        color: on ? "#fff" : "var(--ink-2)",
                        border: `1px solid ${on ? "var(--accent)" : "var(--line)"}`,
                      }}
                    >
                      {k.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Practice-template preview — visibly loads on pick (the plan's hard rule). */}
            {template && matterKind && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, ease }}
                className="rounded-xl p-4"
                style={{ background: "var(--surface-2)", border: "1px solid var(--line)" }}
              >
                <p className="text-[12px] font-semibold text-[var(--ink-2)] mb-0.5">Practice template</p>
                {template.summary && (
                  <p className="text-[12px] text-[var(--ink-3)] mb-3">{template.summary}</p>
                )}
                {/* Seed review-grid columns */}
                <div className="mb-3">
                  <p className="text-[11px] uppercase tracking-wide text-[var(--ink-3)] mb-1.5">
                    Review columns ({template.grid_columns.length})
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {template.grid_columns.map((c) => (
                      <span
                        key={c.key}
                        className="px-2 py-0.5 rounded text-[11px]"
                        style={{ background: "var(--surface-3)", color: "var(--ink-2)", border: "1px solid var(--line)" }}
                      >
                        {c.label}
                      </span>
                    ))}
                  </div>
                </div>
                {/* KB default source chips + flagship pin */}
                <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[11px] uppercase tracking-wide text-[var(--ink-3)]">Knowledge</span>
                    {template.kb_scope.map((s) => (
                      <span key={s} className="text-[11px] text-[var(--ink-2)]">{KB_LABEL[s] ?? s}</span>
                    ))}
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[11px] uppercase tracking-wide text-[var(--ink-3)]">Opens</span>
                    <span className="text-[11px] text-[var(--ink-2)]">
                      {FLAGSHIP_LABEL[template.flagship] ?? template.flagship}
                    </span>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Parties — optional; drives the metadata-only conflict scan. */}
            <div>
              <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-2">
                Parties <span className="text-[var(--ink-3)] font-normal">(optional — used to screen conflicts)</span>
              </label>
              <div className="flex flex-col gap-2">
                {parties.map((p, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <input
                      value={p.name}
                      onChange={(e) => setParties((prev) => prev.map((q, j) => j === i ? { ...q, name: e.target.value } : q))}
                      placeholder="Party name"
                      className="flex-1 px-3 py-2 rounded-lg text-[13px] outline-none focus:border-[var(--accent)]"
                      style={{ background: "var(--surface-2)", border: "1px solid var(--line)", color: "var(--ink)" }}
                    />
                    <input
                      value={p.role ?? ""}
                      onChange={(e) => setParties((prev) => prev.map((q, j) => j === i ? { ...q, role: e.target.value } : q))}
                      placeholder="Role (e.g. client, opposing)"
                      className="w-[170px] px-3 py-2 rounded-lg text-[13px] outline-none focus:border-[var(--accent)]"
                      style={{ background: "var(--surface-2)", border: "1px solid var(--line)", color: "var(--ink)" }}
                    />
                    <button
                      type="button"
                      onClick={() => setParties((prev) => prev.filter((_, j) => j !== i))}
                      className="p-1.5 rounded-lg text-[var(--ink-3)] hover:text-[var(--ink)] hover:bg-[var(--surface-3)] transition-colors"
                      aria-label="Remove party"
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
                <button
                  type="button"
                  onClick={() => setParties((prev) => [...prev, { name: "", role: "" }])}
                  className="self-start flex items-center gap-1 text-[12px] font-medium text-[var(--ink-2)] hover:text-[var(--ink)] transition-colors"
                >
                  <Plus size={13} /> Add party
                </button>
              </div>
            </div>

            {/* Conflict-check banner — non-blocking. Adverse hits use --status-failed tone;
                same-party notes use --status-processing. Never blocks create. */}
            {conflicts.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, ease }}
                className="rounded-xl p-3.5 flex gap-2.5"
                style={{
                  background: "var(--surface-2)",
                  border: `1px solid ${adverseCount > 0 ? "var(--status-failed)" : "var(--status-processing)"}`,
                }}
              >
                <ShieldAlert
                  size={16}
                  className="mt-0.5 flex-shrink-0"
                  style={{ color: adverseCount > 0 ? "var(--status-failed)" : "var(--status-processing)" }}
                />
                <div className="min-w-0">
                  <p className="text-[12px] font-semibold text-[var(--ink)]">
                    {adverseCount > 0
                      ? `Conflict check found ${adverseCount} adverse-party ${adverseCount === 1 ? "match" : "matches"}`
                      : `Conflict check found ${conflicts.length} related ${conflicts.length === 1 ? "matter" : "matters"}`}
                  </p>
                  <ul className="mt-1 flex flex-col gap-0.5">
                    {conflicts.slice(0, 4).map((c, i) => (
                      <li key={i} className="text-[11px] text-[var(--ink-3)]">
                        <span className="text-[var(--ink-2)]">{c.party}</span>
                        {" — "}
                        {c.severity === "adverse" ? "adverse to" : "also on"}{" "}
                        <span className="text-[var(--ink-2)]">{c.matter_name ?? "another matter"}</span>
                      </li>
                    ))}
                  </ul>
                  <p className="text-[11px] text-[var(--ink-3)] mt-1.5">
                    This does not block creating the vault. Route to admin to clear the screen.
                  </p>
                </div>
              </motion.div>
            )}
          </div>
          <DialogFooter>
            <button className="btn-ghost" onClick={() => { setShowNew(false); resetDialog(); }}>Cancel</button>
            <button className="btn-primary" onClick={handleCreate} disabled={!newName.trim() || creating}>
              {creating ? "Creating…" : "Create vault"}
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
