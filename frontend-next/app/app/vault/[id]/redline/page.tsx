"use client";

// /app/vault/[id]/redline — Redline (G6.3 · LEGAL_TASK_CATALOG §2.1 REDLINE verb).
//
// A TWO-PANE review workspace: a compact SETUP rail (left) drives the run; the FINDINGS
// stream fills the working pane (right) as the backend emits one grounded finding per
// clause topic. Each finding is quoted to the document or flagged MISSING/ABSTAIN — never
// a silent gap (the bind-or-flag moat). Compare against EITHER a catalog doc type (the
// §2.1 link — clause topics derived from the doc type's structure + the firm playbook) OR
// the firm playbook directly. The completed redline exports as a tracked-changes .docx.
//
// Monochrome (the B&W law aesthetic, --ink/--line/--surface tokens); colour carries
// verdict meaning ONLY. Scope: the vault [id] is authoritative — read from the route.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  Highlighter, Play, Loader2, FileText, Download, CheckCircle2, AlertTriangle,
  MinusCircle, HelpCircle, ChevronDown, ScrollText, BookMarked, Check,
} from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/auth.store";
import {
  getCollectionDocuments, listDocTypes, exportRedlineDocx,
  type DocumentResponse, type DocTypeCard,
} from "@/lib/api";
import {
  streamRedline, type RedlineFindingEvent, type RedlineDoneEvent,
} from "@/lib/streaming";
import { BackToVault } from "@/components/app/BackToVault";

const ease = [0.23, 1, 0.32, 1] as const;

// Finding status → icon + label + accent. Colour means VERDICT only (the one place the
// monochrome aesthetic admits colour): deviation = attention, conforming = ok, rest neutral.
const STATUS_META: Record<
  RedlineFindingEvent["status"],
  { icon: typeof CheckCircle2; label: string; ink: string; dot: string }
> = {
  deviation: { icon: AlertTriangle, label: "Deviates", ink: "#B45309", dot: "#F59E0B" },
  conforming: { icon: CheckCircle2, label: "Conforms", ink: "#047857", dot: "#10B981" },
  missing: { icon: MinusCircle, label: "Missing", ink: "var(--ink-2)", dot: "var(--ink-3)" },
  abstain: { icon: HelpCircle, label: "Unclear", ink: "var(--ink-3)", dot: "var(--line-2)" },
};

export default function VaultRedlinePage() {
  const params = useParams<{ id: string }>();
  const { token } = useAuthStore();
  const vaultId = params.id;

  const [docs, setDocs] = useState<DocumentResponse[]>([]);
  const [catalog, setCatalog] = useState<DocTypeCard[]>([]);
  const [targetDoc, setTargetDoc] = useState<string>("");
  const [docType, setDocType] = useState<string>("");          // "" = firm playbook
  const [running, setRunning] = useState(false);
  const [findings, setFindings] = useState<RedlineFindingEvent[]>([]);
  const [summary, setSummary] = useState<RedlineDoneEvent | null>(null);
  const [exporting, setExporting] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!token) return;
    getCollectionDocuments(token, vaultId)
      .then((d) => {
        const ready = (d ?? []).filter((x) => x.status === "ready");
        setDocs(ready);
        if (ready.length) setTargetDoc(ready[0].id);
      })
      .catch(() => setDocs([]));
    // You redline a contract, not a will — narrow the catalog to contract-shaped doc types.
    listDocTypes(token)
      .then((cards) =>
        setCatalog(
          (cards ?? []).filter(
            (c) => c.practice_area === "Transactional" || c.practice_area === "Financial services"
          )
        )
      )
      .catch(() => setCatalog([]));
  }, [token, vaultId]);

  const targetName = useMemo(
    () => docs.find((d) => d.id === targetDoc)?.filename ?? "document",
    [docs, targetDoc]
  );
  const referenceLabel = useMemo(
    () => (docType ? catalog.find((c) => c.id === docType)?.title ?? "Doc type" : "My firm playbook"),
    [docType, catalog]
  );

  const run = useCallback(() => {
    if (!token || !targetDoc) {
      toast.error("Pick a document to redline.");
      return;
    }
    setRunning(true);
    setFindings([]);
    setSummary(null);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    streamRedline(
      token,
      {
        collection_id: vaultId,
        doc_id: targetDoc,
        doc_type: docType || null,   // a catalog doc_type derives the clause topics (§2.1)
        title: `Redline — ${targetName}`,
      },
      {
        onFinding: (f) => setFindings((prev) => [...prev, f]),
        onDone: (d) => { setSummary(d); setRunning(false); },
        onError: (msg) => { toast.error(msg); setRunning(false); },
      },
      ctrl.signal
    );
  }, [token, targetDoc, docType, vaultId, targetName]);

  const onExport = useCallback(async () => {
    if (!token || !findings.length) return;
    setExporting(true);
    try {
      const blob = await exportRedlineDocx(
        token, `Redline — ${targetName}`, targetName,
        findings as unknown as Array<Record<string, unknown>>
      );
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `redline-${targetName.replace(/\.[^.]+$/, "")}.docx`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      toast.error("Could not export the redline.");
    } finally {
      setExporting(false);
    }
  }, [token, findings, targetName]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const reviewed = findings.length;

  return (
    <div className="min-h-screen" style={{ background: "var(--canvas)" }}>
      {/* Header band */}
      <div className="border-b" style={{ borderColor: "var(--line)", background: "var(--surface)" }}>
        <div className="mx-auto max-w-6xl px-6 py-4">
          <BackToVault vaultId={vaultId} className="mb-3" />
          <div className="flex items-center gap-2.5">
            <span className="grid h-8 w-8 place-items-center rounded-lg"
                  style={{ background: "var(--ink)", color: "var(--on-ink)" }}>
              <Highlighter size={16} />
            </span>
            <div>
              <h1 className="text-[17px] font-semibold leading-tight text-[var(--ink)]">Redline</h1>
              <p className="text-[12px] text-[var(--ink-3)]">
                Clause-by-clause review — every finding quoted to the document or flagged.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Two-pane workspace. The working pane sizes to its content (no forced viewport
          height — this page lives inside the app shell, so a 100vh calc overflows). */}
      <div className="mx-auto grid max-w-6xl grid-cols-1 items-start gap-6 px-6 py-6 lg:grid-cols-[300px_1fr]">

        {/* ── SETUP rail ──────────────────────────────────────────────────────── */}
        <aside className="lg:sticky lg:top-6 lg:self-start">
          <div className="rounded-xl p-4" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
            <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-[var(--ink-3)]">Setup</h2>

            <FieldLabel icon={ScrollText}>Document to redline</FieldLabel>
            <Dropdown value={targetDoc} onChange={setTargetDoc} disabled={running}
              placeholder={docs.length ? "Select a document" : "No ready documents"}
              options={docs.map((d) => ({ value: d.id, label: d.filename }))} />

            <div className="h-4" />

            <FieldLabel icon={BookMarked}>Compare against</FieldLabel>
            <Dropdown value={docType} onChange={setDocType} disabled={running}
              options={[
                { value: "", label: "My firm playbook" },
                ...catalog.map((c) => ({ value: c.id, label: `${c.title} — standard clauses` })),
              ]} />
            <p className="mt-1.5 text-[11px] leading-relaxed text-[var(--ink-3)]">
              A doc type derives its standard clause set; your stored playbook positions are used where you have them.
            </p>

            <button
              onClick={run}
              disabled={running || !targetDoc}
              className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-[13px] font-medium transition-colors disabled:opacity-50"
              style={{ background: "var(--accent)", color: "var(--on-ink)" }}
            >
              {running ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
              {running ? "Reviewing…" : "Run redline"}
            </button>

            {reviewed > 0 && !running && (
              <button
                onClick={onExport}
                disabled={exporting}
                className="mt-2 inline-flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2 text-[13px] font-medium transition-colors disabled:opacity-50"
                style={{ border: "1px solid var(--line-2)", color: "var(--ink-2)" }}
              >
                {exporting ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
                Export .docx
              </button>
            )}
          </div>

          {/* Summary chips */}
          {summary && (
            <div className="mt-3 rounded-xl p-4" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
              <h2 className="mb-2.5 text-[11px] font-semibold uppercase tracking-wide text-[var(--ink-3)]">Summary</h2>
              <div className="space-y-1.5">
                <SummaryRow meta={STATUS_META.deviation} n={summary.deviations} />
                <SummaryRow meta={STATUS_META.conforming} n={summary.conforming} />
                <SummaryRow meta={STATUS_META.missing} n={summary.missing} />
                <SummaryRow meta={STATUS_META.abstain} n={summary.abstained} />
              </div>
            </div>
          )}
        </aside>

        {/* ── FINDINGS pane ───────────────────────────────────────────────────── */}
        <main>
          {(running || reviewed > 0) && (
            <div className="mb-3 flex items-center gap-2 text-[12px] text-[var(--ink-3)]">
              <FileText size={13} />
              <span className="text-[var(--ink-2)]">{targetName}</span>
              <span>vs</span>
              <span className="text-[var(--ink-2)]">{referenceLabel}</span>
              <span>·</span>
              <span>{reviewed} clause{reviewed === 1 ? "" : "s"} reviewed</span>
            </div>
          )}

          <div className="space-y-3">
            <AnimatePresence initial={false}>
              {findings.map((f, i) => (
                <FindingCard key={`${f.clause_topic}-${i}`} f={f} index={i} />
              ))}
            </AnimatePresence>

            {running && (
              <div className="flex items-center gap-2 rounded-xl px-4 py-4 text-[13px] text-[var(--ink-3)]"
                   style={{ border: "1px dashed var(--line-2)" }}>
                <Loader2 size={14} className="animate-spin" /> reviewing the next clause…
              </div>
            )}

            {!running && reviewed === 0 && <EmptyState />}
          </div>
        </main>
      </div>
    </div>
  );
}

// ─── Pieces ──────────────────────────────────────────────────────────────────

function FieldLabel({ icon: Icon, children }: { icon: typeof ScrollText; children: React.ReactNode }) {
  return (
    <span className="mb-1.5 flex items-center gap-1.5 text-[11px] font-medium text-[var(--ink-2)]">
      <Icon size={12} className="text-[var(--ink-3)]" /> {children}
    </span>
  );
}

// A fully custom dropdown — the native <select>'s OPEN menu is OS-rendered (the black/yellow
// Mac list) and unstyleable, so we render our own trigger + popover in the app's tokens.
function Dropdown({
  value, onChange, options, disabled, placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  disabled?: boolean;
  placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const selected = options.find((o) => o.value === value);
  const display = selected?.label ?? placeholder ?? "Select…";

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDoc); document.removeEventListener("keydown", onKey); };
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2 text-left text-[13px] outline-none transition-colors disabled:opacity-60"
        style={{
          background: "var(--surface-2)",
          border: `1px solid ${open ? "var(--ink)" : "var(--line)"}`,
          color: selected || !placeholder ? "var(--ink)" : "var(--ink-3)",
        }}
      >
        <span className="truncate">{display}</span>
        <ChevronDown size={14} className="shrink-0 text-[var(--ink-3)] transition-transform"
          style={{ transform: open ? "rotate(180deg)" : "none" }} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.12 }}
            className="absolute left-0 right-0 z-30 mt-1.5 max-h-72 overflow-auto rounded-lg py-1 shadow-[var(--shadow-md)]"
            style={{ background: "var(--surface)", border: "1px solid var(--line)" }}
          >
            {options.map((o) => {
              const active = o.value === value;
              return (
                <button
                  key={o.value}
                  type="button"
                  onClick={() => { onChange(o.value); setOpen(false); }}
                  className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-[13px] transition-colors hover:bg-[var(--surface-3)]"
                  style={{ color: active ? "var(--ink)" : "var(--ink-2)" }}
                >
                  <span className="truncate">{o.label}</span>
                  {active && <Check size={14} className="shrink-0 text-[var(--ink)]" />}
                </button>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function SummaryRow({ meta, n }: { meta: (typeof STATUS_META)[keyof typeof STATUS_META]; n: number }) {
  return (
    <div className="flex items-center justify-between text-[12px]">
      <span className="flex items-center gap-2 text-[var(--ink-2)]">
        <span className="h-2 w-2 rounded-full" style={{ background: meta.dot }} />
        {meta.label}
      </span>
      <span className="font-semibold tabular-nums text-[var(--ink)]">{n}</span>
    </div>
  );
}

function FindingCard({ f, index }: { f: RedlineFindingEvent; index: number }) {
  const meta = STATUS_META[f.status];
  const Icon = meta.icon;
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease, delay: Math.min(index * 0.02, 0.2) }}
      className="overflow-hidden rounded-xl"
      style={{ background: "var(--surface)", border: "1px solid var(--line)" }}
    >
      {/* status spine + header */}
      <div className="flex items-center justify-between gap-3 px-4 py-3"
           style={{ borderBottom: f.target_quote || f.deviation || f.suggested_edit ? "1px solid var(--line)" : "none" }}>
        <div className="flex items-center gap-2.5">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: meta.dot }} />
          <h3 className="text-[13.5px] font-semibold text-[var(--ink)]">{f.clause_topic}</h3>
        </div>
        <span className="inline-flex items-center gap-1 text-[11.5px] font-medium" style={{ color: meta.ink }}>
          <Icon size={13} /> {meta.label}
        </span>
      </div>

      {(f.target_quote || f.deviation || f.suggested_edit || f.rationale) && (
        <div className="space-y-2.5 px-4 py-3">
          {f.target_quote && (
            <blockquote className="border-l-2 pl-3 text-[12.5px] italic leading-relaxed text-[var(--ink-2)]"
                        style={{ borderColor: "var(--line-2)" }}>
              “{f.target_quote}”
            </blockquote>
          )}
          {f.deviation && <p className="text-[12.5px] leading-relaxed text-[var(--ink-2)]">{f.deviation}</p>}
          {f.suggested_edit && (
            <div className="rounded-lg p-3" style={{ background: "var(--surface-3)" }}>
              <span className="mb-1 block text-[10.5px] font-semibold uppercase tracking-wide text-[var(--ink-3)]">
                Suggested edit
              </span>
              <span className="text-[12.5px] leading-relaxed text-[var(--ink)]">{f.suggested_edit}</span>
            </div>
          )}
          {f.rationale && <p className="text-[11.5px] leading-relaxed text-[var(--ink-3)]">{f.rationale}</p>}
        </div>
      )}
    </motion.div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl px-6 text-center"
         style={{ border: "1px dashed var(--line)", background: "var(--surface-2)", minHeight: 360 }}>
      <span className="mb-4 grid h-12 w-12 place-items-center rounded-full"
            style={{ background: "var(--surface-3)" }}>
        <Highlighter size={20} className="text-[var(--ink-3)]" />
      </span>
      <p className="text-[13.5px] font-medium text-[var(--ink-2)]">No findings yet</p>
      <p className="mt-1.5 max-w-sm text-[12.5px] leading-relaxed text-[var(--ink-3)]">
        Pick a document and a reference in the panel, then run the redline. Each clause appears
        here as it is reviewed — quoted to the document or flagged.
      </p>
    </div>
  );
}
