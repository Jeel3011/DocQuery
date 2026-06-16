"use client";

// /app/vault/[id]/workflows — Workflow Agents (G7).
//
// A full-page, Harvey-shaped gallery: practice-area sections, cards tagged by OUTPUT TYPE
// (Review / Draft / Output) + step count, search + output-type filter. Picking a card opens
// a clean run drawer (the form is rendered GENERICALLY from params_schema — the §3
// declarative bet). The result routes BY SHAPE into the right surface:
//   · Review (grid)   → a live cell table (same shapes as the Review grid)
//   · Draft (report)  → a streamed, cited markdown deliverable
//   · Output (output) → a streamed freeform deliverable
// Monochrome (the B&W law aesthetic) — no decorative colour; colour means verdict only.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import {
  Search, Play, X, FileText, CheckCircle2, HelpCircle, MinusCircle,
  Table2, PenLine, Sparkles, Loader2,
} from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/auth.store";
import {
  listWorkflows, getCollectionDocuments, WorkflowCard, WorkflowParamSpec, DocumentResponse,
  SourceInfo,
} from "@/lib/api";
import {
  streamWorkflowRun, streamWorkflowReport, GridCellEvent, GridStart, GridDone, WorkflowStep,
} from "@/lib/streaming";
import { ThinkingStream, ThinkingStep } from "@/components/chat/ThinkingStream";
import { BackToVault } from "@/components/app/BackToVault";

const ease = [0.23, 1, 0.32, 1] as const;
const ck = (docId: string, colKey: string) => `${docId}::${colKey}`;

// WorkflowStep (stream) → ThinkingStep (the shared live-timeline component).
function toThinkingStep(s: WorkflowStep): ThinkingStep {
  return { id: s.id, label: s.label, detail: s.detail, status: s.status };
}

// Output-type tag → icon + label (Harvey's Review / Draft / Output). Monochrome.
const OUTPUT_META: Record<string, { icon: typeof Table2; label: string }> = {
  Review: { icon: Table2, label: "Review" },
  Draft: { icon: PenLine, label: "Draft" },
  Output: { icon: Sparkles, label: "Output" },
};

export default function VaultWorkflowsPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const { token } = useAuthStore();
  const vaultId = params.id;

  const [cards, setCards] = useState<WorkflowCard[]>([]);
  const [loading, setLoading] = useState(true);
  const [active, setActive] = useState<WorkflowCard | null>(null);
  const [q, setQ] = useState("");
  const [typeFilter, setTypeFilter] = useState<string | null>(null);

  useEffect(() => {
    if (!token) return;
    listWorkflows(token).then(setCards).catch(() => setCards([])).finally(() => setLoading(false));
  }, [token]);

  const filtered = useMemo(() => {
    return cards.filter((c) => {
      if (typeFilter && c.output_type !== typeFilter) return false;
      if (q.trim()) {
        const s = `${c.title} ${c.description} ${c.practice_area}`.toLowerCase();
        if (!s.includes(q.toLowerCase())) return false;
      }
      return true;
    });
  }, [cards, q, typeFilter]);

  const grouped = useMemo(() => {
    const m = new Map<string, WorkflowCard[]>();
    for (const c of filtered) {
      if (!m.has(c.practice_area)) m.set(c.practice_area, []);
      m.get(c.practice_area)!.push(c);
    }
    return Array.from(m.entries()); // already in backend practice-area order
  }, [filtered]);

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin">
      <div className="max-w-6xl mx-auto px-8 py-9">
        <BackToVault vaultId={vaultId} className="mb-6" />

        {/* Title row */}
        <div className="flex items-end justify-between mb-7">
          <div>
            <h1 style={{ fontFamily: "Fraunces, Georgia, serif", fontSize: 30, fontWeight: 500, letterSpacing: "-0.03em", color: "var(--ink)" }}>
              Workflows
            </h1>
            <p className="text-[13px] text-[var(--ink-3)] mt-1">
              Specialized, repeatable workflows to tackle complex matters — every output cited or flagged.
            </p>
          </div>
        </div>

        {/* Controls — output-type filter + search */}
        <div className="flex items-center gap-2 mb-8">
          <div className="flex items-center gap-1">
            {["Review", "Draft", "Output"].map((t) => {
              const on = typeFilter === t;
              const Icon = OUTPUT_META[t].icon;
              return (
                <button
                  key={t}
                  onClick={() => setTypeFilter(on ? null : t)}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-medium transition-colors"
                  style={{
                    background: on ? "var(--ink)" : "var(--surface)",
                    border: "1px solid var(--line)",
                    color: on ? "var(--surface)" : "var(--ink-2)",
                  }}
                >
                  <Icon size={12} /> {t}
                </button>
              );
            })}
          </div>
          <div className="flex-1" />
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
            <Search size={13} className="text-[var(--ink-3)]" />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search workflows"
              className="bg-transparent outline-none text-[12px] w-44 text-[var(--ink)] placeholder:text-[var(--ink-3)]"
            />
          </div>
        </div>

        {loading ? (
          <div className="py-20 text-center text-[13px] text-[var(--ink-3)]">Loading workflows…</div>
        ) : cards.length === 0 ? (
          <div className="py-20 text-center text-[13px] text-[var(--ink-3)]">
            No workflows available. (The agent core may be off.)
          </div>
        ) : grouped.length === 0 ? (
          <div className="py-20 text-center text-[13px] text-[var(--ink-3)]">No workflows match.</div>
        ) : (
          <div className="space-y-9">
            {grouped.map(([area, list]) => (
              <section key={area}>
                <h2 className="text-[12px] font-semibold tracking-wide text-[var(--ink-2)] mb-3">{area}</h2>
                <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {list.map((c, i) => (
                    <WorkflowCardView key={c.id} card={c} index={i} onRun={() => setActive(c)} />
                  ))}
                </div>
              </section>
            ))}
          </div>
        )}
      </div>

      <AnimatePresence>
        {active && (
          <WorkflowRunner
            key={active.id}
            vaultId={vaultId}
            template={active}
            token={token ?? ""}
            onClose={() => setActive(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

function WorkflowCardView({ card, index, onRun }: { card: WorkflowCard; index: number; onRun: () => void }) {
  const meta = OUTPUT_META[card.output_type] ?? OUTPUT_META.Output;
  const Icon = meta.icon;
  return (
    <motion.button
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index * 0.03, 0.25), ease }}
      onClick={onRun}
      className="text-left flex flex-col rounded-xl p-4 h-full group transition-shadow hover:shadow-[var(--shadow-md)]"
      style={{ background: "var(--surface)", border: "1px solid var(--line)" }}
    >
      <div className="flex items-start justify-between mb-1.5">
        <span className="text-[14px] font-semibold leading-snug pr-2" style={{ color: "var(--ink)" }}>{card.title}</span>
        <span className="inline-flex items-center gap-1 text-[11px] text-[var(--ink)] opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
          <Play size={11} /> Run
        </span>
      </div>
      <p className="text-[12px] leading-relaxed text-[var(--ink-3)] flex-1">{card.description}</p>
      <div className="flex items-center gap-1.5 mt-3 text-[11px] text-[var(--ink-3)]">
        <Icon size={12} />
        <span>{meta.label}</span>
        <span>·</span>
        <span>{card.step_count} step{card.step_count === 1 ? "" : "s"}</span>
      </div>
    </motion.button>
  );
}

// ── The run drawer: a generic form + the result, routed by shape ──────────────────
function WorkflowRunner({
  vaultId, template, token, onClose,
}: { vaultId: string; template: WorkflowCard; token: string; onClose: () => void }) {
  const [docs, setDocs] = useState<DocumentResponse[]>([]);
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState<"form" | "result">("form");

  // grid result
  const [start, setStart] = useState<GridStart | null>(null);
  const [cells, setCells] = useState<Record<string, GridCellEvent>>({});
  const [done, setDone] = useState<GridDone | null>(null);
  // report/output result
  const [report, setReport] = useState("");
  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [steps, setSteps] = useState<WorkflowStep[]>([]);   // live agent timeline

  const [err, setErr] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Merge a streamed step into the timeline: a tool_call adds an active row, its matching
  // tool_result flips the SAME id to done/failed. Order preserved by first-seen.
  const upsertStep = useCallback((s: WorkflowStep) => {
    setSteps((prev) => {
      const i = prev.findIndex((p) => p.id === s.id);
      if (i === -1) {
        // a new step starts → any still-active prior step is now done
        const settled = prev.map((p) => (p.status === "active" ? { ...p, status: "done" as const } : p));
        return [...settled, s];
      }
      const next = [...prev];
      next[i] = { ...next[i], ...s };
      return next;
    });
  }, []);

  useEffect(() => {
    if (!token || !vaultId) return;
    getCollectionDocuments(token, vaultId).then(setDocs).catch(() => setDocs([]));
  }, [token, vaultId]);

  const docIds = useMemo(() => (values["doc_ids"] as string[] | undefined) ?? [], [values]);

  const run = useCallback(async () => {
    if (!token || running) return;
    for (const p of template.params_schema) {
      if (p.required && p.type !== "doc_multiselect" && !String(values[p.name] ?? "").trim()) {
        toast.error(`${p.label} is required`);
        return;
      }
    }
    setRunning(true); setErr(null); setPhase("result");
    setStart(null); setCells({}); setDone(null); setReport(""); setSources([]); setSteps([]);
    abortRef.current = new AbortController();
    const body = { collection_id: vaultId, params: values, doc_ids: docIds };

    if (template.shape === "grid") {
      await streamWorkflowRun(token, template.id, body, {
        onStart: (s) => setStart(s),
        onCell: (c) => setCells((prev) => ({ ...prev, [ck(c.doc_id, c.column_key)]: c })),
        onDone: (d) => { setDone(d); if (d.error) setErr(d.error); setRunning(false); },
        onError: (m) => { setErr(m); setRunning(false); },
      }, abortRef.current.signal);
    } else {
      await streamWorkflowReport(token, template.id, body, {
        onStep: (s) => upsertStep(s),
        onToken: (chunk) => setReport((r) => r + chunk),
        onSources: (s) => setSources(s),
        onDone: () => { setSteps((prev) => prev.map((p) => (p.status === "active" ? { ...p, status: "done" } : p))); setRunning(false); },
        onError: (m) => { setErr(m); setRunning(false); },
      }, abortRef.current.signal);
    }
  }, [token, running, template, values, vaultId, docIds, upsertStep]);

  const meta = OUTPUT_META[template.output_type] ?? OUTPUT_META.Output;

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      // Sit BELOW the app top bar (h-14 = 56px) so the drawer header never slides under it,
      // and use the dedicated drawer z-index (above the sticky bar's z-200).
      className="fixed left-0 right-0 bottom-0 flex justify-end"
      style={{ top: "56px", background: "rgba(0,0,0,0.30)", zIndex: "var(--z-drawer)" as unknown as number }}
      onClick={() => { abortRef.current?.abort(); onClose(); }}
    >
      <motion.div
        initial={{ x: 40, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 40, opacity: 0 }}
        transition={{ ease, duration: 0.22 }}
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-2xl h-full overflow-hidden flex flex-col"
        style={{ background: "var(--surface)", borderLeft: "1px solid var(--line)" }}
      >
        {/* Header */}
        <div className="flex items-start justify-between px-6 py-5 border-b" style={{ borderColor: "var(--line)" }}>
          <div className="min-w-0 pr-3">
            <div className="flex items-center gap-1.5 text-[11px] text-[var(--ink-3)] mb-1">
              <meta.icon size={12} /> {meta.label} · {template.practice_area}
            </div>
            <h2 className="text-[16px] font-semibold leading-snug" style={{ color: "var(--ink)" }}>{template.title}</h2>
          </div>
          <button onClick={() => { abortRef.current?.abort(); onClose(); }} className="p-1.5 rounded-lg hover:bg-[var(--surface-3)] text-[var(--ink-3)]">
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-5">
          {phase === "form" && (
            <div className="space-y-5">
              <p className="text-[12.5px] leading-relaxed text-[var(--ink-2)]">{template.description}</p>
              {template.params_schema.map((p) => (
                <ParamField key={p.name} spec={p} value={values[p.name]} docs={docs}
                            onChange={(v) => setValues((cur) => ({ ...cur, [p.name]: v }))} />
              ))}
              {template.params_schema.some((p) => p.type === "doc_multiselect") && (
                <div className="text-[11px] text-[var(--ink-3)]">
                  {docs.length} document{docs.length === 1 ? "" : "s"} in this vault
                  {docIds.length ? ` · ${docIds.length} selected` : " · all"}.
                </div>
              )}
            </div>
          )}

          {phase === "result" && (
            <>
              {/* Live agent timeline (report/output): the user SEES the agent work —
                  search → read → verify — exactly like the Ask screen, not a spinner. */}
              {template.shape !== "grid" && steps.length > 0 && (
                <div className="mb-5">
                  <ThinkingStream steps={steps.map(toThinkingStep)} keepExpanded />
                </div>
              )}
              {template.shape !== "grid" && steps.length === 0 && running && (
                <div className="flex items-center gap-2 text-[12px] text-[var(--ink-3)] mb-4">
                  <Loader2 size={13} className="animate-spin" /> Starting the agent…
                </div>
              )}
              {template.shape === "grid"
                ? (running && !start
                    ? <div className="flex items-center gap-2 text-[12px] text-[var(--ink-3)]"><Loader2 size={13} className="animate-spin" /> Preparing the review…</div>
                    : start && <ResultGrid start={start} cells={cells} done={done} />)
                : (report || !running) && <ReportView markdown={report} sources={sources} running={running} />}
              {err && <div className="mt-3 text-[12px]" style={{ color: "var(--status-failed)" }}>{err}</div>}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t flex items-center justify-end gap-2" style={{ borderColor: "var(--line)" }}>
          {phase === "result" && !running && (
            <button onClick={() => setPhase("form")} className="text-[13px] px-3 py-1.5 rounded-lg text-[var(--ink-3)] hover:text-[var(--ink)]">
              New run
            </button>
          )}
          {running ? (
            <button onClick={() => { abortRef.current?.abort(); setRunning(false); }}
                    className="text-[13px] px-4 py-2 rounded-lg font-medium"
                    style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-2)" }}>
              Cancel
            </button>
          ) : phase === "form" ? (
            <button onClick={run} className="inline-flex items-center gap-1.5 text-[13px] px-4 py-2 rounded-lg font-medium"
                    style={{ background: "var(--ink)", color: "var(--surface)" }}>
              <Play size={13} /> Run workflow
            </button>
          ) : null}
        </div>
      </motion.div>
    </motion.div>
  );
}

function ParamField({
  spec, value, docs, onChange,
}: { spec: WorkflowParamSpec; value: unknown; docs: DocumentResponse[]; onChange: (v: unknown) => void }) {
  const label = (
    <label className="block text-[12px] font-medium mb-1.5" style={{ color: "var(--ink-2)" }}>
      {spec.label}{spec.required ? " *" : ""}
      {spec.help && <span className="block text-[11px] font-normal text-[var(--ink-3)] mt-0.5">{spec.help}</span>}
    </label>
  );

  if (spec.type === "doc_multiselect") {
    const selected = (value as string[] | undefined) ?? [];
    const toggle = (id: string) => onChange(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id]);
    return (
      <div>
        {label}
        <div className="flex flex-wrap gap-1.5">
          {docs.length === 0 && <span className="text-[12px] text-[var(--ink-3)]">No documents in this vault yet.</span>}
          {docs.map((d) => {
            const on = selected.includes(d.id);
            return (
              <button key={d.id} onClick={() => toggle(d.id)}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[12px] transition-colors"
                      style={{ background: on ? "var(--ink)" : "var(--surface-3)", border: "1px solid var(--line)", color: on ? "var(--surface)" : "var(--ink-2)" }}>
                <FileText size={11} /> <span className="truncate max-w-[200px]">{d.filename}</span>
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  if (spec.type === "textarea") {
    return (
      <div>
        {label}
        <textarea value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} rows={5}
                  className="w-full text-[13px] px-3 py-2 rounded-lg outline-none resize-y"
                  style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink)" }} />
      </div>
    );
  }

  return (
    <div>
      {label}
      <input value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}
             className="w-full text-[13px] px-3 py-2 rounded-lg outline-none"
             style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink)" }} />
    </div>
  );
}

// A streamed cited deliverable (Draft / Output shapes). Monochrome prose + a sources list.
function ReportView({ markdown, sources, running }: { markdown: string; sources: SourceInfo[]; running: boolean }) {
  // The live timeline (above) already shows progress; render the deliverable only once it
  // starts streaming. While the agent is still gathering, show nothing here (no spinner).
  if (!markdown && running) return null;
  return (
    <div>
      <div className="prose-wf text-[13.5px] leading-relaxed" style={{ color: "var(--ink)" }}>
        <ReactMarkdown>{markdown || "_No verifiable content was produced._"}</ReactMarkdown>
      </div>
      {sources.length > 0 && (
        <div className="mt-5 pt-4 border-t" style={{ borderColor: "var(--line)" }}>
          <h4 className="text-[11px] uppercase tracking-wide text-[var(--ink-3)] mb-2">Sources</h4>
          <ul className="space-y-1">
            {sources.map((s, i) => (
              <li key={i} className="text-[12px] text-[var(--ink-2)] flex items-start gap-1.5">
                <FileText size={11} className="mt-0.5 shrink-0 text-[var(--ink-3)]" />
                <span className="truncate">{s.filename ?? "source"}{s.page ? ` · p.${s.page}` : ""}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// The live cell table (Review shape). FOUND shows value + quote tooltip; MISSING/ABSTAIN/
// ERROR shown honestly. Same cite-or-flag contract as the Review grid.
function ResultGrid({ start, cells, done }: { start: GridStart; cells: Record<string, GridCellEvent>; done: GridDone | null }) {
  const docNames = start.doc_names ?? [];
  const colLabels = start.column_labels ?? [];
  const verified = Object.values(cells).filter((c) => c.verified).length;

  const cellAt = (docName: string, colLabel: string) =>
    Object.values(cells).find((c) => (c.doc_name ?? "") === docName && colLabelOf(c, colLabels) === colLabel);

  return (
    <div>
      <div className="flex items-center gap-2 mb-3 text-[11px] text-[var(--ink-3)]">
        <span>{start.rows} docs × {start.columns} columns</span>
        <span>·</span>
        <span style={{ color: "var(--fidelity-good)" }}>{verified} verified</span>
        {done && <span>· done</span>}
      </div>
      <div className="overflow-x-auto rounded-lg" style={{ border: "1px solid var(--line)" }}>
        <table className="w-full text-left text-[12px]" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr className="text-[10px] uppercase tracking-wide text-[var(--ink-3)]">
              <th className="px-3 py-2 font-medium">Document</th>
              {colLabels.map((l) => <th key={l} className="px-3 py-2 font-medium whitespace-nowrap">{l}</th>)}
            </tr>
          </thead>
          <tbody>
            {docNames.map((dn) => (
              <tr key={dn} className="border-t" style={{ borderColor: "var(--line)" }}>
                <td className="px-3 py-2 max-w-[200px]"><span className="truncate block" style={{ color: "var(--ink)" }}>{dn}</span></td>
                {colLabels.map((cl) => <td key={cl} className="px-3 py-2"><CellView cell={cellAt(dn, cl)} /></td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function colLabelOf(c: GridCellEvent, labels: string[]): string {
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z]/g, "");
  const humanized = norm(c.column_key || "");
  return labels.find((l) => norm(l) === humanized) ?? labels[0] ?? "";
}

function CellView({ cell }: { cell?: GridCellEvent }) {
  if (!cell) return <span className="inline-block w-12 h-3 rounded animate-pulse" style={{ background: "var(--surface-3)" }} />;
  if (cell.status === "found") {
    return (
      <span title={cell.quote ?? undefined} className="inline-flex items-center gap-1"
            style={{ color: cell.risk === "non_standard" ? "var(--status-failed)" : "var(--ink)" }}>
        <CheckCircle2 size={11} style={{ color: "var(--fidelity-good)" }} />
        <span className="truncate max-w-[160px]">{cell.value ?? "—"}</span>
      </span>
    );
  }
  if (cell.status === "missing") return <span className="inline-flex items-center gap-1 text-[var(--ink-3)]"><MinusCircle size={11} /> Not found</span>;
  if (cell.status === "error") return <span style={{ color: "var(--status-failed)" }}>Error</span>;
  return <span className="inline-flex items-center gap-1 text-[var(--ink-3)]" title={cell.abstain_reason ?? undefined}><HelpCircle size={11} /> Unclear</span>;
}
