"use client";

// app/app/chat/[id]/page.tsx
// Active conversation — SSE streaming, inline source citations, message history.
// Supports both Standard and Agentic (Deep) query modes.
// Reads ?q= query param from suggestion buttons to auto-submit on mount.

import { useEffect, useRef, useState, useCallback, Suspense } from "react";
import { useParams, useSearchParams } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { ChatInput } from "@/components/chat/ChatInput";
import { ArtifactPanel, detectArtifact, Artifact } from "@/components/chat/ArtifactPanel";
import { ThinkingStreamFixture, ThinkingStream, ThinkingStep } from "@/components/chat/ThinkingStream";
import { SkeletonMessage } from "@/components/ui/Skeleton";
import { MOCK_ANSWER_META, AnswerMeta, ConfidenceLevel } from "@/components/chat/TrustBar";
import { useAuthStore } from "@/stores/auth.store";
import { getMessages, MessageResponse, SourceInfo, exportConversation, compareDocuments, ComparisonResult, DocumentResponse, listDocuments, listCollections } from "@/lib/api";
import { streamQuery, streamAgenticQuery, streamBrainQuery, streamAgentCoreQuery } from "@/lib/streaming";
import { useCollectionStore } from "@/stores/collection.store";
import { toast } from "sonner";
import { Search, Download, FolderOpen, GitCompare, Globe, X, ChevronRight, TrendingUp, BarChart3, ArrowUpRight } from "lucide-react";

// ── Rich agent-run timeline helpers: turn raw tool events into readable steps + chips ──
// A clean doc label from a filename ("goog-20231231.pdf" → "goog 2023"; the SEC accession
// "0000950170-23-035122.pdf" → "MSFT FY23" via a small known map; else the bare stem).
const _DOC_ALIAS: Record<string, string> = { "0000950170-23-035122": "MSFT FY23" };
function _docLabel(raw?: string): string | null {
  if (!raw) return null;
  const stem = raw.replace(/\.(pdf|htm|html|docx?|txt)$/i, "").trim();
  if (_DOC_ALIAS[stem]) return _DOC_ALIAS[stem];
  const m = stem.match(/([a-zA-Z]{2,6}).*?(20\d{2})/);     // issuer + year
  if (m) return `${m[1].toLowerCase()} ${m[2]}`;
  return stem.length > 22 ? stem.slice(0, 22) + "…" : stem;
}
// Pull the `doc`/filename and any page out of a tool's arg/summary string for a chip.
function _chipsFrom(text?: string): string[] {
  if (!text) return [];
  const chips: string[] = [];
  const push = (v: string) => { if (v && chips.indexOf(v) === -1) chips.push(v); };
  // filenames (with extension) — the most reliable token
  const fnRe = /[\w.-]+\.(?:pdf|htm|html|docx?|txt)/gi;
  let m: RegExpExecArray | null;
  while ((m = fnRe.exec(text)) !== null) {
    const d = _docLabel(m[0]); if (d) push(d);
  }
  // bare accession ids in our known map
  Object.keys(_DOC_ALIAS).forEach((k) => { if (text.includes(k)) push(_DOC_ALIAS[k]); });
  // a page reference → append to the last chip if present
  const pg = text.match(/\bp(?:age|\.)?\s*(\d+)/i);
  if (pg && chips.length) {
    chips[chips.length - 1] = `${chips[chips.length - 1]} p.${pg[1]}`;
  }
  return chips.slice(0, 4);
}
// ── Harvey-style action phrasing (G2 Step D) ──
// The timeline reads as clean, human-readable actions (✓ "Searching documents for …",
// ✓ "Looking up research & development") — NOT raw tool signatures. Each agent tool call
// maps to a plain gerund phrase, with the query/metric/doc pulled out of the args for
// specificity. Source/doc chips appear beneath (parsed separately via _chipsFrom).
function _argVal(argsSummary: string | undefined, key: string): string | null {
  if (!argsSummary) return null;
  // args arrive as a JSON-ish string: {"query": "…", "metric": "…", "doc": "…"}
  const m = argsSummary.match(new RegExp(`"${key}"\\s*:\\s*"([^"]+)"`));
  return m ? m[1] : null;
}
function _short(s: string, n = 52): string {
  const t = s.trim();
  return t.length > n ? t.slice(0, n).trimEnd() + "…" : t;
}
// The live phrase while a tool is RUNNING (present continuous).
function toolPhrase(name: string, argsSummary?: string): { label: string; chips: string[] } {
  const chips = _chipsFrom(argsSummary);
  const query = _argVal(argsSummary, "query");
  const metric = _argVal(argsSummary, "metric");
  const docLabel = _docLabel(_argVal(argsSummary, "doc") ?? undefined);
  let label: string;
  switch (name) {
    case "search_vault":
      label = query ? `Searching documents for “${_short(query)}”` : "Searching uploaded documents";
      break;
    case "read_document":
      label = docLabel ? `Reading ${docLabel}` : "Reading the document";
      break;
    case "table_lookup":
      label = metric ? `Looking up ${_short(metric, 40)}` : "Looking up a figure in the tables";
      break;
    case "compute":
      label = "Computing the figure from source cells";
      break;
    case "list_metrics":
      label = "Surveying available metrics";
      break;
    case "verify_numbers":
      label = "Verifying every figure against its source";
      break;
    default:
      label = name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }
  return { label, chips };
}

interface LocalMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceInfo[];
  isStreaming?: boolean;
  isFallback?: boolean;
  subQueries?: string[];
  webSearchUsed?: boolean;
  // Brain (map-reduce synthesis) — real thinking stream + trust meta
  isBrain?: boolean;
  thinkingSteps?: ThinkingStep[];
  thinkingTotalMs?: number;
  answerMeta?: AnswerMeta;
}

// Props let this page be RE-HOMED under /app/vault/[id]/ask/[cid] (G2 Step D) without
// forking the 1,150-line conversation logic into a second, drift-prone copy. When the
// vault route mounts it, `scopedCollectionId` is the route's [id] — the AUTHORITATIVE
// vault scope (§9 risk #1: the URL wins, the store is just its cache). The legacy
// /app/chat/[id] route passes nothing and behaves byte-identically to before (scope
// read from the store). `conversationId` likewise overrides the route param so the
// vault route's [cid] segment drives the conversation.
export interface ChatConversationProps {
  scopedCollectionId?: string;   // route-authoritative vault id (vault route only)
  conversationId?: string;       // override for the [cid] segment (vault route only)
  // G5/G6: which agent-core mode this conversation runs in.
  // "standard" = Ask (default), "deep" = Deep Analysis, "draft" = Draft Deliverable.
  // All three ride the SAME stream consumer; only `mode` in the request body changes.
  analysisMode?: "standard" | "deep" | "draft";
  // G6.1: draft mode only — carried in the URL from the Draft landing and forwarded to
  // the stream body. Ignored when analysisMode != "draft".
  draftDocType?: string | null;
  draftInstructions?: string | null;
}

export default function ChatPage(props: ChatConversationProps = {}) {
  return (
    <Suspense fallback={<div className="flex-1" />}>
      <ChatPageInner {...props} />
    </Suspense>
  );
}

// Exported so the vault route can render the conversation directly with props.
export { ChatPage as ChatConversation };

function ChatPageInner({ scopedCollectionId, conversationId, analysisMode = "standard", draftDocType, draftInstructions }: ChatConversationProps) {
  const routeParams = useParams<{ id: string }>();
  // When re-homed under the vault route, the conversation id arrives as a prop ([cid]);
  // otherwise it's the [id] segment of /app/chat/[id].
  const convId = conversationId ?? routeParams.id;
  const searchParams = useSearchParams();
  const { token, user } = useAuthStore();
  const { activeCollectionId: storeCollectionId, setActiveCollectionId } = useCollectionStore();
  // Scope source of truth: the vault route's [id] (scopedCollectionId) when present,
  // else the store. VaultScopeSync keeps the store mirrored to the route, so on the
  // vault route both agree — but the page DEPENDS on the route prop, never the store,
  // closing the deep-link / second-tab desync window (§9 risk #1).
  const activeCollectionId = scopedCollectionId ?? storeCollectionId;

  const [messages, setMessages] = useState<LocalMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [loading, setLoading] = useState(true);
  // Law-first pivot (2026-06-12): the verified tool-loop Agent is the ONLY user-facing
  // smart mode. Brain (cross-doc map-reduce) is kept as the internal Gate-A comparator,
  // surfaced only when NEXT_PUBLIC_SHOW_DEV_MODES=true. The agentic/multi-hop "Deep"
  // path was retired (its loop is subsumed by the agent core's tool loop).
  const [agenticMode, setAgenticMode] = useState(false);   // retired; retained for dead-branch compat
  const [brainMode, setBrainMode] = useState(false);       // dev-only comparator
  const [agentCoreMode, setAgentCoreMode] = useState(true); // A4: the default smart path
  const [vaultName, setVaultName] = useState<string | null>(null);  // active collection name → composer vault chip
  const [showVaultPicker, setShowVaultPicker] = useState(false);
  const [vaultOptions, setVaultOptions] = useState<{ id: string; name: string }[]>([]);
  const [showExport, setShowExport] = useState(false);
  const [showCompare, setShowCompare] = useState(false);
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [compareDocA, setCompareDocA] = useState("");
  const [compareDocB, setCompareDocB] = useState("");
  const [compareFocus, setCompareFocus] = useState("");
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareResult, setCompareResult] = useState<ComparisonResult | null>(null);
  const [artifact, setArtifact] = useState<Artifact | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const streamingIdRef = useRef<string | null>(null);
  const autoSubmittedRef = useRef(false);
  const exportRef = useRef<HTMLDivElement>(null);
  // Stable refs to avoid re-creating handleSubmit on every state change
  const isStreamingRef = useRef(false);
  const agenticModeRef = useRef(false);
  const brainModeRef = useRef(false);
  const agentCoreModeRef = useRef(false);
  const activeCollectionIdRef = useRef<string | null>(null);
  // G3 Step E: the active vault filter set (doc_type / fiscal_year), carried in the URL
  // (?filters=<json>) from the vault page. EXPLICIT in the request (mirror §9 risk #1 —
  // never a stale store) → sent as the stream body's `filters` so the backend narrows
  // retrieval scope. Parsed once from searchParams (below); null when absent.
  const filtersRef = useRef<Record<string, unknown> | null>(null);
  const toolStepSeq = useRef(0);                 // each tool call = its own timeline step
  const lastToolStepId = useRef<string | null>(null);

  // Keep refs in sync with state
  useEffect(() => { isStreamingRef.current = isStreaming; }, [isStreaming]);
  useEffect(() => { agenticModeRef.current = agenticMode; }, [agenticMode]);
  useEffect(() => { brainModeRef.current = brainMode; }, [brainMode]);
  useEffect(() => { agentCoreModeRef.current = agentCoreMode; }, [agentCoreMode]);
  useEffect(() => { activeCollectionIdRef.current = activeCollectionId; }, [activeCollectionId]);
  // G3 Step E: parse the active vault filter set from the URL (?filters=<json>). Tolerant
  // of a malformed param (→ null, no narrowing) so a bad link never breaks the run.
  useEffect(() => {
    const raw = searchParams.get("filters");
    if (!raw) { filtersRef.current = null; return; }
    try {
      const parsed = JSON.parse(raw);
      filtersRef.current = parsed && typeof parsed === "object" ? parsed : null;
    } catch { filtersRef.current = null; }
  }, [searchParams]);

  // Resolve the active collection's NAME for the composer's vault chip + cache the full
  // list for the vault picker (Harvey-style "Choose vault").
  useEffect(() => {
    if (!token) { setVaultName(null); return; }
    let cancelled = false;
    listCollections(token)
      .then((cols) => {
        if (cancelled) return;
        setVaultOptions(cols.map((c) => ({ id: c.id, name: c.name })));
        const c = cols.find((x) => x.id === activeCollectionId);
        setVaultName(activeCollectionId ? (c?.name ?? null) : null);
      })
      .catch(() => { if (!cancelled) setVaultName(null); });
    return () => { cancelled = true; };
  }, [activeCollectionId, token]);

  // Close export dropdown on outside click
  useEffect(() => {
    if (!showExport) return;
    function handleClick(e: MouseEvent) {
      if (exportRef.current && !exportRef.current.contains(e.target as Node)) {
        setShowExport(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [showExport]);

  const userInitials = user?.email?.slice(0, 2).toUpperCase() ?? "U";
  // First name from the email local-part: take the LEADING alphabetic run, so
  // "jeel15thummar@…" → "Jeel", "john.doe@…" → "John". Avoids the ugly "Jeel15thummar".
  const userName = (() => {
    const local = user?.email?.split("@")[0] ?? "";
    const lead = (local.match(/^[a-zA-Z]+/) ?? [""])[0];
    return lead ? lead.charAt(0).toUpperCase() + lead.slice(1) : "";
  })();

  // ── Stream handler (stable reference — uses refs, not state) ──────────────
  const handleSubmit = useCallback(
    async (question: string) => {
      if (!token || isStreamingRef.current) return;

      abortRef.current?.abort();
      abortRef.current = new AbortController();

      const userMsgId = crypto.randomUUID();
      const assistantMsgId = crypto.randomUUID();
      streamingIdRef.current = assistantMsgId;

      setMessages((prev) => [
        ...prev,
        { id: userMsgId, role: "user", content: question },
        { id: assistantMsgId, role: "assistant", content: "", isStreaming: true },
      ]);
      setIsStreaming(true);

      let isFallback = false;

      const callbacks = {
        onSources: (sources: SourceInfo[]) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId ? { ...m, sources } : m
            )
          );
        },
        onToken: (tok: string) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, content: m.content + tok }
                : m
            )
          );
        },
        onFallback: () => {
          isFallback = true;
        },
        onDone: () => {
          setMessages((prev) => {
            const updated = prev.map((m) =>
              m.id === assistantMsgId ? { ...m, isStreaming: false, isFallback } : m
            );
            // Auto-detect artifact from the completed assistant message. G5: in deep mode
            // a multi-section report opens in the ArtifactPanel (read-as-a-document);
            // standard Ask answers stay inline (sectioned=false → unchanged).
            const completed = updated.find((m) => m.id === assistantMsgId);
            if (completed) {
              const detected = detectArtifact(completed.content, { sectioned: analysisMode === "deep" || analysisMode === "draft" });
              if (detected) setArtifact(detected);
            }
            return updated;
          });
          setIsStreaming(false);
        },
        onError: (msg: string) => {
          toast.error(msg);
          setMessages((prev) => prev.filter((m) => m.id !== assistantMsgId));
          setIsStreaming(false);
        },
      };

      const collId = activeCollectionIdRef.current;

      if (agentCoreModeRef.current) {
        // Agent core (A4): a frontier model calls our verified tools in a loop; the
        // draft is bound to the evidence ledger by non-bypassable output gates. Requires
        // a collection scope and USE_AGENT_CORE=true on the backend (flag off ⇒ 404).
        if (!collId) {
          toast.error("Select a collection to use Agent mode");
          setMessages((prev) => prev.filter((m) => m.id !== assistantMsgId && m.id !== userMsgId));
          setIsStreaming(false);
          return;
        }

        const acStart = Date.now();
        toolStepSeq.current = 0;          // fresh step numbering per run
        lastToolStepId.current = null;
        // The agent loop's steps are emergent (the model decides), so the timeline is
        // built live from tool_call / gate events rather than a fixed pipeline.
        let steps: ThinkingStep[] = [
          { id: "plan", label: "Planning", detail: "Choosing which tools to call", status: "active" },
        ];
        const pushSteps = () => {
          const snapshot = steps.map((s) => ({ ...s }));
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantMsgId ? { ...m, thinkingSteps: snapshot } : m))
          );
        };
        const upsert = (id: string, step: ThinkingStep) => {
          const i = steps.findIndex((s) => s.id === id);
          if (i >= 0) steps[i] = { ...steps[i], ...step };
          else steps = [...steps, step];
          pushSteps();
        };

        setMessages((prev) =>
          prev.map((m) => (m.id === assistantMsgId ? { ...m, isBrain: true } : m))
        );
        pushSteps();

        await streamAgentCoreQuery(
          token,
          {
            // G5/G6: "deep" = sectioned whole-vault report; "draft" = deliverable with
            // doc_type + instructions brief; "standard" = Ask (default).
            question, conversation_id: convId, collection_id: collId, mode: analysisMode,
            // G6.1: draft extras travel only when the mode is "draft".
            ...(analysisMode === "draft" && draftDocType ? { doc_type: draftDocType } : {}),
            ...(analysisMode === "draft" && draftInstructions ? { instructions: draftInstructions } : {}),
            // G3 Step E: the active vault filter narrows retrieval scope conjunctively.
            ...(filtersRef.current ? { filters: filtersRef.current } : {}),
          },
          {
            ...callbacks,
            onDone: () => {
              steps = steps.map((s) =>
                s.status === "active" || s.status === "pending" ? { ...s, status: "done" as const } : s
              );
              const total = Date.now() - acStart;
              const snapshot = steps.map((s) => ({ ...s }));
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId ? { ...m, thinkingSteps: snapshot, thinkingTotalMs: total } : m
                )
              );
              callbacks.onDone();
            },
            onAgentThought: (text) => {
              // The backend streams the model's reasoning here, but it can include
              // answer-shaped prose (e.g. "Alphabet's R&D was 31,562…"). Echoing that
              // into the timeline leaks a half-formed answer above the verified one. A
              // harness log shows the ACTION, not the draft — so keep a stable planning
              // label; the real, accurate signal comes from the tool steps below.
              if (text) upsert("plan", { id: "plan", label: "Planning", detail: "Choosing which tools to call", status: "active" });
            },
            onToolCall: (name, argsSummary) => {
              upsert("plan", { id: "plan", label: "Planning the analysis", status: "done" });
              // Each tool call is its OWN step (id unique per call), rendered as a clean,
              // readable action phrase (Harvey-style) — "Searching documents for …",
              // "Looking up research & development" — with source/doc chips beneath.
              const stepId = `tool-${toolStepSeq.current++}`;
              lastToolStepId.current = stepId;
              const { label, chips } = toolPhrase(name, argsSummary);
              upsert(stepId, { id: stepId, label, chips, status: "active" });
            },
            onToolResult: ({ name, ok, summary, nProvenance }) => {
              const stepId = lastToolStepId.current ?? `tool-${name}`;
              // upsert MERGES, so the readable call-time label (which carries the query/
              // metric) is preserved — we only set the outcome here: a short human detail
              // (N sources cited / withheld) plus any source chips parsed from the result.
              const chips = _chipsFrom(summary);
              const detail = ok
                ? (nProvenance ? `${nProvenance} source${nProvenance !== 1 ? "s" : ""} cited` : undefined)
                : "withheld — could not trace to a source";
              const patch: Partial<ThinkingStep> & { id: string } = {
                id: stepId,
                status: ok ? "done" : "failed",
              };
              if (detail) patch.detail = detail;
              if (chips.length) patch.chips = chips;
              upsert(stepId, patch as ThinkingStep);
            },
            onGate: ({ name, pass, detail }) => {
              // The verify gate is the trust step — one clean line. Pass = every claim
              // traced; fail = the gate withheld something (the honest-abstain path).
              upsert(`gate-${name}`, {
                id: `gate-${name}`,
                label: pass ? "Verified — every figure traces to a source" : "Verifying — withheld unverifiable claims",
                detail: detail || undefined,
                status: pass ? "done" : "failed",
              });
            },
            onAgentMeta: ({ abstained, degrade }) => {
              const level: ConfidenceLevel = abstained || degrade ? "low" : "high";
              const meta: AnswerMeta = { confidence: level, claimTypes: ["fact"] };
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantMsgId ? { ...m, answerMeta: meta } : m))
              );
            },
          },
          abortRef.current.signal
        );
      } else if (brainModeRef.current) {
        // Brain (map-reduce synthesis) requires a collection scope.
        if (!collId) {
          toast.error("Select a collection to use Brain synthesis");
          setMessages((prev) => prev.filter((m) => m.id !== assistantMsgId && m.id !== userMsgId));
          setIsStreaming(false);
          return;
        }

        // Live thinking-step state machine, mirrored into the message.
        const brainStart = Date.now();
        let relevantCount = 0;
        let steps: ThinkingStep[] = [
          { id: "route", label: "Routing", detail: "Selecting relevant documents", status: "active" },
          { id: "read", label: "Reading documents", status: "pending" },
          { id: "verify", label: "Verifying claims", status: "pending" },
          { id: "ground", label: "Checking answer against evidence", status: "pending" },
          { id: "synth", label: "Writing answer", status: "pending" },
        ];
        const pushSteps = () => {
          const snapshot = steps.map((s) => ({ ...s }));
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantMsgId ? { ...m, thinkingSteps: snapshot } : m))
          );
        };
        const setStep = (id: string, patch: Partial<ThinkingStep>) => {
          steps = steps.map((s) => (s.id === id ? { ...s, ...patch } : s));
          pushSteps();
        };

        setMessages((prev) =>
          prev.map((m) => (m.id === assistantMsgId ? { ...m, isBrain: true } : m))
        );
        pushSteps();

        await streamBrainQuery(
          token,
          { question, conversation_id: convId, collection_id: collId },
          {
            ...callbacks,
            onDone: () => {
              // Finalize any still-running steps and stamp total time.
              steps = steps.map((s) =>
                s.status === "active" || s.status === "pending" ? { ...s, status: "done" as const } : s
              );
              const total = Date.now() - brainStart;
              const snapshot = steps.map((s) => ({ ...s }));
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId ? { ...m, thinkingSteps: snapshot, thinkingTotalMs: total } : m
                )
              );
              callbacks.onDone();
            },
            onBrainStart: (docsRouted) => {
              setStep("route", { status: "done", detail: `Selected ${docsRouted} document${docsRouted !== 1 ? "s" : ""}` });
              setStep("read", { status: "active", detail: `0 / ${docsRouted}` });
            },
            onBrainAnalyst: (figures) => {
              // The deterministic Analyst (§4b) already computed these figures from
              // source table cells before synthesis. Surface it as its own done step
              // (inserted after Routing) so the user sees the Analyst woke up and did
              // real, traced arithmetic — not the LLM guessing numbers. Only fires
              // when ≥1 figure was computed, so a non-numeric question never shows it.
              if (figures <= 0) return;
              if (!steps.some((s) => s.id === "analyst")) {
                const routeIdx = steps.findIndex((s) => s.id === "route");
                const analystStep: ThinkingStep = {
                  id: "analyst",
                  label: "Computing figures",
                  detail: `Analyst computed ${figures} figure${figures !== 1 ? "s" : ""} from source tables`,
                  status: "done",
                };
                steps = [
                  ...steps.slice(0, routeIdx + 1),
                  analystStep,
                  ...steps.slice(routeIdx + 1),
                ];
                pushSteps();
              }
            },
            onBrainMap: (ev) => {
              if (ev.relevant) relevantCount += 1;
              setStep("read", { status: "active", detail: `${ev.progress ?? ""} · ${relevantCount} relevant` });
            },
            onBrainVerify: (total, verified) => {
              setStep("read", { status: "done" });
              setStep("verify", { status: "active", detail: `Verified ${verified} of ${total} claim${total !== 1 ? "s" : ""}` });
            },
            onBrainReduce: (docsRelevant, groundedness, unsupported) => {
              setStep("verify", { status: "done" });
              // The §4a.3-step-2 answer-entailment check has already run by the time
              // brain_reduce fires. Surface its result as its own trust step.
              if (groundedness !== undefined) {
                const pct = Math.round(groundedness * 100);
                const allGrounded = !unsupported;
                setStep("ground", {
                  status: allGrounded ? "done" : "failed",
                  detail: allGrounded
                    ? `All sentences grounded (${pct}%)`
                    : `${unsupported} sentence${unsupported !== 1 ? "s" : ""} flagged · ${pct}% grounded`,
                });
              } else {
                setStep("ground", { status: "done" });
              }
              setStep("synth", { status: "active", detail: `Merging ${docsRelevant} source${docsRelevant !== 1 ? "s" : ""}` });
            },
            onBrainMeta: ({ confidence, abstained, coverage }) => {
              setStep("synth", { status: "done" });
              const level: ConfidenceLevel =
                abstained || confidence < 0.45 ? "low" : confidence >= 0.75 ? "high" : "medium";
              const meta: AnswerMeta = {
                confidence: level,
                consulted: coverage?.docs_relevant,
                total: coverage?.docs_routed,
                claimTypes: ["fact"],
              };
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantMsgId ? { ...m, answerMeta: meta } : m))
              );
            },
          },
          abortRef.current.signal
        );
      } else if (agenticModeRef.current) {
        await streamAgenticQuery(
          token,
          // Phase 4.5: the ⚡ agentic toggle now drives the sequential multi-hop loop
          // (retrieve → reason → informed follow-up). multi_hop=true overrides the
          // server default per-request; drop the flag to fall back to parallel decompose.
          { question, conversation_id: convId, collection_id: collId, multi_hop: true },
          {
            ...callbacks,
            onSubQueries: (queries) => {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId ? { ...m, subQueries: queries } : m
                )
              );
            },
            onWebSearch: () => {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId ? { ...m, webSearchUsed: true } : m
                )
              );
            },
          },
          abortRef.current.signal
        );
      } else {
        await streamQuery(
          token,
          { question, conversation_id: convId, collection_id: collId },
          callbacks,
          abortRef.current.signal
        );
      }
    },
    [token, convId, analysisMode] // Stable deps only (analysisMode is a per-mount prop)
  );

  // ── Load documents for the comparison picker (C2: lazy, not on every mount) ──
  // Fetched once, on demand — when the user hovers/opens Compare — instead of
  // eagerly on chat load where most users never open the modal.
  const documentsLoadedRef = useRef(false);
  const loadDocuments = useCallback(() => {
    if (!token || documentsLoadedRef.current) return;
    documentsLoadedRef.current = true;
    listDocuments(token)
      .then((docs) => setDocuments(docs.filter((d) => d.status === "ready")))
      .catch(() => {
        documentsLoadedRef.current = false; // allow retry on next interaction
      });
  }, [token]);

  async function handleCompare() {
    if (!token || !compareDocA || !compareDocB) return;
    if (compareDocA === compareDocB) {
      toast.error("Select two different documents");
      return;
    }
    setCompareLoading(true);
    setCompareResult(null);
    try {
      const result = await compareDocuments(token, compareDocA, compareDocB, compareFocus || undefined);
      setCompareResult(result);
    } catch {
      toast.error("Comparison failed");
    } finally {
      setCompareLoading(false);
    }
  }

  // ── Load message history ──────────────────────────────────────────────────
  useEffect(() => {
    if (!token || !convId) return;
    setLoading(true);
    setMessages([]);
    autoSubmittedRef.current = false;

    getMessages(token, convId)
      .then((msgs) => {
        setMessages(
          msgs.map((m: MessageResponse) => ({
            id: m.id ?? crypto.randomUUID(),
            role: m.role,
            content: m.content,
            // Normalise sources: assign index-based source_id when DB record lacks one
            // (messages saved via /conversations/{id}/messages store sources without source_id)
            sources: m.sources
              ? m.sources.map((s, i) => ({
                  ...s,
                  source_id: s.source_id ?? i + 1,
                }))
              : undefined,
          }))
        );
        return msgs;
      })
      .then((msgs) => {
        // Auto-submit if ?q= param exists and conversation is empty (from the landing).
        // Honor &mode= so a mode picked on the landing is applied here before the run
        // starts. The Agent is the default; "deep" is now an alias for the Agent (the
        // retired multi-hop path), "brain" only resolves in dev builds.
        const q = searchParams.get("q");
        if (q && msgs.length === 0 && !autoSubmittedRef.current) {
          autoSubmittedRef.current = true;
          const mode = searchParams.get("mode");
          if (mode === "brain" && process.env.NEXT_PUBLIC_SHOW_DEV_MODES === "true") {
            setBrainMode(true); brainModeRef.current = true;
            setAgentCoreMode(false); agentCoreModeRef.current = false;
          }
          // "agent" | "deep" | anything else → the default Agent (already on).
          // Small delay to let state settle
          setTimeout(() => handleSubmit(q), 200);
        }
      })
      .catch(() => toast.error("Failed to load messages"))
      .finally(() => setLoading(false));
  }, [convId, token]); // eslint-disable-line react-hooks/exhaustive-deps
  // ^ Intentionally excluding searchParams and handleSubmit to prevent re-fire loops

  // ── Auto-scroll to bottom ─────────────────────────────────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "instant" });
  }, [messages]);

  function handleCancel() {
    abortRef.current?.abort();
    setMessages((prev) =>
      prev.map((m) =>
        m.id === streamingIdRef.current
          ? { ...m, isStreaming: false, content: m.content || "(cancelled)" }
          : m
      )
    );
    setIsStreaming(false);
  }

  async function handleExport(format: "md" | "pdf") {
    if (!token || !convId) return;
    setShowExport(false);
    try {
      const blob = await exportConversation(token, convId, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `conversation.${format}`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`Exported as ${format.toUpperCase()}`);
    } catch {
      toast.error("Export failed");
    }
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Main chat column ── */}
      <div className="relative flex flex-col flex-1 min-w-0 overflow-hidden">
      {/* Top bar — collection badge + export */}
      <div
        className="relative z-10 flex items-center justify-between px-4 py-1.5 border-b flex-shrink-0"
        style={{
          background: "var(--glass-bg-strong)",
          backdropFilter: "blur(16px)",
          WebkitBackdropFilter: "blur(16px)",
          borderColor: "var(--glass-border)",
        }}
      >
        {/* Active collection indicator */}
        {activeCollectionId ? (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[var(--bg-hover)] text-[10px] text-[var(--accent)]">
            <FolderOpen size={11} />
            <span className="font-medium">Scoped to collection</span>
          </div>
        ) : (
          <div className="flex items-center gap-1.5 px-2.5 py-1 text-[10px] text-[var(--text-muted)]">
            All documents
          </div>
        )}

        {/* Right-side actions */}
        <div className="flex items-center gap-1">
          {/* Compare Documents — docs are loaded lazily (C2): prefetch on hover,
              ensure loaded on click. */}
          <button
            onMouseEnter={loadDocuments}
            onClick={() => { loadDocuments(); setShowCompare(true); setCompareResult(null); }}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-[color,background-color]"
          >
            <GitCompare size={11} />
            Compare
          </button>

          {/* Export */}
          {messages.length > 0 && !loading && (
            <div className="relative" ref={exportRef}>
              <button
                onClick={() => setShowExport(!showExport)}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-[color,background-color]"
              >
                <Download size={11} />
                Export
              </button>
              {showExport && (
                <div className="absolute right-0 top-full mt-1 card p-1 min-w-[120px] z-50 shadow-lg">
                  <button
                    onClick={() => handleExport("md")}
                    className="w-full text-left px-3 py-1.5 text-xs text-[var(--text-primary)] hover:bg-[var(--bg-hover)] rounded-md transition-colors"
                  >
                    📝 Markdown
                  </button>
                  <button
                    onClick={() => handleExport("pdf")}
                    className="w-full text-left px-3 py-1.5 text-xs text-[var(--text-primary)] hover:bg-[var(--bg-hover)] rounded-md transition-colors"
                  >
                    📄 PDF
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Messages — scrollable */}
      <div ref={scrollRef} className="relative z-10 flex-1 overflow-y-auto scrollbar-thin">
        {loading ? (
          <div className="py-4 space-y-1">
            <SkeletonMessage />
            <SkeletonMessage />
            <SkeletonMessage />
          </div>
        ) : messages.length === 0 ? (
          // ── Centered empty-state: a calm welcome that lives in the middle of the page
          // until the first message is sent (the composer below centers itself too, then
          // both drop to the normal transcript layout). Harvey/ChatGPT-style entrance. ──
          <motion.div
            key="empty-hero"
            variants={{ show: { transition: { staggerChildren: 0.09, delayChildren: 0.05 } } }}
            initial="hidden"
            animate="show"
            className="h-full flex flex-col items-start justify-center w-full max-w-3xl mx-auto px-4 md:px-8 py-6 text-left select-none"
          >
            {/* The living orb mark: enters, then breathes (slow float + scale) forever, with
                a soft rotating sheen + an outer glow ring that pulses. Monochrome grey depth. */}
            <motion.div
              variants={{ hidden: { opacity: 0, y: 16, scale: 0.7 }, show: { opacity: 1, y: 0, scale: 1 } }}
              transition={{ duration: 0.7, ease: [0.23, 1, 0.32, 1] }}
              className="mb-4 relative"
            >
              {/* outer glow ring — gentle pulse */}
              <motion.div
                aria-hidden
                className="absolute rounded-full"
                style={{ inset: -12, background: "radial-gradient(circle, rgba(0,0,0,0.06), transparent 70%)" }}
                animate={{ scale: [1, 1.12, 1], opacity: [0.5, 0.85, 0.5] }}
                transition={{ duration: 4.5, repeat: Infinity, ease: "easeInOut" }}
              />
              {/* the sphere — small, refined, breathing */}
              <motion.div
                className="w-[52px] h-[52px] rounded-full relative overflow-hidden"
                style={{
                  background: "radial-gradient(circle at 33% 25%, #FFFFFF 0%, #EAEAEA 38%, #B8B8B8 70%, #8E8E8E 100%)",
                  boxShadow: "0 14px 30px -10px rgba(0,0,0,0.36), 0 3px 8px -3px rgba(0,0,0,0.18), inset 0 2px 5px rgba(255,255,255,0.92), inset 0 -9px 16px -5px rgba(0,0,0,0.24)",
                }}
                animate={{ y: [0, -6, 0], scale: [1, 1.03, 1] }}
                transition={{ duration: 5.5, repeat: Infinity, ease: "easeInOut" }}
              >
                {/* primary specular highlight */}
                <span
                  className="absolute rounded-full"
                  style={{ inset: "14% 42% 54% 20%", background: "radial-gradient(circle, rgba(255,255,255,0.95), transparent 70%)", filter: "blur(2px)" }}
                />
                {/* slow drifting sheen across the surface */}
                <motion.span
                  className="absolute inset-0 rounded-full"
                  style={{ background: "linear-gradient(120deg, transparent 35%, rgba(255,255,255,0.45) 50%, transparent 65%)" }}
                  animate={{ x: ["-60%", "60%"], opacity: [0, 0.7, 0] }}
                  transition={{ duration: 6, repeat: Infinity, ease: "easeInOut", repeatDelay: 1.5 }}
                />
              </motion.div>
            </motion.div>

            {/* Big bold two-line greeting — scale & confidence (Genim "Creating Without Limits") */}
            <motion.h2
              variants={{ hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0 } }}
              transition={{ duration: 0.6, ease: [0.23, 1, 0.32, 1] }}
              className="text-[34px] md:text-[40px] leading-[1.08] font-bold text-[var(--text-primary)] tracking-[-0.035em]"
            >
              Hi{userName ? <>, <span className="text-[var(--text-muted)]">{userName}</span></> : " there"}.
              <br />What would you like to know?
            </motion.h2>
            <motion.p
              variants={{ hidden: { opacity: 0, y: 10 }, show: { opacity: 1, y: 0 } }}
              transition={{ duration: 0.6, ease: [0.23, 1, 0.32, 1] }}
              className="text-[14px] text-[var(--text-muted)] mt-3.5 max-w-lg leading-relaxed"
            >
              {vaultName
                ? <>Scoped to <span className="font-medium text-[var(--text-secondary)]">{vaultName}</span> · pick a prompt below or ask your own. Every clause and figure is grounded in a cited source — or honestly withheld.</>
                : "Pick a matter and ask. Every clause and figure is grounded in a cited source — or honestly withheld."}
            </motion.p>

            {/* Suggestion cards — concrete example questions (richer empty state). Clicking
                one submits it. Tuned to contract review (law-first). */}
            <motion.div
              variants={{ hidden: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0 } }}
              transition={{ duration: 0.6, ease: [0.23, 1, 0.32, 1] }}
              className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3 w-full"
            >
              {[
                { icon: Search, title: "Key terms", q: "What are the governing law, term, and termination notice period in this contract? Quote each clause." },
                { icon: GitCompare, title: "Indemnity & liability", q: "Is there an indemnity cap and a limitation of liability? Quote the exact caps, or say if liability is uncapped." },
                { icon: BarChart3, title: "Risk flags", q: "Flag any non-standard or unusual clauses in this contract that a reviewer should look at." },
                { icon: TrendingUp, title: "Dispute resolution", q: "How are disputes resolved — arbitration or courts? What is the seat/venue and which rules apply?" },
              ].map((s) => (
                <button
                  key={s.title}
                  onClick={() => handleSubmit(s.q)}
                  className="group relative text-left rounded-2xl p-3.5 transition-all duration-200 hover:-translate-y-0.5 overflow-hidden"
                  style={{
                    background: "linear-gradient(180deg, #FFFFFF, #F6F6F6)",
                    border: "1px solid rgba(0,0,0,0.10)",
                    boxShadow: "0 6px 20px -10px rgba(0,0,0,0.16), inset 0 1px 0 rgba(255,255,255,0.95)",
                  }}
                >
                  <div className="flex items-center gap-2.5 mb-1.5">
                    <div
                      className="w-8 h-8 rounded-lg flex items-center justify-center text-[var(--text-secondary)] group-hover:text-white group-hover:bg-[var(--accent)] transition-colors"
                      style={{ background: "var(--surface-3)", border: "1px solid var(--line)" }}
                    >
                      <s.icon size={15} />
                    </div>
                    <span className="text-[14px] font-semibold text-[var(--text-primary)] tracking-[-0.01em]">{s.title}</span>
                  </div>
                  <p className="text-[12.5px] text-[var(--text-muted)] leading-snug group-hover:text-[var(--text-secondary)] transition-colors pr-5">
                    {s.q}
                  </p>
                  <ArrowUpRight
                    size={15}
                    className="absolute bottom-3 right-3 text-[var(--text-muted)] opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200"
                  />
                </button>
              ))}
            </motion.div>
          </motion.div>
        ) : (
          <div className="py-4">
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <div key={msg.id}>
                  {/* ── Live agent run — the CENTERPIECE of Ask (G2 Step D). ──
                      An INLINE transcript in the conversation flow (no boxed card): the
                      agent's real tool actions stream as they happen — Searching →
                      Reading goog 2021 → Computing R&D → Verifying against sources — each
                      line completing in place, source/doc chips appearing under it. It
                      STAYS expanded after the answer lands (keepExpanded) so the run reads
                      like a harness log, not a folded summary. The verify gate carries the
                      one meaningful accent (green traced / red withheld) via the step's
                      done/failed state — the only color on this surface. */}
                  {msg.role === "assistant" && msg.isBrain && msg.thinkingSteps && msg.thinkingSteps.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3, ease: [0.23, 1, 0.32, 1] }}
                      className="max-w-3xl mx-auto px-4 md:px-8 mb-2"
                    >
                      {/* Slim inline header — a live pulse + state label, no surface/border */}
                      <div className="flex items-center gap-2 mb-2.5 pl-0.5">
                        <span className="relative flex items-center justify-center w-1.5 h-1.5">
                          {msg.isStreaming && (
                            <span
                              className="absolute inline-flex h-full w-full rounded-full animate-ping"
                              style={{ background: "var(--step-active)", opacity: 0.6 }}
                            />
                          )}
                          <span
                            className="relative inline-flex rounded-full w-1.5 h-1.5"
                            style={{ background: msg.isStreaming ? "var(--step-active)" : "var(--step-done)" }}
                          />
                        </span>
                        {/* Harvey-style state label: Working… while running tools →
                            Answering… once the answer starts streaming → done summary. */}
                        <span className="text-[12px] font-medium" style={{ color: "var(--text-secondary)" }}>
                          {msg.isStreaming
                            ? (msg.content ? "Answering…" : "Working…")
                            : "Reasoned through your question"}
                        </span>
                        {!msg.isStreaming && msg.thinkingTotalMs != null && (
                          <span className="text-[11px] tabular-nums" style={{ color: "var(--text-muted)" }}>
                            · {(msg.thinkingTotalMs / 1000).toFixed(1)}s
                          </span>
                        )}
                      </div>
                      {/* The inline live timeline — stays expanded after completion */}
                      <ThinkingStream
                        steps={msg.thinkingSteps}
                        totalMs={msg.thinkingTotalMs}
                        keepExpanded
                      />
                    </motion.div>
                  )}
                  {/* Agentic thinking stream — mock fixture while streaming */}
                  {msg.role === "assistant" && msg.isStreaming && agenticMode && !msg.isBrain && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="max-w-3xl mx-auto px-4 md:px-8 mb-3"
                    >
                      <div className="card-dotted p-4">
                        <ThinkingStreamFixture />
                      </div>
                    </motion.div>
                  )}
                  {/* Sub-queries indicator for agentic mode (after streaming) */}
                  {msg.subQueries && msg.subQueries.length > 0 && msg.role === "assistant" && !msg.isStreaming && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="max-w-3xl mx-auto px-4 md:px-8 mb-2"
                    >
                      <div className="card-dotted p-3 text-xs">
                        <div className="flex items-center gap-2 text-[var(--text-muted)] font-medium mb-2">
                          <Search size={12} />
                          Decomposed into {msg.subQueries.length} sub-queries
                        </div>
                        <ul className="space-y-1 pl-5">
                          {msg.subQueries.map((sq, i) => (
                            <li key={i} className="text-[var(--text-secondary)] list-disc">
                              {sq}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </motion.div>
                  )}
                  {/* Web search fallback indicator */}
                  {msg.webSearchUsed && msg.role === "assistant" && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="max-w-3xl mx-auto px-4 md:px-8 mb-2"
                    >
                      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[var(--bg-hover)] text-[10px] text-[var(--text-muted)]">
                        <Globe size={11} className="text-[var(--accent)]" />
                        <span>No relevant content found in documents — searched the web for context</span>
                      </div>
                    </motion.div>
                  )}
                  <ChatMessage
                    role={msg.role}
                    content={msg.content}
                    sources={msg.sources}
                    isStreaming={msg.isStreaming}
                    isFallback={msg.isFallback}
                    userInitials={userInitials}
                    showTrust={!msg.isStreaming && msg.role === "assistant" && !!msg.sources?.length}
                    answerMeta={msg.answerMeta ?? MOCK_ANSWER_META}
                  />
                </div>
              ))}
            </AnimatePresence>
            <div ref={bottomRef} className="h-4" />
          </div>
        )}
      </div>

      {/* Input — sticky at bottom */}
      <ChatInput
        onSubmit={handleSubmit}
        onCancel={handleCancel}
        isStreaming={isStreaming}
        agentCoreMode={agentCoreMode}
        onToggleAgentCore={() => { setAgentCoreMode((v) => !v); setBrainMode(false); setAgenticMode(false); }}
        brainMode={brainMode}
        // Brain stays reachable ONLY as the internal Gate-A comparator (dev builds). The
        // agentic/Deep toggle is gone entirely. Normal users see just the Agent pill.
        onToggleBrain={
          process.env.NEXT_PUBLIC_SHOW_DEV_MODES === "true"
            ? () => { setBrainMode((v) => !v); setAgenticMode(false); setAgentCoreMode(false); }
            : undefined
        }
        vaultName={vaultName}
        onChooseVault={() => setShowVaultPicker(true)}
        centered={messages.length === 0 && !loading}
      />

      {/* Vault picker — "Choose vault" is now functional (was display-only) */}
      {showVaultPicker && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
          onClick={() => setShowVaultPicker(false)}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.97, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.18, ease: [0.23, 1, 0.32, 1] }}
            className="card w-full max-w-md mx-4 p-5 space-y-3"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-medium text-[var(--text-primary)]">
                <FolderOpen size={14} /> Choose a vault
              </div>
              <button onClick={() => setShowVaultPicker(false)} className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors">
                <X size={16} />
              </button>
            </div>
            <div className="space-y-1 max-h-[50vh] overflow-y-auto scrollbar-thin">
              <button
                onClick={() => { setActiveCollectionId(null); setShowVaultPicker(false); }}
                className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-colors ${!activeCollectionId ? "bg-[var(--bg-hover)] text-[var(--text-primary)] font-medium" : "text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]"}`}
              >
                All documents
              </button>
              {vaultOptions.map((v) => (
                <button
                  key={v.id}
                  onClick={() => { setActiveCollectionId(v.id); setShowVaultPicker(false); }}
                  className={`w-full text-left px-3 py-2.5 rounded-lg text-sm flex items-center gap-2 transition-colors ${activeCollectionId === v.id ? "bg-[var(--bg-hover)] text-[var(--text-primary)] font-medium" : "text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]"}`}
                >
                  <FolderOpen size={13} /> {v.name}
                </button>
              ))}
              {vaultOptions.length === 0 && (
                <p className="text-xs text-[var(--text-muted)] px-3 py-2">No vaults yet — create one in the sidebar.</p>
              )}
            </div>
          </motion.div>
        </div>
      )}

      {/* Compare Documents Modal */}
      {showCompare && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="card w-full max-w-lg mx-4 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-medium text-[var(--text-primary)]">
                <GitCompare size={14} />
                Compare Documents
              </div>
              <button
                onClick={() => { setShowCompare(false); setCompareResult(null); }}
                className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
              >
                <X size={14} />
              </button>
            </div>

            {!compareResult ? (
              <>
                {documents.length < 2 && (
                  <p className="text-[11px] text-[var(--text-muted)]">
                    You need at least two processed documents to compare.
                  </p>
                )}
                <div className="space-y-3">
                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1 block">Document A</label>
                    <select
                      value={compareDocA}
                      onChange={(e) => setCompareDocA(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-[var(--bg-hover)] border border-[var(--border)] text-xs text-[var(--text-primary)] focus:outline-none"
                    >
                      <option value="">Select a document…</option>
                      {documents.map((d) => (
                        <option key={d.id} value={d.id}>{d.filename}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1 block">Document B</label>
                    <select
                      value={compareDocB}
                      onChange={(e) => setCompareDocB(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-[var(--bg-hover)] border border-[var(--border)] text-xs text-[var(--text-primary)] focus:outline-none"
                    >
                      <option value="">Select a document…</option>
                      {documents.map((d) => (
                        <option key={d.id} value={d.id}>{d.filename}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wide mb-1 block">Focus area <span className="normal-case">(optional)</span></label>
                    <input
                      type="text"
                      placeholder="e.g. pricing, methodology, legal terms"
                      value={compareFocus}
                      onChange={(e) => setCompareFocus(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-[var(--bg-hover)] border border-[var(--border)] text-xs text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none"
                    />
                  </div>
                </div>
                <button
                  onClick={handleCompare}
                  disabled={!compareDocA || !compareDocB || compareLoading}
                  className="w-full py-2 rounded-lg bg-[var(--accent)] text-white text-xs font-medium disabled:opacity-40 hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
                >
                  {compareLoading ? (
                    <span className="animate-pulse">Comparing…</span>
                  ) : (
                    <>Compare <ChevronRight size={12} /></>
                  )}
                </button>
              </>
            ) : (
              <div className="space-y-4 max-h-[60vh] overflow-y-auto scrollbar-thin">
                <div className="flex items-center gap-2 text-[10px] text-[var(--text-muted)]">
                  <span className="font-medium text-[var(--text-primary)]">{compareResult.document_a}</span>
                  <span>vs</span>
                  <span className="font-medium text-[var(--text-primary)]">{compareResult.document_b}</span>
                  {compareResult.focus_area && <span>· {compareResult.focus_area}</span>}
                </div>

                <div>
                  <div className="text-[10px] uppercase tracking-wide text-[var(--text-muted)] mb-2">Similarities</div>
                  <ul className="space-y-1.5">
                    {compareResult.similarities.map((s, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-[var(--text-secondary)]">
                        <span className="mt-0.5 text-green-500 flex-shrink-0">●</span>
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <div className="text-[10px] uppercase tracking-wide text-[var(--text-muted)] mb-2">Differences</div>
                  <ul className="space-y-1.5">
                    {compareResult.differences.map((d, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-[var(--text-secondary)]">
                        <span className="mt-0.5 text-orange-400 flex-shrink-0">●</span>
                        {d}
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="pt-2 border-t border-[var(--border)]">
                  <div className="text-[10px] uppercase tracking-wide text-[var(--text-muted)] mb-1">Summary</div>
                  <p className="text-xs text-[var(--text-secondary)]">{compareResult.summary}</p>
                </div>

                <button
                  onClick={() => setCompareResult(null)}
                  className="w-full py-1.5 rounded-lg border border-[var(--border)] text-xs text-[var(--text-muted)] hover:bg-[var(--bg-hover)] transition-colors"
                >
                  Compare again
                </button>
              </div>
            )}
          </div>
        </div>
      )}
      </div>{/* end main chat column */}

      {/* ── Artifact panel (right, slide-in) ── */}
      <ArtifactPanel artifact={artifact} onClose={() => setArtifact(null)} token={token} />
    </div>
  );
}
