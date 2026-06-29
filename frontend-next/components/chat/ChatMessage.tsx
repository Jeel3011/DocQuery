"use client";

import { memo, useState, useCallback, useMemo, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AlertTriangle, ChevronDown, ChevronUp, FileText } from "lucide-react";
import { SourceInfo } from "@/lib/api";
import { TrustBar, AnswerMeta } from "./TrustBar";
import { SendForReview } from "@/components/app/ReviewChain";
import { OverrideAffordance } from "./OverrideAffordance";
import { AnswerTable } from "./AnswerTable";
import type { Components } from "react-markdown";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  sources?: SourceInfo[] | null;
  isStreaming?: boolean;
  isFallback?: boolean;
  userInitials?: string;
  answerMeta?: AnswerMeta;
  showTrust?: boolean;
  // F2g review/override affordances on an assistant answer. Present once the answer stream surfaces
  // the vault + a stable message ref; an abstain (gateObjection) unlocks the partner override moment.
  vaultId?: string | null;
  messageId?: string | null;
  gateObjection?: string | null;
}

// Convert citation markers → [n](#cite-n) links the markdown `a` renderer turns into
// clickable chips. Handles BOTH formats the system emits:
//   - the old brain path:  [Source: filename, Page: X]
//   - the AGENT CORE:       [doc p.N]  e.g. [amzn-20231231.pdf p.45] / [goog-2022 p.41]
// (Before this, agent-core citations rendered as plain literal text — never clickable —
// because only the [Source:...] form was recognized. That was the "citations don't work"
// bug.) Each marker is resolved to a source by matching its doc token (a distinctive
// substring, ext-insensitive) against the sources list, preferring the same page.
function parseCitations(text: string, sources: SourceInfo[]): {
  cleaned: string;
  citedIds: Set<number>;
} {
  const citedIds = new Set<number>();

  const norm = (s: string) =>
    (s || "").toLowerCase().replace(/\.(pdf|htm|html|docx?|txt)$/i, "").trim();

  // Resolve a (docToken, page?) to the best-matching source id.
  const resolve = (docToken: string, page?: string): number | null => {
    const want = norm(docToken);
    if (!want) return null;
    let bestId: number | null = null;
    let bestScore = -1;
    sources.forEach((s, i) => {
      const fn = norm(s.filename ?? "");
      if (!fn) return;
      const nameMatch = fn === want || fn.includes(want) || want.includes(fn);
      if (!nameMatch) return;
      // prefer a source on the same page when the citation names one
      const pageMatch = page && s.page != null && String(s.page) === String(page);
      const score = (nameMatch ? 1 : 0) + (pageMatch ? 2 : 0);
      if (score > bestScore) { bestScore = score; bestId = s.source_id ?? i + 1; }
    });
    return bestId;
  };

  let cleaned = text;

  // 1) old [Source: filename, Page: X]
  cleaned = cleaned.replace(
    /\[Source:\s*([^,\]]+)(?:,\s*Page:\s*(\d+))?\]/gi,
    (m, filename: string, page?: string) => {
      const id = resolve(filename, page);
      if (id == null) return m;
      citedIds.add(id);
      return `[${id}](#cite-${id})`;
    }
  );

  // 2) agent-core [doc p.N] / [doc p. N] / [doc, p.N] / [doc page N]
  cleaned = cleaned.replace(
    /\[([A-Za-z0-9][\w./-]*?)[\s,]*p(?:age|\.)?\s*(\d+)\]/gi,
    (m, docToken: string, page: string) => {
      const id = resolve(docToken, page);
      if (id == null) return m;
      citedIds.add(id);
      return `[${id}](#cite-${id})`;
    }
  );

  return { cleaned, citedIds };
}

// Recursively flatten a React node (string / number / element / array) to plain
// text. Markdown table cells aren't always plain strings — a cell with bold, a
// link, a citation, or nested formatting arrives as a React element (or array of
// them), and String(<element>) yields the literal "[object Object]" (the bug seen
// in computed-margin cells). This walks children to recover the real text.
function nodeToText(node: unknown): string {
  if (node == null || node === false) return "";
  if (typeof node === "string") return node;
  if (typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(nodeToText).join("");
  if (typeof node === "object" && "props" in (node as Record<string, unknown>)) {
    const props = (node as { props?: { children?: unknown } }).props;
    return nodeToText(props?.children);
  }
  return "";
}

const mdComponents: Components = {
  table: ({ children }: { children?: React.ReactNode }) => {
    let headers: string[] = [];
    const rows: string[][] = [];
    const tableChildren = children as React.ReactElement[];
    tableChildren?.forEach?.((child: React.ReactElement) => {
      const { type, props } = child as { type: string; props: { children: React.ReactElement | React.ReactElement[] } };
      if (type === "thead") {
        const trChildren = (props.children as React.ReactElement)?.props?.children as React.ReactElement[];
        headers = (Array.isArray(trChildren) ? trChildren : [trChildren]).map(
          (th: React.ReactElement) => nodeToText((th?.props as { children?: unknown })?.children)
        );
      }
      if (type === "tbody") {
        const trs = props.children as React.ReactElement[];
        (Array.isArray(trs) ? trs : [trs]).forEach((tr: React.ReactElement) => {
          const tds = (tr?.props as { children?: React.ReactElement[] })?.children as React.ReactElement[];
          rows.push(
            (Array.isArray(tds) ? tds : [tds]).map(
              (td: React.ReactElement) => nodeToText((td?.props as { children?: unknown })?.children)
            )
          );
        });
      }
    });
    if (headers.length > 0) return <AnswerTable headers={headers} rows={rows} />;
    return <table>{children}</table>;
  },
  code: ({ children, className }: { children?: React.ReactNode; className?: string }) => {
    const isBlock = className?.startsWith("language-");
    if (isBlock) {
      return (
        <pre className="bg-[var(--bg-hover)] border border-[var(--border)] rounded-xl px-4 py-3 my-2 overflow-x-auto scrollbar-thin">
          <code className="text-xs font-mono text-[var(--text-primary)] leading-relaxed">{children}</code>
        </pre>
      );
    }
    return (
      <code className="bg-[var(--bg-hover)] text-[var(--text-primary)] px-1.5 py-0.5 rounded text-[11px] font-mono">
        {children}
      </code>
    );
  },
  p: ({ children }: { children?: React.ReactNode }) => <p className="my-1.5 leading-relaxed">{children}</p>,
  h1: ({ children }: { children?: React.ReactNode }) => <h1 className="text-base font-semibold text-[var(--text-primary)] mt-3 mb-1.5">{children}</h1>,
  h2: ({ children }: { children?: React.ReactNode }) => <h2 className="text-sm font-semibold text-[var(--text-primary)] mt-3 mb-1.5">{children}</h2>,
  h3: ({ children }: { children?: React.ReactNode }) => <h3 className="text-xs font-semibold text-[var(--text-primary)] mt-2 mb-1">{children}</h3>,
  ul: ({ children }: { children?: React.ReactNode }) => <ul className="list-disc pl-4 my-1.5 space-y-0.5">{children}</ul>,
  ol: ({ children }: { children?: React.ReactNode }) => <ol className="list-decimal pl-4 my-1.5 space-y-0.5">{children}</ol>,
  li: ({ children }: { children?: React.ReactNode }) => <li className="text-[var(--text-secondary)] leading-relaxed">{children}</li>,
  strong: ({ children }: { children?: React.ReactNode }) => <strong className="font-semibold text-[var(--text-primary)]">{children}</strong>,
  blockquote: ({ children }: { children?: React.ReactNode }) => (
    <blockquote className="border-l-2 border-[var(--border-strong)] pl-3 my-2 text-[var(--text-secondary)] italic">
      {children}
    </blockquote>
  ),
  a: ({ href, children }: { href?: string; children?: React.ReactNode }) => {
    // Citation chips: [n](#cite-n) → clickable superscript that opens/highlights
    // source n (handled by the parent onClick via the citation-chip class).
    const m = href?.match(/^#cite-(\d+)$/);
    if (m) {
      return (
        <sup
          className="citation-chip cursor-pointer select-none text-[10px] font-semibold text-[var(--accent)] bg-[var(--bg-hover)] border border-[var(--border)] rounded px-1 py-0.5 mx-0.5 align-super hover:bg-[var(--accent)] hover:text-white transition-colors"
          data-source-id={m[1]}
          title={`Jump to source ${m[1]}`}
        >
          {m[1]}
        </sup>
      );
    }
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className="text-[var(--accent)] underline underline-offset-2 hover:opacity-70 transition-opacity">
        {children}
      </a>
    );
  },
};

export const ChatMessage = memo(function ChatMessage({
  role,
  content,
  sources,
  isStreaming,
  isFallback,
  userInitials = "U",
  answerMeta,
  showTrust = false,
  vaultId,
  messageId,
  gateObjection,
}: ChatMessageProps) {
  const isUser = role === "user";
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [hoveredSource, setHoveredSource] = useState<number | null>(null);
  const hasSources = !isUser && sources && sources.length > 0 && !isStreaming;

  const { cleaned } = useMemo(() => {
    if (isUser || !sources || isStreaming) return { cleaned: content, citedIds: new Set<number>() };
    return parseCitations(content, sources);
  }, [content, sources, isUser, isStreaming]);

  const handleSourceHover = useCallback((id: number | null) => setHoveredSource(id), []);

  // §2.5.2 — a trace-chip in the live reasoning timeline (rendered by the page, above this
  // message) jumps to the cited source: it dispatches `docquery:open-source` with this
  // message's id + the resolved source_id. We open the sources panel, highlight that row,
  // and scroll it into view — the same affordance an inline citation chip uses, reused so
  // "watch it read → click the span → land on the cited evidence" works end to end.
  useEffect(() => {
    if (isUser || !messageId) return;
    function onOpenSource(e: Event) {
      const det = (e as CustomEvent).detail as { messageId?: string; sourceId?: number } | undefined;
      if (!det || det.messageId !== messageId || !det.sourceId) return;
      setSourcesOpen(true);
      setHoveredSource(det.sourceId);
      requestAnimationFrame(() => {
        const el = document.querySelector(`[data-source-row="${det.sourceId}"]`);
        el?.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    }
    window.addEventListener("docquery:open-source", onOpenSource as EventListener);
    return () => window.removeEventListener("docquery:open-source", onOpenSource as EventListener);
  }, [isUser, messageId]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22, ease: [0.23, 1, 0.32, 1] }}
      className="px-4 md:px-6 py-2.5"
    >
      {/* ── Assistant: flowing full-width answer on the page (Harvey-style), no bubble ── */}
      {!isUser && (
        <div className="flex items-start gap-3 w-full max-w-3xl mx-auto">
          <div
            className="w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center text-[10px] font-bold text-[var(--text-secondary)] mt-0.5"
            aria-hidden="true"
            style={{
              background: "var(--glass-bg-strong)",
              backdropFilter: "blur(8px)",
              WebkitBackdropFilter: "blur(8px)",
              border: "1px solid var(--glass-border)",
              boxShadow: "var(--skeu-raised)",
            }}
          >
            D
          </div>
          <div className="flex flex-col gap-2 min-w-0 flex-1">
            <p className="text-[10px] font-medium text-[var(--text-muted)]">DocQuery</p>
            {/* Answer flows directly on the page — no container box. The reasoning timeline
                (rendered by the page above this) + this open answer = the Harvey transcript. */}
            <div
              className={`${isStreaming ? "streaming-cursor" : ""}`}
              onClick={(e) => {
                const target = e.target as HTMLElement;
                if (target.classList.contains("citation-chip")) {
                  const id = parseInt(target.dataset.sourceId ?? "0", 10);
                  if (id) {
                    // Open the sources, highlight the cited one, and SCROLL it into view
                    // so a citation click actually takes the user to the evidence (the
                    // "verify from the citation" your #2 asked for). The source row is
                    // tagged data-source-id for the scroll target.
                    setSourcesOpen(true);
                    handleSourceHover(id);
                    requestAnimationFrame(() => {
                      const el = document.querySelector(`[data-source-row="${id}"]`);
                      el?.scrollIntoView({ behavior: "smooth", block: "center" });
                    });
                  }
                }
              }}
            >
              {isFallback && (
                <div className="flex items-center gap-1.5 mb-2.5 text-[var(--status-processing)] text-xs bg-amber-50 border border-amber-100 rounded-lg px-2.5 py-1.5">
                  <AlertTriangle size={12} />
                  <span>AI temporarily unavailable. Showing retrieved passages.</span>
                </div>
              )}
              <div className="text-sm text-[var(--text-primary)]">
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
                  {cleaned}
                </ReactMarkdown>
              </div>
            </div>

            {/* TrustBar */}
            {!isStreaming && showTrust && answerMeta && (
              <TrustBar meta={answerMeta} />
            )}

            {/* F2g — review + partner-override affordances on a settled assistant answer. Render only
                when the surface gives us a vault + message ref. SendForReview shows for anyone who
                can send up the chain; the override moment unlocks for a partner when the answer
                abstained (gateObjection present). */}
            {!isUser && !isStreaming && vaultId && messageId && (
              <div className="flex flex-wrap items-center gap-2 mt-2">
                <SendForReview vaultId={vaultId} artifactRef={messageId} />
                {gateObjection && (
                  <OverrideAffordance
                    answerRef={messageId}
                    collectionId={vaultId}
                    gateObjection={gateObjection}
                  />
                )}
              </div>
            )}

            {/* Sources */}
            {hasSources && (
              <div className="ml-1">
                <button
                  onClick={() => setSourcesOpen(!sourcesOpen)}
                  className="flex items-center gap-1.5 text-[11px] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] rounded"
                  aria-expanded={sourcesOpen}
                >
                  <FileText size={11} />
                  <span>{sources!.length} source{sources!.length > 1 ? "s" : ""}</span>
                  {sourcesOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
                </button>
                <AnimatePresence initial={false}>
                  {sourcesOpen && (
                    <motion.div
                      key="sources"
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.2, ease: [0.23, 1, 0.32, 1] }}
                      className="mt-2 space-y-1.5 overflow-hidden"
                    >
                      {sources!.map((src, i) => {
                        const sourceId = src.source_id ?? i + 1;
                        const isHighlighted = hoveredSource === sourceId;
                        return (
                          <motion.div
                            key={i}
                            data-source-row={sourceId}
                            initial={{ opacity: 0, x: -4 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.04, ease: [0.23, 1, 0.32, 1] }}
                            className={`px-3 py-2 text-xs rounded-xl cursor-pointer transition-[border-color,box-shadow,background-color] duration-[120ms] ease-[cubic-bezier(0.23,1,0.32,1)]`}
                            style={{
                              background: isHighlighted ? "var(--bg-hover)" : "var(--glass-bg)",
                              backdropFilter: "blur(8px)",
                              border: isHighlighted ? "1px solid var(--accent)" : "1px solid var(--glass-border)",
                              boxShadow: isHighlighted ? "0 0 0 3px rgba(10,10,10,0.06)" : "var(--glass-shadow)",
                            }}
                            onMouseEnter={() => handleSourceHover(sourceId)}
                            onMouseLeave={() => handleSourceHover(null)}
                          >
                            <div className="flex items-start gap-2">
                              <span
                                className={`w-4 h-4 rounded text-[9px] font-bold flex items-center justify-center flex-shrink-0 mt-0.5 ${isHighlighted ? "bg-[var(--accent)] text-white" : "border border-dashed border-[var(--border-dotted)] text-[var(--text-secondary)]"}`}
                              >
                                {sourceId}
                              </span>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 flex-wrap">
                                  <span className="text-[var(--text-primary)] font-medium truncate">{src.filename ?? "Unknown"}</span>
                                  {src.page && <span className="text-[var(--text-muted)]">p. {src.page}</span>}
                                </div>
                                {src.content && (
                                  <p className="text-[11px] text-[var(--text-secondary)] mt-1 leading-relaxed italic line-clamp-2">&ldquo;{src.content}&rdquo;</p>
                                )}
                              </div>
                            </div>
                          </motion.div>
                        );
                      })}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── User: compact bubble, right-aligned within the same centered column ── */}
      {isUser && (
        <div className="flex items-end gap-2.5 w-full max-w-3xl mx-auto justify-end">
          <div className="flex flex-col items-end gap-1">
            <p className="text-[10px] font-medium text-[var(--text-muted)] mr-1">You</p>
            <div
              className="rounded-2xl rounded-br-sm px-4 py-3"
              style={{
                background: "var(--bg-surface)",
                color: "var(--text-primary)",
                border: "1px solid var(--border)",
                boxShadow: "var(--shadow-sm)",
              }}
            >
              <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
            </div>
          </div>
          <div
            className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center text-[11px] font-bold text-white mb-0.5"
            aria-hidden="true"
            style={{
              background: "var(--accent)",
              boxShadow: "var(--skeu-raised)",
            }}
          >
            {userInitials}
          </div>
        </div>
      )}
    </motion.div>
  );
});
