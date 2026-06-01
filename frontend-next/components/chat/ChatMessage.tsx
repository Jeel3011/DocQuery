"use client";

import { memo, useState, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AlertTriangle, ChevronDown, ChevronUp, FileText } from "lucide-react";
import { SourceInfo } from "@/lib/api";
import { TrustBar, AnswerMeta } from "./TrustBar";
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
}

// Convert [Source: filename, Page: X] → [n] markers; return cleaned text + cited ids
function parseCitations(text: string, sources: SourceInfo[]): {
  cleaned: string;
  citedIds: Set<number>;
} {
  const pattern = /\[Source:\s*([^,\]]+)(?:,\s*Page:\s*(\d+))?\]/gi;
  const citedIds = new Set<number>();
  const sourceMap = new Map<string, number>();
  sources.forEach((s, i) => {
    const key = s.filename?.toLowerCase() ?? "";
    if (!sourceMap.has(key)) sourceMap.set(key, s.source_id ?? i + 1);
  });
  const cleaned = text.replace(pattern, (_, filename) => {
    const id = sourceMap.get(filename.trim().toLowerCase()) ?? 1;
    citedIds.add(id);
    return `[${id}]`;
  });
  return { cleaned, citedIds };
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
          (th: React.ReactElement) => String((th?.props as { children?: unknown })?.children ?? "")
        );
      }
      if (type === "tbody") {
        const trs = props.children as React.ReactElement[];
        (Array.isArray(trs) ? trs : [trs]).forEach((tr: React.ReactElement) => {
          const tds = (tr?.props as { children?: React.ReactElement[] })?.children as React.ReactElement[];
          rows.push(
            (Array.isArray(tds) ? tds : [tds]).map(
              (td: React.ReactElement) => String((td?.props as { children?: unknown })?.children ?? "")
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
  a: ({ href, children }: { href?: string; children?: React.ReactNode }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-[var(--accent)] underline underline-offset-2 hover:opacity-70 transition-opacity">
      {children}
    </a>
  ),
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

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
      className="px-4 md:px-8 py-3"
    >
      <div className="max-w-3xl mx-auto flex gap-3">
        {/* Avatar */}
        <div
          className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5 text-[11px] font-bold
            ${isUser
              ? "bg-[var(--accent)] text-white"
              : "border-2 border-dashed border-[var(--border-dotted)] text-[var(--text-muted)]"
            }`}
          aria-hidden="true"
        >
          {isUser ? userInitials : "D"}
        </div>

        <div className="flex-1 min-w-0">
          {/* Role label */}
          <p className="text-[11px] font-medium mb-1.5 text-[var(--text-muted)]">
            {isUser ? "You" : "DocQuery"}
          </p>

          {/* Bubble */}
          <div
            className={`rounded-xl px-4 py-3 ${
              isUser
                ? "bg-[var(--bg-surface)] border border-[var(--border)]"
                : "bg-[var(--bg-base)] border border-dashed border-[var(--border-dotted)]"
            }`}
          >
            {/* Fallback banner */}
            {isFallback && (
              <div className="flex items-center gap-1.5 mb-2.5 text-[var(--status-processing)] text-xs bg-amber-50 border border-amber-100 rounded-lg px-2.5 py-1.5">
                <AlertTriangle size={12} />
                <span>AI temporarily unavailable — showing retrieved passages</span>
              </div>
            )}

            {isUser ? (
              <p className="text-sm text-[var(--text-primary)] leading-relaxed whitespace-pre-wrap">{content}</p>
            ) : (
              <div
                className={`text-sm text-[var(--text-primary)] ${isStreaming ? "streaming-cursor" : ""}`}
                onClick={(e) => {
                  const target = e.target as HTMLElement;
                  if (target.classList.contains("citation-chip")) {
                    const id = parseInt(target.dataset.sourceId ?? "0", 10);
                    if (id) { setSourcesOpen(true); handleSourceHover(id); }
                  }
                }}
              >
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={mdComponents}
                >
                  {cleaned}
                </ReactMarkdown>
              </div>
            )}
          </div>

          {/* TrustBar — mock-now, wires to real AnswerMeta later */}
          {!isUser && !isStreaming && showTrust && answerMeta && (
            <TrustBar meta={answerMeta} />
          )}

          {/* Sources rail */}
          {hasSources && (
            <div className="mt-2">
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
                    transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                    className="mt-2 space-y-1.5 overflow-hidden"
                  >
                    {sources!.map((src, i) => {
                      const sourceId = src.source_id ?? i + 1;
                      const isHighlighted = hoveredSource === sourceId;
                      return (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, x: -4 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.04 }}
                          className={`card px-3 py-2 text-xs transition-all duration-150 cursor-pointer
                            ${isHighlighted
                              ? "border-[var(--accent)] shadow-[0_0_0_3px_rgba(10,10,10,0.06)] bg-[var(--bg-hover)]"
                              : "hover:border-[var(--border-strong)]"
                            }`}
                          onMouseEnter={() => handleSourceHover(sourceId)}
                          onMouseLeave={() => handleSourceHover(null)}
                        >
                          <div className="flex items-start gap-2">
                            <span
                              className={`w-4 h-4 rounded text-[9px] font-bold flex items-center justify-center flex-shrink-0 transition-all duration-150 mt-0.5
                                ${isHighlighted
                                  ? "bg-[var(--accent)] text-white"
                                  : "border border-dashed border-[var(--border-dotted)] text-[var(--text-secondary)]"
                                }`}
                            >
                              {sourceId}
                            </span>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="text-[var(--text-primary)] font-medium truncate">{src.filename ?? "Unknown"}</span>
                                {src.page && <span className="text-[var(--text-muted)] flex-shrink-0">p. {src.page}</span>}
                                {src.chunk_type && (
                                  <span className="text-[9px] text-[var(--text-muted)] capitalize border border-[var(--border)] rounded px-1 flex-shrink-0">
                                    {src.chunk_type}
                                  </span>
                                )}
                              </div>
                              {src.content && (
                                <p className="text-[11px] text-[var(--text-secondary)] mt-1 leading-relaxed italic line-clamp-2">
                                  &ldquo;{src.content}&rdquo;
                                </p>
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
    </motion.div>
  );
});
