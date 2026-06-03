"use client";

import { useReducedMotion, motion, AnimatePresence } from "framer-motion";
import { X, Copy, Download, Code, FileText, Table2, CheckCheck } from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export type ArtifactKind = "markdown" | "code" | "table" | "text";

export interface Artifact {
  id: string;
  kind: ArtifactKind;
  title: string;
  content: string;
  language?: string;
}

interface ArtifactPanelProps {
  artifact: Artifact | null;
  onClose: () => void;
}

const kindIcon: Record<ArtifactKind, React.ReactNode> = {
  markdown: <FileText size={13} />,
  code: <Code size={13} />,
  table: <Table2 size={13} />,
  text: <FileText size={13} />,
};

export function ArtifactPanel({ artifact, onClose }: ArtifactPanelProps) {
  const rm = useReducedMotion();
  const [copied, setCopied] = useState(false);

  function copyContent() {
    if (!artifact) return;
    navigator.clipboard.writeText(artifact.content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  function downloadContent() {
    if (!artifact) return;
    const ext = artifact.kind === "code" ? (artifact.language ?? "txt") : "md";
    const blob = new Blob([artifact.content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${artifact.title.replace(/\s+/g, "_")}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <AnimatePresence>
      {artifact && (
        <motion.aside
          key="artifact-panel"
          initial={{ x: rm ? 0 : "100%", opacity: rm ? 0 : 1 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: rm ? 0 : "100%", opacity: rm ? 0 : 1 }}
          transition={{ duration: 0.28, ease: [0.32, 0.72, 0, 1] }}
          className="flex flex-col h-full overflow-hidden border-l"
          style={{
            width: "clamp(320px, 48%, 640px)",
            background: "var(--glass-bg-strong)",
            backdropFilter: "blur(20px)",
            WebkitBackdropFilter: "blur(20px)",
            borderColor: "var(--glass-border)",
            boxShadow: "-4px 0 24px rgba(0,0,0,0.06)",
          }}
        >
          {/* Header */}
          <div
            className="flex items-center gap-3 px-4 py-3 border-b flex-shrink-0"
            style={{
              background: "var(--glass-bg-strong)",
              backdropFilter: "blur(12px)",
              borderColor: "var(--glass-border)",
              boxShadow: "var(--skeu-inset)",
            }}
          >
            <div
              className="w-7 h-7 rounded-lg flex items-center justify-center text-[var(--text-secondary)] flex-shrink-0"
              style={{
                background: "var(--glass-bg)",
                border: "1px solid var(--glass-border)",
                boxShadow: "var(--skeu-raised)",
              }}
            >
              {kindIcon[artifact.kind]}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-[var(--text-primary)] truncate">{artifact.title}</p>
              <p className="text-[10px] text-[var(--text-muted)] capitalize">{artifact.kind}{artifact.language ? ` · ${artifact.language}` : ""}</p>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={copyContent}
                aria-label="Copy artifact"
                className="w-7 h-7 rounded-lg flex items-center justify-center text-[var(--text-muted)] transition-[color,background-color] duration-[120ms] ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97]"
                style={{ border: "1px solid var(--glass-border)", background: "var(--glass-bg)", boxShadow: "var(--skeu-raised)" }}
              >
                {copied ? <CheckCheck size={12} className="text-[var(--status-ready)]" /> : <Copy size={12} />}
              </button>
              <button
                onClick={downloadContent}
                aria-label="Download artifact"
                className="w-7 h-7 rounded-lg flex items-center justify-center text-[var(--text-muted)] transition-[color,background-color] duration-[120ms] ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97]"
                style={{ border: "1px solid var(--glass-border)", background: "var(--glass-bg)", boxShadow: "var(--skeu-raised)" }}
              >
                <Download size={12} />
              </button>
              <button
                onClick={onClose}
                aria-label="Close artifact panel"
                className="w-7 h-7 rounded-lg flex items-center justify-center text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-[color,background-color] duration-[120ms] ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97]"
                style={{ border: "1px solid var(--glass-border)", background: "var(--glass-bg)", boxShadow: "var(--skeu-raised)" }}
              >
                <X size={13} />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto scrollbar-thin p-5">
            {artifact.kind === "code" ? (
              <pre
                className="text-xs font-mono text-[var(--text-primary)] leading-relaxed whitespace-pre-wrap break-words rounded-xl p-4"
                style={{
                  background: "rgba(0,0,0,0.04)",
                  border: "1px solid var(--glass-border)",
                  boxShadow: "inset 0 1px 3px rgba(0,0,0,0.04)",
                }}
              >
                <code>{artifact.content}</code>
              </pre>
            ) : (
              <div
                className="prose prose-sm max-w-none text-[var(--text-primary)]"
                style={{ "--tw-prose-body": "var(--text-primary)", "--tw-prose-headings": "var(--text-primary)" } as React.CSSProperties}
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {artifact.content}
                </ReactMarkdown>
              </div>
            )}
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}

/* ── Detect artifact blocks in assistant output ── */
const ARTIFACT_RE = /```artifact(?::(\w+))?(?:\s+title="([^"]*)")?\n([\s\S]*?)```/i;
const CODE_RE = /```(\w+)\n([\s\S]{200,})```/;
const TABLE_RE = /\|.+\|[\s\S]{60,}/;

export function detectArtifact(content: string): Artifact | null {
  // Explicit artifact fence
  const explicit = ARTIFACT_RE.exec(content);
  if (explicit) {
    const [, kind, title, body] = explicit;
    return {
      id: String(Date.now()),
      kind: (kind as ArtifactKind) ?? "markdown",
      title: title ?? "Artifact",
      content: body.trim(),
    };
  }
  // Long code block
  const code = CODE_RE.exec(content);
  if (code) {
    return {
      id: String(Date.now()),
      kind: "code",
      title: `${code[1]} snippet`,
      content: code[2].trim(),
      language: code[1],
    };
  }
  // Large markdown table
  if (TABLE_RE.test(content) && content.length > 400) {
    return {
      id: String(Date.now()),
      kind: "table",
      title: "Table",
      content: content,
    };
  }
  return null;
}
