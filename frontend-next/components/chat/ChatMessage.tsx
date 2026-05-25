"use client";

// components/chat/ChatMessage.tsx — Monochrome
// Both user and assistant left-aligned. Black user avatar, "D" assistant avatar.
// Sources collapsed by default. Clean card borders.

import { memo, useState } from "react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AlertTriangle, ChevronDown, ChevronUp, FileText } from "lucide-react";
import { SourceInfo } from "@/lib/api";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  sources?: SourceInfo[] | null;
  isStreaming?: boolean;
  isFallback?: boolean;
  userInitials?: string;
}

export const ChatMessage = memo(function ChatMessage({
  role, content, sources, isStreaming, isFallback, userInitials = "U",
}: ChatMessageProps) {
  const isUser = role === "user";
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const hasSources = !isUser && sources && sources.length > 0 && !isStreaming;

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="px-4 md:px-8 py-3"
    >
      <div className="max-w-3xl mx-auto flex gap-3">
        {/* Avatar */}
        <div className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-1 text-[11px] font-bold
          ${isUser ? "bg-[var(--accent)] text-white" : "border-2 border-dashed border-[var(--border-dotted)] text-[var(--text-secondary)]"}`}>
          {isUser ? userInitials : "D"}
        </div>

        <div className="flex-1 min-w-0">
          {/* Role */}
          <p className="text-[11px] font-medium mb-1 text-[var(--text-muted)]">
            {isUser ? "You" : "DocQuery"}
          </p>

          {/* Content */}
          <div className={`rounded-xl px-4 py-3 ${
            isUser
              ? "bg-[var(--bg-surface)] border border-[var(--border)]"
              : "bg-[var(--bg-base)] border border-dashed border-[var(--border-dotted)]"
          }`}>
            {isFallback && (
              <div className="flex items-center gap-1.5 mb-2 text-[var(--status-processing)] text-xs">
                <AlertTriangle size={12} />
                <span>AI unavailable — showing retrieved passage</span>
              </div>
            )}

            {isUser ? (
              <p className="text-sm text-[var(--text-primary)] leading-relaxed whitespace-pre-wrap">{content}</p>
            ) : (
              <div className={`text-sm text-[var(--text-primary)] leading-relaxed
                prose prose-sm max-w-none
                prose-p:my-1.5 prose-headings:text-[var(--text-primary)] prose-headings:font-semibold
                prose-code:text-[var(--text-primary)] prose-code:bg-[var(--bg-hover)] prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-code:font-mono
                prose-pre:bg-[var(--bg-hover)] prose-pre:border prose-pre:border-[var(--border)] prose-pre:rounded-xl
                prose-strong:font-semibold prose-a:text-[var(--accent)] prose-a:underline
                ${isStreaming ? "streaming-cursor" : ""}`}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
              </div>
            )}
          </div>

          {/* Sources — collapsed by default */}
          {hasSources && (
            <div className="mt-2">
              <button onClick={() => setSourcesOpen(!sourcesOpen)}
                className="flex items-center gap-1.5 text-[11px] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors">
                <FileText size={11} />
                <span>{sources!.length} source{sources!.length > 1 ? "s" : ""}</span>
                {sourcesOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
              </button>
              {sourcesOpen && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }}
                  className="mt-2 space-y-1.5 overflow-hidden">
                  {sources!.map((src, i) => (
                    <div key={i} className="card px-3 py-2 text-xs">
                      <div className="flex items-center gap-2">
                        <span className="w-4 h-4 rounded border border-dashed border-[var(--border-dotted)] text-[9px] font-bold flex items-center justify-center flex-shrink-0 text-[var(--text-secondary)]">
                          {src.source_id ?? i + 1}
                        </span>
                        <span className="text-[var(--text-primary)] truncate">{src.filename ?? "Unknown"}</span>
                        {src.page && <span className="text-[var(--text-muted)]">p.{src.page}</span>}
                      </div>
                      {src.content && (
                        <p className="text-[11px] text-[var(--text-secondary)] mt-1.5 leading-relaxed italic line-clamp-3">
                          &quot;{src.content}&quot;
                        </p>
                      )}
                    </div>
                  ))}
                </motion.div>
              )}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
});
