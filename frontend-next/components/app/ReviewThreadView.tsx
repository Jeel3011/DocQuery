"use client";

// F2j.1 — ReviewThreadView: the read-only "View full work" surface for a reviewer.
//
// Renders the WHOLE conversation behind a submitted artifact so a senior verifies the work in context
// before Approve / Request changes (the "view full context" code-review principle). READ-ONLY — no
// input box, no edit; the reviewer reads, then decides on the card. The cross-user read is gated +
// audited server-side (db.review_artifact_authority); this is purely the viewer.
//
// DESIGN: B&W answer style — the question is muted ink, the answer is full ink rendered as markdown
// (same react-markdown/remark-gfm as ChatMessage). Source chips reused as plain neutral chips. No new
// trust indicator. Every state designed: loading skeleton, "work unavailable", empty.

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileText } from "lucide-react";
import { useAuthStore } from "@/stores/auth.store";
import { getReviewThread, type ReviewThread } from "@/lib/api";
import { Skeleton } from "@/components/ui/Skeleton";

function sourceLabel(s: unknown): string {
  if (typeof s === "string") return s;
  if (s && typeof s === "object") {
    const o = s as Record<string, unknown>;
    return String(o.filename ?? o.source ?? o.title ?? o.document ?? "source");
  }
  return "source";
}

export function ReviewThreadView({ requestId }: { requestId: string }) {
  const token = useAuthStore((s) => s.token);
  const [thread, setThread] = useState<ReviewThread | null>(null);

  useEffect(() => {
    if (!token) return;
    let alive = true;
    getReviewThread(token, requestId).then((t) => { if (alive) setThread(t); });
    return () => { alive = false; };
  }, [token, requestId]);

  if (thread === null) {
    return (
      <div className="space-y-3 px-5 py-4">
        {[0, 1, 2].map((i) => <Skeleton key={i} className="h-16 w-full rounded-lg" />)}
      </div>
    );
  }

  if (!thread.available || thread.messages.length === 0) {
    return (
      <div className="px-5 py-10 text-center">
        <p className="text-[13px]" style={{ color: "var(--ink-2)" }}>This work is no longer available.</p>
        <p className="text-[11px] mt-1" style={{ color: "var(--ink-3)" }}>
          The conversation may have been deleted after it was sent for review.
        </p>
      </div>
    );
  }

  return (
    <div className="px-5 py-4 max-h-[60vh] overflow-y-auto space-y-4">
      {thread.title && (
        <p className="text-[12px] font-medium" style={{ color: "var(--ink-3)" }}>{thread.title}</p>
      )}
      {thread.messages.map((m, i) => {
        const isUser = m.role === "user";
        return (
          <div key={i} className="space-y-1.5">
            <p className="text-[10px] uppercase tracking-[0.1em]" style={{ color: "var(--ink-3)" }}>
              {isUser ? "Question" : "Answer"}
            </p>
            {isUser ? (
              <p className="text-[13px] whitespace-pre-wrap" style={{ color: "var(--ink-2)" }}>{m.content}</p>
            ) : (
              <div
                className="text-[13px] leading-relaxed markdown-body"
                style={{ color: "var(--ink)" }}
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
              </div>
            )}
            {!isUser && Array.isArray(m.sources) && m.sources.length > 0 && (
              <div className="flex flex-wrap gap-1.5 pt-1">
                {m.sources.slice(0, 8).map((s, j) => (
                  <span key={j}
                    className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-md"
                    style={{ background: "var(--surface-3)", color: "var(--ink-2)" }}>
                    <FileText size={9} /> {sourceLabel(s)}
                  </span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
