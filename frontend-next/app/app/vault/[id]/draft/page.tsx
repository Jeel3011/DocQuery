"use client";

// /app/vault/[id]/draft — Draft Deliverable landing (G6.1).
//
// The third Vault mode (alongside Ask and Deep Analysis). The user picks a document
// type and writes a free-text brief; submitting creates a conversation and routes into
// the streaming draft view at /draft/[cid]?q=…&doc_type=…&instructions=… (same
// ChatConversation consumer, analysisMode="draft", no new orchestrator).
//
// Scope (§9 risk #1): the vault [id] is authoritative — read from the route.

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { FileEdit, ChevronRight } from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/auth.store";
import { createConversation } from "@/lib/api";

const ease = [0.23, 1, 0.32, 1] as const;

const DOC_TYPES = [
  { value: "memo",            label: "Legal memo" },
  { value: "summary",         label: "Summary / executive brief" },
  { value: "letter",          label: "Letter / correspondence" },
  { value: "clause_analysis", label: "Clause analysis" },
  { value: "other",           label: "Other" },
];

export default function VaultDraftPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const { token } = useAuthStore();
  const vaultId = params.id;

  const [docType, setDocType] = useState(DOC_TYPES[0].value);
  const [instructions, setInstructions] = useState("");
  const [creating, setCreating] = useState(false);

  async function startDraft() {
    const brief = instructions.trim();
    if (!token || creating || !brief) return;
    setCreating(true);
    try {
      const c = await createConversation(token, brief.slice(0, 50));
      const params = new URLSearchParams({
        q: brief,
        doc_type: docType,
        instructions: brief,
      });
      router.push(`/app/vault/${vaultId}/draft/${c.id}?${params.toString()}`);
    } catch {
      toast.error("Failed to start draft");
      setCreating(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") startDraft();
  }

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin relative" style={{ background: "var(--canvas)" }}>
      <div className="relative h-full flex flex-col items-center justify-center w-full max-w-2xl mx-auto px-4 md:px-8 py-10 text-left">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease }}
          className="w-full"
        >
          <div className="flex items-center gap-2.5 mb-3">
            <span
              className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
            >
              <FileEdit size={17} />
            </span>
            <span className="text-[11px] font-semibold uppercase tracking-[0.12em]" style={{ color: "var(--text-muted)" }}>
              Draft Document
            </span>
          </div>

          <h2
            className="text-[30px] md:text-[36px] leading-[1.08] font-bold tracking-[-0.035em] mb-3"
            style={{ fontFamily: "Fraunces, Georgia, serif", color: "var(--ink)" }}
          >
            A cited deliverable from the vault.
          </h2>
          <p className="text-[14px] leading-relaxed mb-8" style={{ color: "var(--text-muted)" }}>
            The agent reads the documents, grounds every claim in a cited source, and
            produces a draft — section by section. Anything it cannot trace is honestly
            withheld rather than hallucinated.
          </p>

          {/* Form */}
          <div className="space-y-5">
            {/* Document type */}
            <div>
              <label
                htmlFor="doc-type"
                className="block text-[12px] font-semibold uppercase tracking-[0.08em] mb-1.5"
                style={{ color: "var(--text-muted)" }}
              >
                Document type
              </label>
              <div className="flex flex-wrap gap-2">
                {DOC_TYPES.map((t) => {
                  const active = docType === t.value;
                  return (
                    <button
                      key={t.value}
                      onClick={() => setDocType(t.value)}
                      className="px-3 py-1.5 rounded-full text-[13px] font-medium transition-colors"
                      style={{
                        background: active ? "var(--ink)" : "var(--surface)",
                        border: "1px solid var(--line)",
                        color: active ? "var(--surface)" : "var(--ink-2)",
                      }}
                    >
                      {t.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Instructions */}
            <div>
              <label
                htmlFor="draft-instructions"
                className="block text-[12px] font-semibold uppercase tracking-[0.08em] mb-1.5"
                style={{ color: "var(--text-muted)" }}
              >
                Brief / instructions
              </label>
              <textarea
                id="draft-instructions"
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder={`Describe what you need — e.g. "Draft a legal memo summarising the indemnity and liability cap clauses across these contracts, flagging anything non-standard."`}
                rows={5}
                className="w-full resize-none rounded-2xl px-4 py-3 text-[14px] leading-relaxed outline-none transition-shadow"
                style={{
                  background: "var(--surface)",
                  border: "1px solid var(--line)",
                  color: "var(--text-primary)",
                  boxShadow: "inset 0 1px 3px rgba(0,0,0,0.04)",
                }}
              />
              <p className="mt-1 text-[11px]" style={{ color: "var(--text-muted)" }}>
                ⌘ + Enter to submit
              </p>
            </div>

            {/* Submit */}
            <button
              onClick={startDraft}
              disabled={creating || !instructions.trim()}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-semibold transition-opacity disabled:opacity-40"
              style={{ background: "var(--ink)", color: "var(--surface)" }}
            >
              {creating ? "Starting…" : "Generate draft"}
              {!creating && <ChevronRight size={15} />}
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
