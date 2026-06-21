"use client";

// /app/vault/[id]/draft — Draft Deliverable landing (G6.1).
//
// The third Vault mode (alongside Ask and Deep Analysis). The user picks a document
// type and writes a free-text brief; submitting creates a conversation and routes into
// the streaming draft view at /draft/[cid]?q=…&doc_type=…&instructions=… (same
// ChatConversation consumer, analysisMode="draft", no new orchestrator).
//
// Scope (§9 risk #1): the vault [id] is authoritative — read from the route.

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { FileEdit, ChevronRight } from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/auth.store";
import { createConversation, listDocTypes, type DocTypeCard } from "@/lib/api";
import { BackToVault } from "@/components/app/BackToVault";

const ease = [0.23, 1, 0.32, 1] as const;

// Practice-area display order (matches the catalog's PRACTICE_ORDER / the card grid).
const PRACTICE_ORDER = [
  "Litigation", "Transactional", "Financial services", "Real estate",
  "Employment", "IP & Technology", "Tax", "Compliance (India)", "Family",
];

// Generic fallback (the pre-catalog behaviour) — used only when the catalog is empty
// (USE_AGENT_CORE off ⇒ /doc-types 404s). A catalog doc-type id is expanded server-side
// into the India-correct structure; these generic values are passed through as free text.
const FALLBACK_DOC_TYPES = [
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

  // §2.3: the picker is generated from the catalog table, not authored one-by-one. Add a
  // DocType row → a card appears here. Empty (flag off) ⇒ the generic fallback list.
  const [catalog, setCatalog] = useState<DocTypeCard[]>([]);
  const [docType, setDocType] = useState(FALLBACK_DOC_TYPES[0].value);
  const [docTypeTitle, setDocTypeTitle] = useState<string>("");
  const [query, setQuery] = useState("");   // filter the picker (the catalog is large)
  const [instructions, setInstructions] = useState("");
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    if (!token) return;
    listDocTypes(token)
      .then((cards) => {
        if (cards.length) {
          setCatalog(cards);
          setDocType(cards[0].id);
          setDocTypeTitle(cards[0].title);
        }
      })
      .catch(() => { /* fall back to FALLBACK_DOC_TYPES */ });
  }, [token]);

  // Group catalog cards by practice area (the card grid layout), filtered by the search box.
  const grouped = useMemo(() => {
    const q = query.trim().toLowerCase();
    const match = (c: DocTypeCard) =>
      !q || c.title.toLowerCase().includes(q) || c.practice_area.toLowerCase().includes(q);
    const by: Record<string, DocTypeCard[]> = {};
    for (const c of catalog) if (match(c)) (by[c.practice_area] ??= []).push(c);
    return PRACTICE_ORDER.filter((p) => by[p]?.length).map((p) => ({ area: p, cards: by[p] }));
  }, [catalog, query]);

  const totalCards = catalog.length;

  async function startDraft() {
    const brief = instructions.trim();
    if (!token || creating || !brief) return;
    setCreating(true);
    try {
      const c = await createConversation(token, brief.slice(0, 50));
      const params = new URLSearchParams({
        q: brief,
        mode: "draft",          // explicit: the auto-submit must run the DRAFT path, not Agent
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
      {/* Top-aligned (not vertically centered): with the full catalog the content is taller
          than the viewport, so centering pushes the heading off-screen. Widen for the picker. */}
      <div className="relative min-h-full flex flex-col items-stretch justify-start w-full max-w-3xl mx-auto px-4 md:px-8 py-10 text-left">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease }}
          className="w-full"
        >
          <BackToVault vaultId={vaultId} className="mb-6" />

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
              {totalCards ? (
                // Catalog picker (§2.3): searchable, grouped by practice area. The catalog is
                // large (50+ types), so a flat chip wall is unusable — filter + bounded scroll.
                <div>
                  <div className="flex items-center gap-2 mb-2.5">
                    <input
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder={`Search ${totalCards} document types…`}
                      className="flex-1 rounded-lg px-3 py-2 text-[13px] outline-none"
                      style={{
                        background: "var(--surface)",
                        border: "1px solid var(--line)",
                        color: "var(--text-primary)",
                      }}
                    />
                    {docTypeTitle && (
                      <span
                        className="text-[12px] whitespace-nowrap px-2.5 py-1 rounded-full"
                        style={{ background: "var(--ink)", color: "var(--surface)" }}
                      >
                        {docTypeTitle}
                      </span>
                    )}
                  </div>
                  <div
                    className="max-h-[42vh] overflow-y-auto scrollbar-thin rounded-xl p-3 space-y-4"
                    style={{ background: "var(--surface-2, var(--surface))", border: "1px solid var(--line)" }}
                  >
                    {grouped.length ? grouped.map(({ area, cards }) => (
                      <div key={area}>
                        <div
                          className="text-[10px] font-semibold uppercase tracking-[0.12em] mb-2"
                          style={{ color: "var(--text-muted)" }}
                        >
                          {area} <span style={{ opacity: 0.6 }}>· {cards.length}</span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {cards.map((c) => {
                            const active = docType === c.id;
                            return (
                              <button
                                key={c.id}
                                onClick={() => { setDocType(c.id); setDocTypeTitle(c.title); }}
                                title={c.description}
                                className="px-3 py-1.5 rounded-full text-[13px] font-medium transition-colors"
                                style={{
                                  background: active ? "var(--ink)" : "var(--surface)",
                                  border: active ? "1px solid var(--ink)" : "1px solid var(--line)",
                                  color: active ? "var(--surface)" : "var(--ink-2)",
                                }}
                              >
                                {c.title}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    )) : (
                      <p className="text-[13px] py-2" style={{ color: "var(--text-muted)" }}>
                        No document type matches “{query}”.
                      </p>
                    )}
                  </div>
                </div>
              ) : (
                // Generic fallback (flag off / empty catalog).
                <div className="flex flex-wrap gap-2">
                  {FALLBACK_DOC_TYPES.map((t) => {
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
              )}
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
