"use client";

// UploadZone — the document ingress for a vault (G2 Step C, the NEW hybrid upload UX).
// NOT a big Harvey-style dropzone. Two ingress modes:
//   1. a slim "Add documents" command bar (click → browse), keyboard-first, gets out of
//      the way once the vault has files; and
//   2. a FULL-PAGE drag overlay — drag a file anywhere over the workspace and a clean
//      drop target fades in over the whole page.
// Once files are in, their LIVE progress shows as pipeline tracks (PipelineTrack) driven
// by the parent's polled document list — so the source of truth is the real worker
// progress, not local guesswork. This component owns ingress; the parent owns the docs.
//
// Lifted/reworked from the old 752-line app/app/layout.tsx so there's ONE upload path
// (G2 §9 risk #6). Born in the clean B&W system.

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Plus, UploadCloud } from "lucide-react";
import { toast } from "sonner";
import { uploadDocument, addDocToCollection } from "@/lib/api";

const ACCEPTED = ".pdf,.docx,.pptx,.txt,.xlsx";
const MAX_MB = 50;
const CONCURRENCY = 5;

interface UploadZoneProps {
  token: string;
  collectionId: string | null; // uploaded docs are added to this vault when set
  onUploaded: () => void;        // refetch the document list (drives the pipeline tracks)
  label?: string;
}

export function UploadZone({ token, collectionId, onUploaded, label = "Add documents" }: UploadZoneProps) {
  const [busy, setBusy] = useState(false);
  const [dragDepth, setDragDepth] = useState(0); // window drag-enter/leave counter
  const fileRef = useRef<HTMLInputElement>(null);

  const uploadFiles = useCallback(
    async (files: File[]) => {
      if (!token || files.length === 0) return;

      const valid: File[] = [];
      for (const file of files) {
        const ext = "." + file.name.split(".").pop()?.toLowerCase();
        if (!ACCEPTED.split(",").includes(ext)) { toast.error(`Unsupported type: ${file.name}`); continue; }
        if (file.size > MAX_MB * 1048576) { toast.error(`${file.name} exceeds ${MAX_MB}MB`); continue; }
        valid.push(file);
      }
      if (valid.length === 0) return;

      setBusy(true);
      let idx = 0;
      let anySuccess = false;

      async function uploadOne(file: File) {
        try {
          // Pass the vault so the backend stamps the matter owner (F2m/D0) + auto-links. The
          // addDocToCollection below stays as a harmless idempotent fallback for own vaults.
          const doc = await uploadDocument(token, file, collectionId);
          if (collectionId) {
            try { await addDocToCollection(token, collectionId, doc.id); } catch { /* non-fatal */ }
          }
          anySuccess = true;
        } catch (e: unknown) {
          toast.error(`Upload failed: ${file.name}. ${e instanceof Error ? e.message : "Unknown error"}`);
        }
      }

      async function worker() {
        while (idx < valid.length) {
          await uploadOne(valid[idx++]);
          // Surface each new row as soon as it lands so its pipeline track appears and the
          // parent's status poll starts walking it through Parse→Chunk→Embed→Ready.
          if (anySuccess) onUploaded();
        }
      }

      await Promise.all(Array.from({ length: Math.min(CONCURRENCY, valid.length) }, worker));
      onUploaded();
      setBusy(false);
    },
    [token, collectionId, onUploaded]
  );

  // ── Full-page drag overlay ──────────────────────────────────────────────────
  // Track drag enter/leave at the window level with a depth counter (dragenter/leave
  // fire per descendant, so a single boolean flickers). When a file is dragged anywhere
  // over the page, a clean drop target fades in.
  useEffect(() => {
    function hasFiles(e: DragEvent) {
      return Array.from(e.dataTransfer?.types ?? []).includes("Files");
    }
    function onEnter(e: DragEvent) { if (hasFiles(e)) { e.preventDefault(); setDragDepth((d) => d + 1); } }
    function onLeave(e: DragEvent) { if (hasFiles(e)) setDragDepth((d) => Math.max(0, d - 1)); }
    function onOver(e: DragEvent) { if (hasFiles(e)) e.preventDefault(); }
    function onDrop(e: DragEvent) {
      if (!hasFiles(e)) return;
      e.preventDefault();
      setDragDepth(0);
      const files = Array.from(e.dataTransfer?.files ?? []);
      if (files.length) uploadFiles(files);
    }
    window.addEventListener("dragenter", onEnter);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("dragover", onOver);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onEnter);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("drop", onDrop);
    };
  }, [uploadFiles]);

  return (
    <>
      <input
        ref={fileRef}
        type="file"
        accept={ACCEPTED}
        multiple
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length) uploadFiles(files);
          e.target.value = "";
        }}
      />

      {/* Slim command bar */}
      <button
        type="button"
        onClick={() => fileRef.current?.click()}
        disabled={busy}
        className="w-full flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl text-[13px] font-medium transition-colors hover:bg-[var(--bg-hover)] disabled:opacity-60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
        style={{ background: "var(--surface)", border: "1px dashed var(--border-dotted)", color: "var(--ink-2)" }}
      >
        <span
          className="w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
        >
          {busy ? (
            <span className="w-3 h-3 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
          ) : (
            <Plus size={13} />
          )}
        </span>
        <span>{busy ? "Uploading…" : label}</span>
        <span className="ml-auto text-[11px] text-[var(--text-muted)] hidden sm:inline">
          or drag files anywhere
        </span>
      </button>

      {/* Full-page drop overlay */}
      <AnimatePresence>
        {dragDepth > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.14 }}
            className="fixed inset-0 flex items-center justify-center p-8 pointer-events-none"
            style={{ zIndex: "var(--z-modal)" as unknown as number, background: "rgba(250,250,250,0.78)", backdropFilter: "blur(3px)" }}
          >
            <motion.div
              initial={{ scale: 0.98 }}
              animate={{ scale: 1 }}
              className="w-full max-w-2xl h-[60vh] rounded-3xl flex flex-col items-center justify-center gap-4 text-center"
              style={{ border: "2px dashed var(--ink)", background: "rgba(255,255,255,0.6)" }}
            >
              <motion.span
                className="w-14 h-14 rounded-2xl flex items-center justify-center"
                style={{ background: "var(--ink)", color: "var(--on-ink)" }}
                animate={{ y: [0, -6, 0] }}
                transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
              >
                <UploadCloud size={24} />
              </motion.span>
              <p style={{ fontFamily: "Fraunces, Georgia, serif", fontSize: 22, fontWeight: 500, color: "var(--ink)" }}>
                Drop to add to this vault
              </p>
              <p className="text-[12px] text-[var(--text-muted)]">PDF, DOCX, PPTX, TXT, XLSX · up to {MAX_MB}MB</p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
