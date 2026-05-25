"use client";

// components/sidebar/DocumentCard.tsx
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileText, Trash2, RefreshCw, CheckCircle, AlertCircle } from "lucide-react";
import { DocumentResponse } from "@/lib/api";
import { clsx } from "clsx";

function formatBytes(bytes: number | null): string {
  if (!bytes) return "0 KB";
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

interface DocumentCardProps {
  doc: DocumentResponse;
  onDelete: (id: string) => void;
  onRetry?: (id: string) => void;
}

export function DocumentCard({ doc, onDelete, onRetry }: DocumentCardProps) {
  const [hovered, setHovered] = useState(false);
  const [confirming, setConfirming] = useState(false);

  const statusColors = {
    ready: "bg-status-ready",
    processing: "bg-status-processing animate-pulse",
    failed: "bg-status-failed",
  };

  const truncatedName =
    doc.filename.length > 26
      ? doc.filename.slice(0, 23) + "…"
      : doc.filename;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: -20 }}
      transition={{ duration: 0.2 }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => {
        setHovered(false);
        setConfirming(false);
      }}
      className="glass rounded-xl px-3 py-2.5 mx-2 mb-1.5 group"
    >
      <div className="flex items-start gap-2">
        <FileText size={14} className="text-accent-primary flex-shrink-0 mt-0.5" />

        <div className="flex-1 min-w-0">
          {/* Filename + status dot */}
          <div className="flex items-center gap-1.5">
            <span
              className="text-[10px] font-medium text-text-primary truncate"
              title={doc.filename}
            >
              {truncatedName}
            </span>
            <span
              className={clsx(
                "w-1.5 h-1.5 rounded-full flex-shrink-0",
                statusColors[doc.status]
              )}
            />
          </div>

          {/* Metadata */}
          <p className="text-[9px] text-text-muted mt-0.5">
            {doc.status === "processing" ? (
              <span className="text-status-processing">
                Processing{" "}
                {doc.processing_progress != null
                  ? `(${doc.processing_progress}%)`
                  : "…"}
              </span>
            ) : (
              <>
                {doc.chunk_count} chunks · {formatBytes(doc.file_size_bytes)}
              </>
            )}
          </p>

          {/* Progress bar when processing */}
          {doc.status === "processing" && (
            <div className="mt-1.5 h-0.5 bg-white/10 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-accent-primary to-accent-secondary rounded-full"
                initial={{ width: 0 }}
                animate={{
                  width: `${doc.processing_progress ?? 30}%`,
                }}
                transition={{ duration: 0.5 }}
              />
            </div>
          )}

          {/* Failed state retry */}
          {doc.status === "failed" && onRetry && (
            <button
              onClick={() => onRetry(doc.id)}
              className="mt-1 flex items-center gap-1 text-[9px] text-status-failed hover:opacity-80 transition-opacity"
            >
              <RefreshCw size={9} />
              Retry
            </button>
          )}
        </div>

        {/* Status icon (right side) */}
        <div className="flex-shrink-0">
          {doc.status === "ready" && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="animate-scale-in"
            >
              <CheckCircle size={12} className="text-status-ready" />
            </motion.div>
          )}
          {doc.status === "failed" && (
            <AlertCircle size={12} className="text-status-failed" />
          )}
        </div>
      </div>

      {/* Delete button — appears on hover */}
      <AnimatePresence>
        {hovered && doc.status !== "processing" && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-2 flex items-center gap-2 pt-2 border-t border-white/5">
              {confirming ? (
                <>
                  <span className="text-[9px] text-text-muted flex-1">
                    Delete this document?
                  </span>
                  <button
                    onClick={() => onDelete(doc.id)}
                    className="text-[9px] text-status-failed hover:opacity-80 px-2 py-0.5 rounded border border-status-failed/30"
                  >
                    Delete
                  </button>
                  <button
                    onClick={() => setConfirming(false)}
                    className="text-[9px] text-text-muted hover:opacity-80 px-2 py-0.5 rounded border border-white/10"
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <button
                  onClick={() => setConfirming(true)}
                  className="ml-auto flex items-center gap-1 text-[9px] text-text-muted hover:text-status-failed transition-colors"
                  aria-label="Delete document"
                >
                  <Trash2 size={10} />
                  Delete
                </button>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
