"use client";

// components/sidebar/DocumentUploadZone.tsx
import { useState, useRef, DragEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileText } from "lucide-react";
import { clsx } from "clsx";

const ACCEPTED_TYPES = [".pdf", ".docx", ".pptx", ".txt", ".xlsx"];
const MAX_SIZE_MB = 10;

interface DocumentUploadZoneProps {
  onFileSelect: (file: File) => void;
  isUploading: boolean;
}

export function DocumentUploadZone({
  onFileSelect,
  isUploading,
}: DocumentUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  function validateFile(file: File): string | null {
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!ACCEPTED_TYPES.includes(ext))
      return `Unsupported type. Use: ${ACCEPTED_TYPES.join(", ")}`;
    if (file.size > MAX_SIZE_MB * 1024 * 1024)
      return `File too large (max ${MAX_SIZE_MB}MB)`;
    return null;
  }

  function handleFile(file: File) {
    const err = validateFile(file);
    if (err) {
      setError(err);
      return;
    }
    setError(null);
    onFileSelect(file);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    if (isUploading) return;
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  return (
    <div className="px-2 pb-2">
      <motion.div
        whileHover={isUploading ? {} : { scale: 1.01 }}
        animate={
          isDragging
            ? { scale: 1.01, borderColor: "rgba(99,102,241,0.8)" }
            : { scale: 1, borderColor: "rgba(255,255,255,0.12)" }
        }
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
        className={clsx(
          "relative flex flex-col items-center justify-center gap-1.5",
          "h-20 rounded-xl border-2 border-dashed cursor-pointer transition-colors",
          isDragging
            ? "bg-accent-primary/8 border-accent-primary/80"
            : "hover:bg-white/3 hover:border-white/20",
          isUploading && "opacity-50 cursor-not-allowed pointer-events-none"
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        onClick={() => !isUploading && inputRef.current?.click()}
        role="button"
        aria-label="Upload document"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
      >
        <AnimatePresence mode="wait">
          {isUploading ? (
            <motion.div
              key="uploading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center gap-1"
            >
              <div className="w-4 h-4 border-2 border-accent-primary border-t-transparent rounded-full animate-spin" />
              <span className="text-[10px] text-text-muted">Uploading…</span>
            </motion.div>
          ) : (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center gap-1"
            >
              {isDragging ? (
                <FileText size={18} className="text-accent-primary" />
              ) : (
                <Upload size={16} className="text-text-muted" />
              )}
              <span className="text-[10px] text-text-muted text-center px-2">
                {isDragging
                  ? "Drop to upload"
                  : "Drop file or click to browse"}
              </span>
              <span className="text-[9px] text-text-muted/60">
                PDF · DOCX · PPTX · TXT · Max 10MB
              </span>
            </motion.div>
          )}
        </AnimatePresence>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
            e.target.value = "";
          }}
          aria-label="File input"
        />
      </motion.div>

      {error && (
        <motion.p
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-[10px] text-status-failed mt-1 px-1"
        >
          {error}
        </motion.p>
      )}
    </div>
  );
}
