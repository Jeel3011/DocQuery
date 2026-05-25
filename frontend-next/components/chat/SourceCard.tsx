"use client";

// components/chat/SourceCard.tsx
// Collapsible source citations panel shown after the assistant response.

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileText, ChevronDown, ChevronUp } from "lucide-react";
import { SourceInfo } from "@/lib/api";

interface SourceCardProps {
  source: SourceInfo;
  index: number;
}

export function SourceCard({ source, index }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="glass rounded-xl overflow-hidden"
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/3 transition-colors text-left"
        aria-expanded={expanded}
      >
        {/* Source number badge */}
        <span className="w-5 h-5 rounded-full bg-accent-primary/20 border border-accent-primary/30 flex items-center justify-center text-[10px] font-bold text-accent-primary flex-shrink-0">
          {source.source_id ?? index + 1}
        </span>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <FileText size={12} className="text-accent-primary flex-shrink-0" />
            <span className="text-xs text-text-primary font-medium truncate">
              {source.filename ?? "Unknown file"}
            </span>
          </div>
          <div className="flex items-center gap-3 mt-0.5">
            {source.page && (
              <span className="text-[10px] text-text-muted">
                Page {source.page}
              </span>
            )}
            {source.chunk_type && (
              <span className="text-[10px] text-text-muted capitalize">
                {source.chunk_type}
              </span>
            )}
          </div>
        </div>

        <motion.div
          animate={{ rotate: expanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="text-text-muted flex-shrink-0"
        >
          <ChevronDown size={14} />
        </motion.div>
      </button>

      <AnimatePresence initial={false}>
        {expanded && source.content && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-3 pt-0">
              <div className="border-t border-white/5 pt-3">
                <p className="text-xs text-text-secondary leading-relaxed italic">
                  "{source.content}"
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
