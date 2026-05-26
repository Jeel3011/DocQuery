"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { FileText, Search } from "lucide-react";

export function HeroChatPreview() {
  const [step, setStep] = useState(0);
  const cancelledRef = useRef(false);

  // Auto-playing sequence with proper cleanup
  useEffect(() => {
    cancelledRef.current = false;

    const wait = (ms: number) =>
      new Promise<void>((resolve) => {
        const id = setTimeout(() => {
          if (!cancelledRef.current) resolve();
        }, ms);
        // Store timeout for cleanup
        return () => clearTimeout(id);
      });

    const sequence = async () => {
      while (!cancelledRef.current) {
        setStep(0);
        await wait(1000);
        if (cancelledRef.current) break;
        setStep(1);
        await wait(1500);
        if (cancelledRef.current) break;
        setStep(2);
        await wait(1500);
        if (cancelledRef.current) break;
        setStep(3);
        await wait(6000);
      }
    };
    sequence();

    return () => {
      cancelledRef.current = true;
    };
  }, []);

  return (
    <div className="card w-full max-w-lg p-6 flex flex-col gap-4 relative overflow-hidden shadow-lg">
      {/* Decorative Top Bar */}
      <div className="flex items-center justify-between border-b border-[var(--border)] pb-4 mb-2">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[var(--status-failed)] opacity-60" />
          <div className="w-3 h-3 rounded-full bg-[var(--status-processing)] opacity-60" />
          <div className="w-3 h-3 rounded-full bg-[var(--status-ready)] opacity-60" />
        </div>
        <div className="flex items-center gap-2 px-3 py-1 bg-[var(--bg-hover)] rounded-full text-xs text-[var(--text-secondary)]">
          <FileText size={12} className="text-[var(--text-muted)]" />
          attention_paper.pdf
        </div>
      </div>

      {/* User Message */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: step >= 1 ? 1 : 0, y: step >= 1 ? 0 : 10 }}
        className="self-end max-w-[85%]"
      >
        <div className="bg-[var(--accent)] text-[var(--text-inverse)] px-4 py-3 rounded-2xl rounded-tr-sm text-sm">
          What is the core idea behind the Transformer architecture?
        </div>
      </motion.div>

      {/* AI Searching State */}
      {step === 2 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="self-start flex items-center gap-2 text-xs text-[var(--text-muted)] bg-[var(--bg-hover)] px-3 py-1.5 rounded-full"
        >
          <Search size={12} className="animate-pulse text-[var(--text-secondary)]" />
          Searching 187 chunks...
        </motion.div>
      )}

      {/* AI Answer State */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: step >= 3 ? 1 : 0, y: step >= 3 ? 0 : 10 }}
        className="self-start max-w-[95%] flex flex-col gap-3"
      >
        <div className="bg-[var(--bg-hover)] border border-[var(--border)] text-[var(--text-primary)] px-4 py-3 rounded-2xl rounded-tl-sm text-sm leading-relaxed">
          <span className="font-semibold text-[var(--status-ready)] mb-1 block">DocQuery</span>
          The core idea behind the Transformer is avoiding recurrence entirely. Instead, it relies solely on a mechanism called <span className="bg-[var(--bg-active)] text-[var(--text-primary)] px-1 rounded font-medium">Self-Attention</span> to draw global dependencies between input and output.
          
          {step >= 3 && (
            <motion.span
              initial={{ opacity: 1 }}
              animate={{ opacity: [1, 0, 1] }}
              transition={{ repeat: Infinity, duration: 0.8 }}
              className="inline-block w-1.5 h-3 bg-[var(--accent)] ml-1 translate-y-0.5"
            />
          )}
        </div>

        {/* Source Citation */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: step >= 3 ? 1 : 0, scale: step >= 3 ? 1 : 0.95 }}
          transition={{ delay: 1 }}
          className="card-dotted p-3 flex items-start gap-3"
        >
          <div className="bg-[var(--bg-active)] p-2 rounded-lg">
            <FileText size={16} className="text-[var(--text-secondary)]" />
          </div>
          <div>
            <div className="text-xs font-medium text-[var(--text-primary)] mb-0.5">attention_paper.pdf</div>
            <div className="text-[10px] text-[var(--text-muted)]">Page 1 · Abstract</div>
            <div className="text-xs text-[var(--text-secondary)] mt-1.5 italic line-clamp-2">
              &quot;We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies...&quot;
            </div>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
