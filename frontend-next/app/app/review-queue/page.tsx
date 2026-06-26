"use client";

// app/app/review-queue/page.tsx — F2g surface 6: My Review Queue.
// The work that currently waits on ME. Every firm member can hold a queue (D0 — the chain flows up
// through everyone), so this page is NOT cap-gated to managers; the queue itself is server-filtered
// to requests I own. Each item shows the submitter, the artifact, the owner by name (anti-stall),
// and exactly one next action (Approve / Request changes / Release externally).

import { useRouter } from "next/navigation";
import { motion, useReducedMotion } from "framer-motion";
import { ArrowLeft, Inbox } from "lucide-react";
import { ReviewQueue } from "@/components/app/ReviewChain";

const ease = [0.16, 1, 0.3, 1] as const;

export default function ReviewQueuePage() {
  const router = useRouter();
  const reduce = useReducedMotion();
  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin" style={{ background: "var(--canvas)" }}>
      <div className="max-w-3xl mx-auto px-4 md:px-8 py-8">
        <motion.div
          initial={{ opacity: reduce ? 1 : 0, y: reduce ? 0 : 8 }}
          animate={{ opacity: 1, y: 0 }} transition={{ ease }}
          className="flex items-center gap-3 mb-6">
          <button onClick={() => router.push("/app")}
            className="p-1.5 rounded-lg hover:bg-[var(--bg-hover)] transition-colors" style={{ color: "var(--ink-3)" }}>
            <ArrowLeft size={18} />
          </button>
          <div className="flex-1">
            <h1 style={{ fontFamily: "Fraunces, Georgia, serif", fontSize: "24px", fontWeight: 400, letterSpacing: "-0.025em", color: "var(--ink)", lineHeight: 1.1, textWrap: "balance" }}>
              My review queue
            </h1>
            <p className="text-xs inline-flex items-center gap-1.5" style={{ color: "var(--ink-3)" }}>
              <Inbox size={12} /> Work that is waiting on you
            </p>
          </div>
        </motion.div>
        <ReviewQueue />
      </div>
    </div>
  );
}
