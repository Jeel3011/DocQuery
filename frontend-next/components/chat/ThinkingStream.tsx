"use client";

import { useEffect, useState, useRef } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Check, Loader2, AlertCircle, Clock } from "lucide-react";

export type StepStatus = "pending" | "active" | "done" | "failed";

export interface ThinkingStep {
  id: string;
  label: string;
  detail?: string;
  status: StepStatus;
  durationMs?: number;
}

interface ThinkingStreamProps {
  steps: ThinkingStep[];
  totalMs?: number;
  collapsed?: boolean;
}

const statusIcon: Record<StepStatus, React.ReactNode> = {
  pending: <div className="w-3 h-3 rounded-full border-2 border-[var(--step-pending)]" />,
  active: <Loader2 size={12} className="text-[var(--step-active)] animate-spin" />,
  done: <Check size={11} className="text-[var(--step-done)]" strokeWidth={2.5} />,
  failed: <AlertCircle size={11} className="text-[var(--step-failed)]" />,
};

const statusLineColor: Record<StepStatus, string> = {
  pending: "bg-[var(--bg-active)]",
  active: "bg-[var(--step-active)]",
  done: "bg-[var(--step-done)]",
  failed: "bg-[var(--step-failed)]",
};

export function ThinkingStream({ steps, totalMs, collapsed = false }: ThinkingStreamProps) {
  const shouldReduceMotion = useReducedMotion();
  const allDone = steps.every((s) => s.status === "done" || s.status === "failed");

  if (collapsed || allDone) {
    const doneCount = steps.filter((s) => s.status === "done").length;
    const secs = totalMs != null ? (totalMs / 1000).toFixed(1) : null;
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[var(--bg-hover)] border border-[var(--border)] text-[10px] text-[var(--text-muted)]"
      >
        <Check size={9} className="text-[var(--step-done)]" strokeWidth={2.5} />
        <span>
          Reasoned over {doneCount} step{doneCount !== 1 ? "s" : ""}
          {secs ? ` · ${secs}s` : ""}
        </span>
      </motion.div>
    );
  }

  return (
    <div className="space-y-1.5">
      {steps.map((step, i) => (
        <motion.div
          key={step.id}
          initial={{ opacity: 0, x: shouldReduceMotion ? 0 : -6 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: shouldReduceMotion ? 0 : i * 0.06, duration: shouldReduceMotion ? 0.01 : 0.18, ease: [0.23, 1, 0.32, 1] }}
          className="flex items-start gap-2.5"
        >
          {/* Connector line + icon */}
          <div className="flex flex-col items-center flex-shrink-0 mt-0.5">
            <div className={`w-5 h-5 rounded-full flex items-center justify-center border ${
              step.status === "active"
                ? "border-[var(--step-active)] bg-white"
                : step.status === "done"
                ? "border-[var(--step-done)] bg-green-50"
                : step.status === "failed"
                ? "border-[var(--step-failed)] bg-red-50"
                : "border-[var(--border)] bg-[var(--bg-hover)]"
            }`}>
              {statusIcon[step.status]}
            </div>
            {i < steps.length - 1 && (
              <div className={`w-px flex-1 mt-1 min-h-[12px] transition-colors duration-500 ${
                step.status === "done" ? statusLineColor.done : statusLineColor.pending
              }`} />
            )}
          </div>

          {/* Text */}
          <div className={`pb-2 flex-1 min-w-0 ${step.status === "active" ? "opacity-100" : step.status === "pending" ? "opacity-40" : "opacity-80"}`}>
            <p className={`text-xs font-medium ${step.status === "active" ? "text-[var(--text-primary)]" : "text-[var(--text-secondary)]"}`}>
              {step.label}
            </p>
            {step.detail && step.status !== "pending" && (
              <p className="text-[10px] text-[var(--text-muted)] mt-0.5">{step.detail}</p>
            )}
            {step.status === "done" && step.durationMs != null && (
              <span className="inline-flex items-center gap-1 text-[9px] text-[var(--text-muted)] mt-0.5">
                <Clock size={8} />
                {(step.durationMs / 1000).toFixed(1)}s
              </span>
            )}
          </div>
        </motion.div>
      ))}
    </div>
  );
}

// ── Fixture player: simulates the thinking steps in demo mode ──
const FIXTURE_STEPS: Omit<ThinkingStep, "status">[] = [
  { id: "route", label: "Routing", detail: "Analyzing query intent" },
  { id: "read", label: "Reading documents", detail: "Selected 14 of 34 docs" },
  { id: "synth", label: "Synthesizing", detail: "Merging passages across 14 sources" },
  { id: "verify", label: "Verifying", detail: "Cross-checking 9 claims" },
];

const STEP_DURATIONS = [400, 1800, 1200, 800];

export function ThinkingStreamFixture({ onDone }: { onDone?: (totalMs: number) => void }) {
  const [steps, setSteps] = useState<ThinkingStep[]>(
    FIXTURE_STEPS.map((s) => ({ ...s, status: "pending" }))
  );
  const [done, setDone] = useState(false);
  const [totalMs, setTotalMs] = useState(0);
  const startRef = useRef(Date.now());
  const stepTimings = useRef<number[]>([]);

  useEffect(() => {
    let elapsed = 0;
    STEP_DURATIONS.forEach((dur, i) => {
      // Activate step
      setTimeout(() => {
        setSteps((prev) =>
          prev.map((s, si) => (si === i ? { ...s, status: "active" } : s))
        );
        stepTimings.current[i] = Date.now();
      }, elapsed);
      elapsed += dur;
      // Complete step
      setTimeout(() => {
        const stepDur = Date.now() - stepTimings.current[i];
        setSteps((prev) =>
          prev.map((s, si) => (si === i ? { ...s, status: "done", durationMs: stepDur } : s))
        );
        if (i === FIXTURE_STEPS.length - 1) {
          const total = Date.now() - startRef.current;
          setTotalMs(total);
          setDone(true);
          onDone?.(total);
        }
      }, elapsed);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <ThinkingStream steps={steps} totalMs={done ? totalMs : undefined} collapsed={done} />;
}
