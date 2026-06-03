"use client";

import { Shield, BookOpen, Lock, Server } from "lucide-react";

const TRUST_ITEMS = [
  {
    icon: Lock,
    title: "No training on your data",
    desc: "Your documents never leave to train any model. Queries are inference-only.",
  },
  {
    icon: BookOpen,
    title: "Every answer is cited",
    desc: "Citations link to exact passages in your corpus — no hallucinated facts.",
  },
  {
    icon: Shield,
    title: "Audit trail built in",
    desc: "Every query, step, and source is logged for compliance and review.",
  },
  {
    icon: Server,
    title: "Your infrastructure",
    desc: "Deploy on your own cloud or on-prem. Data residency guaranteed.",
  },
];

export function TrustStrip() {
  return (
    <section className="py-16 px-6 lg:px-12 border-y border-dashed border-[var(--border-dotted)] relative z-10"
      style={{ background: "rgba(255,255,255,0.35)", backdropFilter: "blur(12px) saturate(1.4)", WebkitBackdropFilter: "blur(12px) saturate(1.4)" }}
    >
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-10">
          <p className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-[0.16em] mb-2">Security & Trust</p>
          <h2 className="text-2xl font-bold text-[var(--text-primary)] tracking-tight">
            Built for regulated industries
          </h2>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
          {TRUST_ITEMS.map((item, i) => {
            const Icon = item.icon;
            return (
              <div key={i} className="flex flex-col items-start gap-3">
                <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0" style={{ background: "linear-gradient(165deg,#FFFFFF,#F1EEE9)", border: "1px solid rgba(0,0,0,0.06)", boxShadow: "var(--skeu-raised)" }}>
                  <Icon size={16} className="text-[var(--text-secondary)]" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-[var(--text-primary)] mb-0.5">{item.title}</p>
                  <p className="text-[11px] text-[var(--text-muted)] leading-relaxed">{item.desc}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
