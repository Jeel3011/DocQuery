"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Table2, MessageSquare, Workflow as WorkflowIcon, Brain, ArrowRight } from "lucide-react";
import { Reveal } from "@/components/landing/Reveal";
import { PlatformPageShell, PlatformCTA, ease } from "@/components/landing/PlatformPageShell";

const PILLARS = [
  {
    href: "/platform/brain",
    icon: Brain,
    title: "Brain",
    description: "Route → Read → Synthesise → Verify. Every run.",
  },
  {
    href: "/platform/vault",
    icon: Table2,
    title: "Vault",
    description: "Every document verified on the way in.",
  },
  {
    href: "/platform/agent",
    icon: MessageSquare,
    title: "Agent",
    description: "Cites every figure, or abstains.",
  },
  {
    href: "/platform/workflows",
    icon: WorkflowIcon,
    title: "Workflows",
    description: "Repeatable playbooks, cited end to end.",
  },
];

export default function PlatformOverviewPage() {
  return (
    <PlatformPageShell>
      <section
        className="relative overflow-hidden"
        style={{ paddingTop: "clamp(140px, 16vw, 200px)", paddingBottom: "clamp(70px, 9vw, 110px)" }}
      >
        <div className="section-container">
          <Reveal>
            <span className="tag">Platform</span>
          </Reveal>
          <Reveal delay={0.06}>
            <h1
              className="font-display font-light mt-6"
              style={{
                fontSize: "clamp(44px, 7vw, 84px)",
                lineHeight: "1.0",
                letterSpacing: "-0.03em",
                color: "var(--ink)",
                maxWidth: "16ch",
              }}
            >
              One system. Every claim traced.
            </h1>
          </Reveal>
        </div>
      </section>

      <section className="relative z-10" style={{ paddingBottom: "clamp(100px, 12vw, 160px)" }}>
        <div className="section-container">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4" style={{ borderTop: "1px solid var(--line)" }}>
            {PILLARS.map((p, i) => (
              <motion.div
                key={p.href}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "0px 0px -10% 0px" }}
                transition={{ duration: 0.5, delay: i * 0.06, ease }}
                style={{
                  borderRight: "1px solid var(--line)",
                  borderBottom: "1px solid var(--line)",
                }}
              >
                <Link href={p.href} className="group flex flex-col h-full justify-between p-10 transition-colors" style={{ minHeight: 260 }}>
                  <div
                    className="w-12 h-12 rounded-xl flex items-center justify-center"
                    style={{ background: "var(--ink)", color: "var(--on-ink)" }}
                  >
                    <p.icon size={20} strokeWidth={1.6} />
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <h2 className="font-display" style={{ fontSize: "30px", fontWeight: 500, letterSpacing: "-0.02em", color: "var(--ink)" }}>
                        {p.title}
                      </h2>
                      <ArrowRight size={16} className="opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" style={{ color: "var(--ink-3)" }} />
                    </div>
                    <p style={{ fontSize: "15px", color: "var(--ink-3)" }}>{p.description}</p>
                  </div>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <PlatformCTA />
    </PlatformPageShell>
  );
}
