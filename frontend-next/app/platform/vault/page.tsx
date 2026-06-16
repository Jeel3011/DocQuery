"use client";

import { motion } from "framer-motion";
import { FileText, CheckCircle2, Shield, Filter, Tag, Loader2 } from "lucide-react";
import { Reveal } from "@/components/landing/Reveal";
import { PlatformPageShell, PlatformHero, PlatformCTA, ease } from "@/components/landing/PlatformPageShell";

const PIPELINE = ["Parse", "Chunk", "Embed", "Ready"];

const DOCS = [
  { name: "MSA_Sezzle_v3.pdf", type: "Contract", fidelity: "good", size: "1.2 MB", status: "ready" },
  { name: "10-K_2023_Microsoft.pdf", type: "Filing", fidelity: "good", size: "8.4 MB", status: "ready" },
  { name: "NDA_redline_draft.docx", type: "Contract", fidelity: "partial", size: "340 KB", status: "ready" },
  { name: "board_pack_q3.pdf", type: "Mixed", fidelity: "good", size: "2.9 MB", status: "processing" },
];

const FEATURES = [
  {
    title: "Live fidelity pipeline",
    description: "Every upload walks through four real, visible stages — Parse, Chunk, Embed, Ready. A failed document highlights exactly which stage it died on, instead of disappearing into a queue.",
  },
  {
    title: "Structural document classification",
    description: "Documents are classified financial_filing, legal_contract, mixed, or generic — by clause-heading density and $-token frequency, not an LLM call. Zero cost, deterministic, and it gates which extraction path runs (table-grid vs. clause-aware chunking).",
  },
  {
    title: "Metadata-driven filtering",
    description: "Filters populate only from documents actually in the vault — no dead chips. Filter by document type or fiscal year, and the same filter scopes what the Agent and Review grid retrieve.",
  },
  {
    title: "Fidelity trust dot",
    description: "Green, amber, or hollow — a self-reported per-document extraction-completeness score, computed by cross-checking extracted grids against the PDF's own text layer at ingestion time. Never blocks upload; always visible.",
  },
];

function FidelityDot({ level }: { level: "good" | "partial" | "hollow" }) {
  const color = level === "good" ? "var(--fidelity-good)" : level === "partial" ? "var(--fidelity-partial)" : "var(--ink-3)";
  return <span className="inline-block w-2 h-2 rounded-full shrink-0" style={{ background: level === "hollow" ? "transparent" : color, border: level === "hollow" ? `1.5px solid ${color}` : "none" }} />;
}

function VaultDemo() {
  return (
    <Reveal delay={0.2}>
      <div
        className="rounded-[22px] overflow-hidden mt-4"
        style={{
          background: "rgba(255,255,255,0.72)",
          backdropFilter: "blur(22px) saturate(1.0)",
          WebkitBackdropFilter: "blur(22px) saturate(1.0)",
          border: "1px solid rgba(255,255,255,0.80)",
          boxShadow: "var(--shadow-xl), inset 0 1px 0 rgba(255,255,255,0.90)",
        }}
      >
        <div
          className="flex items-center gap-3 px-5 py-3.5"
          style={{
            background: "linear-gradient(180deg, rgba(255,255,255,0.90) 0%, rgba(255,255,255,0.55) 100%)",
            borderBottom: "1px solid var(--glass-border)",
          }}
        >
          <div className="flex gap-1.5">
            {["#C4C4C4", "#A8A8A8", "#8C8C8C"].map((c, i) => (
              <div key={i} className="w-3 h-3 rounded-full opacity-90" style={{ background: c }} />
            ))}
          </div>
          <div className="flex-1 mx-6 flex items-center justify-center gap-2 px-4 py-1.5 rounded-lg" style={{ background: "rgba(244,244,244,0.70)", border: "1px solid var(--line)" }}>
            <Shield size={10} className="opacity-40" />
            <span className="text-[11px] font-medium" style={{ color: "var(--ink-3)" }}>Vault — Acquisition Diligence</span>
          </div>
          <div className="flex items-center gap-1.5 text-[11px]" style={{ color: "var(--ink-3)" }}>
            <Filter size={12} /> Contract <span>·</span> FY2023
          </div>
        </div>

        <div className="p-5">
          {/* Processing doc — pipeline track */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, ease }}
            className="rounded-xl p-4 mb-3"
            style={{ background: "var(--surface-2)", border: "1px solid var(--line)" }}
          >
            <div className="flex items-center gap-2 mb-3">
              <Loader2 size={13} className="animate-spin" style={{ color: "var(--ink-3)" }} />
              <span className="text-[13px] font-medium" style={{ color: "var(--ink)" }}>board_pack_q3.pdf</span>
            </div>
            <div className="flex items-center gap-0">
              {PIPELINE.map((stage, i) => (
                <div key={stage} className="flex items-center flex-1">
                  <div className="flex flex-col items-center gap-1.5">
                    <div
                      className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-semibold"
                      style={{
                        background: i <= 2 ? "var(--ink)" : "var(--surface-3)",
                        color: i <= 2 ? "var(--on-ink)" : "var(--ink-3)",
                        border: i === 2 ? "2px solid var(--ink)" : "none",
                      }}
                    >
                      {i < 2 ? <CheckCircle2 size={12} /> : i + 1}
                    </div>
                    <span className="text-[10px]" style={{ color: i <= 2 ? "var(--ink-2)" : "var(--ink-3)" }}>{stage}</span>
                  </div>
                  {i < PIPELINE.length - 1 && (
                    <div className="flex-1 h-[2px] mx-1 -mt-4" style={{ background: i < 2 ? "var(--ink)" : "var(--line)" }} />
                  )}
                </div>
              ))}
            </div>
          </motion.div>

          {/* Ready docs table */}
          <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--line)" }}>
            <table className="w-full text-left text-[12.5px]">
              <thead>
                <tr className="text-[10px] uppercase tracking-wide" style={{ color: "var(--ink-3)" }}>
                  <th className="px-3 py-2 font-medium">Document</th>
                  <th className="px-3 py-2 font-medium">Type</th>
                  <th className="px-3 py-2 font-medium">Fidelity</th>
                  <th className="px-3 py-2 font-medium">Size</th>
                </tr>
              </thead>
              <tbody>
                {DOCS.filter((d) => d.status === "ready").map((d, i) => (
                  <motion.tr
                    key={d.name}
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.1 + i * 0.05, duration: 0.4 }}
                    className="border-t"
                    style={{ borderColor: "var(--line)" }}
                  >
                    <td className="px-3 py-2.5">
                      <span className="inline-flex items-center gap-1.5" style={{ color: "var(--ink)" }}>
                        <FileText size={12} style={{ color: "var(--ink-3)" }} />
                        {d.name}
                      </span>
                    </td>
                    <td className="px-3 py-2.5">
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px]" style={{ background: "var(--surface-3)", color: "var(--ink-2)" }}>
                        <Tag size={9} /> {d.type}
                      </span>
                    </td>
                    <td className="px-3 py-2.5">
                      <FidelityDot level={d.fidelity as "good" | "partial"} />
                    </td>
                    <td className="px-3 py-2.5" style={{ color: "var(--ink-3)" }}>{d.size}</td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Reveal>
  );
}

export default function VaultPlatformPage() {
  return (
    <PlatformPageShell>
      <PlatformHero
        breadcrumb="Platform Overview / Vault"
        title={<>Every document, <span style={{ color: "var(--accent-taupe)" }}>verified</span> on the way in.</>}
        description="Upload contracts, filings, and board packs. Watch them classify, chunk, and embed in real time — with a fidelity score that tells you when extraction is incomplete, before you ever ask a question."
      />

      <section className="relative z-10" style={{ paddingBottom: "clamp(60px, 8vw, 100px)" }}>
        <div className="section-container">
          <VaultDemo />
        </div>
      </section>

      <section className="relative z-10" style={{ paddingTop: "clamp(40px, 6vw, 70px)", paddingBottom: "clamp(80px, 10vw, 130px)" }}>
        <div className="section-container">
          <Reveal className="text-center mb-14">
            <p className="eyebrow mb-4">Why it&apos;s different</p>
            <h2 className="font-display font-light" style={{ fontSize: "clamp(30px, 4vw, 46px)", lineHeight: "1.08", letterSpacing: "-0.03em", color: "var(--ink)" }}>
              Trust starts before the first question.
            </h2>
          </Reveal>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {FEATURES.map((f, i) => (
              <Reveal key={f.title} delay={i * 0.05}>
                <div className="h-full rounded-[20px] p-7" style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-md)" }}>
                  <h3 className="font-semibold mb-2.5" style={{ color: "var(--ink)", fontSize: "16px", letterSpacing: "-0.01em" }}>{f.title}</h3>
                  <p className="leading-relaxed" style={{ color: "var(--ink-2)", fontSize: "14px" }}>{f.description}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <PlatformCTA />
    </PlatformPageShell>
  );
}
