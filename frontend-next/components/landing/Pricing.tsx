"use client";

import Link from "next/link";
import { Check, ArrowRight, Mail } from "lucide-react";
import { Reveal } from "./Reveal";

const FREE_FEATURES = [
  "Up to 3 collections",
  "50 documents total",
  "Hybrid retrieval + cited answers",
  "Inline tables & citations",
  "Community support",
];

const ENTERPRISE_FEATURES = [
  "Unlimited collections & documents",
  "Multi-agent Brain on large corpora",
  "Deploy on your own VPC (AWS / GCP / on-prem)",
  "SSO, audit logs & access controls",
  "RAGAS evaluation reports",
  "Dedicated support & SLA",
];

// Where the Enterprise "Contact us" CTA points. Pre-fills a sales email.
const CONTACT_HREF =
  "mailto:jeel15thummar@gmail.com?subject=DocQuery%20Enterprise%20enquiry&body=Hi%20DocQuery%20team%2C%0A%0AWe%27d%20like%20to%20talk%20about%20an%20Enterprise%20plan.%0A%0ACompany%3A%0ATeam%20size%3A%0ADocuments%2Fmonth%3A%0ADeployment%20(cloud%2Fon-prem)%3A%0A%0AThanks%21";

export function Pricing() {
  return (
    <section
      id="pricing"
      className="relative z-10 scroll-mt-24"
      style={{
        paddingTop: "clamp(80px, 10vw, 140px)",
        paddingBottom: "clamp(80px, 10vw, 140px)",
      }}
    >
      <div className="section-container">
        <Reveal className="text-center mb-14 max-w-2xl mx-auto">
          <p className="eyebrow mb-4">Pricing</p>
          <h2
            className="font-display font-light mb-5"
            style={{
              fontSize: "clamp(36px, 5vw, 58px)",
              lineHeight: "1.05",
              letterSpacing: "-0.03em",
              color: "var(--ink)",
            }}
          >
            Start free.
            <br />
            Scale on your terms.
          </h2>
          <p
            className="text-[17px] leading-relaxed"
            style={{ color: "var(--ink-2)", letterSpacing: "-0.01em" }}
          >
            A genuinely useful free tier to evaluate DocQuery. Enterprise is sold directly — so we can tailor deployment, security, and scale to your team.
          </p>
        </Reveal>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-4xl mx-auto items-stretch">
          {/* Free */}
          <Reveal>
            <div
              className="flex flex-col h-full p-8 rounded-[24px]"
              style={{
                background: "var(--surface)",
                border: "1px solid var(--line)",
                boxShadow: "var(--shadow-md)",
              }}
            >
              <p className="eyebrow mb-3">Free</p>
              <div className="flex items-end gap-2 mb-1">
                <span
                  className="font-display"
                  style={{
                    fontSize: "56px",
                    fontWeight: 300,
                    lineHeight: 1,
                    letterSpacing: "-0.04em",
                    color: "var(--ink)",
                  }}
                >
                  ₹0
                </span>
                <span className="text-[14px] mb-2" style={{ color: "var(--ink-3)" }}>
                  forever
                </span>
              </div>
              <p className="text-[14px] mb-7" style={{ color: "var(--ink-2)" }}>
                For individuals evaluating DocQuery on a small set of documents.
              </p>

              <ul className="flex flex-col gap-3 mb-8">
                {FREE_FEATURES.map((f) => (
                  <li key={f} className="flex items-start gap-2.5 text-[14px]" style={{ color: "var(--ink-2)" }}>
                    <span
                      className="w-4 h-4 rounded-full flex items-center justify-center shrink-0 mt-0.5"
                      style={{ background: "var(--surface-3)", color: "var(--ink)" }}
                    >
                      <Check size={11} strokeWidth={2.4} />
                    </span>
                    {f}
                  </li>
                ))}
              </ul>

              <Link
                href="/login"
                className="mt-auto inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-[14px] font-semibold text-[15px] transition-colors"
                style={{
                  background: "var(--surface-3)",
                  color: "var(--ink)",
                  border: "1px solid var(--line)",
                }}
              >
                Start for free
                <ArrowRight size={16} strokeWidth={2.2} />
              </Link>
            </div>
          </Reveal>

          {/* Enterprise */}
          <Reveal delay={0.08}>
            <div
              className="flex flex-col h-full p-8 rounded-[24px] relative overflow-hidden"
              style={{
                background: "var(--ink)",
                boxShadow: "var(--shadow-lg)",
              }}
            >
              <div
                aria-hidden
                className="absolute inset-0 pointer-events-none"
                style={{ background: "radial-gradient(ellipse 70% 50% at 80% 0%, rgba(255,255,255,0.08) 0%, transparent 70%)" }}
              />
              <div className="relative flex flex-col h-full">
                <div className="flex items-center justify-between mb-3">
                  <p className="eyebrow" style={{ color: "rgba(250,250,250,0.55)" }}>
                    Enterprise
                  </p>
                  <span
                    className="text-[11px] font-medium px-2.5 py-1 rounded-full"
                    style={{ background: "rgba(250,250,250,0.12)", color: "var(--on-ink)" }}
                  >
                    Sold directly
                  </span>
                </div>
                <div className="flex items-end gap-2 mb-1">
                  <span
                    className="font-display"
                    style={{
                      fontSize: "46px",
                      fontWeight: 300,
                      lineHeight: 1,
                      letterSpacing: "-0.03em",
                      color: "var(--on-ink)",
                    }}
                  >
                    Let's talk
                  </span>
                </div>
                <p className="text-[14px] mb-7" style={{ color: "rgba(250,250,250,0.65)" }}>
                  Custom pricing based on scale, deployment, and security needs. Built around your stack.
                </p>

                <ul className="flex flex-col gap-3 mb-8">
                  {ENTERPRISE_FEATURES.map((f) => (
                    <li key={f} className="flex items-start gap-2.5 text-[14px]" style={{ color: "rgba(250,250,250,0.88)" }}>
                      <span
                        className="w-4 h-4 rounded-full flex items-center justify-center shrink-0 mt-0.5"
                        style={{ background: "rgba(250,250,250,0.15)", color: "var(--on-ink)" }}
                      >
                        <Check size={11} strokeWidth={2.4} />
                      </span>
                      {f}
                    </li>
                  ))}
                </ul>

                <a
                  href={CONTACT_HREF}
                  className="mt-auto inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-[14px] font-semibold text-[15px]"
                  style={{
                    background: "var(--on-ink)",
                    color: "var(--ink)",
                    boxShadow: "0 4px 20px -4px rgba(0,0,0,0.30)",
                  }}
                >
                  <Mail size={16} strokeWidth={2.2} />
                  Contact us for Enterprise
                </a>
              </div>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
