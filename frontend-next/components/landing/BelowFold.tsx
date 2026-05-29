"use client";

// C4: below-the-fold landing sections, code-split and client-loaded so their
// JS (and the framer-motion animation work they do) doesn't block first paint.
// HeroSection stays eagerly server-rendered for LCP/SEO.
import dynamic from "next/dynamic";

const FeaturesBento = dynamic(
  () => import("./FeaturesBento").then((m) => m.FeaturesBento),
  { ssr: false }
);
const HowItWorks = dynamic(
  () => import("./HowItWorks").then((m) => m.HowItWorks),
  { ssr: false }
);
const RagasMetrics = dynamic(
  () => import("./RagasMetrics").then((m) => m.RagasMetrics),
  { ssr: false }
);

export function BelowFold() {
  return (
    <>
      <FeaturesBento />
      <HowItWorks />
      <RagasMetrics />
    </>
  );
}
