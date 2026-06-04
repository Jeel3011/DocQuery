"use client";

// C4: below-the-fold landing sections, code-split so they don't block first paint.
import dynamic from "next/dynamic";

const FeaturesBento = dynamic(
  () => import("./FeaturesBento").then((m) => m.FeaturesBento),
  { ssr: false }
);
const HowItWorks = dynamic(
  () => import("./HowItWorks").then((m) => m.HowItWorks),
  { ssr: false }
);
const AudienceTabs = dynamic(
  () => import("./AudienceTabs").then((m) => m.AudienceTabs),
  { ssr: false }
);
const RagasMetrics = dynamic(
  () => import("./RagasMetrics").then((m) => m.RagasMetrics),
  { ssr: false }
);
const Pricing = dynamic(
  () => import("./Pricing").then((m) => m.Pricing),
  { ssr: false }
);
const TrustStrip = dynamic(
  () => import("./TrustStrip").then((m) => m.TrustStrip),
  { ssr: false }
);

export function BelowFold() {
  return (
    <>
      <FeaturesBento />
      <AudienceTabs />
      <HowItWorks />
      <RagasMetrics />
      <Pricing />
      <TrustStrip />
    </>
  );
}
