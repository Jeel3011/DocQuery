# Product

## Register

product

> Dual-surface project. The default register is **product** (the `/app/*` chat + Trust UI shell, where design serves the workflow). The marketing landing at `/` is a **brand** surface and should be treated as `brand` per-task when worked on directly. When a task names the landing, override to `brand`; everything else defaults to `product`.

## Users

Legal and compliance professionals, primarily in India: lawyers, in-house compliance teams, company secretaries, and chartered accountants reviewing contracts, regulations, and filings (DPDP, SEBI, ICAI, BCI context). They arrive with a pile of documents and a high-stakes question. Their context is adversarial-careful: a wrong answer asserted confidently is worse than no answer. They need to *trust* what they read, which means seeing where every claim came from and whether it was verified, not just getting a fluent paragraph.

## Product Purpose

DocQuery answers hard questions across collections of documents and returns cited, verified answers. A map-reduce "Brain" reads every relevant document, checks each claim against its source, drops what it can't prove, and synthesizes one grounded answer. Success is not "fast chat" — it is a professional closing the loop on a question they could defend to a regulator or a court, with the receipts visible. The Trust UI (streaming Routing → Reading → Verifying → Synthesizing, confidence bar, coverage badge, clickable citation chips with quoted source spans) is the product's reason to exist, not decoration.

## Brand Personality

Precise, trustworthy, calm. A precision instrument, not a chatbot. Quiet confidence: the interface never shouts, never over-animates, never asks to be admired. Trust is earned through citations, verification, and restraint rather than asserted through copy. Monochrome (black `#0A0A0A` is the only accent) with a warm-neutral aurora that stays in the background. Motion is crisp and exponential, never bouncy or playful — the stakes are serious.

## Anti-references

- **Generic ChatGPT clone.** No undifferentiated chat-bubble-in-a-box. The Trust UI — citations, claim verification, coverage ledger — must visibly be the differentiator. If a screen looks like any LLM wrapper, it has failed.
- **Cluttered enterprise SaaS.** No dense toolbars, no hero-metric template (big number + small label + gradient accent), no identical icon-heading-text card grids, no busy dashboards. Hold the precision-instrument restraint.
- **Flashy AI-startup gradient.** No purple/blue gradients, no glow-everything, no glassmorphism-as-default. Monochrome with purposeful, sparing glass only (the aurora is the one sanctioned exception, kept quiet on working surfaces).
- **Playful / consumer-cute.** No mascots, no emoji-led UI, no bouncy springs, no round-everything. Motion stays crisp; radii stay restrained (cards 12–16px, never 24px+).

## Design Principles

1. **Show the receipts, don't assert trust.** Every answer surfaces its sources, its verification state, and its coverage. The UI's job is to make grounding legible, not to sound confident.
2. **Restraint is the brand.** Monochrome, one accent, quiet motion. When in doubt, remove. The precision-instrument feel comes from what's absent.
3. **Calm under high stakes.** The user is making a defensible decision. Nothing in the interface should rush, distract, or over-celebrate. Feedback is immediate but understated.
4. **Differentiate on the Trust UI, not the chat.** Effort goes into citations, the thinking stream, coverage, and verification states — the parts no LLM-wrapper has — not into making bubbles prettier.
5. **Every pressable feels heard.** Uniform, immediate press feedback (`active:scale-[0.97]`), explicit transitions, never `transition-all`. The craft is in the invisible consistency.

## Accessibility & Inclusion

Target **WCAG 2.1 AA**. Body text ≥4.5:1 contrast against its background (large/bold text ≥3:1), including placeholder text — never light gray "for elegance." Full keyboard navigation with visible `focus-visible` rings on every interactive element. Reduced motion is not optional: every animation has a `@media (prefers-reduced-motion: reduce)` alternative (crossfade or instant), keeping opacity/color cues that aid comprehension while removing movement. Status is never conveyed by color alone (the thinking stream pairs color with icons and labels).
