"use client";

// app/app/settings/firm/_shared.tsx — F2g shared vocabulary for the Firm Console.
// Role / capability LABELS and ranks. These are display strings + ordering only — they are NOT
// the authorization source of truth (that is the server's caps payload, surface 10). Keeping the
// labels here keeps every console surface reading the SAME human vocabulary (DESIGN.md: verb+object
// labels, no buzzwords, no em-dashes). The role ORDER mirrors schemas.py ROLE_RANK exactly.

import type { Capability, FirmRole } from "@/lib/api";

// Mirror of schemas.py ROLES / ROLE_RANK (lower index = more senior). Used to surface the T1 guard
// ("you cannot pick a role at or above your own") as a reasoned disabled control — never as the
// security itself (the server enforces it).
export const ROLE_ORDER: FirmRole[] = [
  "managing_partner", "senior_partner", "partner", "senior_associate",
  "associate", "paralegal", "assistant", "client", "guest",
];

export const ROLE_RANK: Record<FirmRole, number> = Object.fromEntries(
  ROLE_ORDER.map((r, i) => [r, i])
) as Record<FirmRole, number>;

export const ROLE_LABEL: Record<FirmRole, string> = {
  managing_partner: "Managing Partner",
  senior_partner: "Senior Partner",
  partner: "Partner",
  senior_associate: "Senior Associate",
  associate: "Associate",
  paralegal: "Paralegal",
  assistant: "Assistant",
  client: "Client",
  guest: "Guest",
};

// The roles a firm assigns INTERNALLY (external client/guest are scoped per-shared-vault, never via
// the console invite flow). Onboarding + promote/demote pick from these.
export const INTERNAL_ROLES: FirmRole[] = ROLE_ORDER.filter(
  (r) => r !== "client" && r !== "guest"
);

export const EXTERNAL_ROLES: FirmRole[] = ["client", "guest"];

// Human, verb+object capability labels (for the access matrix + the delegation verb picker).
// Order matters for the matrix columns: working toolkit first, then the held/outbound gates last.
export const CAP_LABEL: Record<Capability, string> = {
  ask: "Ask the agent",
  draft: "Draft & redline",
  grids: "Build grids",
  run_workflow: "Run workflows",
  ingest: "Add documents",
  create_vault: "Create matters",
  send_for_review: "Send for review",
  manage_matter_team: "Staff a matter",
  release_external: "Release externally",
  override_abstain: "Override an abstain",
  delete: "Delete",
  sign_certificate: "Sign certificate",
  manage_members: "Manage members",
  edit_playbooks: "Edit playbooks",
  publish_to_firm_brain: "Publish to firm brain",
  run_sentinel: "Run sentinel",
  view_billing: "View billing",
};

// The columns the Matters & Access matrix shows, in display order (the working toolkit a staffed
// member ENJOYS first — D0 positive framing — then the few held outbound/admin gates).
export const MATRIX_VERBS: Capability[] = [
  "ask", "draft", "grids", "run_workflow", "ingest", "send_for_review",
  "manage_matter_team", "release_external", "override_abstain", "manage_members",
];

// The full default capability matrix, MIRRORING authz.py ROLE_CAPS exactly. This drives the
// READ-ONLY "Matters & Access" matrix (a transparency view of the firm's default policy). It is
// NOT used to authorize anything — every action is server-gated, and the viewer's OWN caps come
// from the server (surface 10). A drift here would only mis-PAINT the reference matrix, never grant
// access; the gate asserts it stays lockstep with authz.py.
const WORKING_TOOLKIT: Capability[] = [
  "create_vault", "ingest", "ask", "draft", "run_workflow", "grids", "send_for_review",
];
const FIRM_GOVERNANCE: Capability[] = [
  "manage_members", "view_billing", "edit_playbooks", "publish_to_firm_brain",
];
const PARTNER_ELEVATED: Capability[] = [
  "release_external", "manage_matter_team", "delete", "sign_certificate",
  "override_abstain", "run_sentinel",
];

export const ROLE_CAPS: Record<FirmRole, Set<Capability>> = {
  managing_partner: new Set([...WORKING_TOOLKIT, ...FIRM_GOVERNANCE, ...PARTNER_ELEVATED]),
  senior_partner: new Set([...WORKING_TOOLKIT, ...FIRM_GOVERNANCE, ...PARTNER_ELEVATED]),
  partner: new Set([...WORKING_TOOLKIT, ...FIRM_GOVERNANCE, ...PARTNER_ELEVATED]),
  senior_associate: new Set<Capability>([
    ...WORKING_TOOLKIT, "release_external", "manage_matter_team", "delete",
  ]),
  associate: new Set(WORKING_TOOLKIT),
  paralegal: new Set(WORKING_TOOLKIT),
  assistant: new Set(WORKING_TOOLKIT),
  client: new Set<Capability>(["ask"]),
  guest: new Set<Capability>([]),
};

export function shortId(id: string | null | undefined): string {
  if (!id) return "—";
  return id.length > 10 ? `${id.slice(0, 8)}…` : id;
}

// A member's display handle: prefer email, fall back to a short user-id. Used everywhere a person
// is named (the "by name" anti-stall requirement — chain preview, owner labels, remove consequence).
export function memberLabel(
  m: { email?: string | null; user_id?: string | null } | null | undefined
): string {
  if (!m) return "—";
  if (m.email) return m.email;
  return shortId(m.user_id);
}
