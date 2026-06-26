// lib/firmSetup.ts — F2g onboarding glue between the auth screen and the firm.
//
// The signup form collects an optional firm NAME (create a new firm) or an invite TOKEN (join an
// existing one). But Supabase signup may not return a session immediately (email confirmation), so
// we can't always call the authenticated firm endpoints right away. The robust flow:
//   • If we already have a token → apply the intent now (rename the backfilled solo firm, or accept
//     the invite).
//   • Otherwise → STASH the intent in localStorage; the app layout applies it on the first
//     authenticated load (consumeStashedFirmSetup), then clears it. One-shot, best-effort.
//
// The backfill always gives a new user a solo firm named "<local>'s Firm"; naming at signup just
// renames that firm to the chosen name. Joining by invite re-points the membership to the inviter's
// firm. Neither is load-bearing for security — the server enforces roles + the invite (T4); this is
// pure onboarding convenience.

import { acceptInvite, renameFirm, getFirm } from "@/lib/api";

const KEY = "docquery-firm-setup";

export interface FirmSetupIntent {
  firmName?: string;
  inviteToken?: string;
}

export function stashFirmSetup(intent: FirmSetupIntent) {
  if (typeof window === "undefined") return;
  if (!intent.firmName && !intent.inviteToken) return;
  try { window.localStorage.setItem(KEY, JSON.stringify(intent)); } catch { /* ignore */ }
}

function clearStash() {
  if (typeof window === "undefined") return;
  try { window.localStorage.removeItem(KEY); } catch { /* ignore */ }
}

/** Read the stashed intent WITHOUT consuming it (the login page uses this to pre-fill the
 *  "Join with invite" field after a /invite redirect). Returns null if none. */
export function peekStashedFirmSetup(): FirmSetupIntent | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as FirmSetupIntent) : null;
  } catch {
    return null;
  }
}

/** Apply a firm-setup intent against a live session. Returns true if it did something.
 *
 *  SECURITY (fail-closed): an invite-JOIN that fails THROWS — a bad/expired/wrong-email token must
 *  NEVER be swallowed and silently fall through to a solo firm (the exact defect that let a wrong
 *  token mint a new firm). The caller surfaces that error to the user. A firm-RENAME failure is
 *  cosmetic and stays best-effort (it never affects access). */
export async function setupFirm(token: string, intent: FirmSetupIntent): Promise<boolean> {
  if (!token) return false;
  if (intent.inviteToken) {
    // No try/catch here on purpose — a rejected invite must propagate (fail closed).
    await acceptInvite(token, intent.inviteToken);
    return true;
  }
  if (intent.firmName && intent.firmName.trim()) {
    try {
      // The backfill already created a solo firm; rename it to the chosen name (cosmetic).
      const firm = await getFirm(token);
      if (firm?.id) {
        await renameFirm(token, firm.id, intent.firmName.trim());
        return true;
      }
    } catch {
      /* rename is convenience only — never blocks the user */
    }
  }
  return false;
}

/** Consume a stashed intent (after an email-confirm signup) on the first authenticated load.
 *  Returns {applied, error}: a failed invite-join surfaces `error` (the app layer toasts it) so a
 *  bad token is never silently dropped. A rename failure is swallowed inside setupFirm (cosmetic). */
export async function consumeStashedFirmSetup(
  token: string
): Promise<{ applied: boolean; error: string | null }> {
  if (typeof window === "undefined" || !token) return { applied: false, error: null };
  let raw: string | null = null;
  try { raw = window.localStorage.getItem(KEY); } catch { return { applied: false, error: null }; }
  if (!raw) return { applied: false, error: null };
  clearStash();
  let intent: FirmSetupIntent;
  try { intent = JSON.parse(raw) as FirmSetupIntent; }
  catch { return { applied: false, error: null }; }
  try {
    const applied = await setupFirm(token, intent);
    return { applied, error: null };
  } catch (e) {
    // The invite-join was rejected (bad/expired/wrong-email). Surface it — do NOT fall through.
    const msg = (e as { detail?: string; message?: string })?.detail
      ?? (e as { message?: string })?.message
      ?? "That invite could not be accepted. It may have expired or be addressed to a different email.";
    return { applied: false, error: msg };
  }
}
