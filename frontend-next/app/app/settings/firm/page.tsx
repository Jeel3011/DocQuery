"use client";

// app/app/settings/firm/page.tsx — THE FIRM CONSOLE (F2g).
// Makes the whole F2a–F2f engine visible, usable, trustable. Every surface here is wired to an
// already-built, server-cap-gated route; the UI never authorizes — it surfaces the server's
// decision and avoids SHOWING what will 403 (the route guard + every action are the security).
//
// Surfaces on this page (the §2.2 completeness contract):
//   1  Onboarding ........ Invite-member dialog (accept-invite is its own landing page)
//   2  People tab ........ member list + promote/demote (T1 guard surfaced) + remove (last-MP guard)
//   3  Matters & Access .. the read-only role × verb default matrix (transparency)
//   4  Ethical Walls ..... screen list + add/lift (required reason) + conflict-scan hand-off
//   7  Delegation / PA ... grant-authority (verbs + expiry) → time-boxed chip → one-click revoke
//   9  Firm switcher ..... the active firm + (multi-firm seam) re-scope
//   10 Caps source ....... the server-resolved cap banner drives what renders
// (5 matter-team, 6 review chain/queue, 8 override live on the vault/answer surfaces, not here.)
//
// D0 — POSITIVE, never punitive. The People tab leads with "your team and what they can do",
// the matrix frames a paralegal's full working toolkit as ENABLED, and held controls
// (release_external / manage-firm) read as the firm's outbound/admin gates, not a demotion.

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, useReducedMotion } from "framer-motion";
import {
  ArrowLeft, Users, Grid3x3, Lock, KeyRound, Building2, ShieldCheck, Plus, Trash2,
  Check, Clock, AlertCircle, UserPlus, Copy, ChevronDown,
} from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/auth.store";
import { useFirmStore } from "@/stores/firm.store";
import {
  inviteMember, listInvites, resendInvite, revokeInvite, acceptInviteUrl,
  setRole, removeMember, createScreen, removeScreen,
  grantAuthority, revokeAuthority, listDelegations, listCollections,
  APIError,
  type FirmRole, type Capability, type InviteResponse, type DelegationResponse,
  type CollectionResponse, type MemberResponse,
} from "@/lib/api";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/Dialog";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";
import { Select } from "@/components/ui/Select";
import {
  ROLE_LABEL, INTERNAL_ROLES, ROLE_RANK, CAP_LABEL, MATRIX_VERBS, ROLE_CAPS, ROLE_ORDER,
  memberLabel,
} from "./_shared";

const ease = [0.16, 1, 0.3, 1] as const;

// ── A neutral role chip (lifecycle = INFORMATION, neutral ink — NOT a trust color, DESIGN.md) ──
function RoleChip({ role }: { role: FirmRole | null }) {
  if (!role) return null;
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-medium tracking-wide"
      style={{ background: "var(--surface-3)", color: "var(--ink-2)", border: "1px solid var(--line)" }}
    >
      {ROLE_LABEL[role]}
    </span>
  );
}

// ════════════════════════════════════════════════════════════════════════════════
// SURFACE 1 — Invite member dialog (onboarding)
// ════════════════════════════════════════════════════════════════════════════════
function InviteDialog({ onInvited }: { onInvited: () => void }) {
  const token = useAuthStore((s) => s.token);
  const { role: myRole, onForbidden } = useFirmStore();
  const [open, setOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [role, setRoleSel] = useState<FirmRole>("associate");
  const [busy, setBusy] = useState(false);
  const [created, setCreated] = useState<InviteResponse | null>(null);

  // T1 surfaced: you cannot invite at a role AT OR ABOVE your own rank. We DISABLE those options and
  // say why — the server enforces it regardless; this just keeps the UI honest about the boundary.
  const myRank = myRole ? ROLE_RANK[myRole] : 0;
  const roleAllowed = (r: FirmRole) => ROLE_RANK[r] > myRank;

  async function submit() {
    if (!token || !email.trim()) return;
    setBusy(true);
    try {
      const inv = await inviteMember(token, email.trim(), role);
      setCreated(inv);          // show the one-time link to deliver out-of-band
      setEmail("");
      toast.success("Invite created");
      // NOTE: do NOT refresh the parent list here. PeopleTab swaps its EmptyState branch for the
      // list branch the moment an invite exists, which REMOUNTS this dialog and would wipe the
      // one-time link before you can copy it. We defer the refresh to close() instead.
    } catch (err) {
      const was403 = await onForbidden(token, err);
      const detail = err instanceof APIError ? err.detail : undefined;
      toast.error(was403 ? "Your permission changed. Refreshed." : detail || "Could not create the invite.");
    } finally {
      setBusy(false);
    }
  }

  function close() {
    setOpen(false);
    const hadInvite = !!created;
    setTimeout(() => { setCreated(null); }, 200);
    // Refresh the list AFTER the dialog closes (see submit() — refreshing while open remounts us).
    if (hadInvite) onInvited();
  }

  return (
    <Dialog open={open} onOpenChange={(v) => (v ? setOpen(true) : close())}>
      <Button variant="primary" size="sm" onClick={() => setOpen(true)}>
        <UserPlus size={14} className="mr-1.5" /> Invite member
      </Button>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{created ? "Invite created" : "Invite a member"}</DialogTitle>
        </DialogHeader>
        {created ? (
          // Success: deliver the one-time invite LINK. There is no email server wired yet (F2j is
          // deferred), so you copy this link and send it to the invitee. It is shown once; if you
          // lose it, use "Copy link" on the pending invite to mint a fresh one (rotates the token).
          <div className="px-5 py-5 space-y-3">
            <p className="text-[13px]" style={{ color: "var(--ink-2)" }}>
              Send this single-use link to <strong style={{ color: "var(--ink)" }}>{created.email}</strong>.
              They join as <RoleChip role={created.role} />. It is shown once; you can re-copy it from
              the pending invites list.
            </p>
            <div
              className="flex items-center gap-2 p-2.5 rounded-lg text-[12px] font-mono break-all"
              style={{ background: "var(--surface-3)", color: "var(--ink)", border: "1px solid var(--line)" }}
            >
              <span className="flex-1">{created.token ? acceptInviteUrl(created.token) : ""}</span>
            </div>
            <Button
              variant="primary" size="sm"
              onClick={() => {
                navigator.clipboard?.writeText(created.token ? acceptInviteUrl(created.token) : "");
                toast.success("Invite link copied");
              }}
            >
              <Copy size={13} className="mr-1.5" /> Copy invite link
            </Button>
          </div>
        ) : (
          <div className="px-5 py-5 space-y-4">
            <label className="block">
              <span className="text-[11px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--ink-3)" }}>
                Email
              </span>
              <input
                type="email" value={email} onChange={(e) => setEmail(e.target.value)}
                placeholder="colleague@firm.com"
                className="mt-1 w-full px-3 py-2 rounded-lg text-[14px] outline-none"
                style={{ background: "var(--surface-2)", border: "1px solid var(--line-2)", color: "var(--ink)" }}
              />
            </label>
            <div>
              <span className="text-[11px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--ink-3)" }}>
                Role
              </span>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {INTERNAL_ROLES.map((r) => {
                  const ok = roleAllowed(r);
                  const active = role === r;
                  return (
                    <button
                      key={r}
                      disabled={!ok}
                      onClick={() => setRoleSel(r)}
                      title={ok ? undefined : "You cannot grant a role at or above your own."}
                      className="px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors disabled:cursor-not-allowed"
                      style={{
                        background: active ? "var(--ink)" : "var(--surface-3)",
                        color: active ? "var(--on-ink)" : ok ? "var(--ink-2)" : "var(--ink-3)",
                        border: `1px solid ${active ? "var(--ink)" : "var(--line)"}`,
                        opacity: ok ? 1 : 0.4,
                      }}
                    >
                      {ROLE_LABEL[r]}
                    </button>
                  );
                })}
              </div>
              <p className="text-[10px] mt-1.5" style={{ color: "var(--ink-3)" }}>
                The joiner cannot change this. You can promote them later, never above your own role.
              </p>
            </div>
          </div>
        )}
        <DialogFooter>
          {created ? (
            <Button variant="primary" size="sm" onClick={close}>Done</Button>
          ) : (
            <>
              <Button variant="ghost" size="sm" onClick={close}>Cancel</Button>
              <Button variant="primary" size="sm" loading={busy} disabled={!email.trim()} onClick={submit}>
                Send invite
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ════════════════════════════════════════════════════════════════════════════════
// SURFACE 2 — People tab (members + promote/demote + remove)
// ════════════════════════════════════════════════════════════════════════════════
function PeopleTab() {
  const token = useAuthStore((s) => s.token);
  const me = useAuthStore((s) => s.user);
  const {
    members, loadingMembers, loadMembers, role: myRole, onForbidden, refreshCaps,
  } = useFirmStore();
  const [invites, setInvites] = useState<InviteResponse[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const myRank = myRole ? ROLE_RANK[myRole] : 0;
  const mpCount = members.filter((m) => m.role === "managing_partner").length;

  const reload = useCallback(async () => {
    if (!token) return;
    setErr(null);
    try {
      await loadMembers(token);
      setInvites(await listInvites(token));
    } catch {
      setErr("Could not load your team. Check your connection and try again.");
    }
  }, [token, loadMembers]);

  useEffect(() => { reload(); }, [reload]);

  async function changeRole(userId: string, role: FirmRole) {
    if (!token) return;
    try {
      await setRole(token, userId, role);
      toast.success(`Role updated to ${ROLE_LABEL[role]}`);
      await reload();
      await refreshCaps(token);    // my own caps may change if I changed myself
    } catch (e) {
      const was403 = await onForbidden(token, e);
      toast.error(was403 ? "Your permission changed. Refreshed." :
        (e instanceof APIError ? e.detail : "Could not change the role."));
      await reload();              // reconcile optimistic intent with server truth
    }
  }

  async function remove(userId: string, label: string) {
    if (!token) return;
    try {
      await removeMember(token, userId);
      toast.success(`Removed ${label}. Their access ended immediately.`);
      await reload();
    } catch (e) {
      const was403 = await onForbidden(token, e);
      toast.error(was403 ? "Your permission changed. Refreshed." :
        (e instanceof APIError ? e.detail : "Could not remove the member."));
    }
  }

  if (loadingMembers && members.length === 0) {
    return (
      <div className="space-y-2">
        {[0, 1, 2].map((i) => <Skeleton key={i} className="h-14 w-full rounded-xl" />)}
      </div>
    );
  }

  if (err) {
    return (
      <div className="p-5 rounded-xl flex items-center gap-3" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
        <AlertCircle size={16} style={{ color: "var(--fidelity-failed)" }} />
        <div className="flex-1"><p className="text-[13px]" style={{ color: "var(--ink-2)" }}>{err}</p></div>
        <Button variant="outline" size="sm" onClick={reload}>Retry</Button>
      </div>
    );
  }

  if (members.length <= 1 && invites.length === 0) {
    return (
      <EmptyState
        icon={<Users size={18} />}
        title="You're the only member"
        description="Invite your team to collaborate. Everyone you add gets a role you choose and the full working toolkit on the matters they're staffed on."
        action={<InviteDialog onInvited={reload} />}
      />
    );
  }

  const pendingInvites = invites.filter((i) => !i.accepted_at);

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <p className="text-[12px]" style={{ color: "var(--ink-3)" }}>
          <span className="tabular-nums">{members.length}</span> {members.length === 1 ? "member" : "members"}
        </p>
        <InviteDialog onInvited={reload} />
      </div>

      {/* Member list */}
      <div className="rounded-2xl overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
        {members.map((m, i) => {
          const isSelf = me?.id === m.user_id;
          const isSoleMP = m.role === "managing_partner" && mpCount <= 1;
          // T1 surfaced: I can only set a role strictly below my own. The last-MP guard disables a
          // remove that would orphan the firm. The server enforces both; this explains the boundary.
          return (
            <div
              key={m.user_id}
              className="flex items-center gap-3 px-4 py-3"
              style={{ borderTop: i ? "1px solid var(--line)" : "none" }}
            >
              <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 text-[11px] font-semibold"
                   style={{ background: "var(--surface-3)", color: "var(--ink-2)" }}>
                {(m.email ?? m.user_id).slice(0, 2).toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-[13px] font-medium truncate" style={{ color: "var(--ink)" }}>
                  {memberLabel(m)} {isSelf && <span style={{ color: "var(--ink-3)" }}>(you)</span>}
                </p>
              </div>
              <RolePicker
                current={m.role}
                myRank={myRank}
                disabled={isSelf && isSoleMP}
                onPick={(r) => changeRole(m.user_id, r)}
              />
              <RemoveMemberButton
                disabled={isSoleMP}
                label={memberLabel(m)}
                reason={isSoleMP ? "The firm must keep at least one Managing Partner." : undefined}
                onConfirm={() => remove(m.user_id, memberLabel(m))}
              />
            </div>
          );
        })}
      </div>

      {/* Pending invites */}
      {pendingInvites.length > 0 && (
        <div>
          <p className="text-[11px] font-medium uppercase tracking-[0.1em] mb-2" style={{ color: "var(--ink-3)" }}>
            Pending invites
          </p>
          <div className="rounded-2xl overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
            {pendingInvites.map((inv, i) => (
              <PendingInviteRow key={inv.id} invite={inv} first={i === 0} onChange={reload} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Promote/demote control — surfaces the T1 guard as reasoned disabled options.
function RolePicker({
  current, myRank, disabled, onPick,
}: { current: FirmRole; myRank: number; disabled?: boolean; onPick: (r: FirmRole) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium transition-[color,background-color,transform] active:scale-[0.96] disabled:cursor-not-allowed disabled:opacity-50"
        style={{ background: "var(--surface-3)", color: "var(--ink-2)", border: "1px solid var(--line)" }}
        title={disabled ? "The firm must keep at least one Managing Partner." : "Change role"}
      >
        {ROLE_LABEL[current]} <ChevronDown size={12} />
      </button>
      {open && !disabled && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-full mt-1 z-20 w-44 py-1 rounded-xl"
               style={{ background: "var(--surface)", border: "1px solid var(--line-2)", boxShadow: "var(--shadow-md)" }}>
            {INTERNAL_ROLES.map((r) => {
              const allowed = ROLE_RANK[r] > myRank;     // strictly below my own rank (T1)
              const isCurrent = r === current;
              return (
                <button
                  key={r}
                  disabled={!allowed}
                  onClick={() => { setOpen(false); if (allowed && !isCurrent) onPick(r); }}
                  title={allowed ? undefined : "You cannot set a role at or above your own."}
                  className="w-full flex items-center justify-between px-3 py-1.5 text-[12px] text-left transition-colors disabled:cursor-not-allowed hover:bg-[var(--bg-hover)]"
                  style={{ color: allowed ? "var(--ink-2)" : "var(--ink-3)", opacity: allowed ? 1 : 0.45 }}
                >
                  {ROLE_LABEL[r]} {isCurrent && <Check size={12} />}
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

function RemoveMemberButton({
  disabled, label, reason, onConfirm,
}: { disabled?: boolean; label: string; reason?: string; onConfirm: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        disabled={disabled}
        onClick={() => setOpen(true)}
        title={disabled ? reason : "Remove member"}
        className="p-1.5 rounded-lg transition-[color,background-color,transform] active:scale-[0.96] disabled:cursor-not-allowed disabled:opacity-30 disabled:active:scale-100 hover:bg-[var(--bg-hover)]"
        style={{ color: "var(--ink-3)" }}
      >
        <Trash2 size={14} />
      </button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent maxWidth="420px">
          <DialogHeader><DialogTitle>Remove {label}?</DialogTitle></DialogHeader>
          <div className="px-5 py-5">
            <p className="text-[13px]" style={{ color: "var(--ink-2)" }}>
              This revokes all of their access immediately and reassigns their matters to the firm.
              They will be signed out of the firm on their next request.
            </p>
          </div>
          <DialogFooter>
            <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>Cancel</Button>
            <Button variant="primary" size="sm" onClick={() => { setOpen(false); onConfirm(); }}>
              Remove member
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

// A pending invite row — re-copy the link (rotates the token, the researched recovery for a
// hash-stored invite) or revoke it. No email server is wired (F2j deferred), so the admin delivers
// the link out-of-band; "Copy link" mints a FRESH one-time link each time.
function PendingInviteRow({
  invite, first, onChange,
}: { invite: InviteResponse; first: boolean; onChange: () => void }) {
  const token = useAuthStore((s) => s.token);
  const { onForbidden } = useFirmStore();
  const [busy, setBusy] = useState(false);

  async function copyLink() {
    if (!token) return;
    setBusy(true);
    try {
      const rotated = await resendInvite(token, invite.id);
      if (rotated.token) {
        await navigator.clipboard?.writeText(acceptInviteUrl(rotated.token));
        toast.success("New invite link copied. Send THIS one; any earlier link no longer works.");
      }
    } catch (e) {
      const was403 = await onForbidden(token, e);
      toast.error(was403 ? "Your permission changed. Refreshed." :
        (e instanceof APIError ? e.detail : "Could not refresh the invite link."));
    } finally { setBusy(false); }
  }

  async function revoke() {
    if (!token) return;
    try {
      await revokeInvite(token, invite.id);
      toast.success("Invite revoked");
      onChange();
    } catch (e) {
      const was403 = await onForbidden(token, e);
      toast.error(was403 ? "Your permission changed. Refreshed." :
        (e instanceof APIError ? e.detail : "Could not revoke the invite."));
    }
  }

  return (
    <div className="flex items-center gap-3 px-4 py-2.5"
         style={{ borderTop: first ? "none" : "1px solid var(--line)" }}>
      <Clock size={13} style={{ color: "var(--ink-3)" }} />
      <span className="text-[13px] flex-1 truncate" style={{ color: "var(--ink-2)" }}>{invite.email}</span>
      <RoleChip role={invite.role} />
      <button
        onClick={copyLink}
        disabled={busy}
        title="Generates a NEW link and copies it. Any link you sent before this stops working."
        className="inline-flex items-center gap-1 text-[11px] font-medium px-2 py-1 rounded-md transition-[color,background-color,transform] active:scale-[0.96] hover:bg-[var(--bg-hover)] disabled:opacity-50"
        style={{ color: "var(--ink-2)", border: "1px solid var(--line)" }}
      >
        <Copy size={11} /> {busy ? "Generating…" : "Get new link"}
      </button>
      <button
        onClick={revoke}
        aria-label="Revoke invite"
        className="p-1.5 rounded-lg transition-[color,background-color,transform] active:scale-[0.96] hover:bg-[var(--bg-hover)]"
        style={{ color: "var(--ink-3)" }}
      >
        <Trash2 size={13} />
      </button>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════════
// SURFACE 3 — Matters & Access matrix (read-only transparency of the default policy)
// ════════════════════════════════════════════════════════════════════════════════
function AccessMatrixTab() {
  // The matrix is the firm's DEFAULT role→capability policy (mirrors authz.py ROLE_CAPS). It is a
  // transparency view — read-only. Internal roles only (external client/guest are per-vault).
  const rows = ROLE_ORDER.filter((r) => r !== "client" && r !== "guest");
  return (
    <div>
      <p className="text-[12px] mb-3" style={{ color: "var(--ink-3)" }}>
        What each role can do by default. A team member also gets the full working toolkit on every
        matter they are staffed on. Releasing work outside the firm is the one tightly-held gate.
      </p>
      <div className="rounded-2xl overflow-x-auto" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
        <table className="w-full text-[11px] border-collapse">
          <thead>
            <tr>
              <th className="sticky left-0 px-3 py-2.5 text-left font-medium z-10"
                  style={{ background: "var(--surface-2)", color: "var(--ink-3)", borderBottom: "1px solid var(--line)" }}>
                Role
              </th>
              {MATRIX_VERBS.map((v) => (
                <th key={v} className="px-2 py-2.5 text-left font-medium whitespace-nowrap"
                    style={{ color: "var(--ink-3)", borderBottom: "1px solid var(--line)" }}>
                  {CAP_LABEL[v]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((role, ri) => (
              <tr key={role}>
                <td className="sticky left-0 px-3 py-2 font-medium z-10"
                    style={{ background: "var(--surface)", color: "var(--ink)", borderTop: ri ? "1px solid var(--line)" : "none" }}>
                  {ROLE_LABEL[role]}
                </td>
                {MATRIX_VERBS.map((v) => {
                  const allow = ROLE_CAPS[role].has(v);
                  return (
                    <td key={v} className="px-2 py-2 text-center"
                        style={{ borderTop: ri ? "1px solid var(--line)" : "none" }}>
                      {allow
                        ? <Check size={13} style={{ color: "var(--ink)" }} className="inline" />
                        : <span style={{ color: "var(--ink-3)" }}>·</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════════
// SURFACE 4 — Ethical Walls (screens) + conflict-scan hand-off
// ════════════════════════════════════════════════════════════════════════════════
function WallsTab() {
  const token = useAuthStore((s) => s.token);
  const { screens, loadingScreens, loadScreens, members, onForbidden } = useFirmStore();
  const [collections, setCollections] = useState<CollectionResponse[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const reload = useCallback(async () => {
    if (!token) return;
    setErr(null);
    try {
      await loadScreens(token);
      setCollections(await listCollections(token));
    } catch {
      setErr("Could not load the ethical walls. Try again.");
    }
  }, [token, loadScreens]);

  useEffect(() => { reload(); }, [reload]);

  async function lift(screenId: string) {
    if (!token) return;
    try {
      await removeScreen(token, screenId);
      toast.success("Wall lifted. The member regains access on their next request.");
      await reload();
    } catch (e) {
      const was403 = await onForbidden(token, e);
      toast.error(was403 ? "Your permission changed. Refreshed." :
        (e instanceof APIError ? e.detail : "Could not lift the wall."));
    }
  }

  const active = screens.filter((s) => !s.removed_at);
  const collName = (id: string) => collections.find((c) => c.id === id)?.name ?? id.slice(0, 8) + "…";
  const memLabel = (uid: string) => memberLabel(members.find((m) => m.user_id === uid)) || uid.slice(0, 8) + "…";

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <p className="text-[12px] flex-1" style={{ color: "var(--ink-3)" }}>
          A wall blocks one member from one matter, for a documented conflict reason. It overrides
          every role grant, even a partner&apos;s. The screened member cannot see the wall on them.
        </p>
        <AddScreenDialog members={members} collections={collections} onDone={reload} />
      </div>

      {loadingScreens && screens.length === 0 ? (
        <div className="space-y-2">{[0, 1].map((i) => <Skeleton key={i} className="h-14 w-full rounded-xl" />)}</div>
      ) : err ? (
        <div className="p-5 rounded-xl flex items-center gap-3" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
          <AlertCircle size={16} style={{ color: "var(--fidelity-failed)" }} />
          <p className="text-[13px] flex-1" style={{ color: "var(--ink-2)" }}>{err}</p>
          <Button variant="outline" size="sm" onClick={reload}>Retry</Button>
        </div>
      ) : active.length === 0 ? (
        <EmptyState
          icon={<Lock size={18} />}
          title="No ethical walls in place"
          description="If a conflict requires it, screen a member off a specific matter. The wall is enforced in the data layer, not just hidden."
        />
      ) : (
        <div className="rounded-2xl overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
          {active.map((s, i) => (
            <div key={s.id} className="flex items-start gap-3 px-4 py-3"
                 style={{ borderTop: i ? "1px solid var(--line)" : "none" }}>
              {/* A wall is a lock-glyph BOUNDARY marker — neutral ink, NOT a red alarm (DESIGN.md) */}
              <Lock size={15} style={{ color: "var(--ink-2)" }} className="mt-0.5 shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-[13px]" style={{ color: "var(--ink)" }}>
                  <strong>{memLabel(s.user_id)}</strong> is screened off <strong>{collName(s.vault_id)}</strong>
                </p>
                <p className="text-[11px] mt-0.5" style={{ color: "var(--ink-3)" }}>{s.reason}</p>
              </div>
              <button
                onClick={() => lift(s.id)}
                className="text-[11px] font-medium px-2 py-1 rounded-md transition-[color,background-color,transform] active:scale-[0.96] hover:bg-[var(--bg-hover)] shrink-0"
                style={{ color: "var(--ink-2)", border: "1px solid var(--line)" }}
              >
                Lift wall
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function AddScreenDialog({
  members, collections, onDone,
}: { members: MemberResponse[]; collections: CollectionResponse[]; onDone: () => void }) {
  const token = useAuthStore((s) => s.token);
  const { onForbidden } = useFirmStore();
  const [open, setOpen] = useState(false);
  const [userId, setUserId] = useState("");
  const [vaultId, setVaultId] = useState("");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (!token || !userId || !vaultId || !reason.trim()) return;
    setBusy(true);
    try {
      await createScreen(token, userId, vaultId, reason.trim());
      toast.success("Wall created");
      setOpen(false); setUserId(""); setVaultId(""); setReason("");
      onDone();
    } catch (e) {
      const was403 = await onForbidden(token, e);
      toast.error(was403 ? "Your permission changed. Refreshed." :
        (e instanceof APIError ? e.detail : "Could not create the wall."));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
        <Plus size={14} className="mr-1" /> Add wall
      </Button>
      <DialogContent>
        <DialogHeader><DialogTitle>Screen a member off a matter</DialogTitle></DialogHeader>
        <div className="px-5 py-5 space-y-4 overflow-y-auto">
          <div className="block">
            <span className="text-[11px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--ink-3)" }}>Member</span>
            <Select
              className="mt-1" ariaLabel="Member to screen" placeholder="Select a member…"
              value={userId} onChange={setUserId}
              options={members.map((m) => ({ value: m.user_id, label: memberLabel(m),
                hint: m.role ? ROLE_LABEL[m.role] : undefined }))}
            />
          </div>
          <div className="block">
            <span className="text-[11px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--ink-3)" }}>Matter</span>
            <Select
              className="mt-1" ariaLabel="Matter to screen off" placeholder="Select a matter…"
              value={vaultId} onChange={setVaultId}
              options={collections.map((c) => ({ value: c.id, label: c.name }))}
            />
          </div>
          <label className="block">
            <span className="text-[11px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--ink-3)" }}>
              Reason (required)
            </span>
            <textarea value={reason} onChange={(e) => setReason(e.target.value)} rows={2}
              placeholder="The documented conflict this wall addresses."
              className="mt-1 w-full px-3 py-2 rounded-lg text-[13px] outline-none resize-none"
              style={{ background: "var(--surface-2)", border: "1px solid var(--line-2)", color: "var(--ink)" }} />
          </label>
        </div>
        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>Cancel</Button>
          <Button variant="primary" size="sm" loading={busy}
            disabled={!userId || !vaultId || !reason.trim()} onClick={submit}>
            Create wall
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ════════════════════════════════════════════════════════════════════════════════
// SURFACE 7 — Delegation / PA (grant authority, time-boxed, revocable)
// ════════════════════════════════════════════════════════════════════════════════
function DelegationTab() {
  const token = useAuthStore((s) => s.token);
  const { members, onForbidden, caps } = useFirmStore();
  const [delegations, setDelegations] = useState<DelegationResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const reload = useCallback(async () => {
    if (!token) return;
    setLoading(true); setErr(null);
    try {
      setDelegations(await listDelegations(token));
    } catch {
      setErr("Could not load delegations. Try again.");
    } finally { setLoading(false); }
  }, [token]);

  useEffect(() => { reload(); }, [reload]);

  async function revoke(id: string) {
    if (!token) return;
    try {
      await revokeAuthority(token, id);
      toast.success("Authority revoked");
      await reload();
    } catch (e) {
      const was403 = await onForbidden(token, e);
      toast.error(was403 ? "Your permission changed. Refreshed." :
        (e instanceof APIError ? e.detail : "Could not revoke."));
    }
  }

  const now = Date.now();
  const active = delegations.filter((d) => !d.revoked_at && (!d.expires_at || new Date(d.expires_at).getTime() > now));
  const memLabel = (uid: string) => memberLabel(members.find((m) => m.user_id === uid)) || uid.slice(0, 8) + "…";

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <p className="text-[12px] flex-1" style={{ color: "var(--ink-3)" }}>
          Hand a colleague a time-boxed set of actions on your behalf, for example an assistant who
          triages your review queue while you are away. You stay accountable; it expires automatically.
        </p>
        <GrantAuthorityDialog members={members} myCaps={caps} onDone={reload} />
      </div>

      {loading && delegations.length === 0 ? (
        <div className="space-y-2">{[0, 1].map((i) => <Skeleton key={i} className="h-14 w-full rounded-xl" />)}</div>
      ) : err ? (
        <div className="p-5 rounded-xl flex items-center gap-3" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
          <AlertCircle size={16} style={{ color: "var(--fidelity-failed)" }} />
          <p className="text-[13px] flex-1" style={{ color: "var(--ink-2)" }}>{err}</p>
          <Button variant="outline" size="sm" onClick={reload}>Retry</Button>
        </div>
      ) : active.length === 0 ? (
        <EmptyState
          icon={<KeyRound size={18} />}
          title="No active delegations"
          description="Grant a colleague a time-boxed subset of your actions when you need cover. It is logged and you can revoke it any time."
        />
      ) : (
        <div className="rounded-2xl overflow-hidden" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
          {active.map((d, i) => (
            <div key={d.id} className="flex items-start gap-3 px-4 py-3"
                 style={{ borderTop: i ? "1px solid var(--line)" : "none" }}>
              <KeyRound size={15} style={{ color: "var(--ink-2)" }} className="mt-0.5 shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-[13px]" style={{ color: "var(--ink)" }}>
                  <strong>{memLabel(d.delegate_id)}</strong> can {d.verbs.map((v) => CAP_LABEL[v]).join(", ")} for you
                </p>
                <p className="text-[11px] mt-0.5 inline-flex items-center gap-1 tabular-nums" style={{ color: "var(--ink-3)" }}>
                  <Clock size={11} />
                  {d.expires_at ? `until ${new Date(d.expires_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}` : "no expiry"}
                </p>
              </div>
              <button onClick={() => revoke(d.id)}
                className="text-[11px] font-medium px-2 py-1 rounded-md transition-[color,background-color,transform] active:scale-[0.96] hover:bg-[var(--bg-hover)] shrink-0"
                style={{ color: "var(--ink-2)", border: "1px solid var(--line)" }}>
                Revoke
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function GrantAuthorityDialog({
  members, myCaps, onDone,
}: { members: MemberResponse[]; myCaps: Set<Capability>; onDone: () => void }) {
  const token = useAuthStore((s) => s.token);
  const me = useAuthStore((s) => s.user);
  const { onForbidden } = useFirmStore();
  const [open, setOpen] = useState(false);
  const [delegateId, setDelegateId] = useState("");
  const [verbs, setVerbs] = useState<Set<Capability>>(new Set());
  const [expiry, setExpiry] = useState("");
  const [busy, setBusy] = useState(false);

  // You can only delegate verbs you hold yourself (the server re-bounds this; we don't even SHOW
  // verbs you can't grant). send_for_review/release_external/override_abstain are the useful ones.
  const grantable = useMemo(
    () => (["send_for_review", "release_external", "override_abstain", "run_workflow", "draft", "grids"] as Capability[])
      .filter((v) => myCaps.has(v)),
    [myCaps]
  );

  function toggle(v: Capability) {
    setVerbs((prev) => {
      const n = new Set(prev);
      if (n.has(v)) n.delete(v); else n.add(v);
      return n;
    });
  }

  async function submit() {
    if (!token || !delegateId || verbs.size === 0 || !expiry) return;
    setBusy(true);
    try {
      await grantAuthority(token, delegateId, Array.from(verbs), new Date(expiry).toISOString());
      toast.success("Authority granted");
      setOpen(false); setDelegateId(""); setVerbs(new Set()); setExpiry("");
      onDone();
    } catch (e) {
      const was403 = await onForbidden(token, e);
      toast.error(was403 ? "Your permission changed. Refreshed." :
        (e instanceof APIError ? e.detail : "Could not grant authority."));
    } finally { setBusy(false); }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
        <Plus size={14} className="mr-1" /> Grant authority
      </Button>
      <DialogContent>
        <DialogHeader><DialogTitle>Grant authority to a colleague</DialogTitle></DialogHeader>
        <div className="px-5 py-5 space-y-4 overflow-y-auto">
          <div className="block">
            <span className="text-[11px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--ink-3)" }}>To</span>
            <Select
              className="mt-1" ariaLabel="Colleague to delegate to" placeholder="Select a colleague…"
              value={delegateId} onChange={setDelegateId}
              options={members.filter((m) => m.user_id !== me?.id).map((m) => ({
                value: m.user_id, label: memberLabel(m), hint: m.role ? ROLE_LABEL[m.role] : undefined,
              }))}
            />
          </div>
          <div>
            <span className="text-[11px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--ink-3)" }}>
              They may
            </span>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {grantable.map((v) => {
                const on = verbs.has(v);
                return (
                  <button key={v} onClick={() => toggle(v)}
                    className="px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors"
                    style={{
                      background: on ? "var(--ink)" : "var(--surface-3)",
                      color: on ? "var(--on-ink)" : "var(--ink-2)",
                      border: `1px solid ${on ? "var(--ink)" : "var(--line)"}`,
                    }}>
                    {CAP_LABEL[v]}
                  </button>
                );
              })}
              {grantable.length === 0 && (
                <p className="text-[11px]" style={{ color: "var(--ink-3)" }}>You have no delegatable actions.</p>
              )}
            </div>
          </div>
          <label className="block">
            <span className="text-[11px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--ink-3)" }}>
              Until (expiry)
            </span>
            <input type="date" value={expiry} onChange={(e) => setExpiry(e.target.value)}
              className="mt-1 w-full px-3 py-2 rounded-lg text-[13px] outline-none"
              style={{ background: "var(--surface-2)", border: "1px solid var(--line-2)", color: "var(--ink)" }} />
          </label>
        </div>
        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>Cancel</Button>
          <Button variant="primary" size="sm" loading={busy}
            disabled={!delegateId || verbs.size === 0 || !expiry} onClick={submit}>
            Grant
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ════════════════════════════════════════════════════════════════════════════════
// SURFACE 9 + 10 — Firm switcher + caps banner (the page header)
// ════════════════════════════════════════════════════════════════════════════════
function ConsoleHeader() {
  const { firm, role } = useFirmStore();
  // Firm switcher: the active firm is shown with its role. Multi-firm ENUMERATION is a forward seam
  // (no list-my-firms endpoint exists yet — get_user_firm resolves one active firm server-side, and
  // /auth/capabilities re-scopes to it). When that endpoint lands, the chevron opens the list and a
  // pick re-runs init(token, firm_id). Logged-not-hidden per the "no silent caps" rule.
  return (
    <div className="flex items-center gap-2.5">
      <div className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
           style={{ background: "var(--ink)", color: "var(--on-ink)" }}>
        <Building2 size={17} strokeWidth={1.7} />
      </div>
      <div>
        <p className="text-[15px] font-semibold leading-tight" style={{ color: "var(--ink)" }}>
          {firm?.name ?? "Your firm"}
        </p>
        <p className="text-[11px] inline-flex items-center gap-1.5" style={{ color: "var(--ink-3)" }}>
          You are <RoleChip role={role} />
        </p>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════════
// THE CONSOLE PAGE
// ════════════════════════════════════════════════════════════════════════════════
const TABS = [
  { key: "people", label: "People", icon: Users },
  { key: "access", label: "Matters & Access", icon: Grid3x3 },
  { key: "walls", label: "Ethical Walls", icon: Lock },
  { key: "delegation", label: "Delegation", icon: KeyRound },
] as const;

export default function FirmConsolePage() {
  const router = useRouter();
  const token = useAuthStore((s) => s.token);
  const { init, initialized, loadingCaps, can, isExternal, error } = useFirmStore();
  const [tab, setTab] = useState<(typeof TABS)[number]["key"]>("people");
  const reduce = useReducedMotion();

  useEffect(() => { if (token) init(token); }, [token, init]);

  // The page is a SECOND line of defense behind middleware (T8): if the server says this user lacks
  // manage_members (or is external), we do not render the console even if they reached the route.
  const blocked = initialized && (isExternal || !can("manage_members"));

  if (!initialized || loadingCaps) {
    return (
      <div className="flex-1 overflow-y-auto" style={{ background: "var(--canvas)" }}>
        <div className="max-w-4xl mx-auto px-4 md:px-8 py-8 space-y-4">
          <Skeleton className="h-10 w-64 rounded-xl" />
          <Skeleton className="h-9 w-full rounded-xl" />
          <Skeleton className="h-32 w-full rounded-2xl" />
        </div>
      </div>
    );
  }

  if (blocked) {
    return (
      <div className="flex-1 flex items-center justify-center" style={{ background: "var(--canvas)" }}>
        <EmptyState
          icon={<ShieldCheck size={18} />}
          title="The firm console is for firm administrators"
          description="Your role doesn't include managing the firm. If you need access, ask a partner at your firm."
          action={<Button variant="outline" size="sm" onClick={() => router.push("/app/settings")}>Back to settings</Button>}
        />
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin" style={{ background: "var(--canvas)" }}>
      <div className="max-w-4xl mx-auto px-4 md:px-8 py-8">
        {/* Header — back · title · firm switcher (surface 9/10) */}
        <motion.div
          initial={{ opacity: reduce ? 1 : 0, y: reduce ? 0 : 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ ease }}
          className="flex items-center gap-3 mb-6"
        >
          <button onClick={() => router.push("/app/settings")}
            className="p-1.5 rounded-lg hover:bg-[var(--bg-hover)] transition-[color,background-color,transform] active:scale-[0.96]" style={{ color: "var(--ink-3)" }}>
            <ArrowLeft size={18} />
          </button>
          <div className="flex-1">
            <h1 style={{ fontFamily: "Fraunces, Georgia, serif", fontSize: "24px", fontWeight: 400, letterSpacing: "-0.025em", color: "var(--ink)", lineHeight: 1.1, textWrap: "balance" }}>
              Firm console
            </h1>
            <p className="text-xs" style={{ color: "var(--ink-3)" }}>Members, access, walls, and delegation</p>
          </div>
          <ConsoleHeader />
        </motion.div>

        {error && (
          <div className="mb-4 p-3 rounded-xl flex items-center gap-2.5 text-[12px]"
               style={{ background: "var(--surface)", border: "1px solid var(--line)", color: "var(--ink-2)" }}>
            <AlertCircle size={14} style={{ color: "var(--fidelity-partial)" }} /> {error}
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-1 mb-5 p-1 rounded-xl" style={{ background: "var(--surface-2)", border: "1px solid var(--line)" }}>
          {TABS.map((t) => {
            const active = tab === t.key;
            const Icon = t.icon;
            return (
              <button key={t.key} onClick={() => setTab(t.key)}
                className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-[12px] font-medium transition-[color,background-color,transform] active:scale-[0.96]"
                style={{
                  background: active ? "var(--surface)" : "transparent",
                  color: active ? "var(--ink)" : "var(--ink-3)",
                  boxShadow: active ? "var(--shadow-sm)" : "none",
                }}>
                <Icon size={13} /> {t.label}
              </button>
            );
          })}
        </div>

        {/* Tab content */}
        <motion.div key={tab}
          initial={{ opacity: reduce ? 1 : 0, y: reduce ? 0 : 6 }}
          animate={{ opacity: 1, y: 0 }} transition={{ ease, duration: 0.3 }}>
          {tab === "people" && <PeopleTab />}
          {tab === "access" && <AccessMatrixTab />}
          {tab === "walls" && <WallsTab />}
          {tab === "delegation" && <DelegationTab />}
        </motion.div>
      </div>
    </div>
  );
}
