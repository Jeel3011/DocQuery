"use client";

// components/app/MatterTeamPanel.tsx — F2g surface 5 (matter team, D3) + surface 6 partial
// (customize review chain). Mounts on the vault page. The seniors on the matter staff it; everyone
// staffed gets the FULL working toolkit on it — and the panel SAYS SO PLAINLY ("Full access on this
// matter"), making D0 visible: this is a positive grant, never a wall of disabled buttons.
//
// Security: every action is server-cap-gated (manage_matter_team). The store's caps decide whether
// the "Add to team" / "Customize chain" affordances RENDER; the route is the boundary. A 403 (a
// permission changed mid-session) refetches caps via the store (T7).

import { useCallback, useEffect, useState } from "react";
import { Users, Plus, Trash2, ListOrdered, Check } from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/auth.store";
import { useFirmStore } from "@/stores/firm.store";
import {
  getMatterTeam, addMatterTeam, removeMatterTeam, listMembers, setReviewChain,
  APIError, type MatterTeamMember, type MemberResponse,
} from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { Select } from "@/components/ui/Select";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/Dialog";
import { ROLE_LABEL, ROLE_RANK, memberLabel } from "@/app/app/settings/firm/_shared";
import type { FirmRole } from "@/lib/api";

export function MatterTeamPanel({ vaultId }: { vaultId: string }) {
  const token = useAuthStore((s) => s.token);
  const { can, initialized, init, onForbidden } = useFirmStore();
  const [team, setTeam] = useState<MatterTeamMember[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { if (token && !initialized) init(token); }, [token, initialized, init]);

  const reload = useCallback(async () => {
    if (!token) return;
    setLoading(true); setErr(null);
    try {
      setTeam(await getMatterTeam(token, vaultId));
    } catch (e) {
      // A non-firm / legacy vault returns 403/404 from the team route — treat as "no team yet",
      // not an error wall (the matter just isn't firm-staffed).
      if (e instanceof APIError && (e.status === 403 || e.status === 404)) setTeam([]);
      else setErr("Could not load the matter team.");
    } finally { setLoading(false); }
  }, [token, vaultId]);

  useEffect(() => { reload(); }, [reload]);

  const canStaff = can("manage_matter_team");

  async function remove(userId: string, label: string) {
    if (!token) return;
    const prev = team;
    setTeam((t) => t.filter((m) => m.user_id !== userId));   // optimistic
    try {
      await removeMatterTeam(token, vaultId, userId);
      toast.success(`Removed ${label} from this matter.`);
    } catch (e) {
      setTeam(prev);                                         // roll back (F1e lesson)
      const was403 = await onForbidden(token, e);
      toast.error(was403 ? "Your permission changed. Refreshed." :
        (e instanceof APIError ? e.detail : "Could not remove from the matter."));
    }
  }

  return (
    <div className="rounded-2xl p-4" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-[12px] font-medium uppercase tracking-[0.12em] inline-flex items-center gap-2" style={{ color: "var(--ink-3)" }}>
          <Users size={13} /> Matter team
        </h3>
        <div className="flex items-center gap-2">
          {canStaff && team.length > 1 && (
            <CustomizeChainDialog vaultId={vaultId} team={team} />
          )}
          {canStaff && <AddTeamDialog vaultId={vaultId} team={team} onDone={reload} />}
        </div>
      </div>

      {loading ? (
        <div className="space-y-2">{[0, 1].map((i) => <Skeleton key={i} className="h-10 w-full rounded-lg" />)}</div>
      ) : err ? (
        <div className="flex items-center justify-between text-[12px]" style={{ color: "var(--ink-2)" }}>
          <span>{err}</span>
          <button onClick={reload} className="font-medium" style={{ color: "var(--ink)" }}>Retry</button>
        </div>
      ) : team.length === 0 ? (
        <p className="text-[12px]" style={{ color: "var(--ink-3)" }}>
          {canStaff
            ? "No one is staffed yet. Add a colleague to give them full access on this matter."
            : "No one is staffed on this matter yet."}
        </p>
      ) : (
        <>
          <div className="space-y-1.5">
            {team.map((m) => (
              <div key={m.user_id} className="flex items-center gap-2.5 px-2 py-1.5 rounded-lg"
                   style={{ background: "var(--surface-2)" }}>
                <div className="w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-[9px] font-semibold"
                     style={{ background: "var(--surface-3)", color: "var(--ink-2)" }}>
                  {(m.user_id).slice(0, 2).toUpperCase()}
                </div>
                <span className="text-[12px] flex-1 truncate" style={{ color: "var(--ink)" }}>
                  {memberLabel(m)}
                </span>
                {m.role && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: "var(--surface-3)", color: "var(--ink-3)" }}>
                    {ROLE_LABEL[m.role as FirmRole] ?? m.role}
                  </span>
                )}
                {canStaff && (
                  <button onClick={() => remove(m.user_id, memberLabel(m))}
                    className="p-1 rounded hover:bg-[var(--bg-hover)] transition-transform active:scale-[0.96]" style={{ color: "var(--ink-3)" }}
                    aria-label="Remove from matter">
                    <Trash2 size={12} />
                  </button>
                )}
              </div>
            ))}
          </div>
          {/* D0 made visible — the positive grant, stated plainly. */}
          <p className="text-[11px] mt-3 inline-flex items-center gap-1.5" style={{ color: "var(--ink-2)" }}>
            <Check size={12} style={{ color: "var(--fidelity-good)" }} />
            Everyone here has full access on this matter.
          </p>
        </>
      )}
    </div>
  );
}

function AddTeamDialog({
  vaultId, team, onDone,
}: { vaultId: string; team: MatterTeamMember[]; onDone: () => void }) {
  const token = useAuthStore((s) => s.token);
  const { onForbidden } = useFirmStore();
  const [open, setOpen] = useState(false);
  const [members, setMembers] = useState<MemberResponse[]>([]);
  const [pick, setPick] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (open && token) listMembers(token).then(setMembers).catch(() => setMembers([]));
  }, [open, token]);

  const onTeam = new Set(team.map((m) => m.user_id));
  const candidates = members.filter((m) => !onTeam.has(m.user_id));

  async function submit() {
    if (!token || !pick) return;
    setBusy(true);
    try {
      await addMatterTeam(token, vaultId, pick);
      toast.success("Added to the matter. They now have full access on it.");
      setOpen(false); setPick("");
      onDone();
    } catch (e) {
      const was403 = await onForbidden(token, e);
      toast.error(was403 ? "Your permission changed. Refreshed." :
        (e instanceof APIError ? e.detail : "Could not add to the matter."));
    } finally { setBusy(false); }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
        <Plus size={13} className="mr-1" /> Add to team
      </Button>
      <DialogContent maxWidth="420px">
        <DialogHeader><DialogTitle>Add a colleague to this matter</DialogTitle></DialogHeader>
        <div className="px-5 py-5">
          <p className="text-[12px] mb-3" style={{ color: "var(--ink-3)" }}>
            They get full access on this matter right away. The matter boundary and the ethical
            walls are the controls, never a stripped-down toolkit.
          </p>
          <Select
            ariaLabel="Colleague to add" placeholder="Select a colleague…"
            value={pick} onChange={setPick}
            options={candidates.map((m) => ({
              value: m.user_id, label: memberLabel(m),
              hint: m.role ? ROLE_LABEL[m.role as FirmRole] : undefined,
            }))}
          />
        </div>
        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>Cancel</Button>
          <Button variant="primary" size="sm" loading={busy} disabled={!pick} onClick={submit}>Add</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// Surface 6 — customize the review chain (matter leads only). The default chain routes UP by rank;
// for a big matter, a lead can pin an explicit ordered reviewer list. Empty/clear → rank default.
function CustomizeChainDialog({
  vaultId, team,
}: { vaultId: string; team: MatterTeamMember[] }) {
  const token = useAuthStore((s) => s.token);
  const { onForbidden } = useFirmStore();
  const [open, setOpen] = useState(false);
  const [order, setOrder] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  // Default ordering preview = up by rank (most-junior first → most-senior last), matching the
  // server's rank default so the lead sees what "default" means before they customize it.
  const byRank = [...team].sort((a, b) =>
    (ROLE_RANK[(b.role as FirmRole)] ?? 99) - (ROLE_RANK[(a.role as FirmRole)] ?? 99));

  function toggle(uid: string) {
    setOrder((prev) => prev.includes(uid) ? prev.filter((x) => x !== uid) : [...prev, uid]);
  }

  async function save(clear = false) {
    if (!token) return;
    setBusy(true);
    try {
      await setReviewChain(token, vaultId, clear ? null : order);
      toast.success(clear ? "Reverted to the default chain (up by rank)." : "Custom review chain saved.");
      setOpen(false); setOrder([]);
    } catch (e) {
      const was403 = await onForbidden(token, e);
      toast.error(was403 ? "Your permission changed. Refreshed." :
        (e instanceof APIError ? e.detail : "Could not save the chain."));
    } finally { setBusy(false); }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <Button variant="ghost" size="sm" onClick={() => setOpen(true)}>
        <ListOrdered size={13} className="mr-1" /> Customize chain
      </Button>
      <DialogContent>
        <DialogHeader><DialogTitle>Customize the review chain</DialogTitle></DialogHeader>
        <div className="px-5 py-5 overflow-y-auto">
          <p className="text-[12px] mb-3" style={{ color: "var(--ink-3)" }}>
            By default, work routes up by rank: {byRank.map((m) => memberLabel(m)).join(" → ")}.
            For this matter, click reviewers in the order work should flow.
          </p>
          <div className="space-y-1.5">
            {team.map((m) => {
              const idx = order.indexOf(m.user_id);
              const on = idx >= 0;
              return (
                <button key={m.user_id} onClick={() => toggle(m.user_id)}
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-colors"
                  style={{
                    background: on ? "var(--ink)" : "var(--surface-2)",
                    color: on ? "var(--on-ink)" : "var(--ink-2)",
                    border: `1px solid ${on ? "var(--ink)" : "var(--line)"}`,
                  }}>
                  <span className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0"
                        style={{ background: on ? "var(--on-ink)" : "var(--surface-3)", color: on ? "var(--ink)" : "var(--ink-3)" }}>
                    {on ? idx + 1 : ""}
                  </span>
                  <span className="text-[12px] flex-1 truncate">{memberLabel(m)}</span>
                  {m.role && <span className="text-[10px]" style={{ opacity: 0.7 }}>{ROLE_LABEL[m.role as FirmRole] ?? m.role}</span>}
                </button>
              );
            })}
          </div>
        </div>
        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={() => save(true)}>Use default</Button>
          <Button variant="primary" size="sm" loading={busy} disabled={order.length === 0} onClick={() => save(false)}>
            Save chain
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
