"use client";

// components/app/ReviewChain.tsx — F2g surface 6 (the review chain, D5).
//   • SendForReview ... a button on a draft/answer/grid → submit UP the chain; shows the chain
//     preview ("goes to: Sr Assoc Priya → Partner Mehta") so the submitter sees who owns it next.
//   • ReviewQueue ..... a list of the requests I currently OWN, each with Approve / Request changes;
//     current_owner is ALWAYS shown by name (anti-stall made visible). At the chain's end a partner
//     sees Release externally (the one outbound gate, release_external cap).
//
// Security: send_for_review is held by everyone on a matter (D0). approve/changes require being the
// current_owner (server-checked). release is release_external (partners). The store caps decide what
// RENDERS; the route is the boundary; a 403 refetches caps (T7).

import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Send, ArrowRight, Check, RotateCcw, Upload, Inbox, AlertCircle } from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/auth.store";
import { useFirmStore } from "@/stores/firm.store";
import {
  submitForReview, getReviewQueue, approveReview, requestChanges, releaseExternal, listMembers,
  APIError, type ReviewRequest, type MemberResponse,
} from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/Dialog";
import { memberLabel } from "@/app/app/settings/firm/_shared";

const ease = [0.16, 1, 0.3, 1] as const;

function useMemberMap() {
  const token = useAuthStore((s) => s.token);
  const [map, setMap] = useState<Record<string, MemberResponse>>({});
  useEffect(() => {
    if (!token) return;
    listMembers(token)
      .then((ms) => setMap(Object.fromEntries(ms.map((m) => [m.user_id, m]))))
      .catch(() => {});
  }, [token]);
  return map;
}

// ── Send for review (on a draft/answer/grid) ─────────────────────────────────────────────────
export function SendForReview({
  vaultId, artifactRef, label = "Send for review",
}: { vaultId: string; artifactRef: string; label?: string }) {
  const token = useAuthStore((s) => s.token);
  const { can, initialized, init, onForbidden } = useFirmStore();
  const members = useMemberMap();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ReviewRequest | null>(null);

  useEffect(() => { if (token && !initialized) init(token); }, [token, initialized, init]);

  if (initialized && !can("send_for_review")) return null;   // not a renderer for external/guest

  const name = (uid: string | null) => (uid ? memberLabel(members[uid] ?? { user_id: uid }) : "—");

  async function submit() {
    if (!token) return;
    setBusy(true);
    try {
      const req = await submitForReview(token, vaultId, artifactRef);
      setResult(req);
      toast.success("Sent for review");
    } catch (e) {
      const was403 = await onForbidden(token, e);
      toast.error(was403 ? "Your permission changed. Refreshed." :
        (e instanceof APIError ? e.detail : "Could not send for review."));
    } finally { setBusy(false); }
  }

  return (
    <Dialog open={open} onOpenChange={(v) => { setOpen(v); if (!v) setTimeout(() => setResult(null), 200); }}>
      <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
        <Send size={13} className="mr-1.5" /> {label}
      </Button>
      <DialogContent maxWidth="440px">
        <DialogHeader><DialogTitle>{result ? "Sent for review" : "Send this for review"}</DialogTitle></DialogHeader>
        <div className="px-5 py-5">
          {result ? (
            <div className="space-y-3">
              <p className="text-[13px]" style={{ color: "var(--ink-2)" }}>
                It now sits with <strong style={{ color: "var(--ink)" }}>{name(result.current_owner)}</strong>.
                They own the next step.
              </p>
              {result.chain.length > 0 && (
                <ChainPreview chain={result.chain} ownerId={result.current_owner} nameOf={name} />
              )}
            </div>
          ) : (
            <p className="text-[13px]" style={{ color: "var(--ink-2)" }}>
              This routes the work up your matter&apos;s review chain. The next reviewer owns it; you can
              see where it stands in your review queue at every step.
            </p>
          )}
        </div>
        <DialogFooter>
          {result ? (
            <Button variant="primary" size="sm" onClick={() => setOpen(false)}>Done</Button>
          ) : (
            <>
              <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>Cancel</Button>
              <Button variant="primary" size="sm" loading={busy} onClick={submit}>Send</Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ChainPreview({
  chain, ownerId, nameOf,
}: { chain: string[]; ownerId: string | null; nameOf: (uid: string | null) => string }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 p-2.5 rounded-lg" style={{ background: "var(--surface-2)" }}>
      {chain.map((uid, i) => (
        <span key={uid} className="inline-flex items-center gap-1.5">
          {i > 0 && <ArrowRight size={11} style={{ color: "var(--ink-3)" }} />}
          <span className="text-[11px] px-1.5 py-0.5 rounded-md"
            style={{
              background: uid === ownerId ? "var(--ink)" : "var(--surface-3)",
              color: uid === ownerId ? "var(--on-ink)" : "var(--ink-2)",
              fontWeight: uid === ownerId ? 600 : 400,
            }}>
            {nameOf(uid)}
          </span>
        </span>
      ))}
    </div>
  );
}

// ── My Review Queue (a list surface) ─────────────────────────────────────────────────────────
const STATUS_LABEL: Record<ReviewRequest["status"], string> = {
  pending: "Awaiting your review",
  approved: "Approved — ready to release",
  changes_requested: "Changes requested",
  released: "Released externally",
};

export function ReviewQueue() {
  const token = useAuthStore((s) => s.token);
  const me = useAuthStore((s) => s.user);
  const { can, initialized, init, onForbidden } = useFirmStore();
  const members = useMemberMap();
  const [queue, setQueue] = useState<ReviewRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { if (token && !initialized) init(token); }, [token, initialized, init]);

  const reload = useCallback(async () => {
    if (!token) return;
    setLoading(true); setErr(null);
    try { setQueue(await getReviewQueue(token)); }
    catch { setErr("Could not load your review queue."); }
    finally { setLoading(false); }
  }, [token]);

  useEffect(() => { reload(); }, [reload]);

  const name = (uid: string | null) => (uid ? memberLabel(members[uid] ?? { user_id: uid }) : "—");
  const canRelease = can("release_external");

  async function act(fn: () => Promise<ReviewRequest>, ok: string) {
    if (!token) return;
    try { await fn(); toast.success(ok); await reload(); }
    catch (e) {
      const was403 = await onForbidden(token, e);
      toast.error(was403 ? "Your permission changed. Refreshed." :
        (e instanceof APIError ? e.detail : "That action could not complete."));
      await reload();
    }
  }

  if (loading && queue.length === 0) {
    return <div className="space-y-2">{[0, 1].map((i) => <Skeleton key={i} className="h-20 w-full rounded-xl" />)}</div>;
  }
  if (err) {
    return (
      <div className="p-5 rounded-xl flex items-center gap-3" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
        <AlertCircle size={16} style={{ color: "var(--fidelity-failed)" }} />
        <p className="text-[13px] flex-1" style={{ color: "var(--ink-2)" }}>{err}</p>
        <Button variant="outline" size="sm" onClick={reload}>Retry</Button>
      </div>
    );
  }
  if (queue.length === 0) {
    return (
      <EmptyState
        icon={<Inbox size={18} />}
        title="Your review queue is clear"
        description="When a colleague sends work up the chain to you, it shows here with one next action. Nothing is waiting on you right now."
      />
    );
  }

  return (
    <div className="space-y-3">
      {queue.map((r, i) => {
        const owned = r.current_owner === me?.id;
        const atChainEnd = r.status === "approved";
        return (
          <motion.div key={r.id}
            initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
            transition={{ ease, delay: Math.min(i * 0.04, 0.2) }}
            className="rounded-2xl p-4" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-[13px] font-medium truncate" style={{ color: "var(--ink)" }}>{r.artifact_ref}</p>
                <p className="text-[11px] mt-0.5" style={{ color: "var(--ink-3)" }}>
                  From {name(r.submitted_by)} · {STATUS_LABEL[r.status]}
                </p>
              </div>
              {/* current_owner ALWAYS shown by name — the anti-stall property made visible */}
              <span className="text-[10px] px-2 py-0.5 rounded-md shrink-0"
                    style={{ background: "var(--surface-3)", color: "var(--ink-2)" }}>
                Owner: {name(r.current_owner)}
              </span>
            </div>
            {r.chain.length > 0 && (
              <div className="mt-3"><ChainPreview chain={r.chain} ownerId={r.current_owner} nameOf={name} /></div>
            )}
            {owned && r.status !== "released" && (
              <div className="flex items-center gap-2 mt-3">
                {atChainEnd ? (
                  canRelease ? (
                    <Button variant="primary" size="sm" onClick={() => act(() => releaseExternal(token!, r.id), "Released externally")}>
                      <Upload size={13} className="mr-1.5" /> Release externally
                    </Button>
                  ) : (
                    <span className="text-[11px]" style={{ color: "var(--ink-3)" }}>
                      Approved. A partner releases it outside the firm.
                    </span>
                  )
                ) : (
                  <>
                    <Button variant="primary" size="sm" onClick={() => act(() => approveReview(token!, r.id), "Approved")}>
                      <Check size={13} className="mr-1.5" /> Approve
                    </Button>
                    <Button variant="ghost" size="sm" onClick={() => act(() => requestChanges(token!, r.id), "Changes requested")}>
                      <RotateCcw size={13} className="mr-1.5" /> Request changes
                    </Button>
                  </>
                )}
              </div>
            )}
          </motion.div>
        );
      })}
    </div>
  );
}
