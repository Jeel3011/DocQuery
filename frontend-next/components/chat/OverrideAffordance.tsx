"use client";

// components/chat/OverrideAffordance.tsx — F2g surface 8 (the abstain-override moment, F2d/T6).
// Load-bearing. When the agent ABSTAINS on an answer, a partner who holds `override_abstain` may
// consciously take accountability for releasing it. This is a HIGH-TRUST event:
//   • the affordance renders ONLY for a holder of the cap (the store's server-resolved caps) — a
//     non-partner sees the abstain with NO override control (the POST is the gate, not the dialog);
//   • the dialog STATES WHAT THE GATE OBJECTED TO before you can confirm, and requires a reason;
//   • on confirm the answer flips to the reserved "Overridden" trust state (the indigo token, never
//     green — it was NOT auto-verified) and the server writes the F3 hash-chain audit row.

import { useEffect, useState } from "react";
import { ShieldAlert, Lock } from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/auth.store";
import { useFirmStore } from "@/stores/firm.store";
import { overrideAbstain, APIError } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/Dialog";

export type OverrideState = "abstained" | "overridden";

export function OverrideAffordance({
  answerRef, collectionId, gateObjection, onOverridden,
}: {
  answerRef: string;
  collectionId: string;
  gateObjection?: string;
  onOverridden?: () => void;
}) {
  const token = useAuthStore((s) => s.token);
  const { can, initialized, init, onForbidden } = useFirmStore();
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [state, setState] = useState<OverrideState>("abstained");

  useEffect(() => { if (token && !initialized) init(token); }, [token, initialized, init]);

  // Already overridden — show the reserved trust state, no further action.
  if (state === "overridden") {
    return (
      <span className="inline-flex items-center gap-1.5 text-[11px] font-medium px-2 py-0.5 rounded-md"
        style={{ background: "rgba(67,56,202,0.08)", color: "var(--fidelity-overridden)", border: "1px solid rgba(67,56,202,0.25)" }}>
        <ShieldAlert size={11} /> Overridden by partner
      </span>
    );
  }

  // A non-holder sees the abstain with NO override affordance (the cap is the gate). Render nothing
  // until caps are resolved, then nothing for a non-holder.
  if (!initialized || !can("override_abstain")) return null;

  async function confirm() {
    if (!token || !reason.trim()) return;
    setBusy(true);
    try {
      await overrideAbstain(token, answerRef, collectionId, reason.trim(), gateObjection);
      setState("overridden");
      setOpen(false); setReason("");
      onOverridden?.();
      toast.success("Override recorded. You now own this answer.");
    } catch (e) {
      const was403 = await onForbidden(token, e);
      // A 403 here is most often an ethical wall on this vault (a screen beats even a partner's
      // override grant) — surface the server's reason, never silently swallow it.
      toast.error(was403 ? "You cannot override on this matter." :
        (e instanceof APIError ? e.detail : "Could not record the override."));
    } finally { setBusy(false); }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <button onClick={() => setOpen(true)}
        className="inline-flex items-center gap-1.5 text-[11px] font-medium px-2 py-0.5 rounded-md transition-[color,background-color,transform] active:scale-[0.96]"
        style={{ color: "var(--fidelity-overridden)", border: "1px solid rgba(67,56,202,0.3)", background: "rgba(67,56,202,0.05)" }}>
        <ShieldAlert size={11} /> Override (partner)
      </button>
      <DialogContent maxWidth="460px">
        <DialogHeader><DialogTitle>Override this abstain</DialogTitle></DialogHeader>
        <div className="px-5 py-5 space-y-4">
          {/* State the gate's objection BEFORE confirm — nothing consequential without understanding. */}
          <div className="p-3 rounded-lg flex items-start gap-2.5" style={{ background: "var(--surface-2)", border: "1px solid var(--line)" }}>
            <Lock size={14} style={{ color: "var(--ink-2)" }} className="mt-0.5 shrink-0" />
            <div>
              <p className="text-[11px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--ink-3)" }}>
                What the gate objected to
              </p>
              <p className="text-[12px] mt-0.5" style={{ color: "var(--ink-2)" }}>
                {gateObjection || "The agent could not fully ground this answer and abstained."}
              </p>
            </div>
          </div>
          <p className="text-[12px]" style={{ color: "var(--ink-2)" }}>
            Overriding records that you, as a partner, take accountability for releasing this answer
            despite the abstain. It is logged with your reason and cannot be edited afterward.
          </p>
          <label className="block">
            <span className="text-[11px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--ink-3)" }}>
              Reason (required)
            </span>
            <textarea value={reason} onChange={(e) => setReason(e.target.value)} rows={3}
              placeholder="Why this answer is sound despite the gate's objection."
              className="mt-1 w-full px-3 py-2 rounded-lg text-[13px] outline-none resize-none"
              style={{ background: "var(--surface-2)", border: "1px solid var(--line-2)", color: "var(--ink)" }} />
          </label>
        </div>
        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>Cancel</Button>
          <Button variant="primary" size="sm" loading={busy} disabled={!reason.trim()} onClick={confirm}>
            Record override
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
