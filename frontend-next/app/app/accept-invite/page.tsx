"use client";

// app/app/accept-invite/page.tsx — F2g surface 1 (onboarding: the accept-invite landing).
// A user who received an invite token lands here (logged in — their VERIFIED email must match the
// invite, enforced server-side, T4). The token comes from the ?token= query. On accept they see a
// clear "you're joining <firm> as <role>" confirmation, then enter the app re-scoped to the firm.

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { motion } from "framer-motion";
import { Building2, Check, AlertCircle, Loader2 } from "lucide-react";
import { useAuthStore } from "@/stores/auth.store";
import { useFirmStore } from "@/stores/firm.store";
import { acceptInvite, extractInviteToken, APIError, type Firm } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { ROLE_LABEL } from "@/app/app/settings/firm/_shared";
import type { FirmRole } from "@/lib/api";

const ease = [0.16, 1, 0.3, 1] as const;

function AcceptInviteInner() {
  const router = useRouter();
  const params = useSearchParams();
  const token = useAuthStore((s) => s.token);
  const initFirm = useFirmStore((s) => s.init);
  // Accept either a bare token or a full link that got pasted into ?token= (double-paste).
  const inviteToken = extractInviteToken(params.get("token") ?? "");

  const [status, setStatus] = useState<"idle" | "accepting" | "done" | "error">("idle");
  const [firm, setFirm] = useState<Firm | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function accept() {
    if (!token || !inviteToken) return;
    setStatus("accepting"); setError(null);
    try {
      const f = await acceptInvite(token, inviteToken);
      setFirm(f);
      setStatus("done");
      await initFirm(token);   // re-scope caps/firm to the firm just joined
    } catch (e) {
      setStatus("error");
      const fallback = "This invite could not be accepted. It may have expired, already been used, or be addressed to a different email.";
      setError((e instanceof APIError ? e.detail : null) ?? fallback);
    }
  }

  // Auto-attempt once we have a token + a session (the common path: click the link → land here).
  useEffect(() => {
    if (token && inviteToken && status === "idle") accept();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, inviteToken]);

  return (
    <div className="flex-1 flex items-center justify-center px-4" style={{ background: "var(--canvas)" }}>
      <motion.div
        initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ ease }}
        className="w-full max-w-md rounded-2xl p-8 text-center"
        style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-md)" }}>

        {!inviteToken ? (
          <>
            <AlertCircle size={28} className="mx-auto mb-3" style={{ color: "var(--fidelity-partial)" }} />
            <h1 className="text-[17px] font-semibold mb-1" style={{ color: "var(--ink)" }}>No invite token</h1>
            <p className="text-[13px] mb-5" style={{ color: "var(--ink-3)" }}>
              This link is missing its invite token. Ask whoever invited you to resend the full link.
            </p>
            <Button variant="outline" size="sm" onClick={() => router.push("/app")}>Go to the app</Button>
          </>
        ) : status === "accepting" || status === "idle" ? (
          <>
            <Loader2 size={28} className="mx-auto mb-3 animate-spin" style={{ color: "var(--ink-2)" }} />
            <h1 className="text-[17px] font-semibold mb-1" style={{ color: "var(--ink)" }}>Joining your firm…</h1>
            <p className="text-[13px]" style={{ color: "var(--ink-3)" }}>Confirming your invite.</p>
          </>
        ) : status === "done" && firm ? (
          <>
            <div className="w-12 h-12 rounded-2xl flex items-center justify-center mx-auto mb-4"
                 style={{ background: "var(--ink)", color: "var(--on-ink)" }}>
              <Building2 size={22} strokeWidth={1.7} />
            </div>
            <h1 className="text-[18px] font-semibold mb-1 [text-wrap:balance]" style={{ color: "var(--ink)" }}>
              You&apos;ve joined {firm.name}
            </h1>
            <p className="text-[13px] mb-5 inline-flex items-center gap-1.5" style={{ color: "var(--ink-2)" }}>
              <Check size={13} style={{ color: "var(--fidelity-good)" }} />
              as {firm.role ? ROLE_LABEL[firm.role as FirmRole] : "a member"}
            </p>
            <div><Button variant="primary" size="sm" onClick={() => router.push("/app")}>Enter the workspace</Button></div>
          </>
        ) : (
          <>
            <AlertCircle size={28} className="mx-auto mb-3" style={{ color: "var(--fidelity-failed)" }} />
            <h1 className="text-[17px] font-semibold mb-1" style={{ color: "var(--ink)" }}>Couldn&apos;t accept the invite</h1>
            <p className="text-[13px] mb-5" style={{ color: "var(--ink-3)" }}>{error}</p>
            <div className="flex items-center justify-center gap-2">
              <Button variant="ghost" size="sm" onClick={() => router.push("/app")}>Go to the app</Button>
              <Button variant="outline" size="sm" onClick={accept}>Try again</Button>
            </div>
          </>
        )}
      </motion.div>
    </div>
  );
}

export default function AcceptInvitePage() {
  return (
    <Suspense fallback={null}>
      <AcceptInviteInner />
    </Suspense>
  );
}
