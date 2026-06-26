"use client";

// app/invite/page.tsx — PUBLIC invite landing (F2g). This route is intentionally OUTSIDE /app/*
// so the Next middleware does NOT bounce a logged-out invitee to /login and drop the token.
//
// Flow:
//   • Capture the one-time token from ?token= (accepts a bare token or a pasted full link).
//   • STASH it (firmSetup) so it survives the auth round-trip.
//   • If already signed in → go straight to the in-app accept page (which validates email-binding).
//   • If signed OUT → send to /login with a clear "log in or sign up to join" message; after auth,
//     the stashed token is applied on first app load (consumeStashedFirmSetup in the app layout),
//     OR the signup "Join with invite" field is pre-filled from the same stash.

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { motion } from "framer-motion";
import { Ticket, Loader2 } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { extractInviteToken } from "@/lib/api";
import { stashFirmSetup } from "@/lib/firmSetup";

const ease = [0.16, 1, 0.3, 1] as const;

function InviteInner() {
  const router = useRouter();
  const params = useSearchParams();
  const inviteToken = extractInviteToken(params.get("token") ?? "");
  const [msg, setMsg] = useState("Checking your invite…");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!inviteToken) { setMsg("This invite link is missing its token."); return; }
      // Stash so the token survives signup/login (applied on first authed load).
      stashFirmSetup({ inviteToken });
      const { data: { session } } = await supabase.auth.getSession();
      if (cancelled) return;
      if (session) {
        // Already signed in → validate + accept in-app (email-binding enforced server-side).
        router.replace(`/app/accept-invite?token=${encodeURIComponent(inviteToken)}`);
      } else {
        // Signed out → onboard. The signup "Join with invite" tab + the stash both carry the token.
        setMsg("Sign in or create your account with the invited email to join the firm.");
        setTimeout(() => router.replace("/login"), 1200);
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inviteToken]);

  return (
    <div className="min-h-dvh flex items-center justify-center p-4" style={{ background: "var(--bg-base)" }}>
      <motion.div
        initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ ease }}
        className="w-full max-w-md rounded-2xl p-8 text-center"
        style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-md)" }}>
        <div className="w-12 h-12 rounded-2xl flex items-center justify-center mx-auto mb-4"
             style={{ background: "var(--ink)", color: "var(--on-ink)" }}>
          <Ticket size={20} strokeWidth={1.7} />
        </div>
        <h1 className="text-[18px] font-semibold mb-2 [text-wrap:balance]" style={{ color: "var(--ink)" }}>
          You&apos;ve been invited to a firm
        </h1>
        <p className="text-[13px] inline-flex items-center gap-2" style={{ color: "var(--ink-3)" }}>
          <Loader2 size={13} className="animate-spin" /> {msg}
        </p>
      </motion.div>
    </div>
  );
}

export default function InvitePage() {
  return (
    <Suspense fallback={null}>
      <InviteInner />
    </Suspense>
  );
}
