"use client";

// app/login/page.tsx — Monochrome login
// White card, black buttons, dotted border accents, dot-grid background

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Eye, EyeOff, Loader2, Building2, Ticket } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { setupFirm, stashFirmSetup, peekStashedFirmSetup } from "@/lib/firmSetup";
import { extractInviteToken } from "@/lib/api";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";

const schema = z.object({
  email: z.string().email("Invalid email"),
  password: z.string().min(6, "At least 6 characters"),
  // F2g onboarding (signup only) — optional. Either name your new firm, or paste an invite to join
  // an existing one. Both blank ⇒ a solo firm named after you (the legacy default).
  firmName: z.string().max(120).optional(),
  inviteToken: z.string().optional(),
});
type Form = z.infer<typeof schema>;

export default function LoginPage() {
  const router = useRouter();
  const [tab, setTab] = useState<"signin" | "signup">("signin");
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [signedUp, setSignedUp] = useState(false);
  const [countdown, setCountdown] = useState(3);
  // Signup-only: create a new firm (name it) or join an existing one (invite token).
  const [joinMode, setJoinMode] = useState<"create" | "join">("create");

  const { register, handleSubmit, formState: { errors }, reset, setValue } = useForm<Form>({
    resolver: zodResolver(schema),
  });

  // If the user arrived from an invite link (/invite stashed the token), jump to the signup
  // "Join with invite" tab and pre-fill the token — they don't have to paste anything.
  useEffect(() => {
    const stashed = peekStashedFirmSetup();
    if (stashed?.inviteToken) {
      setTab("signup");
      setJoinMode("join");
      setValue("inviteToken", stashed.inviteToken);
    }
  }, [setValue]);

  // Reset form errors when switching tabs (#19)
  function switchTab(t: "signin" | "signup") {
    setTab(t);
    reset();
  }

  // Auto-redirect 3s after signup (#18)
  useEffect(() => {
    if (!signedUp) return;
    if (countdown <= 0) { router.push("/app"); return; }
    const t = setTimeout(() => setCountdown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [signedUp, countdown, router]);

  async function onSubmit(data: Form) {
    setLoading(true);
    try {
      if (tab === "signin") {
        const { data: result, error } = await supabase.auth.signInWithPassword(data);
        if (error) throw error;
        // If they arrived from an invite link, accept it now (email-binding enforced server-side).
        const tok = result.session?.access_token;
        const stashedToken = data.inviteToken ? extractInviteToken(data.inviteToken) : peekStashedFirmSetup()?.inviteToken;
        if (tok && stashedToken) await setupFirm(tok, { inviteToken: stashedToken });
        toast.success("Welcome back!");
        router.push("/app");
      } else {
        const { data: result, error } = await supabase.auth.signUp({
          email: data.email, password: data.password,
        });
        if (error) throw error;
        // F2g onboarding: name the new firm, or join one by invite. People paste the whole invite
        // LINK here, not the bare token — extractInviteToken handles either. If a session came back
        // (email confirmation off) apply it now; otherwise stash it for the first authed load.
        const intent = {
          firmName: data.firmName,
          inviteToken: data.inviteToken ? extractInviteToken(data.inviteToken) : undefined,
        };
        const token = result.session?.access_token;
        if (token) {
          // setupFirm THROWS on a rejected invite (fail-closed). If the join fails we must NOT
          // pretend it worked — the account exists, but they did not join the firm.
          try {
            await setupFirm(token, intent);
          } catch (joinErr) {
            const detail = (joinErr as { detail?: string })?.detail
              ?? "That invite could not be accepted (expired, already used, or a different email).";
            toast.error(`Account created, but ${detail}`);
            setSignedUp(true);
            return;
          }
        } else {
          stashFirmSetup(intent);
        }
        toast.success("Account created! Check your email to confirm.");
        setSignedUp(true);
      }
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Auth failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-dvh flex items-center justify-center p-4 dot-grid bg-[var(--bg-base)]">
      <motion.div
        initial={{ opacity: 0, y: 16, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.3 }}
        className="card p-8 w-full max-w-[420px] shadow-lg"
      >
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="w-11 h-11 rounded-xl bg-[var(--accent)] flex items-center justify-center mx-auto mb-4 shadow-md">
            <span className="text-lg font-bold text-white">D</span>
          </div>
          <h1 className="text-lg font-semibold text-[var(--text-primary)]">DocQuery</h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">Intelligent Document Q&A</p>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 p-1 bg-[var(--bg-base)] rounded-xl mb-6 border border-[var(--border)]">
          {(["signin", "signup"] as const).map((t) => (
            <button key={t} onClick={() => switchTab(t)}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition-[background-color,color] duration-[120ms]
                ${tab === t ? "bg-[var(--accent)] text-white shadow-sm" : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"}`}>
              {t === "signin" ? "Sign In" : "Sign Up"}
            </button>
          ))}
        </div>

        {/* Post-signup confirmation banner */}
        {signedUp && (
          <div className="mb-4 px-4 py-3 rounded-xl bg-[var(--bg-hover)] border border-[var(--border)] text-center">
            <p className="text-xs text-[var(--text-primary)] font-medium">Check your email to confirm your account.</p>
            <p className="text-[10px] text-[var(--text-muted)] mt-1">Redirecting in {countdown}s…</p>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <label className="block text-xs text-[var(--text-secondary)] mb-1.5 font-medium">Email</label>
            <input {...register("email")} type="email" autoComplete="email" placeholder="you@example.com"
              className={`w-full card rounded-xl px-4 py-3 text-sm text-[var(--text-primary)]
                placeholder:text-[var(--text-muted)] outline-none transition-[border-color,box-shadow]
                focus:border-[var(--accent)] focus:shadow-[0_0_0_3px_rgba(10,10,10,0.06)]
                ${errors.email ? "border-[var(--status-failed)]" : ""}`} />
            {errors.email && <p className="text-[10px] text-[var(--status-failed)] mt-1">{errors.email.message}</p>}
          </div>

          <div>
            <label className="block text-xs text-[var(--text-secondary)] mb-1.5 font-medium">Password</label>
            <div className="relative">
              <input {...register("password")} type={showPw ? "text" : "password"}
                autoComplete={tab === "signin" ? "current-password" : "new-password"} placeholder="••••••••"
                className={`w-full card rounded-xl px-4 py-3 pr-10 text-sm text-[var(--text-primary)]
                  placeholder:text-[var(--text-muted)] outline-none transition-[border-color,box-shadow]
                  focus:border-[var(--accent)] focus:shadow-[0_0_0_3px_rgba(10,10,10,0.06)]
                  ${errors.password ? "border-[var(--status-failed)]" : ""}`} />
              <button type="button" onClick={() => setShowPw(!showPw)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--text-secondary)]">
                {showPw ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
            </div>
            {errors.password && <p className="text-[10px] text-[var(--status-failed)] mt-1">{errors.password.message}</p>}
          </div>

          {/* F2g onboarding — only on signup. Choose to start a firm or join one by invite. */}
          <AnimatePresence initial={false}>
            {tab === "signup" && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.2, ease: [0.2, 0, 0, 1] }}
                className="overflow-hidden"
              >
                <div className="pt-1">
                  <label className="block text-xs text-[var(--text-secondary)] mb-1.5 font-medium">Your firm</label>
                  <div className="flex gap-1 p-1 bg-[var(--bg-base)] rounded-xl mb-3 border border-[var(--border)]">
                    {(["create", "join"] as const).map((m) => (
                      <button key={m} type="button" onClick={() => setJoinMode(m)}
                        className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-[background-color,color] duration-[120ms] active:scale-[0.96]
                          ${joinMode === m ? "bg-[var(--accent)] text-white shadow-sm" : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"}`}>
                        {m === "create" ? "Start a firm" : "Join with invite"}
                      </button>
                    ))}
                  </div>
                  {joinMode === "create" ? (
                    <div className="relative">
                      <Building2 size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
                      <input {...register("firmName")} type="text" placeholder="Firm name (optional)" maxLength={120}
                        className="w-full card rounded-xl pl-9 pr-4 py-3 text-sm text-[var(--text-primary)]
                          placeholder:text-[var(--text-muted)] outline-none transition-[border-color,box-shadow]
                          focus:border-[var(--accent)] focus:shadow-[0_0_0_3px_rgba(10,10,10,0.06)]" />
                    </div>
                  ) : (
                    <div className="relative">
                      <Ticket size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
                      <input {...register("inviteToken")} type="text" placeholder="Paste your invite link or token"
                        className="w-full card rounded-xl pl-9 pr-4 py-3 text-sm text-[var(--text-primary)]
                          placeholder:text-[var(--text-muted)] outline-none transition-[border-color,box-shadow]
                          focus:border-[var(--accent)] focus:shadow-[0_0_0_3px_rgba(10,10,10,0.06)]" />
                    </div>
                  )}
                  <p className="text-[10px] text-[var(--text-muted)] mt-1.5 leading-relaxed">
                    {joinMode === "create"
                      ? "You become the Managing Partner. You can rename the firm and invite your team later."
                      : "Join your firm at the role your inviter chose. The invite is tied to this email."}
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <button type="submit" disabled={loading}
            className="w-full btn-primary py-3 flex items-center justify-center gap-2 text-sm">
            {loading && <Loader2 size={14} className="animate-spin" />}
            {tab === "signin" ? "Sign In" : "Create Account"}
          </button>
        </form>

        <p className="text-center text-[11px] text-[var(--text-muted)] mt-6">
          {tab === "signin" ? (
            <>No account?{" "}<button onClick={() => switchTab("signup")} className="text-[var(--accent)] underline">Sign up</button></>
          ) : (
            <>Have an account?{" "}<button onClick={() => switchTab("signin")} className="text-[var(--accent)] underline">Sign in</button></>
          )}
        </p>
      </motion.div>
    </div>
  );
}
