"use client";

// app/login/page.tsx — Monochrome login
// White card, black buttons, dotted border accents, dot-grid background

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Eye, EyeOff, Loader2 } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { toast } from "sonner";
import { motion } from "framer-motion";

const schema = z.object({
  email: z.string().email("Invalid email"),
  password: z.string().min(6, "At least 6 characters"),
});
type Form = z.infer<typeof schema>;

export default function LoginPage() {
  const router = useRouter();
  const [tab, setTab] = useState<"signin" | "signup">("signin");
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);

  const { register, handleSubmit, formState: { errors } } = useForm<Form>({
    resolver: zodResolver(schema),
  });

  async function onSubmit(data: Form) {
    setLoading(true);
    try {
      if (tab === "signin") {
        const { error } = await supabase.auth.signInWithPassword(data);
        if (error) throw error;
        toast.success("Welcome back!");
        router.push("/app/chat");
      } else {
        const { error } = await supabase.auth.signUp(data);
        if (error) throw error;
        toast.success("Account created! Check your email.");
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
            <button key={t} onClick={() => setTab(t)}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all duration-150
                ${tab === t ? "bg-[var(--accent)] text-white shadow-sm" : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"}`}>
              {t === "signin" ? "Sign In" : "Sign Up"}
            </button>
          ))}
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <label className="block text-xs text-[var(--text-secondary)] mb-1.5 font-medium">Email</label>
            <input {...register("email")} type="email" autoComplete="email" placeholder="you@example.com"
              className={`w-full card rounded-xl px-4 py-3 text-sm text-[var(--text-primary)]
                placeholder:text-[var(--text-muted)] outline-none transition-all
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
                  placeholder:text-[var(--text-muted)] outline-none transition-all
                  focus:border-[var(--accent)] focus:shadow-[0_0_0_3px_rgba(10,10,10,0.06)]
                  ${errors.password ? "border-[var(--status-failed)]" : ""}`} />
              <button type="button" onClick={() => setShowPw(!showPw)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--text-secondary)]">
                {showPw ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
            </div>
            {errors.password && <p className="text-[10px] text-[var(--status-failed)] mt-1">{errors.password.message}</p>}
          </div>

          <button type="submit" disabled={loading}
            className="w-full btn-primary py-3 flex items-center justify-center gap-2 text-sm">
            {loading && <Loader2 size={14} className="animate-spin" />}
            {tab === "signin" ? "Sign In" : "Create Account"}
          </button>
        </form>

        <p className="text-center text-[11px] text-[var(--text-muted)] mt-6">
          {tab === "signin" ? (
            <>No account?{" "}<button onClick={() => setTab("signup")} className="text-[var(--accent)] underline">Sign up</button></>
          ) : (
            <>Have an account?{" "}<button onClick={() => setTab("signin")} className="text-[var(--accent)] underline">Sign in</button></>
          )}
        </p>
      </motion.div>
    </div>
  );
}
