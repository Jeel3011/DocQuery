"use client";

import { useState, useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Sparkles } from "lucide-react";
import { useProfileStore } from "@/stores/profile.store";
import { useAuthStore } from "@/stores/auth.store";
import { updatePreferredName } from "@/lib/api";

const ease = [0.16, 1, 0.3, 1] as const;

/* First-run prompt: asks the user what DocQuery should call them.
   Persisted server-side (Supabase user_metadata) so the assistant can use it,
   with the local store as an instant cache. Shows once. */
export function NamePrompt({ fallback }: { fallback: string }) {
  const { preferredName, asked, setPreferredName, markAsked } = useProfileStore();
  const { token } = useAuthStore();
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState("");
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Only prompt once, and only if they haven't set a name yet.
    if (!asked && !preferredName) {
      const t = setTimeout(() => setOpen(true), 600);
      return () => clearTimeout(t);
    }
  }, [asked, preferredName]);

  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 120);
  }, [open]);

  async function save() {
    const name = value.trim() || fallback;
    setSaving(true);
    // Optimistic local update so the UI reflects it immediately.
    setPreferredName(name);
    setOpen(false);
    // Persist server-side so the assistant can address the user by name.
    if (token) {
      try { await updatePreferredName(token, name); } catch { /* local cache still holds it */ }
    }
    setSaving(false);
  }
  function skip() {
    markAsked();
    setOpen(false);
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[400] flex items-center justify-center px-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0"
            style={{ background: "rgba(14,14,14,0.32)", backdropFilter: "blur(4px)" }}
            onClick={skip}
          />

          {/* Card */}
          <motion.div
            initial={{ opacity: 0, y: 16, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.98 }}
            transition={{ duration: 0.28, ease }}
            className="relative w-full max-w-sm p-7 rounded-[24px]"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--line)",
              boxShadow: "var(--shadow-xl)",
            }}
          >
            <div
              className="w-11 h-11 rounded-2xl flex items-center justify-center mb-5"
              style={{ background: "var(--ink)", color: "var(--on-ink)" }}
            >
              <Sparkles size={20} strokeWidth={1.7} />
            </div>

            <h2
              className="mb-2"
              style={{
                fontFamily: "Fraunces, Georgia, serif",
                fontSize: "23px",
                fontWeight: 400,
                letterSpacing: "-0.025em",
                color: "var(--ink)",
                lineHeight: 1.15,
              }}
            >
              What should I call you?
            </h2>
            <p className="text-[14px] leading-relaxed mb-5" style={{ color: "var(--ink-3)" }}>
              DocQuery will use this name when greeting you. You can change it anytime in Settings.
            </p>

            <input
              ref={inputRef}
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") save();
                if (e.key === "Escape") skip();
              }}
              placeholder={fallback}
              maxLength={40}
              className="w-full px-4 py-3 rounded-xl text-[15px] outline-none mb-4"
              style={{
                background: "var(--surface-2)",
                border: "1px solid var(--line)",
                color: "var(--ink)",
              }}
            />

            <div className="flex items-center gap-2.5">
              <button
                onClick={save}
                disabled={saving}
                className="flex-1 inline-flex items-center justify-center px-5 py-3 rounded-xl font-semibold text-[14px] disabled:opacity-60"
                style={{ background: "var(--ink)", color: "var(--on-ink)" }}
              >
                {saving ? "Saving…" : "Save"}
              </button>
              <button
                onClick={skip}
                className="px-5 py-3 rounded-xl font-medium text-[14px]"
                style={{ background: "var(--surface-3)", color: "var(--ink-2)", border: "1px solid var(--line)" }}
              >
                Skip
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
