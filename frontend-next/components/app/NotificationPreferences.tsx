"use client";

// F2j — NotificationPreferences: the anti-nag controls (mute a category / quiet hours / daily digest).
// A self-contained section mounted on the Settings page. Loads + saves its own state; optimistic with
// rollback on failure (the F1e delete-404 lesson). Dormant-safe: when migration 017 is unapplied the
// API returns defaults and a save is a best-effort no-op — the UI never breaks.
//
// DESIGN: monochrome ink, neutral chrome (preferences are not a trust state — no trust color). Every
// state designed: loading skeleton, error toast, optimistic toggles that reconcile.

import { useEffect, useState, useCallback } from "react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/auth.store";
import {
  getNotificationPreferences, setNotificationPreferences,
  type NotificationPreferences as Prefs,
} from "@/lib/api";

const ease = [0.16, 1, 0.3, 1] as const;

// The three notification categories (lockstep with notifications.CATEGORIES on the server).
const CATEGORIES: { key: string; label: string; hint: string }[] = [
  { key: "review", label: "Review chain", hint: "Work awaiting you, approvals, changes, releases" },
  { key: "matter", label: "Matter staffing", hint: "When you're added to a matter" },
  { key: "governance", label: "Governance", hint: "Abstain overrides on your matters" },
];

const HOURS = Array.from({ length: 24 }, (_, h) => h);

function hourLabel(h: number): string {
  const am = h < 12;
  const base = h % 12 === 0 ? 12 : h % 12;
  return `${base}:00 ${am ? "AM" : "PM"}`;
}

export function NotificationPreferences() {
  const token = useAuthStore((s) => s.token);
  const [prefs, setPrefs] = useState<Prefs | null>(null);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    if (!token) return;
    try {
      setPrefs(await getNotificationPreferences(token));
    } catch {
      // Degrade to defaults — the section still renders, just at the "notify normally" baseline.
      setPrefs({ muted_categories: [], quiet_start: null, quiet_end: null, digest_mode: false });
    }
  }, [token]);

  useEffect(() => { load(); }, [load]);

  // Persist a patch optimistically; roll back + toast on failure.
  const save = useCallback(async (patch: Partial<Prefs>) => {
    if (!token || !prefs) return;
    const prev = prefs;
    const next = { ...prefs, ...patch };
    setPrefs(next);
    setSaving(true);
    try {
      const saved = await setNotificationPreferences(token, patch);
      setPrefs(saved);
    } catch {
      setPrefs(prev);
      toast.error("Could not save notification preferences.");
    } finally {
      setSaving(false);
    }
  }, [token, prefs]);

  function toggleCategory(key: string) {
    if (!prefs) return;
    const muted = new Set(prefs.muted_categories);
    if (muted.has(key)) muted.delete(key);
    else muted.add(key);
    save({ muted_categories: Array.from(muted) });
  }

  const quietOn = prefs != null && prefs.quiet_start != null && prefs.quiet_end != null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ ease, duration: 0.5 }}
      className="p-5 rounded-2xl"
      style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-sm)" }}
    >
      {prefs === null ? (
        // Loading skeleton.
        <div className="space-y-3">
          {[0, 1, 2].map((i) => (
            <div key={i} className="h-9 rounded-md animate-pulse" style={{ background: "var(--surface-3)" }} />
          ))}
        </div>
      ) : (
        <div className="space-y-5">
          {/* Per-category mute */}
          <div className="space-y-2.5">
            {CATEGORIES.map((c) => {
              const on = !prefs.muted_categories.includes(c.key);
              return (
                <div key={c.key} className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-[13px] font-medium" style={{ color: "var(--ink)" }}>{c.label}</p>
                    <p className="text-[11px]" style={{ color: "var(--ink-3)" }}>{c.hint}</p>
                  </div>
                  <button
                    onClick={() => toggleCategory(c.key)}
                    disabled={saving}
                    role="switch"
                    aria-checked={on}
                    aria-label={`${c.label} notifications`}
                    className="relative w-10 h-[22px] rounded-full transition-colors flex-shrink-0 active:scale-[0.97] disabled:opacity-50"
                    style={{ background: on ? "var(--ink)" : "var(--surface-3)", border: "1px solid var(--line)" }}
                  >
                    <span
                      className="absolute top-[2px] w-[16px] h-[16px] rounded-full transition-all"
                      style={{ left: on ? "20px" : "2px", background: on ? "var(--on-ink)" : "var(--ink-3)" }}
                    />
                  </button>
                </div>
              );
            })}
          </div>

          {/* Quiet hours */}
          <div className="pt-4 border-t" style={{ borderColor: "var(--line)" }}>
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-[13px] font-medium" style={{ color: "var(--ink)" }}>Quiet hours</p>
                <p className="text-[11px]" style={{ color: "var(--ink-3)" }}>
                  Hold email during these hours. In-app notifications still arrive.
                </p>
              </div>
              <button
                onClick={() => save(quietOn ? { quiet_start: null, quiet_end: null } : { quiet_start: 21, quiet_end: 7 })}
                disabled={saving}
                role="switch"
                aria-checked={quietOn}
                aria-label="Quiet hours"
                className="relative w-10 h-[22px] rounded-full transition-colors flex-shrink-0 active:scale-[0.97] disabled:opacity-50"
                style={{ background: quietOn ? "var(--ink)" : "var(--surface-3)", border: "1px solid var(--line)" }}
              >
                <span
                  className="absolute top-[2px] w-[16px] h-[16px] rounded-full transition-all"
                  style={{ left: quietOn ? "20px" : "2px", background: quietOn ? "var(--on-ink)" : "var(--ink-3)" }}
                />
              </button>
            </div>
            {quietOn && (
              <div className="flex items-center gap-2 mt-3 text-[12px]" style={{ color: "var(--ink-2)" }}>
                <span>From</span>
                <select
                  value={prefs.quiet_start ?? 21}
                  onChange={(e) => save({ quiet_start: Number(e.target.value) })}
                  className="px-2 py-1 rounded-md bg-[var(--bg-surface)] border border-[var(--line)] text-[12px]"
                >
                  {HOURS.map((h) => <option key={h} value={h}>{hourLabel(h)}</option>)}
                </select>
                <span>to</span>
                <select
                  value={prefs.quiet_end ?? 7}
                  onChange={(e) => save({ quiet_end: Number(e.target.value) })}
                  className="px-2 py-1 rounded-md bg-[var(--bg-surface)] border border-[var(--line)] text-[12px]"
                >
                  {HOURS.map((h) => <option key={h} value={h}>{hourLabel(h)}</option>)}
                </select>
              </div>
            )}
          </div>

          {/* Daily digest */}
          <div className="pt-4 border-t flex items-center justify-between gap-3" style={{ borderColor: "var(--line)" }}>
            <div>
              <p className="text-[13px] font-medium" style={{ color: "var(--ink)" }}>Daily digest</p>
              <p className="text-[11px]" style={{ color: "var(--ink-3)" }}>
                Batch low-signal updates into one daily email instead of one per event.
              </p>
            </div>
            <button
              onClick={() => save({ digest_mode: !prefs.digest_mode })}
              disabled={saving}
              role="switch"
              aria-checked={prefs.digest_mode}
              aria-label="Daily digest"
              className="relative w-10 h-[22px] rounded-full transition-colors flex-shrink-0 active:scale-[0.97] disabled:opacity-50"
              style={{ background: prefs.digest_mode ? "var(--ink)" : "var(--surface-3)", border: "1px solid var(--line)" }}
            >
              <span
                className="absolute top-[2px] w-[16px] h-[16px] rounded-full transition-all"
                style={{ left: prefs.digest_mode ? "20px" : "2px", background: prefs.digest_mode ? "var(--on-ink)" : "var(--ink-3)" }}
              />
            </button>
          </div>
        </div>
      )}
    </motion.div>
  );
}
