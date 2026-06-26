"use client";

// F2j — NotificationBell: the in-app inbox surface (the bell + dropdown) in the TopBar.
//
// Makes the F2e review chain VISIBLE: when work lands in your queue, you're staffed onto a matter, or
// your work is approved/returned/released, a notification appears here. v1 is IN-APP ONLY (email is a
// later seam) — the bell polls the cheap unread-count, and opening the dropdown loads the inbox.
//
// DESIGN (DESIGN.md): monochrome ink; the unread badge is the ONE accent (it's an attention signal,
// not a trust state — no trust color). Every state designed: loading skeleton, empty ("all caught
// up"), error degrades SILENTLY (the bell is non-critical chrome — it shows 0, never a broken UI).
// Named transition, active:scale, useReducedMotion-friendly via framer's reduced-motion handling.

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { Bell, CheckCheck } from "lucide-react";
import { useAuthStore } from "@/stores/auth.store";
import {
  getUnreadCount, getNotifications, markNotificationsRead,
  type AppNotification,
} from "@/lib/api";

const POLL_MS = 30_000; // the bell badge refresh cadence (cheap unread-count; no SSE for v1)

function relativeTime(iso: string | null): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const secs = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (secs < 60) return "just now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export function NotificationBell() {
  const router = useRouter();
  const token = useAuthStore((s) => s.token);
  const [open, setOpen] = useState(false);
  const [unread, setUnread] = useState(0);
  const [items, setItems] = useState<AppNotification[] | null>(null); // null = not loaded yet
  const [loading, setLoading] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Poll the cheap unread count on an interval (the badge). Degrades silently to 0 on error.
  useEffect(() => {
    if (!token) return;
    let alive = true;
    const tick = async () => {
      const n = await getUnreadCount(token);
      if (alive) setUnread(n);
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => { alive = false; clearInterval(id); };
  }, [token]);

  const loadInbox = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const res = await getNotifications(token, false, 20);
      setItems(res.notifications);
      setUnread(res.unread);
    } catch {
      setItems([]); // silent degrade — non-critical chrome
    } finally {
      setLoading(false);
    }
  }, [token]);

  // Open the dropdown → load the inbox.
  function toggle() {
    const next = !open;
    setOpen(next);
    if (next) loadInbox();
  }

  // Close on outside-click / Escape.
  useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) { if (e.key === "Escape") setOpen(false); }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  async function markAllRead() {
    if (!token) return;
    // Optimistic: clear the badge + flip rows read; reconcile on failure (the F1e delete-404 lesson).
    const prevUnread = unread;
    const prevItems = items;
    setUnread(0);
    setItems((cur) => (cur ?? []).map((n) => ({ ...n, read: true })));
    try {
      await markNotificationsRead(token);
    } catch {
      setUnread(prevUnread);
      setItems(prevItems);
    }
  }

  function openItem(n: AppNotification) {
    setOpen(false);
    // Opening a notification marks it read (optimistically — the badge clears, the row un-bolds).
    // Best-effort: a failed mark just leaves it unread, never blocks the navigation.
    if (token && !n.read) {
      setUnread((u) => Math.max(0, u - 1));
      setItems((cur) => (cur ?? []).map((x) => (x.id === n.id ? { ...x, read: true } : x)));
      markNotificationsRead(token, [n.id]).catch(() => {});
    }
    // A review notification routes to the review queue (where I own / submitted the work).
    if (n.category === "review" || n.resource_type === "review_request") {
      router.push("/app/review-queue");
    } else if (n.vault_id) {
      router.push(`/app/vault/${n.vault_id}`);
    }
  }

  if (!token) return null;

  return (
    <div ref={ref} className="relative flex-shrink-0">
      <button
        onClick={toggle}
        className="relative flex items-center justify-center w-8 h-8 rounded-lg hover:bg-[var(--bg-hover)] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] active:scale-[0.97]"
        aria-haspopup="menu"
        aria-expanded={open}
        title="Notifications"
      >
        <Bell size={16} className="text-[var(--text-muted)]" />
        {unread > 0 && (
          <span
            className="absolute -top-0.5 -right-0.5 min-w-[16px] h-[16px] px-1 rounded-full flex items-center justify-center text-[10px] font-semibold leading-none"
            style={{ background: "var(--accent)", color: "var(--on-ink)" }}
            aria-label={`${unread} unread notifications`}
          >
            {unread > 9 ? "9+" : unread}
          </span>
        )}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98 }}
            transition={{ duration: 0.12, ease: [0.23, 1, 0.32, 1] }}
            className="absolute right-0 top-full mt-1.5 w-[340px] rounded-xl overflow-hidden bg-[var(--bg-surface)] border border-[var(--border)]"
            style={{ zIndex: "var(--z-dropdown)" as unknown as number, boxShadow: "var(--shadow-lg)" }}
            role="menu"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-3 py-2.5 border-b border-[var(--border)]">
              <p className="text-[13px] font-semibold text-[var(--text-primary)]">Notifications</p>
              {(items?.some((n) => !n.read) ?? false) && (
                <button
                  onClick={markAllRead}
                  className="flex items-center gap-1 text-[11px] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
                  title="Mark all read"
                >
                  <CheckCheck size={12} />
                  <span>Mark all read</span>
                </button>
              )}
            </div>

            {/* Body */}
            <div className="max-h-[380px] overflow-y-auto">
              {loading && items === null ? (
                // Loading skeleton (never a dead spinner).
                <div className="p-3 space-y-2">
                  {[0, 1, 2].map((i) => (
                    <div key={i} className="h-10 rounded-md animate-pulse" style={{ background: "var(--surface-3)" }} />
                  ))}
                </div>
              ) : (items?.length ?? 0) === 0 ? (
                // Empty — situation then (implicit) action.
                <div className="px-3 py-8 text-center">
                  <p className="text-[13px] text-[var(--text-secondary)]">You&rsquo;re all caught up.</p>
                  <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
                    Review requests and matter updates show up here.
                  </p>
                </div>
              ) : (
                <ul>
                  {(items ?? []).map((n) => (
                    <li key={n.id}>
                      <button
                        onClick={() => openItem(n)}
                        className="w-full text-left px-3 py-2.5 flex gap-2.5 transition-colors hover:bg-[var(--bg-hover)] border-b border-[var(--border)] last:border-b-0"
                        role="menuitem"
                      >
                        {/* Unread dot */}
                        <span
                          className="mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0"
                          style={{ background: n.read ? "transparent" : "var(--accent)" }}
                          aria-hidden
                        />
                        <span className="min-w-0 flex-1">
                          <span className="flex items-center justify-between gap-2">
                            <span className="text-[12.5px] font-medium text-[var(--text-primary)] truncate">
                              {n.title ?? "Update"}
                            </span>
                            <span className="text-[10px] text-[var(--text-muted)] flex-shrink-0">
                              {relativeTime(n.created_at)}
                            </span>
                          </span>
                          {n.body && (
                            <span className="block text-[11.5px] text-[var(--text-secondary)] mt-0.5 line-clamp-2">
                              {n.body}
                            </span>
                          )}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
