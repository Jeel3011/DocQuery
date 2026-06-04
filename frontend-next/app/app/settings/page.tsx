"use client";

// app/app/settings/page.tsx — Analytics Dashboard & Settings
// Phase 4: Usage stats, query analytics, export controls + profile prefs

import { useEffect, useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { motion, useInView, animate } from "framer-motion";
import {
  BarChart3,
  Activity,
  Clock,
  Database,
  FileText,
  MessageSquare,
  FolderOpen,
  TrendingUp,
  Zap,
  Globe,
  ArrowLeft,
  Shield,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  Check,
  Pencil,
} from "lucide-react";
import { useAuthStore } from "@/stores/auth.store";
import { useProfileStore } from "@/stores/profile.store";
import {
  getAnalyticsSummary,
  getUsageSummary,
  getAuditLog,
  getMe,
  updatePreferredName,
  AnalyticsSummary,
  UsageSummary,
  AuditEntry,
} from "@/lib/api";
import { toast } from "sonner";

const ease = [0.16, 1, 0.3, 1] as const;

// ── Count-up number — animates from 0 to value when scrolled into view ──
function CountUp({ value, suffix = "", decimals = 0 }: { value: number; suffix?: string; decimals?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "-40px" });
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    if (!inView) return;
    const controls = animate(0, value, {
      duration: 0.9,
      ease: "easeOut",
      onUpdate: (v) => setDisplay(v),
    });
    return () => controls.stop();
  }, [inView, value]);

  return (
    <span ref={ref} className="tabular-nums">
      {display.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}
      {suffix}
    </span>
  );
}

// ── Stat card — animated count, hover lift, optional selected state ──
function StatCard({
  label,
  value,
  rawValue,
  suffix,
  decimals,
  icon: Icon,
  sub,
  selected,
  onClick,
  delay = 0,
}: {
  label: string;
  value?: string;
  rawValue?: number | null;
  suffix?: string;
  decimals?: number;
  icon: React.ElementType;
  sub?: string;
  selected?: boolean;
  onClick?: () => void;
  delay?: number;
}) {
  const interactive = !!onClick;
  return (
    <motion.button
      type="button"
      onClick={onClick}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, ease, duration: 0.5 }}
      whileHover={interactive ? { y: -3 } : { y: -2 }}
      className="text-left p-4 rounded-2xl transition-shadow"
      style={{
        background: "var(--surface)",
        border: `1px solid ${selected ? "var(--ink)" : "var(--line)"}`,
        boxShadow: selected ? "var(--shadow-md)" : "var(--shadow-sm)",
        cursor: interactive ? "pointer" : "default",
      }}
    >
      <div className="flex items-center gap-2 mb-2">
        <Icon size={14} style={{ color: selected ? "var(--ink)" : "var(--ink-3)" }} />
        <span className="text-[10px] font-medium uppercase tracking-[0.12em]" style={{ color: "var(--ink-3)" }}>
          {label}
        </span>
      </div>
      <p className="text-2xl font-bold" style={{ color: "var(--ink)" }}>
        {rawValue != null ? <CountUp value={rawValue} suffix={suffix} decimals={decimals} /> : value ?? "—"}
      </p>
      {sub && <p className="text-[10px] mt-1" style={{ color: "var(--ink-3)" }}>{sub}</p>}
    </motion.button>
  );
}

// ── Interactive bar chart — hover tooltip, baseline, highlight on hover ──
function BarChart({ data }: { data: { date: string; count: number }[] }) {
  const [hover, setHover] = useState<number | null>(null);
  if (!data.length) return <p className="text-xs text-center py-10" style={{ color: "var(--ink-3)" }}>No query data yet</p>;
  const recent = data.slice(-21);
  const max = Math.max(...recent.map((d) => d.count), 1);

  return (
    <div className="relative">
      <div className="flex items-end gap-[3px] h-32 px-1" onMouseLeave={() => setHover(null)}>
        {recent.map((d, i) => {
          const pct = (d.count / max) * 100;
          const active = hover === i;
          return (
            <div
              key={i}
              className="flex-1 flex flex-col items-center justify-end h-full cursor-pointer group"
              onMouseEnter={() => setHover(i)}
            >
              {/* Tooltip */}
              {active && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="absolute -top-1 z-10 px-2.5 py-1.5 rounded-lg text-[10px] whitespace-nowrap pointer-events-none"
                  style={{ background: "var(--ink)", color: "var(--on-ink)", boxShadow: "var(--shadow-md)" }}
                >
                  <strong>{d.count}</strong> {d.count === 1 ? "query" : "queries"} · {d.date.slice(5)}
                </motion.div>
              )}
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: `${Math.max(pct, 2)}%` }}
                transition={{ delay: i * 0.025, ease, duration: 0.5 }}
                className="w-full rounded-t-[3px] min-h-[3px]"
                style={{
                  background: active ? "var(--ink)" : "var(--ink-3)",
                  opacity: active ? 1 : 0.55,
                  transition: "background 140ms ease, opacity 140ms ease",
                }}
              />
            </div>
          );
        })}
      </div>
      {/* Baseline + sparse labels */}
      <div className="flex justify-between mt-2 px-1">
        {recent.map((d, i) => (
          <span
            key={i}
            className="text-[8px] tabular-nums"
            style={{ color: "var(--ink-3)", visibility: i % 5 === 0 ? "visible" : "hidden" }}
          >
            {d.date.slice(5)}
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Editable "what should we call you" name preference ──
function NamePreference({ fallback, token }: { fallback: string; token: string | null }) {
  const { preferredName, setPreferredName } = useProfileStore();
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(preferredName ?? "");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { if (editing) setTimeout(() => inputRef.current?.focus(), 60); }, [editing]);

  async function save() {
    const name = value.trim() || fallback;
    setPreferredName(name);          // optimistic local cache
    setEditing(false);
    if (token) {
      try {
        await updatePreferredName(token, name);   // server-side source of truth
        toast.success(`I'll call you ${name}`);
      } catch {
        toast.error("Couldn't save your name. Please try again.");
      }
    }
  }

  const shown = preferredName ?? fallback;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ ease, duration: 0.5 }}
      className="flex items-center gap-4 p-5 rounded-2xl mb-8"
      style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-sm)" }}
    >
      <div
        className="w-11 h-11 rounded-2xl flex items-center justify-center shrink-0"
        style={{ background: "var(--ink)", color: "var(--on-ink)" }}
      >
        <Sparkles size={19} strokeWidth={1.7} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-[11px] font-medium uppercase tracking-[0.12em] mb-1" style={{ color: "var(--ink-3)" }}>
          What DocQuery calls you
        </p>
        {editing ? (
          <input
            ref={inputRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") save(); if (e.key === "Escape") setEditing(false); }}
            placeholder={fallback}
            maxLength={40}
            className="w-full max-w-xs px-3 py-1.5 rounded-lg text-[15px] outline-none"
            style={{ background: "var(--surface-2)", border: "1px solid var(--ink)", color: "var(--ink)" }}
          />
        ) : (
          <p className="text-[16px] font-semibold" style={{ color: "var(--ink)" }}>{shown}</p>
        )}
      </div>
      {editing ? (
        <button
          onClick={save}
          className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-[13px] font-semibold shrink-0"
          style={{ background: "var(--ink)", color: "var(--on-ink)" }}
        >
          <Check size={14} /> Save
        </button>
      ) : (
        <button
          onClick={() => { setValue(preferredName ?? ""); setEditing(true); }}
          className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-[13px] font-medium shrink-0"
          style={{ background: "var(--surface-3)", color: "var(--ink-2)", border: "1px solid var(--line)" }}
        >
          <Pencil size={13} /> Edit
        </button>
      )}
    </motion.div>
  );
}

function SectionTitle({ icon: Icon, children }: { icon: React.ElementType; children: React.ReactNode }) {
  return (
    <h2 className="text-xs font-medium uppercase tracking-[0.12em] mb-3 flex items-center gap-2" style={{ color: "var(--ink-3)" }}>
      <Icon size={12} /> {children}
    </h2>
  );
}

export default function SettingsPage() {
  const router = useRouter();
  const { user, token } = useAuthStore();
  const fallbackName = user?.email?.split("@")[0] ?? "there";
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([]);
  const [auditTotal, setAuditTotal] = useState(0);
  const [auditPage, setAuditPage] = useState(1);
  const [auditLoading, setAuditLoading] = useState(false);
  const AUDIT_PER_PAGE = 20;

  const hydrateFromServer = useProfileStore((s) => s.hydrateFromServer);

  const load = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const [a, u] = await Promise.all([
        getAnalyticsSummary(token, 30),
        getUsageSummary(token),
      ]);
      setAnalytics(a);
      setUsage(u);
      // Reconcile preferred name with server source of truth (non-blocking).
      getMe(token).then((m) => hydrateFromServer(m.preferred_name)).catch(() => {});
    } catch {
      toast.error("Failed to load analytics");
    } finally {
      setLoading(false);
    }
  }, [token, hydrateFromServer]);

  const loadAudit = useCallback(async (page: number) => {
    if (!token) return;
    setAuditLoading(true);
    try {
      const res = await getAuditLog(token, page, AUDIT_PER_PAGE, 30);
      setAuditEntries(res.entries);
      setAuditTotal(res.total);
      setAuditPage(page);
    } catch {
      toast.error("Failed to load audit log");
    } finally {
      setAuditLoading(false);
    }
  }, [token]);

  useEffect(() => { load(); loadAudit(1); }, [load, loadAudit]);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin" style={{ background: "var(--canvas)" }}>
      <div className="max-w-4xl mx-auto px-4 md:px-8 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ ease }}
          className="flex items-center gap-3 mb-8"
        >
          <button
            onClick={() => router.push("/app/chat")}
            className="p-1.5 rounded-lg hover:bg-[var(--bg-hover)] transition-colors"
            style={{ color: "var(--ink-3)" }}
          >
            <ArrowLeft size={18} />
          </button>
          <div>
            <h1
              style={{ fontFamily: "Fraunces, Georgia, serif", fontSize: "24px", fontWeight: 400, letterSpacing: "-0.025em", color: "var(--ink)", lineHeight: 1.1 }}
            >
              Analytics & Settings
            </h1>
            <p className="text-xs" style={{ color: "var(--ink-3)" }}>Last 30 days overview</p>
          </div>
        </motion.div>

        {/* Name preference */}
        <NamePreference fallback={fallbackName} token={token} />

        {/* Usage Stats */}
        <section className="mb-8">
          <SectionTitle icon={Database}>Platform Usage</SectionTitle>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <StatCard label="Documents" rawValue={usage?.documents_count ?? 0} icon={FileText} sub={`${usage?.total_chunks ?? 0} chunks indexed`} delay={0.02} />
            <StatCard label="Collections" rawValue={usage?.collections_count ?? 0} icon={FolderOpen} delay={0.06} />
            <StatCard label="Conversations" rawValue={usage?.conversations_count ?? 0} icon={MessageSquare} delay={0.10} />
            <StatCard label="Messages" rawValue={usage?.total_messages ?? 0} icon={MessageSquare} delay={0.14} />
            <StatCard label="Queries (30d)" rawValue={analytics?.total_queries ?? 0} icon={BarChart3} selected delay={0.18} />
          </div>
        </section>

        {/* Query Analytics */}
        <section className="mb-8">
          <SectionTitle icon={Activity}>Query Performance</SectionTitle>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard
              label="Avg Latency"
              rawValue={analytics?.avg_latency_ms ?? null}
              suffix="ms"
              decimals={1}
              icon={Clock}
              sub="Response time"
              delay={0.02}
            />
            <StatCard
              label="Cache Hit Rate"
              rawValue={analytics?.cache_hit_rate ?? null}
              suffix="%"
              decimals={1}
              icon={Zap}
              sub="Semantic cache"
              delay={0.06}
            />
            <StatCard
              label="Agentic Mode"
              rawValue={analytics?.agentic_query_rate ?? null}
              suffix="%"
              decimals={1}
              icon={TrendingUp}
              sub="Deep query usage"
              delay={0.10}
            />
            <StatCard
              label="Web Search"
              rawValue={analytics?.web_search_rate ?? null}
              suffix="%"
              decimals={1}
              icon={Globe}
              sub="Fallback searches"
              delay={0.14}
            />
          </div>
        </section>

        {/* Activity Chart */}
        <section className="mb-8">
          <SectionTitle icon={BarChart3}>Query Activity</SectionTitle>
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ease, duration: 0.5 }}
            className="p-5 rounded-2xl"
            style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-sm)" }}
          >
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-medium" style={{ color: "var(--ink-2)" }}>Queries per day</span>
              <div className="flex gap-4 text-[11px]" style={{ color: "var(--ink-3)" }}>
                <span>Today <strong style={{ color: "var(--ink)" }}>{analytics?.queries_today ?? 0}</strong></span>
                <span>This week <strong style={{ color: "var(--ink)" }}>{analytics?.queries_this_week ?? 0}</strong></span>
              </div>
            </div>
            <BarChart data={analytics?.daily_queries ?? []} />
          </motion.div>
        </section>

        {/* Top Queries */}
        {analytics?.top_queries && analytics.top_queries.length > 0 && (
          <section className="mb-8">
            <SectionTitle icon={MessageSquare}>Most Common Queries</SectionTitle>
            <div
              className="rounded-2xl overflow-hidden"
              style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-sm)" }}
            >
              {analytics.top_queries.map((item, i) => {
                const topCount = analytics.top_queries[0]?.count || 1;
                return (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05, ease }}
                    className="relative flex items-center justify-between px-4 py-3"
                    style={{ borderTop: i ? "1px solid var(--line)" : "none" }}
                  >
                    {/* Frequency bar behind the row */}
                    <div
                      className="absolute inset-y-0 left-0 pointer-events-none"
                      style={{ width: `${(item.count / topCount) * 100}%`, background: "var(--surface-3)", opacity: 0.6 }}
                    />
                    <span className="relative text-[13px] truncate flex-1 mr-4" style={{ color: "var(--ink)" }}>
                      {item.query}
                    </span>
                    <span
                      className="relative text-[11px] font-semibold tabular-nums px-2 py-0.5 rounded-md shrink-0"
                      style={{ background: "var(--ink)", color: "var(--on-ink)" }}
                    >
                      {item.count}×
                    </span>
                  </motion.div>
                );
              })}
            </div>
          </section>
        )}

        {/* Audit Log */}
        <section className="mb-8">
          <div className="flex items-center justify-between mb-3">
            <SectionTitle icon={Shield}>Audit Log</SectionTitle>
            <span className="text-[10px]" style={{ color: "var(--ink-3)" }}>{auditTotal} events (last 30 days)</span>
          </div>

          {auditLoading ? (
            <div className="p-6 flex justify-center rounded-2xl" style={{ background: "var(--surface)", border: "1px solid var(--line)" }}>
              <div className="w-5 h-5 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
            </div>
          ) : auditEntries.length === 0 ? (
            <div className="p-6 text-center text-xs rounded-2xl" style={{ background: "var(--surface)", border: "1px solid var(--line)", color: "var(--ink-3)" }}>No audit events yet</div>
          ) : (
            <>
              <div
                className="rounded-2xl overflow-hidden"
                style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-sm)" }}
              >
                {auditEntries.map((entry, idx) => (
                  <motion.div
                    key={entry.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: Math.min(idx * 0.02, 0.3) }}
                    className="flex items-start gap-3 px-4 py-2.5 transition-colors hover:bg-[var(--surface-2)]"
                    style={{ borderTop: idx ? "1px solid var(--line)" : "none" }}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: "var(--surface-3)", color: "var(--ink)" }}>
                          {entry.action}
                        </span>
                        {entry.resource_type && (
                          <span className="text-[10px]" style={{ color: "var(--ink-3)" }}>{entry.resource_type}</span>
                        )}
                        {entry.metadata?.filename != null && (
                          <span className="text-[10px] truncate max-w-[200px]" style={{ color: "var(--ink-2)" }}>
                            {String(entry.metadata.filename)}
                          </span>
                        )}
                        {entry.metadata?.question != null && (
                          <span className="text-[10px] truncate max-w-[200px]" style={{ color: "var(--ink-2)" }}>
                            {(() => { const q = String(entry.metadata.question); return `“${q.slice(0, 60)}${q.length > 60 ? "…" : ""}”`; })()}
                          </span>
                        )}
                      </div>
                    </div>
                    <span className="text-[10px] flex-shrink-0 tabular-nums" style={{ color: "var(--ink-3)" }}>
                      {entry.created_at ? new Date(entry.created_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—"}
                    </span>
                  </motion.div>
                ))}
              </div>

              {/* Pagination */}
              {auditTotal > AUDIT_PER_PAGE && (
                <div className="flex items-center justify-between mt-2 text-[10px]" style={{ color: "var(--ink-3)" }}>
                  <span>Page {auditPage} of {Math.ceil(auditTotal / AUDIT_PER_PAGE)}</span>
                  <div className="flex gap-1">
                    <button
                      onClick={() => loadAudit(auditPage - 1)}
                      disabled={auditPage <= 1}
                      className="p-1 rounded hover:bg-[var(--bg-hover)] disabled:opacity-30 transition-colors"
                    >
                      <ChevronLeft size={12} />
                    </button>
                    <button
                      onClick={() => loadAudit(auditPage + 1)}
                      disabled={auditPage >= Math.ceil(auditTotal / AUDIT_PER_PAGE)}
                      className="p-1 rounded hover:bg-[var(--bg-hover)] disabled:opacity-30 transition-colors"
                    >
                      <ChevronRight size={12} />
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </div>
  );
}
