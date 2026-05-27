"use client";

// app/app/settings/page.tsx — Analytics Dashboard & Settings
// Phase 4: Usage stats, query analytics, and export controls

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
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
} from "lucide-react";
import { useAuthStore } from "@/stores/auth.store";
import {
  getAnalyticsSummary,
  getUsageSummary,
  AnalyticsSummary,
  UsageSummary,
} from "@/lib/api";
import { toast } from "sonner";

// Stat card component
function StatCard({
  label,
  value,
  icon: Icon,
  sub,
  accent,
}: {
  label: string;
  value: string | number;
  icon: React.ElementType;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`card p-4 ${accent ? "border-[var(--accent)]" : ""}`}
    >
      <div className="flex items-center gap-2 mb-2">
        <Icon size={14} className="text-[var(--text-muted)]" />
        <span className="text-[10px] font-medium text-[var(--text-muted)] uppercase tracking-[0.12em]">
          {label}
        </span>
      </div>
      <p className="text-2xl font-bold text-[var(--text-primary)]">{value}</p>
      {sub && <p className="text-[10px] text-[var(--text-muted)] mt-1">{sub}</p>}
    </motion.div>
  );
}

// Simple bar chart using CSS
function MiniBarChart({ data }: { data: { date: string; count: number }[] }) {
  if (!data.length) return <p className="text-xs text-[var(--text-muted)] text-center py-8">No query data yet</p>;
  const max = Math.max(...data.map((d) => d.count), 1);
  // Show last 14 days
  const recent = data.slice(-14);

  return (
    <div className="flex items-end gap-1 h-24 px-1">
      {recent.map((d, i) => (
        <div key={i} className="flex-1 flex flex-col items-center gap-1">
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: `${(d.count / max) * 100}%` }}
            transition={{ delay: i * 0.03 }}
            className="w-full bg-[var(--accent)] rounded-sm min-h-[2px]"
            title={`${d.date}: ${d.count} queries`}
          />
          {i % 3 === 0 && (
            <span className="text-[7px] text-[var(--text-muted)] whitespace-nowrap">
              {d.date.slice(5)}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

export default function SettingsPage() {
  const router = useRouter();
  const { token } = useAuthStore();
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [loading, setLoading] = useState(true);

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
    } catch {
      toast.error("Failed to load analytics");
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => { load(); }, [load]);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin">
      <div className="max-w-4xl mx-auto px-4 md:px-8 py-8">
        {/* Header */}
        <div className="flex items-center gap-3 mb-8">
          <button
            onClick={() => router.push("/app/chat")}
            className="p-1.5 rounded-lg hover:bg-[var(--bg-hover)] text-[var(--text-muted)] transition-colors"
          >
            <ArrowLeft size={18} />
          </button>
          <div>
            <h1 className="text-lg font-semibold text-[var(--text-primary)]">Analytics & Settings</h1>
            <p className="text-xs text-[var(--text-muted)]">Last 30 days overview</p>
          </div>
        </div>

        {/* Usage Stats */}
        <section className="mb-8">
          <h2 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-[0.12em] mb-3 flex items-center gap-2">
            <Database size={12} /> Platform Usage
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <StatCard label="Documents" value={usage?.documents_count ?? 0} icon={FileText} sub={`${usage?.total_chunks ?? 0} chunks indexed`} />
            <StatCard label="Collections" value={usage?.collections_count ?? 0} icon={FolderOpen} />
            <StatCard label="Conversations" value={usage?.conversations_count ?? 0} icon={MessageSquare} />
            <StatCard label="Messages" value={usage?.total_messages ?? 0} icon={MessageSquare} />
            <StatCard label="Queries (30d)" value={analytics?.total_queries ?? 0} icon={BarChart3} accent />
          </div>
        </section>

        {/* Query Analytics */}
        <section className="mb-8">
          <h2 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-[0.12em] mb-3 flex items-center gap-2">
            <Activity size={12} /> Query Performance
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard
              label="Avg Latency"
              value={analytics?.avg_latency_ms ? `${analytics.avg_latency_ms}ms` : "—"}
              icon={Clock}
              sub="Response time"
            />
            <StatCard
              label="Cache Hit Rate"
              value={analytics?.cache_hit_rate ? `${analytics.cache_hit_rate}%` : "—"}
              icon={Zap}
              sub="Semantic cache"
            />
            <StatCard
              label="Agentic Mode"
              value={analytics?.agentic_query_rate ? `${analytics.agentic_query_rate}%` : "—"}
              icon={TrendingUp}
              sub="Deep query usage"
            />
            <StatCard
              label="Web Search"
              value={analytics?.web_search_rate ? `${analytics.web_search_rate}%` : "—"}
              icon={Globe}
              sub="Fallback searches"
            />
          </div>
        </section>

        {/* Activity Chart */}
        <section className="mb-8">
          <h2 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-[0.12em] mb-3 flex items-center gap-2">
            <BarChart3 size={12} /> Query Activity
          </h2>
          <div className="card p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs text-[var(--text-secondary)]">Queries per day</span>
              <div className="flex gap-3 text-[10px] text-[var(--text-muted)]">
                <span>Today: <strong className="text-[var(--text-primary)]">{analytics?.queries_today ?? 0}</strong></span>
                <span>This week: <strong className="text-[var(--text-primary)]">{analytics?.queries_this_week ?? 0}</strong></span>
              </div>
            </div>
            <MiniBarChart data={analytics?.daily_queries ?? []} />
          </div>
        </section>

        {/* Top Queries */}
        {analytics?.top_documents && analytics.top_documents.length > 0 && (
          <section className="mb-8">
            <h2 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-[0.12em] mb-3">
              Most Common Queries
            </h2>
            <div className="card divide-y divide-[var(--border)]">
              {analytics.top_documents.map((item, i) => (
                <div key={i} className="flex items-center justify-between px-4 py-2.5">
                  <span className="text-xs text-[var(--text-primary)] truncate flex-1 mr-4">{item.query}</span>
                  <span className="text-[10px] text-[var(--text-muted)] flex-shrink-0">{item.count}×</span>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
