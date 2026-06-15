"use client";

// /app — Vault Home (G2 Step B).
// A card grid of vaults (= collections): name · file count · last activity · a subtle
// type-mix indicator. "+ New Vault" creates one (own dialog); empty state invites the
// first vault. Card click → /app/vault/[id]. Born in the clean B&W system; status color
// appears only on the fidelity-style type dots (meaning, not decoration).
//
// Scope rule (G2 §9 risk #1): this page is OFF a vault route, so it deliberately does NOT
// touch collection.store — the route becomes the source of truth only once you open a
// vault (VaultScopeSync handles that in the shell).

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { FolderOpen, FileText, Plus } from "lucide-react";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/auth.store";
import { listCollections, createCollection, CollectionResponse } from "@/lib/api";
import { EmptyState } from "@/components/ui/EmptyState";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/Dialog";

const ease = [0.23, 1, 0.32, 1] as const;

function timeAgo(d: string | null): string {
  if (!d) return "";
  const m = Math.floor((Date.now() - new Date(d).getTime()) / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export default function VaultHomePage() {
  const router = useRouter();
  const { token } = useAuthStore();
  const [collections, setCollections] = useState<CollectionResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNew, setShowNew] = useState(false);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);

  const load = useCallback(async () => {
    if (!token) return;
    try {
      const cols = await listCollections(token);
      cols.sort(
        (a, b) => new Date(b.updated_at ?? 0).getTime() - new Date(a.updated_at ?? 0).getTime()
      );
      setCollections(cols);
    } catch {
      toast.error("Failed to load vaults");
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    load();
  }, [load]);

  async function handleCreate() {
    if (!token || !newName.trim() || creating) return;
    setCreating(true);
    try {
      const c = await createCollection(token, newName.trim());
      setNewName("");
      setShowNew(false);
      toast.success(`Created vault "${c.name}"`);
      router.push(`/app/vault/${c.id}`);
    } catch {
      toast.error("Failed to create vault");
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin relative">

      <div className="relative max-w-5xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="flex items-end justify-between mb-8">
          <div>
            <h1
              style={{
                fontFamily: "Fraunces, Georgia, serif",
                fontSize: 32,
                fontWeight: 500,
                letterSpacing: "-0.03em",
                color: "var(--ink)",
                lineHeight: 1.05,
              }}
            >
              Vaults
            </h1>
            <p className="text-[13px] mt-1.5" style={{ color: "var(--ink-3)" }}>
              {loading
                ? "Loading…"
                : collections.length === 0
                ? "Create a vault to organize a matter, deal, or document set."
                : `${collections.length} vault${collections.length === 1 ? "" : "s"}`}
            </p>
          </div>
          {!loading && collections.length > 0 && (
            <button className="btn-primary flex items-center gap-1.5" onClick={() => setShowNew(true)}>
              <Plus size={14} /> New vault
            </button>
          )}
        </div>

        {/* Body */}
        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <div
                key={i}
                className="h-[124px] rounded-2xl animate-pulse"
                style={{ background: "var(--surface-3)", border: "1px solid var(--line)" }}
              />
            ))}
          </div>
        ) : collections.length === 0 ? (
          <div
            className="rounded-2xl"
            style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow-sm)" }}
          >
            <EmptyState
              icon={<FolderOpen size={18} />}
              title="No vaults yet"
              description="A vault holds the documents for one matter — a deal, a contract set, or a filing. Create your first to begin."
              action={
                <button className="btn-primary flex items-center gap-1.5" onClick={() => setShowNew(true)}>
                  <Plus size={14} /> Create your first vault
                </button>
              }
            />
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {collections.map((c, i) => (
              <motion.button
                key={c.id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: Math.min(i * 0.04, 0.3), duration: 0.45, ease }}
                whileHover={{ y: -3 }}
                onClick={() => router.push(`/app/vault/${c.id}`)}
                className="group text-left p-5 rounded-2xl flex flex-col gap-4 transition-shadow card hover:shadow-[var(--shadow-lg)]"
                style={{ borderColor: "var(--line)" }}
              >
                <div className="flex items-start justify-between">
                  <span
                    className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                    style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
                  >
                    <FolderOpen size={17} />
                  </span>
                  {/* Type-mix indicator — kept a single neutral "documents present" dot.
                      listCollections returns only document_count, not per-doc doc_type, so a
                      real type-mix would cost an extra getCollectionDocuments per card on the
                      home grid. The real type/fidelity surfacing (Step F) lives in the vault
                      doc table where the data is already fetched; the home dot stays neutral. */}
                  {(c.document_count ?? 0) > 0 && (
                    <span
                      className="w-2 h-2 rounded-full mt-1.5"
                      style={{ background: "var(--line-2)" }}
                      title="Document set"
                    />
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <p
                    className="text-[15px] font-semibold truncate"
                    style={{ color: "var(--ink)", fontFamily: "Fraunces, Georgia, serif", letterSpacing: "-0.01em" }}
                  >
                    {c.name}
                  </p>
                  <p className="text-[12px] text-[var(--text-muted)] flex items-center gap-1.5 mt-1.5">
                    <FileText size={12} />
                    {c.document_count ?? 0} file{(c.document_count ?? 0) === 1 ? "" : "s"}
                    {c.updated_at && <span aria-hidden>·</span>}
                    {c.updated_at && <span>{timeAgo(c.updated_at)}</span>}
                  </p>
                </div>
              </motion.button>
            ))}
          </div>
        )}
      </div>

      {/* New-vault dialog (this page's own entry point — the switcher has its own in the shell) */}
      <Dialog open={showNew} onOpenChange={setShowNew}>
        <DialogContent maxWidth="420px">
          <DialogHeader>
            <DialogTitle>New vault</DialogTitle>
          </DialogHeader>
          <div className="px-5 py-5">
            <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-2">Vault name</label>
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleCreate();
                if (e.key === "Escape") setShowNew(false);
              }}
              placeholder="e.g. Acme acquisition contracts"
              autoFocus
              className="w-full px-3 py-2.5 rounded-xl text-[14px] outline-none focus:border-[var(--accent)]"
              style={{ background: "var(--surface-2)", border: "1px solid var(--line)", color: "var(--ink)" }}
            />
          </div>
          <DialogFooter>
            <button className="btn-ghost" onClick={() => setShowNew(false)}>Cancel</button>
            <button className="btn-primary" onClick={handleCreate} disabled={!newName.trim() || creating}>
              {creating ? "Creating…" : "Create vault"}
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
