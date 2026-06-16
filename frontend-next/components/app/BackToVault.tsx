"use client";

// BackToVault — the consistent "you are in <vault>" header for every vault sub-mode
// (Deep Analysis, Draft, Workflows, …). Names the active vault AND links back to it, so
// the user always knows which vault scopes the work and can return in one click.
//
// Scope rule (§9 risk #1): the vault id is passed in by the caller from the route's [id]
// (the AUTHORITATIVE scope) — never read from the persisted store here.

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, FolderOpen } from "lucide-react";
import { useAuthStore } from "@/stores/auth.store";
import { listCollections } from "@/lib/api";

export function BackToVault({ vaultId, className = "" }: { vaultId: string; className?: string }) {
  const router = useRouter();
  const { token } = useAuthStore();
  const [vaultName, setVaultName] = useState<string | null>(null);

  useEffect(() => {
    if (!token || !vaultId) return;
    let cancelled = false;
    listCollections(token)
      .then((cols) => { if (!cancelled) setVaultName(cols.find((c) => c.id === vaultId)?.name ?? null); })
      .catch(() => { if (!cancelled) setVaultName(null); });
    return () => { cancelled = true; };
  }, [token, vaultId]);

  return (
    <button
      onClick={() => router.push(`/app/vault/${encodeURIComponent(vaultId)}`)}
      className={`inline-flex items-center gap-1.5 text-[12px] text-[var(--ink-3)] hover:text-[var(--ink)] transition-colors ${className}`}
    >
      <ArrowLeft size={13} />
      <span className="inline-flex items-center gap-1">
        Back to
        <FolderOpen size={12} className="opacity-70" />
        <span className="font-medium text-[var(--ink-2)]">{vaultName ?? "vault"}</span>
      </span>
    </button>
  );
}
