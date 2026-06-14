"use client";

// app/app/layout.tsx — G2 app shell (the de-sidebar).
// Slim top bar (logo · vault switcher · ⌘K · account) over a full-bleed body on the
// clean black-and-white canvas. NO file sidebar — the document list now lives inside a
// vault (Step C). This layout's job is the shell + the shared, app-wide providers:
// CommandPalette, NamePrompt, toast, the vault-scope source-of-truth sync, and the
// minimal data the top bar needs (collections + profile name).
//
// Upload + the document table moved out of here into the vault workspace (Steps C/F).
// We keep a single hidden file input + upload handler so the ⌘K "Upload" command still
// works during the A→H transition; uploaded docs land in the active vault if one is set.

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";

import { useAuthStore } from "@/stores/auth.store";
import { useProfileStore } from "@/stores/profile.store";
import { useCollectionStore } from "@/stores/collection.store";
import {
  listConversations,
  listCollections,
  createConversation,
  createCollection,
  uploadDocument,
  addDocToCollection,
  getMe,
  ConversationResponse,
  CollectionResponse,
} from "@/lib/api";
import { toast } from "sonner";
import { CommandPalette } from "@/components/ui/CommandPalette";
import { NamePrompt } from "@/components/app/NamePrompt";
import { TopBar } from "@/components/app/TopBar";
import { VaultScopeSync } from "@/components/app/VaultScopeSync";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/Dialog";

const ACCEPTED = ".pdf,.docx,.pptx,.txt,.xlsx";
const MAX_MB = 50;

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const { user, token, logout } = useAuthStore();
  const { preferredName, hydrateFromServer } = useProfileStore();
  // Scope is DERIVED from the route by VaultScopeSync; here we only READ it (for the
  // switcher's active highlight and to drop uploads into the current vault).
  const { activeCollectionId } = useCollectionStore();

  const [convs, setConvs] = useState<ConversationResponse[]>([]);
  const [collections, setCollections] = useState<CollectionResponse[]>([]);
  const [uploadQueue, setUploadQueue] = useState<{ total: number; done: number } | null>(null);
  const uploadDoneRef = useRef(0);
  const [showNewVault, setShowNewVault] = useState(false);
  const [newVaultName, setNewVaultName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  // First name fallback so the assistant can greet the user.
  const fallbackName = (user?.email?.split("@")[0].match(/^[a-zA-Z]+/) ?? ["there"])[0];
  const displayName = preferredName ?? null;

  const loadConvs = useCallback(async () => {
    if (!token) return;
    try {
      const d = await listConversations(token);
      setConvs(
        d.sort((a, b) => new Date(b.updated_at ?? 0).getTime() - new Date(a.updated_at ?? 0).getTime())
      );
    } catch {
      /* non-fatal — ⌘K just won't list chats */
    }
  }, [token]);

  const loadCollections = useCallback(async () => {
    if (!token) return;
    try {
      setCollections(await listCollections(token));
    } catch {
      toast.error("Failed to load vaults");
    }
  }, [token]);

  useEffect(() => {
    loadConvs();
    loadCollections();
  }, [loadConvs, loadCollections]);

  // Reconcile the preferred name with the server (source of truth) once per session.
  useEffect(() => {
    if (!token) return;
    getMe(token).then((m) => hydrateFromServer(m.preferred_name)).catch(() => {});
  }, [token, hydrateFromServer]);

  async function newChat() {
    if (!token) return;
    // Step H: Ask is now vault-scoped (/app/vault/[id]/ask). A "New chat" needs a vault to
    // honour the scope source-of-truth (§9 risk #1). If a vault is active, start there;
    // otherwise send the user to Vault Home to pick one — never a vault-less conversation.
    if (!activeCollectionId) {
      router.push("/app");
      return;
    }
    try {
      const c = await createConversation(token, "New Chat");
      setConvs((p) => [c, ...p]);
      router.push(`/app/vault/${activeCollectionId}/ask/${c.id}`);
    } catch {
      toast.error("Failed to create conversation");
    }
  }

  async function handleCreateVault() {
    if (!token || !newVaultName.trim()) return;
    try {
      const c = await createCollection(token, newVaultName.trim());
      setCollections((p) => [c, ...p]);
      setNewVaultName("");
      setShowNewVault(false);
      toast.success(`Created vault "${c.name}"`);
      router.push(`/app/vault/${c.id}`);
    } catch {
      toast.error("Failed to create vault");
    }
  }

  // Lightweight upload (⌘K → Upload during the transition). The full drag-drop
  // UploadZone with live per-file progress lands in the vault workspace (Step C).
  async function uploadFiles(files: File[]) {
    if (!token || files.length === 0) return;

    const valid: File[] = [];
    for (const file of files) {
      const ext = "." + file.name.split(".").pop()?.toLowerCase();
      if (!ACCEPTED.split(",").includes(ext)) { toast.error(`Unsupported type: ${file.name}`); continue; }
      if (file.size > MAX_MB * 1048576) { toast.error(`${file.name} exceeds ${MAX_MB}MB`); continue; }
      valid.push(file);
    }
    if (valid.length === 0) return;

    uploadDoneRef.current = 0;
    setUploadQueue({ total: valid.length, done: 0 });

    const CONCURRENCY = 5;
    let idx = 0;
    const vaultId = activeCollectionId;

    async function uploadOne(file: File) {
      try {
        const doc = await uploadDocument(token!, file);
        if (vaultId) {
          try { await addDocToCollection(token!, vaultId, doc.id); } catch { /* non-fatal */ }
        }
      } catch (e: unknown) {
        toast.error(`Upload failed: ${file.name}. ${e instanceof Error ? e.message : "Unknown error"}`);
      } finally {
        uploadDoneRef.current += 1;
        setUploadQueue({ total: valid.length, done: uploadDoneRef.current });
      }
    }

    async function worker() {
      while (idx < valid.length) {
        const file = valid[idx++];
        await uploadOne(file);
      }
    }

    await Promise.all(Array.from({ length: Math.min(CONCURRENCY, valid.length) }, worker));
    toast.success(valid.length === 1 ? `Uploaded ${valid[0].name}` : `Uploaded ${valid.length} files`);
    setUploadQueue(null);
  }

  return (
    <div className="relative flex flex-col h-dvh overflow-hidden" style={{ background: "var(--canvas)" }}>
      {/* Soft aurora wash so glass surfaces (top bar, panels, input) blur real depth */}
      <div className="aurora aurora-soft" />

      {/* Route → store scope sync (the URL is the authoritative vault scope) */}
      <VaultScopeSync />

      {/* App-wide overlays */}
      <CommandPalette
        onNewChat={newChat}
        onUpload={() => fileRef.current?.click()}
        conversations={convs}
        collections={collections}
      />
      <NamePrompt fallback={fallbackName} />

      {/* Hidden file input for the ⌘K Upload command (transition-only) */}
      <input
        ref={fileRef}
        type="file"
        accept={ACCEPTED}
        multiple
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length) uploadFiles(files);
          e.target.value = "";
        }}
      />

      {/* New-vault dialog (opened from the vault switcher) */}
      <Dialog open={showNewVault} onOpenChange={setShowNewVault}>
        <DialogContent maxWidth="420px">
          <DialogHeader>
            <DialogTitle>New vault</DialogTitle>
          </DialogHeader>
          <div className="px-5 py-5">
            <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-2">
              Vault name
            </label>
            <input
              value={newVaultName}
              onChange={(e) => setNewVaultName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleCreateVault();
                if (e.key === "Escape") setShowNewVault(false);
              }}
              placeholder="e.g. Acme acquisition contracts"
              autoFocus
              className="w-full px-3 py-2.5 rounded-xl text-[14px] outline-none focus:border-[var(--accent)]"
              style={{ background: "var(--surface-2)", border: "1px solid var(--line)", color: "var(--ink)" }}
            />
          </div>
          <DialogFooter>
            <button className="btn-ghost" onClick={() => setShowNewVault(false)}>Cancel</button>
            <button className="btn-primary" onClick={handleCreateVault} disabled={!newVaultName.trim()}>
              Create vault
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Top bar */}
      <div className="relative" style={{ zIndex: 20 }}>
        <TopBar
          collections={collections}
          activeId={activeCollectionId}
          email={user?.email}
          name={displayName}
          onNewVault={() => setShowNewVault(true)}
          onLogout={async () => { await logout(); router.push("/login"); }}
        />
        {/* Upload progress ribbon — appears only mid-upload */}
        {uploadQueue && (
          <div className="absolute left-0 right-0 top-full flex items-center justify-center gap-2 px-4 py-1.5 text-[11px] border-b border-[var(--border)] bg-[var(--bg-surface)] text-[var(--text-muted)]">
            <div className="w-3 h-3 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
            <span>Uploading… ({uploadQueue.done}/{uploadQueue.total} done)</span>
          </div>
        )}
      </div>

      {/* Body — full-bleed children on the clean canvas */}
      <main className="flex-1 flex flex-col min-h-0 overflow-hidden relative" style={{ zIndex: 10 }}>
        {children}
      </main>
    </div>
  );
}
