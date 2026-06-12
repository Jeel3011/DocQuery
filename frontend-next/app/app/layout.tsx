"use client";

// app/app/layout.tsx — Monochrome app shell
// White dominant, black accents, dotted borders

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter, useParams } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import {
  Plus,
  FileText,
  LogOut,
  Menu,
  X,
  MessageSquare,
  Trash2,
  Upload,
  CheckCircle,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Pencil,
  FolderOpen,
  FolderPlus,
  BarChart3,
} from "lucide-react";

import { useAuthStore } from "@/stores/auth.store";
import {
  listConversations,
  listDocuments,
  createConversation,
  deleteConversation,
  renameConversation,
  uploadDocument,
  deleteDocument,
  listCollections,
  createCollection,
  deleteCollection,
  addDocToCollection,
  removeDocFromCollection,
  getCollectionDocuments,
  ConversationResponse,
  DocumentResponse,
  CollectionResponse,
} from "@/lib/api";
import { useCollectionStore } from "@/stores/collection.store";
import { toast } from "sonner";
import { CommandPalette } from "@/components/ui/CommandPalette";

const ACCEPTED = ".pdf,.docx,.pptx,.txt,.xlsx";
const MAX_MB = 50;

function timeAgo(d: string | null): string {
  if (!d) return "";
  const m = Math.floor((Date.now() - new Date(d).getTime()) / 60000);
  if (m < 1) return "now";
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h`;
  return `${Math.floor(h / 24)}d`;
}

function fmtBytes(b: number | null): string {
  if (b == null || b === 0) return "—";
  return b < 1048576 ? `${(b / 1024).toFixed(0)} KB` : `${(b / 1048576).toFixed(1)} MB`;
}

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const params = useParams();
  const { user, token, logout } = useAuthStore();

  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [convs, setConvs] = useState<ConversationResponse[]>([]);
  const [docs, setDocs] = useState<DocumentResponse[]>([]);
  const [uploadQueue, setUploadQueue] = useState<{ total: number; done: number } | null>(null);
  const uploadDoneRef = useRef(0);
  const [delDocId, setDelDocId] = useState<string | null>(null);
  const [delConvId, setDelConvId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [collections, setCollections] = useState<CollectionResponse[]>([]);
  const { activeCollectionId, setActiveCollectionId } = useCollectionStore();
  const [showNewCollection, setShowNewCollection] = useState(false);
  const [newCollectionName, setNewCollectionName] = useState("");
  const [delCollId, setDelCollId] = useState<string | null>(null);
  const [collectionDocIds, setCollectionDocIds] = useState<Set<string>>(new Set());
  const [isMobile, setIsMobile] = useState(false);
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const renameRef = useRef<HTMLInputElement>(null);

  const activeId = (params?.id as string) ?? null;

  const loadConvs = useCallback(async () => {
    if (!token) return;
    try {
      const d = await listConversations(token);
      setConvs(d.sort((a, b) => new Date(b.updated_at ?? 0).getTime() - new Date(a.updated_at ?? 0).getTime()));
    } catch (e) {
      console.error("Failed to load conversations", e);
      toast.error("Failed to load conversations");
    }
  }, [token]);

  const loadDocs = useCallback(async () => {
    if (!token) return;
    try { setDocs(await listDocuments(token)); }
    catch (e) {
      console.error("Failed to load documents", e);
      toast.error("Failed to load documents");
    }
  }, [token]);

  const loadCollections = useCallback(async () => {
    if (!token) return;
    try { setCollections(await listCollections(token)); }
    catch (e) {
      console.error("Failed to load collections", e);
      toast.error("Failed to load collections");
    }
  }, [token]);

  useEffect(() => { loadConvs(); loadDocs(); loadCollections(); }, [loadConvs, loadDocs, loadCollections]);

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 768);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  // Load docs in active collection to show checkmarks
  useEffect(() => {
    if (!token || !activeCollectionId) {
      setCollectionDocIds(new Set());
      return;
    }
    getCollectionDocuments(token, activeCollectionId)
      .then((docs) => setCollectionDocIds(new Set(docs.map((d) => d.id))))
      .catch(() => setCollectionDocIds(new Set()));
  }, [token, activeCollectionId]);

  useEffect(() => {
    if (docs.some((d) => d.status === "processing")) {
      pollRef.current = setTimeout(loadDocs, 3000);
    }
    return () => { if (pollRef.current) clearTimeout(pollRef.current); };
  }, [docs, loadDocs]);

  async function newChat() {
    if (!token) return;
    try {
      const c = await createConversation(token, "New Chat");
      setConvs((p) => [c, ...p]);
      router.push(`/app/chat/${c.id}`);
      setMobileOpen(false);
    } catch { toast.error("Failed to create conversation"); }
  }

  async function handleCreateCollection() {
    if (!token || !newCollectionName.trim()) return;
    try {
      const c = await createCollection(token, newCollectionName.trim());
      setCollections((p) => [c, ...p]);
      setNewCollectionName("");
      setShowNewCollection(false);
      toast.success(`Created collection "${c.name}"`);
    } catch { toast.error("Failed to create collection"); }
  }

  async function handleDeleteCollection(id: string) {
    if (!token) return;
    setDelCollId(null);
    const prev = collections;
    setCollections((p) => p.filter((c) => c.id !== id));
    if (activeCollectionId === id) setActiveCollectionId(null);
    try {
      await deleteCollection(token, id);
    } catch {
      setCollections(prev);
      toast.error("Failed to delete collection");
    }
  }

  async function handleToggleDocInCollection(docId: string) {
    if (!token || !activeCollectionId) return;
    const isInCollection = collectionDocIds.has(docId);
    // Optimistic update
    setCollectionDocIds((prev) => {
      const next = new Set(prev);
      if (isInCollection) next.delete(docId);
      else next.add(docId);
      return next;
    });
    // Update collection doc count optimistically
    setCollections((prev) =>
      prev.map((c) =>
        c.id === activeCollectionId
          ? { ...c, document_count: (c.document_count ?? 0) + (isInCollection ? -1 : 1) }
          : c
      )
    );
    try {
      if (isInCollection) {
        await removeDocFromCollection(token, activeCollectionId, docId);
      } else {
        await addDocToCollection(token, activeCollectionId, docId);
      }
    } catch {
      // Revert on failure
      setCollectionDocIds((prev) => {
        const next = new Set(prev);
        if (isInCollection) next.add(docId);
        else next.delete(docId);
        return next;
      });
      setCollections((prev) =>
        prev.map((c) =>
          c.id === activeCollectionId
            ? { ...c, document_count: (c.document_count ?? 0) + (isInCollection ? 1 : -1) }
            : c
        )
      );
      toast.error(isInCollection ? "Failed to remove from collection" : "Failed to add to collection");
    }
  }

  function startRename(c: ConversationResponse) {
    setRenamingId(c.id);
    setRenameValue(c.title || "");
    setTimeout(() => renameRef.current?.select(), 50);
  }

  async function commitRename(id: string) {
    const title = renameValue.trim();
    setRenamingId(null);
    if (!title || !token) return;
    const prev = convs;
    setConvs((p) => p.map((c) => c.id === id ? { ...c, title } : c));
    try {
      await renameConversation(token, id, title);
    } catch {
      setConvs(prev);
      toast.error("Failed to rename");
    }
  }

  async function delConv(id: string) {
    if (!token) return;
    setDelConvId(null);
    // Optimistic: remove immediately
    const prev = convs;
    setConvs((p) => p.filter((c) => c.id !== id));
    if (activeId === id) router.push("/app/chat");
    try {
      await deleteConversation(token, id);
    } catch {
      // Restore on failure
      setConvs(prev);
      toast.error("Failed to delete conversation");
    }
  }

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

    async function uploadOne(file: File) {
      try {
        const doc = await uploadDocument(token!, file);
        setDocs((p) => [doc, ...p]);
        if (activeCollectionId) {
          try {
            await addDocToCollection(token!, activeCollectionId, doc.id);
            setCollectionDocIds((prev) => { const next = new Set(prev); next.add(doc.id); return next; });
            setCollections((prev) =>
              prev.map((c) =>
                c.id === activeCollectionId
                  ? { ...c, document_count: (c.document_count ?? 0) + 1 }
                  : c
              )
            );
          } catch { /* non-fatal */ }
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
    loadDocs();
  }

  async function delDoc(id: string) {
    if (!token) return;
    setDelDocId(null);
    // Optimistic: remove immediately
    const prev = docs;
    setDocs((p) => p.filter((d) => d.id !== id));
    try {
      await deleteDocument(token, id);
    } catch {
      // Restore on failure
      setDocs(prev);
      toast.error("Failed to delete document");
    }
  }

  // ── Sidebar ────────────────────────────────────────────────────────────────

  function sidebar() {
    return (
      <div
        className="flex flex-col h-full"
        style={{
          background: "linear-gradient(180deg, rgba(250,250,250,0.72), rgba(244,244,244,0.58))",
          backdropFilter: "blur(22px) saturate(1.5)",
          WebkitBackdropFilter: "blur(22px) saturate(1.5)",
        }}
      >
        {/* Header */}
        <div className="flex items-center gap-2 px-4 py-4 border-b border-[var(--glass-border)] flex-shrink-0">
          {!collapsed && (
            <div className="flex items-center gap-2 flex-1">
              <div
                className="w-6 h-6 rounded-[7px] flex items-center justify-center font-bold text-xs shadow-sm flex-shrink-0"
                style={{ background: "var(--ink)", color: "var(--on-ink)", fontFamily: "Fraunces, Georgia, serif" }}
              >D</div>
              <span
                className="text-[15px] font-semibold tracking-tight"
                style={{ color: "var(--ink)", fontFamily: "Fraunces, Georgia, serif", letterSpacing: "-0.025em" }}
              >
                DocQuery
              </span>
            </div>
          )}
          {!collapsed && !isMobile && (
            <button
              onClick={() => document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true, bubbles: true }))}
              title="Command palette (⌘K)"
              className="px-1.5 py-0.5 rounded text-[10px] font-mono text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--accent)] hover:text-[var(--text-primary)] transition-colors"
            >
              ⌘K
            </button>
          )}
          <button
            onClick={() => isMobile ? setMobileOpen(false) : setCollapsed(!collapsed)}
            className="w-8 h-8 flex items-center justify-center rounded-lg bg-[var(--bg-hover)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {isMobile ? <X size={16} /> : collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
        </div>

        {/* New Chat */}
        <div className="p-3 flex-shrink-0">
          <button
            onClick={newChat}
            className={`w-full btn-primary flex items-center gap-2 justify-center ${collapsed ? "!px-2" : ""}`}
          >
            <Plus size={14} />
            {!collapsed && <span>New Chat</span>}
          </button>
        </div>

        {/* Collections */}
        {!collapsed && (
          <div className="px-2 pb-1">
            <div className="flex items-center gap-2 px-2 py-1.5">
              <FolderOpen size={12} className="text-[var(--text-muted)]" />
              <span className="text-[10px] font-medium text-[var(--text-muted)] uppercase tracking-[0.12em]">Collections</span>
              <button
                onClick={() => setShowNewCollection(!showNewCollection)}
                className="ml-auto p-0.5 rounded text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors"
                title="New Collection"
              >
                <FolderPlus size={12} />
              </button>
            </div>

            {/* New collection input */}
            {showNewCollection && (
              <div className="px-2 pb-2">
                <div className="flex gap-1">
                  <input
                    value={newCollectionName}
                    onChange={(e) => setNewCollectionName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleCreateCollection();
                      if (e.key === "Escape") { setShowNewCollection(false); setNewCollectionName(""); }
                    }}
                    placeholder="Collection name..."
                    className="flex-1 text-xs px-2 py-1 rounded-lg bg-[var(--bg-surface)] border border-[var(--border)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none focus:border-[var(--accent)] transition-colors"
                    autoFocus
                  />
                  <button
                    onClick={handleCreateCollection}
                    className="text-[10px] px-2 py-1 rounded-lg bg-[var(--accent)] text-white hover:opacity-90 transition-opacity"
                  >
                    Add
                  </button>
                </div>
              </div>
            )}

            {/* "All Documents" button */}
            <button
              onClick={() => setActiveCollectionId(null)}
              className={`w-full flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs transition-[background-color,color,box-shadow] ${
                activeCollectionId === null
                  ? "text-[var(--text-primary)] font-medium"
                  : "text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]"
              }`}
              style={activeCollectionId === null ? {
                background: "rgba(255,255,255,0.85)",
                border: "1px solid var(--line)",
                boxShadow: "var(--shadow-sm)",
              } : undefined}
            >
              <span className="truncate">All Documents</span>
              <span className="ml-auto text-[10px] text-[var(--text-muted)]">{docs.length}</span>
            </button>

            {/* Collection list */}
            {collections.map((coll) => (
              <div
                key={coll.id}
                className={`group flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs cursor-pointer border transition-[background-color,color,box-shadow] ${
                  activeCollectionId === coll.id
                    ? "text-[var(--text-primary)] font-medium border-[rgba(10,10,10,0.18)]"
                    : "text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] border-transparent"
                }`}
                style={activeCollectionId === coll.id ? {
                  background: "rgba(255,255,255,0.85)",
                  border: "1px solid var(--line)",
                  boxShadow: "var(--shadow-sm)",
                } : undefined}
                onClick={() => setActiveCollectionId(activeCollectionId === coll.id ? null : coll.id)}
              >
                <FolderOpen size={12} className={activeCollectionId === coll.id ? "text-[var(--accent)]" : "text-[var(--text-muted)]"} />
                <span className="truncate flex-1">{coll.name}</span>
                <span className="text-[10px] text-[var(--text-muted)]">{coll.document_count ?? 0}</span>
                {delCollId === coll.id ? (
                  <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
                    <button onClick={() => handleDeleteCollection(coll.id)} className="text-[9px] text-[var(--status-failed)] px-1.5 py-0.5 rounded border border-red-200">Yes</button>
                    <button onClick={() => setDelCollId(null)} className="text-[9px] text-[var(--text-muted)] px-1.5 py-0.5 rounded border border-[var(--border)]">No</button>
                  </div>
                ) : (
                  <button
                    onClick={(e) => { e.stopPropagation(); setDelCollId(coll.id); }}
                    className="p-0.5 rounded text-[var(--text-muted)] hover:text-[var(--status-failed)] opacity-0 group-hover:opacity-100 transition-[color,opacity]"
                  >
                    <Trash2 size={10} />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        {!collapsed && <hr className="divider-dotted mx-3" />}

        {/* Conversations */}
        {!collapsed && (
          <div className="flex-1 overflow-y-auto scrollbar-thin min-h-0">
            <div className="px-4 py-1.5">
              <span className="text-[10px] font-medium text-[var(--text-muted)] uppercase tracking-[0.12em]">
                Chats
              </span>
            </div>
            {convs.length === 0 ? (
              <p className="text-[var(--text-muted)] text-xs text-center mt-6 px-4">No conversations yet.</p>
            ) : (
              <ul className="px-2 space-y-0.5 pb-2">
                {convs.map((c) => (
                  <li key={c.id}>
                    <div
                      onClick={() => { router.push(`/app/chat/${c.id}`); setMobileOpen(false); }}
                      className={`flex items-center gap-2 px-3 py-2 rounded-xl cursor-pointer transition-[background-color,border-color] duration-[100ms] ease-[cubic-bezier(0.23,1,0.32,1)] group
                        ${activeId === c.id
                          ? "bg-[var(--bg-surface)] border border-[var(--accent)] shadow-[0_0_0_3px_rgba(10,10,10,0.06)]"
                          : "hover:bg-[var(--bg-hover)] border border-transparent"
                        }`}
                    >
                      <MessageSquare size={13} className={activeId === c.id ? "text-[var(--accent)]" : "text-[var(--text-muted)]"} />
                      <div className="flex-1 min-w-0">
                        {renamingId === c.id ? (
                          <input
                            ref={renameRef}
                            value={renameValue}
                            onChange={(e) => setRenameValue(e.target.value)}
                            onBlur={() => commitRename(c.id)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") commitRename(c.id);
                              if (e.key === "Escape") setRenamingId(null);
                            }}
                            onClick={(e) => e.stopPropagation()}
                            className="text-xs text-[var(--text-primary)] bg-[var(--bg-surface)] border border-[var(--accent)] rounded px-1 py-0.5 w-full outline-none"
                          />
                        ) : (
                          <p
                            className="text-xs text-[var(--text-primary)] truncate"
                            onDoubleClick={(e) => { e.stopPropagation(); startRename(c); }}
                            title="Double-click to rename"
                          >
                            {c.title || "Untitled"}
                          </p>
                        )}
                        <p className="text-[10px] text-[var(--text-muted)]">{timeAgo(c.updated_at)}</p>
                      </div>
                      {delConvId === c.id ? (
                        <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
                          <button onClick={() => delConv(c.id)} className="text-[9px] text-[var(--status-failed)] px-1.5 py-0.5 rounded border border-red-200">Yes</button>
                          <button onClick={() => setDelConvId(null)} className="text-[9px] text-[var(--text-muted)] px-1.5 py-0.5 rounded border border-[var(--border)]">No</button>
                        </div>
                      ) : (
                        <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button
                            onClick={(e) => { e.stopPropagation(); startRename(c); }}
                            className="p-1 rounded text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors"
                            title="Rename"
                          >
                            <Pencil size={11} />
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); setDelConvId(c.id); }}
                            className="p-1 rounded text-[var(--text-muted)] hover:text-[var(--status-failed)] transition-colors"
                            title="Delete"
                          >
                            <Trash2 size={11} />
                          </button>
                        </div>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* ── Dotted divider ── */}
        {!collapsed && <hr className="divider-dotted mx-3" />}

        {/* Documents */}
        {!collapsed && (
          <div className="flex-shrink-0 max-h-[35vh] flex flex-col">
            <div className="flex items-center gap-2 px-4 py-2.5 flex-shrink-0">
              <FileText size={12} className="text-[var(--text-muted)]" />
              <span className="text-[10px] font-medium text-[var(--text-muted)] uppercase tracking-[0.12em]">Documents</span>
              <span className="ml-auto text-[10px] text-[var(--text-muted)]">{docs.length}</span>
            </div>

            {/* Upload */}
            <div
              className={`card-dotted mx-3 mb-2 flex flex-col items-center justify-center gap-1 py-4 cursor-pointer transition-[border-color,background-color,opacity]
                ${uploadQueue ? "opacity-50 pointer-events-none" : ""}`}
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); e.currentTarget.style.borderColor = "var(--ink)"; e.currentTarget.style.background = "var(--accent-soft)"; }}
              onDragLeave={(e) => { e.currentTarget.style.borderColor = ""; e.currentTarget.style.background = ""; }}
              onDrop={(e) => { e.preventDefault(); e.currentTarget.style.borderColor = ""; e.currentTarget.style.background = ""; const files = Array.from(e.dataTransfer.files); if (files.length) uploadFiles(files); }}
            >
              {uploadQueue
                ? <div className="w-4 h-4 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
                : <Upload size={14} className="text-[var(--text-muted)]" />
              }
              <span className="text-[10px] text-[var(--text-muted)]">
                {uploadQueue ? `Uploading… (${uploadQueue.done}/${uploadQueue.total} done)` : "Drop files or click"}
              </span>
              <span className="text-[9px] text-[var(--text-muted)]/60">PDF · DOCX · PPTX · TXT · XLSX · Max 50MB each</span>
              <input ref={fileRef} type="file" accept={ACCEPTED} multiple className="hidden"
                onChange={(e) => { const files = Array.from(e.target.files ?? []); if (files.length) uploadFiles(files); e.target.value = ""; }} />
            </div>

            {/* Doc list */}
            <div className="overflow-y-auto scrollbar-thin px-3 pb-2 flex-1 min-h-0 space-y-1">
              {docs.length === 0 && !uploadQueue && (
                <p className="text-[10px] text-[var(--text-muted)] text-center py-4">
                  No documents yet. Upload one above.
                </p>
              )}
              {docs.map((doc) => (
                <div key={doc.id} className="card px-3 py-2 group">
                  <div className="flex items-center gap-2">
                    <FileText size={12} className="text-[var(--text-muted)] flex-shrink-0" />
                    <span className="text-[11px] text-[var(--text-primary)] truncate flex-1" title={doc.filename}>{doc.filename}</span>
                    {doc.status === "ready" && <CheckCircle size={11} className="text-[var(--status-ready)] flex-shrink-0" />}
                    {doc.status === "processing" && <div className="w-3 h-3 border-2 border-[var(--status-processing)] border-t-transparent rounded-full animate-spin flex-shrink-0" />}
                    {doc.status === "failed" && <AlertCircle size={11} className="text-[var(--status-failed)] flex-shrink-0" />}
                  </div>
                  <div className="flex items-center justify-between mt-1">
                    <span className={`text-[9px] ${doc.status === "failed" ? "text-[var(--status-failed)]" : "text-[var(--text-muted)]"}`}>
                      {doc.status === "processing"
                        ? `Processing${doc.processing_progress != null ? ` (${doc.processing_progress}%)` : "…"}`
                        : doc.status === "failed"
                        ? "Processing failed. Delete and re-upload."
                        : `${doc.chunk_count} chunks · ${fmtBytes(doc.file_size_bytes)}`}
                    </span>
                    <div className="flex items-center gap-1">
                      {/* Add/Remove from collection button */}
                      {activeCollectionId && doc.status === "ready" && (
                        <button
                          onClick={() => handleToggleDocInCollection(doc.id)}
                          className={`p-0.5 rounded transition-[color,opacity] ${
                            collectionDocIds.has(doc.id)
                              ? "text-[var(--accent)] opacity-100"
                              : "text-[var(--text-muted)] hover:text-[var(--accent)] opacity-0 group-hover:opacity-100"
                          }`}
                          title={collectionDocIds.has(doc.id) ? "Remove from collection" : "Add to collection"}
                        >
                          {collectionDocIds.has(doc.id)
                            ? <CheckCircle size={11} />
                            : <FolderPlus size={11} />
                          }
                        </button>
                      )}
                      {doc.status !== "processing" && (
                        delDocId === doc.id ? (
                          <div className="flex gap-1">
                            <button onClick={() => delDoc(doc.id)} className="text-[8px] text-[var(--status-failed)] px-1 py-0.5 rounded border border-red-200">Delete</button>
                            <button onClick={() => setDelDocId(null)} className="text-[8px] text-[var(--text-muted)] px-1 py-0.5 rounded border border-[var(--border)]">Cancel</button>
                          </div>
                        ) : (
                          <button onClick={() => setDelDocId(doc.id)} className="opacity-0 group-hover:opacity-100 text-[var(--text-muted)] hover:text-[var(--status-failed)] transition-[color,opacity]">
                            <Trash2 size={10} />
                          </button>
                        )
                      )}
                    </div>
                  </div>
                  {doc.status === "processing" && (
                    <div className="mt-1.5 h-0.5 bg-[var(--bg-active)] rounded-full overflow-hidden">
                      <div className="h-full bg-[var(--accent)] rounded-full transition-[width] duration-[500ms] ease-[cubic-bezier(0.23,1,0.32,1)]" style={{ width: `${doc.processing_progress ?? 30}%` }} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* User footer */}
        <div className="flex items-center gap-2 px-4 py-3 border-t border-[var(--border)] flex-shrink-0 mt-auto">
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <p className="text-xs text-[var(--text-primary)] truncate">{user?.email}</p>
            </div>
          )}
          <button onClick={() => router.push("/app/settings")}
            className="p-1.5 rounded-lg hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors"
            title="Analytics & Settings">
            <BarChart3 size={14} />
          </button>
          <button onClick={async () => { await logout(); router.push("/login"); }}
            className="p-1.5 rounded-lg hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--status-failed)] transition-colors">
            <LogOut size={14} />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex h-dvh overflow-hidden" style={{ background: "var(--canvas)" }}>
      {/* Soft aurora wash so glass surfaces (input, top bar, panels) blur real colour */}
      <div className="aurora aurora-soft" />
      <CommandPalette
        onNewChat={newChat}
        onUpload={() => fileRef.current?.click()}
        conversations={convs}
        collections={collections}
      />

      {/* Desktop Sidebar */}
      <aside
        className="sidebar-desktop flex-shrink-0 h-full border-r border-[var(--border)] transition-[width] duration-[200ms] ease-[cubic-bezier(0.23,1,0.32,1)]"
        style={{ width: collapsed ? 64 : 280, zIndex: 20 }}
      >
        {sidebar()}
      </aside>

      {/* Mobile Overlay */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              onClick={() => setMobileOpen(false)} className="fixed inset-0 bg-black/20 z-30 md:hidden" />
            <motion.aside initial={{ x: -300 }} animate={{ x: 0 }} exit={{ x: -300 }}
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
              className="fixed top-0 left-0 h-dvh w-[280px] bg-[var(--bg-surface)] z-40 md:hidden border-r border-[var(--border)] shadow-xl">
              {sidebar()}
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Main */}
      <main className="flex-1 flex flex-col h-dvh overflow-hidden relative" style={{ zIndex: 10 }}>
        {/* Mobile top bar */}
        <div className="sidebar-mobile-toggle items-center gap-2 px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-surface)]">
          <button onClick={() => setMobileOpen(true)} className="p-1.5 rounded-lg hover:bg-[var(--bg-hover)] text-[var(--text-muted)]">
            <Menu size={18} />
          </button>
          <span className="text-sm font-medium text-[var(--text-primary)]">DocQuery</span>
        </div>
        {children}
      </main>
    </div>
  );
}
