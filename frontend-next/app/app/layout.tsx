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
  ConversationResponse,
  DocumentResponse,
} from "@/lib/api";
import { toast } from "sonner";

const ACCEPTED = ".pdf,.docx,.pptx,.txt,.xlsx";
const MAX_MB = 10;

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
  if (!b) return "0 KB";
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
  const [uploading, setUploading] = useState(false);
  const [delDocId, setDelDocId] = useState<string | null>(null);
  const [delConvId, setDelConvId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const renameRef = useRef<HTMLInputElement>(null);

  const activeId = (params?.id as string) ?? null;

  const loadConvs = useCallback(async () => {
    if (!token) return;
    try {
      const d = await listConversations(token);
      setConvs(d.sort((a, b) => new Date(b.updated_at ?? 0).getTime() - new Date(a.updated_at ?? 0).getTime()));
    } catch {}
  }, [token]);

  const loadDocs = useCallback(async () => {
    if (!token) return;
    try { setDocs(await listDocuments(token)); } catch {}
  }, [token]);

  useEffect(() => { loadConvs(); loadDocs(); }, [loadConvs, loadDocs]);

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

  async function upload(file: File) {
    if (!token) return;
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!ACCEPTED.split(",").includes(ext)) { toast.error(`Unsupported: ${ext}`); return; }
    if (file.size > MAX_MB * 1048576) { toast.error(`Max ${MAX_MB}MB`); return; }
    setUploading(true);
    try {
      const doc = await uploadDocument(token, file);
      setDocs((p) => [doc, ...p]);
      toast.success(`Uploaded ${file.name}`);
      loadDocs();
    } catch (e: unknown) {
      toast.error(`Upload failed: ${e instanceof Error ? e.message : "Error"}`);
    } finally { setUploading(false); }
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
      <div className="flex flex-col h-full bg-[var(--bg-sidebar)]">
        {/* Header */}
        <div className="flex items-center gap-2 px-4 py-4 border-b border-[var(--border)] flex-shrink-0">
          {!collapsed && (
            <span className="text-sm font-semibold text-[var(--text-primary)] tracking-tight">
              DocQuery
            </span>
          )}
          <button
            onClick={() => typeof window !== "undefined" && window.innerWidth < 768 ? setMobileOpen(false) : setCollapsed(!collapsed)}
            className="w-8 h-8 flex items-center justify-center rounded-lg bg-[var(--bg-hover)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
          >
            {typeof window !== "undefined" && window.innerWidth < 768 ? <X size={16} /> : collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
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
                      className={`flex items-center gap-2 px-3 py-2 rounded-xl cursor-pointer transition-all duration-100 group
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
                        <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-all">
                          <button
                            onClick={(e) => { e.stopPropagation(); startRename(c); }}
                            className="p-1 rounded text-[var(--text-muted)] hover:text-[var(--accent)] transition-all"
                            title="Rename"
                          >
                            <Pencil size={11} />
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); setDelConvId(c.id); }}
                            className="p-1 rounded text-[var(--text-muted)] hover:text-[var(--status-failed)] transition-all"
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
              className={`card-dotted mx-3 mb-2 flex flex-col items-center justify-center gap-1 py-4 cursor-pointer transition-all
                ${uploading ? "opacity-50 pointer-events-none" : ""}`}
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); e.currentTarget.style.borderColor = "#0A0A0A"; e.currentTarget.style.background = "#F5F5F5"; }}
              onDragLeave={(e) => { e.currentTarget.style.borderColor = ""; e.currentTarget.style.background = ""; }}
              onDrop={(e) => { e.preventDefault(); e.currentTarget.style.borderColor = ""; e.currentTarget.style.background = ""; const f = e.dataTransfer.files[0]; if (f) upload(f); }}
            >
              {uploading
                ? <div className="w-4 h-4 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
                : <Upload size={14} className="text-[var(--text-muted)]" />
              }
              <span className="text-[10px] text-[var(--text-muted)]">{uploading ? "Uploading…" : "Drop file or click"}</span>
              <span className="text-[9px] text-[var(--text-muted)]/60">PDF · DOCX · PPTX · TXT · Max 10MB</span>
              <input ref={fileRef} type="file" accept={ACCEPTED} className="hidden"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) upload(f); e.target.value = ""; }} />
            </div>

            {/* Doc list */}
            <div className="overflow-y-auto scrollbar-thin px-3 pb-2 flex-1 min-h-0 space-y-1">
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
                    <span className="text-[9px] text-[var(--text-muted)]">
                      {doc.status === "processing" ? `Processing${doc.processing_progress != null ? ` (${doc.processing_progress}%)` : "…"}` : `${doc.chunk_count} chunks · ${fmtBytes(doc.file_size_bytes)}`}
                    </span>
                    {doc.status !== "processing" && (
                      delDocId === doc.id ? (
                        <div className="flex gap-1">
                          <button onClick={() => delDoc(doc.id)} className="text-[8px] text-[var(--status-failed)] px-1 py-0.5 rounded border border-red-200">Delete</button>
                          <button onClick={() => setDelDocId(null)} className="text-[8px] text-[var(--text-muted)] px-1 py-0.5 rounded border border-[var(--border)]">Cancel</button>
                        </div>
                      ) : (
                        <button onClick={() => setDelDocId(doc.id)} className="opacity-0 group-hover:opacity-100 text-[var(--text-muted)] hover:text-[var(--status-failed)] transition-all">
                          <Trash2 size={10} />
                        </button>
                      )
                    )}
                  </div>
                  {doc.status === "processing" && (
                    <div className="mt-1.5 h-0.5 bg-[var(--bg-active)] rounded-full overflow-hidden">
                      <div className="h-full bg-[var(--accent)] rounded-full transition-all duration-500" style={{ width: `${doc.processing_progress ?? 30}%` }} />
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
          <button onClick={async () => { await logout(); router.push("/login"); }}
            className="p-1.5 rounded-lg hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--status-failed)] transition-colors ml-auto">
            <LogOut size={14} />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex h-dvh overflow-hidden bg-[var(--bg-base)]">
      {/* Desktop Sidebar */}
      <aside
        className="sidebar-desktop flex-shrink-0 h-full border-r border-[var(--border)] transition-all duration-200"
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
