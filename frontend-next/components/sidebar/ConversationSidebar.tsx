"use client";

// components/sidebar/ConversationSidebar.tsx
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Plus,
  MessageSquare,
  Trash2,
  Pencil,
  Check,
  X,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { ConversationResponse } from "@/lib/api";
import { clsx } from "clsx";

function timeAgo(dateStr: string | null): string {
  if (!dateStr) return "";
  const diff = Date.now() - new Date(dateStr).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

interface ConversationSidebarProps {
  conversations: ConversationResponse[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onNewChat: () => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export function ConversationSidebar({
  conversations,
  activeId,
  onSelect,
  onDelete,
  onRename,
  onNewChat,
  collapsed,
  onToggleCollapse,
}: ConversationSidebarProps) {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);

  function startEdit(conv: ConversationResponse) {
    setEditingId(conv.id);
    setEditValue(conv.title);
  }

  function commitEdit(id: string) {
    if (editValue.trim()) onRename(id, editValue.trim());
    setEditingId(null);
  }

  return (
    <motion.aside
      animate={{ width: collapsed ? 64 : 280 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className="relative flex flex-col h-full glass-elevated border-r border-white/5 overflow-hidden flex-shrink-0"
      style={{ zIndex: 10 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-white/5">
        {!collapsed && (
          <span className="text-sm font-semibold text-text-primary tracking-wide">
            DocQuery
          </span>
        )}
        <button
          onClick={onToggleCollapse}
          className="ml-auto p-1.5 rounded-lg hover:bg-white/5 text-text-muted hover:text-text-primary transition-colors"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      {/* New Chat Button */}
      <div className="p-2">
        <button
          onClick={onNewChat}
          className={clsx(
            "w-full flex items-center gap-2 p-2.5 rounded-xl text-sm font-medium transition-all duration-200",
            "glass-accent hover:bg-accent-primary/20 text-text-primary",
            "hover:scale-[1.02] active:scale-[0.98]",
            collapsed ? "justify-center" : ""
          )}
          aria-label="New chat"
        >
          <Plus size={16} className="flex-shrink-0 text-accent-primary" />
          {!collapsed && <span>New Chat</span>}
        </button>
      </div>

      {/* Conversation List */}
      {!collapsed && (
        <div className="flex-1 overflow-y-auto scrollbar-thin px-2 pb-2">
          {conversations.length === 0 ? (
            <p className="text-text-muted text-xs text-center mt-8 px-4">
              No conversations yet.
              <br />
              Start a new chat!
            </p>
          ) : (
            <ul className="space-y-0.5">
              <AnimatePresence initial={false}>
                {conversations.map((conv) => (
                  <motion.li
                    key={conv.id}
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.15 }}
                    onMouseEnter={() => setHoveredId(conv.id)}
                    onMouseLeave={() => {
                      setHoveredId(null);
                      if (deletingId === conv.id) setDeletingId(null);
                    }}
                  >
                    <div
                      className={clsx(
                        "relative flex items-center gap-2 px-3 py-2 rounded-xl cursor-pointer transition-all duration-150 group",
                        activeId === conv.id
                          ? "glass-accent border-l-2 border-accent-primary"
                          : "hover:bg-white/5 border-l-2 border-transparent"
                      )}
                      onClick={() => {
                        if (!editingId) onSelect(conv.id);
                      }}
                    >
                      <MessageSquare
                        size={14}
                        className={clsx(
                          "flex-shrink-0",
                          activeId === conv.id
                            ? "text-accent-primary"
                            : "text-text-muted"
                        )}
                      />

                      {editingId === conv.id ? (
                        <div className="flex-1 flex items-center gap-1">
                          <input
                            autoFocus
                            value={editValue}
                            onChange={(e) => setEditValue(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") commitEdit(conv.id);
                              if (e.key === "Escape") setEditingId(null);
                            }}
                            className="flex-1 bg-transparent text-text-primary text-xs outline-none border-b border-accent-primary/50 pb-0.5"
                            onClick={(e) => e.stopPropagation()}
                          />
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              commitEdit(conv.id);
                            }}
                            className="text-status-ready hover:opacity-80"
                          >
                            <Check size={12} />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setEditingId(null);
                            }}
                            className="text-text-muted hover:opacity-80"
                          >
                            <X size={12} />
                          </button>
                        </div>
                      ) : (
                        <div className="flex-1 min-w-0">
                          <p className="text-xs text-text-primary truncate">
                            {conv.title || "Untitled"}
                          </p>
                          <p className="text-[10px] text-text-muted mt-0.5">
                            {timeAgo(conv.updated_at)}
                          </p>
                        </div>
                      )}

                      {/* Action buttons on hover */}
                      {hoveredId === conv.id && editingId !== conv.id && (
                        <div className="flex items-center gap-1 flex-shrink-0">
                          {deletingId === conv.id ? (
                            <>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onDelete(conv.id);
                                  setDeletingId(null);
                                }}
                                className="text-[10px] text-status-failed hover:opacity-80 px-1.5 py-0.5 rounded border border-status-failed/40"
                              >
                                Delete
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setDeletingId(null);
                                }}
                                className="text-[10px] text-text-muted hover:opacity-80 px-1.5 py-0.5 rounded border border-white/10"
                              >
                                Cancel
                              </button>
                            </>
                          ) : (
                            <>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  startEdit(conv);
                                }}
                                className="p-1 rounded text-text-muted hover:text-text-primary hover:bg-white/10 transition-colors"
                                aria-label="Rename conversation"
                              >
                                <Pencil size={11} />
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setDeletingId(conv.id);
                                }}
                                className="p-1 rounded text-text-muted hover:text-status-failed hover:bg-status-failed/10 transition-colors"
                                aria-label="Delete conversation"
                              >
                                <Trash2 size={11} />
                              </button>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  </motion.li>
                ))}
              </AnimatePresence>
            </ul>
          )}
        </div>
      )}
    </motion.aside>
  );
}
