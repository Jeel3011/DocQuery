"use client";

// /app/vault/[id]/ask/[cid] — Ask, re-homed under the vault (G2 Step D).
//
// This is a THIN wrapper. The 1,150-line conversation logic (agentcore.ev → timeline
// mapping, gate handling, ?q= auto-submit, export/compare) lives in ONE place —
// app/app/chat/[id]/page.tsx, exported as ChatConversation — so the old /app/chat/[id]
// route and this re-homed route share a single source of truth (no drift; plan §6 #6).
//
// Scope (§9 risk #1): the vault [id] segment is the AUTHORITATIVE scope; we pass it as
// scopedCollectionId so the conversation depends on the URL, not the persisted store.
// The [cid] segment is the conversation id, passed as conversationId.

import { useParams } from "next/navigation";
import { ChatConversation } from "@/app/app/chat/[id]/page";

export default function VaultAskConversationPage() {
  const params = useParams<{ id: string; cid: string }>();
  return (
    <ChatConversation
      scopedCollectionId={params.id}
      conversationId={params.cid}
    />
  );
}
