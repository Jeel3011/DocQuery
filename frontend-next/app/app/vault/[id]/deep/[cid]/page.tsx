"use client";

// /app/vault/[id]/deep/[cid] — Deep Analysis report, re-homed under the vault (G5).
//
// A THIN wrapper, exactly like the Ask [cid] route: the 1,150-line conversation logic
// (agentcore.ev → live timeline, gate handling, ?q= auto-submit, report rendering, the
// ArtifactPanel) lives in ONE place — app/app/chat/[id]/page.tsx, exported as
// ChatConversation. Deep Analysis reuses it verbatim and only sets analysisMode="deep",
// which makes the request carry mode:"deep" → the backend uses the Deep Analysis prompt
// (sectioned report + survey_collection breadth pass) + the deep budget + the PER-SECTION
// citation gate. No second consumer, no forked streaming — one engine, one renderer.
//
// Scope (§9 risk #1): the vault [id] segment is the AUTHORITATIVE scope (passed as
// scopedCollectionId); [cid] is the conversation id (conversationId).

import { useParams } from "next/navigation";
import { ChatConversation } from "@/app/app/chat/[id]/page";

export default function VaultDeepAnalysisConversationPage() {
  const params = useParams<{ id: string; cid: string }>();
  return (
    <ChatConversation
      scopedCollectionId={params.id}
      conversationId={params.cid}
      analysisMode="deep"
    />
  );
}
