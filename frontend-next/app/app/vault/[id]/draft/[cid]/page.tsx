"use client";

// /app/vault/[id]/draft/[cid] — Draft Deliverable conversation view (G6.1).
//
// THIN wrapper — same pattern as ask/[cid] and deep/[cid]. All streaming logic lives
// in ChatConversation (app/app/chat/[id]/page.tsx). Draft only sets analysisMode="draft"
// and passes the doc_type + instructions from the URL so ChatConversation forwards them
// in the stream body. No new orchestrator; one engine, one renderer.
//
// Scope (§9 risk #1): the vault [id] is passed as scopedCollectionId.

import { useParams, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { ChatConversation } from "@/app/app/chat/[id]/page";

function VaultDraftConversationInner() {
  const params = useParams<{ id: string; cid: string }>();
  const searchParams = useSearchParams();
  return (
    <ChatConversation
      scopedCollectionId={params.id}
      conversationId={params.cid}
      analysisMode="draft"
      draftDocType={searchParams.get("doc_type")}
      draftInstructions={searchParams.get("instructions")}
    />
  );
}

export default function VaultDraftConversationPage() {
  return (
    <Suspense fallback={<div className="flex-1" />}>
      <VaultDraftConversationInner />
    </Suspense>
  );
}
