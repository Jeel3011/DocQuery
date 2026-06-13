"use client";

// /app/vault/[id]/review — Review grid, re-homed under the vault (G2 Step E).
//
// This is a THIN wrapper. The grid logic (streamReviewGrid, live skeleton→value cells,
// clickable FOUND sources, coverage headline, CSV export) lives in ONE place —
// app/app/grid/page.tsx, exported as ReviewGrid — so the legacy /app/grid route and
// this re-homed route share a single source of truth (no drift; plan §6 #6).
//
// Scope (§9 risk #1): the vault [id] segment is the AUTHORITATIVE scope; we pass it as
// scopedCollectionId so the grid binds to the URL, not the persisted store. The in-page
// collection picker is dropped on this path — the route IS the scope.

import { useParams } from "next/navigation";
import { ReviewGrid } from "@/app/app/grid/page";

export default function VaultReviewPage() {
  const params = useParams<{ id: string }>();
  return <ReviewGrid scopedCollectionId={params.id} />;
}
