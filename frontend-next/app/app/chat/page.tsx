"use client";

// /app/chat — RETIRED (G2 Step H).
//
// The standalone chat composer landing has been re-homed under the vault: Ask now lives
// at /app/vault/[id]/ask (Step D), scoped to a vault by the route [id] (§9 risk #1: the
// URL is the authoritative scope). A vault-less /app/chat entry can't honour that scope,
// so this legacy route now redirects to Vault Home — the user picks a vault, then asks.
// We redirect rather than 404 so old links / bookmarks degrade gracefully (plan §4b,
// mirroring the USE_AGENT_CORE 404→fallback behaviour).

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function ChatIndexRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/app");
  }, [router]);
  return null;
}
