"use client";

// VaultScopeSync — the vault-scope source-of-truth enforcer (G2 Step A, §9 risk #1).
//
// THE RULE: the route segment /app/vault/[id] is the AUTHORITATIVE vault scope.
// collection.store.activeCollectionId is DERIVED from it, never the reverse. On a
// deep-link, a second tab, or a back/forward navigation, the persisted store can hold
// a STALE id while the URL points at a different vault — and Ask/Review read the store
// at query time (streamAgentCoreQuery / streamReviewGrid pass collection_id). A
// wrong-vault answer looks perfectly confident: it is exactly the silent-wrong failure
// class this product exists to kill. So we make the URL win.
//
// Mount this once inside the app shell. Whenever the pathname carries a vault id, it
// pushes that id into the store. (Pages OUTSIDE a vault route — /app, /app/settings —
// deliberately leave the store untouched so the last-chosen vault persists as a hint
// for the composer; query-issuing surfaces all live UNDER /app/vault/[id], where the
// route id is present and therefore authoritative.)

import { useEffect } from "react";
import { usePathname } from "next/navigation";
import { useCollectionStore } from "@/stores/collection.store";

// Pull the vault id out of /app/vault/<id>/... — returns null off a vault route.
function vaultIdFromPath(pathname: string | null): string | null {
  if (!pathname) return null;
  const m = pathname.match(/^\/app\/vault\/([^/]+)/);
  return m ? decodeURIComponent(m[1]) : null;
}

export function VaultScopeSync() {
  const pathname = usePathname();
  const setActiveCollectionId = useCollectionStore((s) => s.setActiveCollectionId);

  useEffect(() => {
    const routeId = vaultIdFromPath(pathname);
    // Only sync when ON a vault route. The route is the source of truth there; the
    // store is just its cache. Off a vault route we don't clear the store — the
    // last-vault hint is harmless because no query surface reads it off-route.
    if (routeId) {
      // Read latest store value at run time (not via subscription) to avoid an extra
      // re-render loop; only write when it actually differs.
      const current = useCollectionStore.getState().activeCollectionId;
      if (current !== routeId) setActiveCollectionId(routeId);
    }
  }, [pathname, setActiveCollectionId]);

  return null;
}
