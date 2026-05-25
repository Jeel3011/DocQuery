"use client";

// components/AuthProvider.tsx
// Bootstraps the Zustand auth store on first mount.
// Placed in the root layout so auth is available app-wide.
// No loading spinner — each page/component handles its own loading state.

import { useEffect } from "react";
import { useAuthStore } from "@/stores/auth.store";

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const initialize = useAuthStore((s) => s.initialize);

  useEffect(() => {
    initialize();
  }, [initialize]);

  return <>{children}</>;
}
