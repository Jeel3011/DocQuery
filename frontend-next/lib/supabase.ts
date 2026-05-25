// lib/supabase.ts
// Singleton Supabase browser client for the Next.js App Router.
// Uses @supabase/ssr so sessions are stored in cookies (SSR-compatible)
// and tokens auto-refresh before expiry.

import { createBrowserClient } from "@supabase/ssr";

function createSafeBrowserClient() {
  if (typeof window === "undefined") {
    return {
      auth: {
        getSession: async () => ({ data: { session: null }, error: null }),
        onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
        signInWithPassword: async () => ({ error: new Error("SSR") }),
        signUp: async () => ({ error: new Error("SSR") }),
        signOut: async () => ({ error: null }),
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any;
  }

  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );
}

// Singleton — one client per browser tab
export const supabase = createSafeBrowserClient();

// Re-export for use in middleware (server-side, reads cookies)
export { createServerClient } from "@supabase/ssr";
