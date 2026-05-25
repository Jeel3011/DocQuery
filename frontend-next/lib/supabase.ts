// lib/supabase.ts
// Singleton Supabase browser client for the Next.js App Router.
// Uses @supabase/ssr so sessions are stored in cookies (SSR-compatible)
// and tokens auto-refresh before expiry.

import { createBrowserClient } from "@supabase/ssr";

// Singleton — one client per browser tab (not re-created on every render)
export const supabase = createBrowserClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

// Re-export for use in middleware (server-side, reads cookies)
export { createServerClient } from "@supabase/ssr";
