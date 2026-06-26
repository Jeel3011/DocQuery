// middleware.ts — MUST be at project root (not inside app/)
// Route protection: runs on every request BEFORE the page renders.
// Equivalent to Streamlit's `if not user: st.stop()`.
//
// Why cookies must be threaded through the response:
// Supabase SSR stores the session in cookies. When the SDK refreshes a token
// during a middleware check, it writes the new token to the response cookies.
// If we don't thread cookies through, the browser never gets the updated token
// and subsequent requests will fail with 401.

import { createServerClient } from "@supabase/ssr";
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Protected: must be authenticated
  const isProtected = pathname.startsWith("/app");

  // Guest-only: redirect to /app/chat if already logged in
  const isAuthPage =
    pathname.startsWith("/login") || pathname.startsWith("/signup");

  // Create a mutable response to thread cookies through
  const response = NextResponse.next({
    request: { headers: request.headers },
  });

  // Server-side session check (reads from cookies, never localStorage)
  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        get(name) {
          return request.cookies.get(name)?.value;
        },
        set(name, value, options) {
          // Thread new/refreshed cookies into BOTH request and response
          request.cookies.set({ name, value, ...options });
          response.cookies.set({ name, value, ...options });
        },
        remove(name, options) {
          request.cookies.set({ name, value: "", ...options });
          response.cookies.set({ name, value: "", ...options });
        },
      },
    }
  );

  const {
    data: { session },
  } = await supabase.auth.getSession();

  // Protect /app/* — redirect unauthenticated users to login
  if (isProtected && !session) {
    return NextResponse.redirect(new URL("/login", request.url));
  }

  // Redirect already-authenticated users away from /login and /signup
  if (isAuthPage && session) {
    return NextResponse.redirect(new URL("/app/chat", request.url));
  }

  // ── F2g (T8) — Firm Console route guard: BLOCK, don't just hide ──────────────────────────────
  // The console is gated server-side on `manage_members` (and external client/guest are excluded
  // entirely). The store decides what RENDERS, but a paralegal who *types* /app/settings/firm must
  // be REDIRECTED, not shown a hidden link — defense in depth, the server is the boundary.
  //
  // We resolve the SERVER's cap decision via /auth/capabilities (the same path require_cap trusts,
  // so the middleware can't drift from authz.py ROLE_CAPS). If the user lacks the cap (or is an
  // external role), redirect to /app/settings (the non-firm settings they CAN see). A network
  // hiccup fails OPEN to the page — the page itself re-checks via the store + every action 403s
  // server-side, so a transient miss can never grant access to a real ACTION.
  if (session && pathname.startsWith("/app/settings/firm")) {
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/v1/auth/capabilities`,
        { headers: { Authorization: `Bearer ${session.access_token}` }, cache: "no-store" }
      );
      if (res.ok) {
        const caps = await res.json();
        const allowed =
          !caps.is_external && Array.isArray(caps.caps) && caps.caps.includes("manage_members");
        if (!allowed) {
          return NextResponse.redirect(new URL("/app/settings", request.url));
        }
      }
      // res not ok → fall through (the page re-checks; actions stay server-gated).
    } catch {
      // Network error → fall through. The page + every action remain server-gated.
    }
  }

  return response;
}

// Only run on these paths — skip static files and API routes for performance
export const config = {
  matcher: ["/app/:path*", "/login", "/signup"],
};
