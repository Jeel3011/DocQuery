// stores/auth.store.ts
// Zustand auth store — replaces st.session_state for user identity.
// Session is restored from Supabase cookie on mount via initialize().
// Token auto-refresh is handled by the Supabase SDK via onAuthStateChange.

"use client";

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { supabase } from "@/lib/supabase";

let _authUnsub: (() => void) | null = null;

interface User {
  id: string;
  email: string;
}

interface AuthStore {
  user: User | null;
  token: string | null;
  isLoading: boolean;

  // Actions
  setUser: (user: User | null) => void;
  setToken: (token: string | null) => void;
  logout: () => Promise<void>;
  initialize: () => Promise<void>;
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isLoading: true,

      setUser: (user) => set({ user }),
      setToken: (token) => set({ token }),

      logout: async () => {
        await supabase.auth.signOut();
        set({ user: null, token: null });
        // Clear cached profile (in-memory + persisted) so the next user on this
        // browser never sees the previous user's preferred name. The server
        // (/auth/me) is the source of truth and re-hydrates on next login.
        try {
          const { useProfileStore } = await import("./profile.store");
          useProfileStore.getState().reset();
        } catch { /* ignore */ }
        if (typeof window !== "undefined") {
          try { window.localStorage.removeItem("docquery-profile"); } catch { /* ignore */ }
        }
      },

      initialize: async () => {
        set({ isLoading: true });

        // Restore session from cookie (works on page refresh / SSR)
        const {
          data: { session },
        } = await supabase.auth.getSession();

        if (session) {
          set({
            user: {
              id: session.user.id,
              email: session.user.email ?? "",
            },
            token: session.access_token,
          });
        }

        set({ isLoading: false });

        // Listen for token refresh + sign-out from any tab
        // Unsubscribe any previous listener before registering a new one
        const { data: { subscription } } = supabase.auth.onAuthStateChange((event: any, session: any) => {
          if (session) {
            set({
              user: {
                id: session.user.id,
                email: session.user.email ?? "",
              },
              token: session.access_token,
            });
          } else {
            set({ user: null, token: null });
          }
        });
        // Unsubscribe previous listener so repeated initialize() calls don't leak
        _authUnsub?.();
        _authUnsub = () => subscription.unsubscribe();
      },
    }),
    {
      name: "docquery-auth",
      // Only persist user object — NOT the token.
      // Token is always restored from the Supabase cookie on initialize().
      partialize: (state) => ({ user: state.user }),
      storage: createJSONStorage(() => 
        typeof window !== 'undefined' ? localStorage : {
          getItem: () => null,
          setItem: () => {},
          removeItem: () => {}
        }
      ),
    }
  )
);
