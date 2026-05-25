// stores/auth.store.ts
// Zustand auth store — replaces st.session_state for user identity.
// Session is restored from Supabase cookie on mount via initialize().
// Token auto-refresh is handled by the Supabase SDK via onAuthStateChange.

"use client";

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { supabase } from "@/lib/supabase";

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
        supabase.auth.onAuthStateChange((event: any, session: any) => {
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
