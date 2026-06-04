// stores/profile.store.ts
// Lightweight per-user profile preferences (client-side only).
// Currently holds the name the assistant should address the user by.
// Persisted in localStorage so it survives reloads without a backend change.

"use client";

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

interface ProfileStore {
  // Name the assistant uses to address the user. null = never set (prompt them).
  preferredName: string | null;
  // Set true once we've asked, so we don't nag a user who chose to skip.
  asked: boolean;
  // True once we've reconciled with the server (/auth/me) this session.
  hydrated: boolean;

  setPreferredName: (name: string | null) => void;
  markAsked: () => void;
  // Set the locally-cached value from the server without re-marking "asked".
  hydrateFromServer: (name: string | null) => void;
  // Wipe all cached profile state (called on logout — privacy boundary).
  reset: () => void;
}

export const useProfileStore = create<ProfileStore>()(
  persist(
    (set) => ({
      preferredName: null,
      asked: false,
      hydrated: false,
      setPreferredName: (name) =>
        set({ preferredName: name && name.trim() ? name.trim() : null, asked: true }),
      markAsked: () => set({ asked: true }),
      hydrateFromServer: (name) =>
        set((s) => ({
          preferredName: name ?? s.preferredName,
          // If the server already has a name, we never need to prompt.
          asked: name ? true : s.asked,
          hydrated: true,
        })),
      reset: () => set({ preferredName: null, asked: false, hydrated: false }),
    }),
    {
      name: "docquery-profile",
      storage: createJSONStorage(() =>
        typeof window !== "undefined"
          ? localStorage
          : { getItem: () => null, setItem: () => {}, removeItem: () => {} }
      ),
    }
  )
);
