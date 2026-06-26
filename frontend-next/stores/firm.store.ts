// stores/firm.store.ts
// F2g — the Firm Console store. Holds the per-session view of the firm the user is acting in:
// the active firm, the server-RESOLVED effective capability set, and the lists the console
// renders (members, screens, my review queue).
//
// THE SECURITY MODEL (plan §2.1 — defense in depth, the server is ALWAYS the boundary):
//   • Caps are the SERVER's decision (getCapabilities → resolve_membership + authz.caps_for_role),
//     loaded ONCE per session to decide WHAT RENDERS. They are NOT a hardcoded role→button map in
//     JS that could drift from authz.py ROLE_CAPS (surface 10, the caps source of truth).
//   • The route guard — not this store — is the security. Every action re-checks server-side; a
//     stale render can never grant a revoked permission, because the POST/PATCH/DELETE is the gate.
//   • Stale-cap invalidation (T7): a 403 from any gated action, or a role-change/offboard the UI
//     performs, triggers refreshCaps() — the store can never out-live a revoked permission. Call
//     onForbidden() from a catch block when an action 403s.
//
// No new SSE: override + review actions are POSTs the existing gate/meta stream reflects on the
// next answer. This store is plain client state hydrated from the typed lib/api.ts wrappers.

"use client";

import { create } from "zustand";
import {
  getCapabilities, getFirm, listMembers, listScreens, getReviewQueue,
  type Capability, type FirmRole, type Firm, type CapabilitiesResponse,
  type MemberResponse, type ScreenResponse, type ReviewRequest,
  APIError,
} from "@/lib/api";

interface FirmStore {
  // Active firm + the server-resolved effective caps (surface 9 + 10).
  firm: Firm | null;
  role: FirmRole | null;
  caps: Set<Capability>;
  isExternal: boolean;
  delegatedVerbs: Set<Capability>;

  // The console lists (server-filtered to the caller's firm before they reach the client, T2).
  members: MemberResponse[];
  screens: ScreenResponse[];
  reviewQueue: ReviewRequest[];

  // Per-section loading + error so each surface designs its own loading/empty/error state.
  loadingCaps: boolean;
  loadingMembers: boolean;
  loadingScreens: boolean;
  loadingQueue: boolean;
  error: string | null;
  initialized: boolean;

  // Derived (never a hardcoded role map — always reads the server-resolved caps).
  can: (verb: Capability) => boolean;

  // Lifecycle.
  init: (token: string) => Promise<void>;
  // Latency: hydrate caps + firm from an already-fetched bootstrap payload (no extra round-trip).
  hydrate: (capsRes: CapabilitiesResponse, firm: Firm | null) => void;
  refreshCaps: (token: string) => Promise<void>;
  loadMembers: (token: string) => Promise<void>;
  loadScreens: (token: string) => Promise<void>;
  loadQueue: (token: string) => Promise<void>;
  // T7: call from a 403 catch — re-resolve caps from the server so the UI reconciles with the
  // (possibly revoked) permission immediately. Returns true if it ran (the error WAS a 403).
  onForbidden: (token: string, err: unknown) => Promise<boolean>;
  reset: () => void;
}

const EMPTY = new Set<Capability>();

export const useFirmStore = create<FirmStore>((set, get) => ({
  firm: null,
  role: null,
  caps: EMPTY,
  isExternal: false,
  delegatedVerbs: EMPTY,
  members: [],
  screens: [],
  reviewQueue: [],
  loadingCaps: false,
  loadingMembers: false,
  loadingScreens: false,
  loadingQueue: false,
  error: null,
  initialized: false,

  can: (verb) => get().caps.has(verb),

  init: async (token) => {
    if (!token) return;
    set({ loadingCaps: true, error: null });
    try {
      // Caps + firm in parallel — both are cheap server reads. Caps is the source of truth for
      // rendering; firm carries the display name + the active-firm identity (surface 9).
      const [capsRes, firm] = await Promise.all([
        getCapabilities(token),
        getFirm(token),
      ]);
      set({
        caps: new Set(capsRes.caps),
        role: capsRes.role,
        isExternal: capsRes.is_external,
        delegatedVerbs: new Set(capsRes.delegated_verbs),
        firm: firm,
        loadingCaps: false,
        initialized: true,
      });
    } catch {
      set({ loadingCaps: false, error: "Could not load your firm permissions.", initialized: true });
    }
  },

  hydrate: (capsRes, firm) => {
    // Populate from a bootstrap payload already fetched by the layout — no extra round-trip.
    set({
      caps: new Set(capsRes.caps),
      role: capsRes.role,
      isExternal: capsRes.is_external,
      delegatedVerbs: new Set(capsRes.delegated_verbs),
      firm: firm,
      loadingCaps: false,
      initialized: true,
    });
  },

  refreshCaps: async (token) => {
    if (!token) return;
    try {
      const capsRes = await getCapabilities(token);
      set({
        caps: new Set(capsRes.caps),
        role: capsRes.role,
        isExternal: capsRes.is_external,
        delegatedVerbs: new Set(capsRes.delegated_verbs),
      });
    } catch {
      /* leave the last-known caps; the route guard is still the boundary */
    }
  },

  loadMembers: async (token) => {
    if (!token) return;
    set({ loadingMembers: true });
    try {
      const members = await listMembers(token);
      set({ members, loadingMembers: false });
    } catch (err) {
      await get().onForbidden(token, err);
      set({ loadingMembers: false });
      throw err;
    }
  },

  loadScreens: async (token) => {
    if (!token) return;
    set({ loadingScreens: true });
    try {
      const screens = await listScreens(token);
      set({ screens, loadingScreens: false });
    } catch (err) {
      await get().onForbidden(token, err);
      set({ loadingScreens: false });
      throw err;
    }
  },

  loadQueue: async (token) => {
    if (!token) return;
    set({ loadingQueue: true });
    try {
      const reviewQueue = await getReviewQueue(token);
      set({ reviewQueue, loadingQueue: false });
    } catch (err) {
      await get().onForbidden(token, err);
      set({ loadingQueue: false });
      throw err;
    }
  },

  onForbidden: async (token, err) => {
    if (err instanceof APIError && err.status === 403) {
      // The server revoked a permission mid-session (role change / offboard / new screen). Re-resolve
      // the caps so the UI reconciles to reality on the spot — never out-live a revoked permission.
      await get().refreshCaps(token);
      return true;
    }
    return false;
  },

  reset: () =>
    set({
      firm: null, role: null, caps: EMPTY, isExternal: false, delegatedVerbs: EMPTY,
      members: [], screens: [], reviewQueue: [],
      loadingCaps: false, loadingMembers: false, loadingScreens: false, loadingQueue: false,
      error: null, initialized: false,
    }),
}));
