import { create } from "zustand";

interface CollectionStore {
  activeCollectionId: string | null;
  setActiveCollectionId: (id: string | null) => void;
}

export const useCollectionStore = create<CollectionStore>((set) => ({
  activeCollectionId: null,
  setActiveCollectionId: (id) => set({ activeCollectionId: id }),
}));
