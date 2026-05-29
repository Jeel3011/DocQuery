import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

interface CollectionStore {
  activeCollectionId: string | null;
  setActiveCollectionId: (id: string | null) => void;
}

export const useCollectionStore = create<CollectionStore>()(
  persist(
    (set) => ({
      activeCollectionId: null,
      setActiveCollectionId: (id) => set({ activeCollectionId: id }),
    }),
    {
      name: "docquery-collection",
      storage: createJSONStorage(() =>
        typeof window !== "undefined" ? localStorage : {
          getItem: () => null,
          setItem: () => {},
          removeItem: () => {},
        }
      ),
    }
  )
);
