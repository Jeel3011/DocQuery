"use client";

import { createContext, useContext, useState, ReactNode, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { clsx } from "clsx";

interface SheetContextValue {
  open: boolean;
  setOpen: (v: boolean) => void;
}

const SheetCtx = createContext<SheetContextValue>({ open: false, setOpen: () => {} });

export function Sheet({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return <SheetCtx.Provider value={{ open, setOpen }}>{children}</SheetCtx.Provider>;
}

export function SheetTrigger({ children, asChild }: { children: ReactNode; asChild?: boolean }) {
  const { setOpen } = useContext(SheetCtx);
  if (asChild) {
    return <span onClick={() => setOpen(true)} className="contents">{children}</span>;
  }
  return (
    <button onClick={() => setOpen(true)} className="contents">
      {children}
    </button>
  );
}

interface SheetContentProps {
  children: ReactNode;
  className?: string;
  side?: "right" | "left";
  width?: string;
}

export function SheetContent({ children, className, side = "right", width = "420px" }: SheetContentProps) {
  const { open, setOpen } = useContext(SheetCtx);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/20"
            style={{ zIndex: "var(--z-drawer)" as unknown as number }}
            onClick={() => setOpen(false)}
            aria-hidden="true"
          />
          <motion.div
            key="sheet"
            initial={{ x: side === "right" ? "100%" : "-100%", opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: side === "right" ? "100%" : "-100%", opacity: 0 }}
            transition={{ type: "spring", stiffness: 320, damping: 32 }}
            className={clsx(
              "fixed top-0 bottom-0 flex flex-col bg-[var(--bg-surface)] border-l border-[var(--border)] shadow-xl overflow-hidden",
              side === "right" ? "right-0" : "left-0",
              className
            )}
            style={{ width, zIndex: "var(--z-drawer)" as unknown as number }}
            role="dialog"
            aria-modal="true"
          >
            <button
              onClick={() => setOpen(false)}
              className="absolute top-4 right-4 p-1.5 rounded-lg hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
              aria-label="Close"
            >
              <X size={16} />
            </button>
            {children}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

export function SheetHeader({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={clsx("px-6 pt-6 pb-4 border-b border-[var(--border)] flex-shrink-0", className)}>
      {children}
    </div>
  );
}

export function SheetTitle({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <h2 className={clsx("text-sm font-semibold text-[var(--text-primary)]", className)}>
      {children}
    </h2>
  );
}
