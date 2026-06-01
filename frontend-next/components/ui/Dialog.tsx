"use client";

import { createContext, useContext, useState, ReactNode, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { clsx } from "clsx";

interface DialogContextValue {
  open: boolean;
  setOpen: (v: boolean) => void;
}

const DialogCtx = createContext<DialogContextValue>({ open: false, setOpen: () => {} });

export function Dialog({ children, open: controlledOpen, onOpenChange }: {
  children: ReactNode;
  open?: boolean;
  onOpenChange?: (v: boolean) => void;
}) {
  const [internalOpen, setInternalOpen] = useState(false);
  const open = controlledOpen ?? internalOpen;
  const setOpen = onOpenChange ?? setInternalOpen;
  return <DialogCtx.Provider value={{ open, setOpen }}>{children}</DialogCtx.Provider>;
}

export function DialogTrigger({ children, asChild }: { children: ReactNode; asChild?: boolean }) {
  const { setOpen } = useContext(DialogCtx);
  if (asChild) {
    return <span onClick={() => setOpen(true)} className="contents">{children}</span>;
  }
  return <button onClick={() => setOpen(true)} className="contents">{children}</button>;
}

interface DialogContentProps {
  children: ReactNode;
  className?: string;
  maxWidth?: string;
}

export function DialogContent({ children, className, maxWidth = "480px" }: DialogContentProps) {
  const { open, setOpen } = useContext(DialogCtx);

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
            transition={{ duration: 0.15 }}
            className="fixed inset-0 bg-black/40"
            style={{ zIndex: "var(--z-dialog)" as unknown as number }}
            onClick={() => setOpen(false)}
            aria-hidden="true"
          />
          <motion.div
            key="dialog"
            initial={{ opacity: 0, scale: 0.97, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: 8 }}
            transition={{ type: "spring", stiffness: 400, damping: 35 }}
            role="dialog"
            aria-modal="true"
            className={clsx(
              "fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2",
              "bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl shadow-xl",
              "w-full mx-4 flex flex-col",
              className
            )}
            style={{ maxWidth, zIndex: "var(--z-dialog)" as unknown as number }}
          >
            <button
              onClick={() => setOpen(false)}
              className="absolute top-4 right-4 p-1.5 rounded-lg hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
              aria-label="Close"
            >
              <X size={15} />
            </button>
            {children}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

export function DialogHeader({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={clsx("px-5 pt-5 pb-4 border-b border-[var(--border)]", className)}>
      {children}
    </div>
  );
}

export function DialogTitle({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <h2 className={clsx("text-sm font-semibold text-[var(--text-primary)] pr-6", className)}>
      {children}
    </h2>
  );
}

export function DialogFooter({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={clsx("px-5 py-4 border-t border-[var(--border)] flex items-center justify-end gap-2", className)}>
      {children}
    </div>
  );
}
