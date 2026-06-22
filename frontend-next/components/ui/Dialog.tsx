"use client";

import { createContext, useContext, useState, ReactNode, useEffect } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
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
  const shouldReduceMotion = useReducedMotion();

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
          {/* Full-screen flex wrapper centers the dialog via flexbox — NOT via a transform —
              so Framer Motion's scale/opacity animation on the panel can't override the
              centering (the old `-translate-1/2` bug: Framer's inline transform wiped the
              Tailwind translate, anchoring the panel's top-left at screen-center → it drifted
              to the bottom-right). p-4 keeps a margin on small screens. */}
          <div
            className="fixed inset-0 flex items-center justify-center p-4 pointer-events-none"
            style={{ zIndex: "var(--z-dialog)" as unknown as number }}
          >
            <motion.div
              key="dialog"
              initial={{ opacity: 0, scale: shouldReduceMotion ? 1 : 0.97, y: shouldReduceMotion ? 0 : 8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: shouldReduceMotion ? 1 : 0.97, y: shouldReduceMotion ? 0 : 4 }}
              transition={{ duration: shouldReduceMotion ? 0.1 : 0.2, ease: [0.23, 1, 0.32, 1] }}
              role="dialog"
              aria-modal="true"
              className={clsx(
                "relative pointer-events-auto",
                "bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl shadow-xl",
                // Cap to the viewport and manage overflow as a flex column — a tall body
                // (the F1c create form) scrolls INSIDE while header/footer stay pinned.
                "w-full flex flex-col max-h-[calc(100vh-2rem)] overflow-hidden",
                className
              )}
              style={{ maxWidth }}
            >
              <button
                onClick={() => setOpen(false)}
                className="absolute top-4 right-4 p-1.5 rounded-lg hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors z-10"
                aria-label="Close"
              >
                <X size={15} />
              </button>
              {children}
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}

export function DialogHeader({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={clsx("px-5 pt-5 pb-4 border-b border-[var(--border)] shrink-0", className)}>
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
    <div className={clsx("px-5 py-4 border-t border-[var(--border)] flex items-center justify-end gap-2 shrink-0", className)}>
      {children}
    </div>
  );
}
