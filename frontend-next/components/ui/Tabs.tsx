"use client";

import { createContext, useContext, useState, ReactNode, HTMLAttributes } from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

interface TabsContextValue {
  active: string;
  setActive: (v: string) => void;
}

const TabsCtx = createContext<TabsContextValue>({ active: "", setActive: () => {} });

interface TabsProps {
  defaultValue: string;
  children: ReactNode;
  className?: string;
}

export function Tabs({ defaultValue, children, className }: TabsProps) {
  const [active, setActive] = useState(defaultValue);
  return (
    <TabsCtx.Provider value={{ active, setActive }}>
      <div className={clsx("flex flex-col", className)}>{children}</div>
    </TabsCtx.Provider>
  );
}

export function TabsList({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      role="tablist"
      className={twMerge(
        "flex items-center gap-1 border-b border-[var(--border)] pb-0",
        className
      )}
    >
      {children}
    </div>
  );
}

interface TabsTriggerProps {
  value: string;
  children: ReactNode;
  className?: string;
}

export function TabsTrigger({ value, children, className }: TabsTriggerProps) {
  const { active, setActive } = useContext(TabsCtx);
  const isActive = active === value;
  return (
    <button
      role="tab"
      aria-selected={isActive}
      onClick={() => setActive(value)}
      className={twMerge(
        clsx(
          "px-3 py-2 text-xs font-medium transition-all border-b-2 -mb-px focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]",
          isActive
            ? "text-[var(--text-primary)] border-[var(--accent)]"
            : "text-[var(--text-muted)] border-transparent hover:text-[var(--text-secondary)]",
          className
        )
      )}
    >
      {children}
    </button>
  );
}

interface TabsContentProps extends HTMLAttributes<HTMLDivElement> {
  value: string;
}

export function TabsContent({ value, children, className, ...props }: TabsContentProps) {
  const { active } = useContext(TabsCtx);
  if (active !== value) return null;
  return (
    <div role="tabpanel" className={clsx("pt-4", className)} {...props}>
      {children}
    </div>
  );
}
