"use client";

import { ReactNode } from "react";
import { clsx } from "clsx";

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}

export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div className={clsx("flex flex-col items-center justify-center text-center py-12 px-6", className)}>
      {icon && (
        <div className="w-10 h-10 rounded-2xl bg-[var(--bg-hover)] border border-dashed border-[var(--border-dotted)] flex items-center justify-center mb-4 text-[var(--text-muted)]">
          {icon}
        </div>
      )}
      <p className="text-sm font-medium text-[var(--text-primary)] mb-1">{title}</p>
      {description && (
        <p className="text-xs text-[var(--text-muted)] max-w-[240px] leading-relaxed mb-4">{description}</p>
      )}
      {action}
    </div>
  );
}
