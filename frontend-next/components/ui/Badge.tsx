"use client";

import { HTMLAttributes } from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

type BadgeVariant = "default" | "success" | "warning" | "error" | "warm";

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

const variants: Record<BadgeVariant, string> = {
  default: "bg-[var(--bg-hover)] text-[var(--text-secondary)] border-[var(--border)]",
  success: "bg-green-50 text-[var(--status-ready)] border-green-200",
  warning: "bg-amber-50 text-[var(--status-processing)] border-amber-200",
  error: "bg-red-50 text-[var(--status-failed)] border-red-200",
  warm: "bg-[var(--warm-50)] text-[var(--warm-700)] border-[var(--warm-300)]",
};

export function Badge({ variant = "default", className, children, ...props }: BadgeProps) {
  return (
    <span
      className={twMerge(
        clsx(
          "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium border",
          variants[variant],
          className
        )
      )}
      {...props}
    >
      {children}
    </span>
  );
}
