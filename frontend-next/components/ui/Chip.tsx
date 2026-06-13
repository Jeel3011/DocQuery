"use client";

import { HTMLAttributes } from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

type ChipVariant = "default" | "active" | "citation" | "fact" | "argument" | "strategy";

interface ChipProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: ChipVariant;
  num?: number;
}

const variants: Record<ChipVariant, string> = {
  default:
    "bg-[var(--bg-hover)] text-[var(--text-secondary)] border-[var(--border)] hover:bg-[var(--bg-active)] hover:text-[var(--text-primary)]",
  active:
    "bg-[var(--accent)] text-white border-[var(--accent)]",
  citation:
    "bg-[var(--bg-hover)] text-[var(--text-secondary)] border-[var(--border)] hover:bg-[var(--accent)] hover:text-white hover:border-[var(--accent)] hover:-translate-y-px",
  fact:
    "text-[var(--text-secondary)] border-[var(--border)] bg-transparent",
  argument:
    "text-[var(--accent)] border-[var(--accent)] bg-transparent",
  strategy:
    "text-[var(--ink-2)] border-[var(--line-2)] bg-[var(--surface-3)]",
};

export function Chip({ variant = "default", num, className, children, ...props }: ChipProps) {
  const isCitation = variant === "citation";
  return (
    <span
      className={twMerge(
        clsx(
          "inline-flex items-center justify-center border rounded transition-[background-color,color,border-color,transform] duration-[120ms] ease-[cubic-bezier(0.23,1,0.32,1)] cursor-default select-none",
          isCitation
            ? "w-[18px] h-[18px] text-[10px] font-bold rounded-[4px]"
            : "px-1.5 py-px text-[10px] font-semibold tracking-wide uppercase rounded-[4px]",
          variants[variant],
          className
        )
      )}
      {...props}
    >
      {isCitation ? num ?? children : children}
    </span>
  );
}
