"use client";

import { HTMLAttributes } from "react";
import { clsx } from "clsx";

interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {
  lines?: number;
  height?: string;
}

export function Skeleton({ lines, height, className, ...props }: SkeletonProps) {
  if (lines && lines > 1) {
    return (
      <div className={clsx("space-y-2", className)} {...props}>
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className="skeleton rounded"
            style={{ height: height ?? "12px", width: i === lines - 1 ? "60%" : "100%" }}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      className={clsx("skeleton rounded", className)}
      style={{ height: height ?? "12px" }}
      {...props}
    />
  );
}

export function SkeletonMessage() {
  return (
    <div className="flex gap-3 px-4 md:px-8 py-3">
      <div className="w-7 h-7 rounded-lg skeleton flex-shrink-0 mt-1" />
      <div className="flex-1 space-y-2">
        <div className="h-2.5 skeleton rounded w-16" />
        <div className="h-[72px] skeleton rounded-xl" />
      </div>
    </div>
  );
}
