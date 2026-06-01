"use client";

import { HTMLAttributes } from "react";
import { clsx } from "clsx";

export function Surface({ className, children, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={clsx("surface", className)} {...props}>
      {children}
    </div>
  );
}

export function Panel({ className, children, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={clsx("panel", className)} {...props}>
      {children}
    </div>
  );
}
