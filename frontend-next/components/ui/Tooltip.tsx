"use client";

import { useState, useRef, useEffect, ReactNode } from "react";
import { clsx } from "clsx";

interface TooltipProps {
  content: ReactNode;
  children: ReactNode;
  side?: "top" | "bottom" | "left" | "right";
  delay?: number;
}

export function Tooltip({ content, children, side = "top", delay = 600 }: TooltipProps) {
  const [visible, setVisible] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  function show() {
    timerRef.current = setTimeout(() => setVisible(true), delay);
  }
  function hide() {
    if (timerRef.current) clearTimeout(timerRef.current);
    setVisible(false);
  }

  useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current); }, []);

  const positions: Record<string, string> = {
    top: "bottom-full left-1/2 -translate-x-1/2 mb-2",
    bottom: "top-full left-1/2 -translate-x-1/2 mt-2",
    left: "right-full top-1/2 -translate-y-1/2 mr-2",
    right: "left-full top-1/2 -translate-y-1/2 ml-2",
  };

  return (
    <span className="relative inline-flex items-center" onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}>
      {children}
      {visible && (
        <span
          role="tooltip"
          className={clsx(
            "absolute z-[var(--z-dropdown)] pointer-events-none whitespace-nowrap",
            "bg-[var(--text-primary)] text-white text-[11px] font-medium px-2 py-1 rounded-lg shadow-md",
            "animate-fade-in-up",
            positions[side]
          )}
        >
          {content}
        </span>
      )}
    </span>
  );
}
