"use client";

import { useState, useRef, useEffect, ReactNode } from "react";
import { clsx } from "clsx";

let tooltipGroupOpen = false;

interface TooltipProps {
  content: ReactNode;
  children: ReactNode;
  side?: "top" | "bottom" | "left" | "right";
  delay?: number;
}

export function Tooltip({ content, children, side = "top", delay = 600 }: TooltipProps) {
  const [visible, setVisible] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [instant, setInstant] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const rafRef = useRef<number | null>(null);

  function show() {
    if (tooltipGroupOpen) {
      // Subsequent hover within the group: appear instantly, no enter animation.
      setInstant(true);
      setVisible(true);
      setMounted(true);
    } else {
      setInstant(false);
      timerRef.current = setTimeout(() => {
        setVisible(true);
        tooltipGroupOpen = true;
        // Flip to the mounted (settled) state on the next frame so the
        // scale(0.97)→1 + opacity enter transition actually plays.
        rafRef.current = requestAnimationFrame(() => setMounted(true));
      }, delay);
    }
  }
  function hide() {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    setVisible(false);
    setMounted(false);
    setInstant(false);
    tooltipGroupOpen = false;
  }

  useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  }, []);

  const positions: Record<string, string> = {
    top: "bottom-full left-1/2 -translate-x-1/2 mb-2",
    bottom: "top-full left-1/2 -translate-x-1/2 mt-2",
    left: "right-full top-1/2 -translate-y-1/2 mr-2",
    right: "left-full top-1/2 -translate-y-1/2 ml-2",
  };

  const originMap: Record<string, string> = {
    top: "bottom center",
    bottom: "top center",
    left: "center right",
    right: "center left",
  };

  return (
    <span className="relative inline-flex items-center" onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}>
      {children}
      {visible && (
        <span
          role="tooltip"
          style={{
            transformOrigin: originMap[side],
            opacity: instant || mounted ? 1 : 0,
            transform: instant || mounted ? "scale(1)" : "scale(0.97)",
            transition: instant ? "none" : "opacity 125ms cubic-bezier(0.23,1,0.32,1), transform 125ms cubic-bezier(0.23,1,0.32,1)",
          }}
          className={clsx(
            "absolute z-[var(--z-dropdown)] pointer-events-none whitespace-nowrap",
            "bg-[var(--text-primary)] text-white text-[11px] font-medium px-2 py-1 rounded-lg shadow-md",
            positions[side]
          )}
        >
          {content}
        </span>
      )}
    </span>
  );
}
