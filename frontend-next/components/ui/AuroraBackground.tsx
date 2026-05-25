// components/ui/AuroraBackground.tsx

import React from "react";

type AuroraBackgroundProps = React.HTMLAttributes<HTMLDivElement>;

export function AuroraBackground({ className = "", ...props }: AuroraBackgroundProps) {
  return (
    <div className={`pointer-events-none overflow-hidden bg-[#050811] ${className}`} {...props}>
      <div className="absolute -top-[200px] -left-[100px] w-[600px] h-[600px] bg-indigo-600 rounded-full blur-[120px] opacity-35 animate-aurora-drift" />
      <div className="absolute -bottom-[150px] -right-[100px] w-[500px] h-[500px] bg-sky-500 rounded-full blur-[120px] opacity-30 animate-aurora-drift" style={{ animationDelay: "-7s" }} />
      <div className="absolute top-[40%] left-[50%] -translate-x-1/2 w-[400px] h-[400px] bg-teal-500 rounded-full blur-[100px] opacity-25 animate-aurora-drift" style={{ animationDelay: "-14s" }} />
    </div>
  );
}
