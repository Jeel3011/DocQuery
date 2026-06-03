import React from "react";

export type GlassCardVariant = "default" | "elevated" | "accent";

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: GlassCardVariant;
  children: React.ReactNode;
}

export function GlassCard({
  variant = "default",
  children,
  className = "",
  ...props
}: GlassCardProps) {
  const baseStyles =
    "will-change-transform transition-[transform,box-shadow,border-color] duration-[120ms] ease-[cubic-bezier(0.23,1,0.32,1)]";

  const variantStyles = {
    default: "card",
    elevated: "card shadow-md",
    accent: "card-dotted",
  };

  return (
    <div
      className={`${baseStyles} ${variantStyles[variant]} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}
