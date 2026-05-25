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
    "will-change-transform transition-all duration-200 ease-in-out";

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
