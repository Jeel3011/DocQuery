import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // ── Pure monochrome canvas system (black / white / grey) ──
        canvas: "#F4F4F4",
        surface: "#FFFFFF",
        "surface-2": "#FAFAFA",
        "surface-3": "#EFEFEF",

        ink: "#0E0E0E",
        "ink-2": "#525252",
        "ink-3": "#8A8A8A",
        "on-ink": "#FAFAFA",

        line: "#E4E4E4",
        "line-2": "#D4D4D4",

        "accent-taupe": "#525252",
        "accent-soft": "#EDEDED",

        // Legacy aliases
        "bg-base": "#F4F4F4",
        "bg-surface": "#FFFFFF",
        "bg-hover": "#EFEFEF",
        "bg-active": "#E6E6E6",
        "bg-sidebar": "#F4F4F4",
        "text-primary": "#0E0E0E",
        "text-secondary": "#525252",
        "text-muted": "#8A8A8A",
        "border-default": "#E4E4E4",
        "border-strong": "#D4D4D4",
        "border-dotted": "#C4C4C4",
        accent: "#0E0E0E",
        "accent-hover": "#262626",
        "accent-subtle": "#EDEDED",

        // Neutral grey ramp
        "warm-50": "#F7F7F7",
        "warm-100": "#EDEDED",
        "warm-300": "#D4D4D4",
        "warm-500": "#969696",
        "warm-700": "#5C5C5C",
        "warm-900": "#2E2E2E",

        "status-ready": "#2E2E2E",
        "status-processing": "#6B6B6B",
        "status-failed": "#C0392B",

        "conf-high": "#2E2E2E",
        "conf-med": "#6B6B6B",
        "conf-low": "#C0392B",

        "step-pending": "#8A8A8A",
        "step-active": "#0E0E0E",
        "step-done": "#2E2E2E",
        "step-failed": "#C0392B",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
        display: ["Fraunces", "Georgia", "Times New Roman", "serif"],
      },
      borderRadius: {
        xl: "12px",
        "2xl": "16px",
      },
      boxShadow: {
        sm: "0 1px 2px rgba(0,0,0,0.04)",
        md: "0 2px 8px rgba(0,0,0,0.06)",
        lg: "0 4px 16px rgba(0,0,0,0.08)",
        xl: "0 8px 32px rgba(0,0,0,0.10)",
        ring: "0 0 0 3px rgba(10,10,10,0.08)",
      },
      keyframes: {
        "cursor-blink": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "aurora-drift": {
          "0%, 100%": { transform: "translate(0, 0) scale(1)" },
          "33%": { transform: "translate(30px, -20px) scale(1.05)" },
          "66%": { transform: "translate(-20px, 30px) scale(0.95)" },
        },
        "fade-in-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "slide-in-right": {
          "0%": { opacity: "0", transform: "translateX(16px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        pulse: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.4" },
        },
      },
      animation: {
        "cursor-blink": "cursor-blink 0.7s ease-in-out infinite",
        shimmer: "shimmer 2s linear infinite",
        "aurora-drift": "aurora-drift 20s ease-in-out infinite",
        "fade-in-up": "fade-in-up 0.22s ease-out",
        "slide-in-right": "slide-in-right 0.22s ease-out",
        pulse: "pulse 2s ease-in-out infinite",
      },
      zIndex: {
        dropdown: "100",
        sticky: "200",
        drawer: "300",
        dialog: "400",
        toast: "500",
      },
    },
  },
  plugins: [],
};

export default config;
