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
        // Monochrome system
        "bg-base": "#FAFAFA",
        "bg-surface": "#FFFFFF",
        "bg-hover": "#F5F5F5",
        "bg-active": "#F0F0F0",
        "bg-sidebar": "#FAFAFA",

        "text-primary": "#0A0A0A",
        "text-secondary": "#525252",
        "text-muted": "#A3A3A3",

        "border-default": "#E5E5E5",
        "border-strong": "#D4D4D4",
        "border-dotted": "#C4C4C4",

        accent: "#0A0A0A",
        "accent-hover": "#262626",
        "accent-subtle": "#F5F5F5",

        "status-ready": "#16A34A",
        "status-processing": "#CA8A04",
        "status-failed": "#DC2626",

        // Warm-neutral ramp — used sparingly for assets/illustrations/landing accents
        "warm-50": "#FAF8F5",
        "warm-100": "#F0EBE3",
        "warm-300": "#D9CFC0",
        "warm-500": "#A1907A",
        "warm-700": "#6B5B4D",
        "warm-900": "#3A312A",

        // Confidence semantic tokens (primary rendering is mono bar; color is secondary)
        "conf-high": "#16A34A",
        "conf-med": "#CA8A04",
        "conf-low": "#DC2626",

        // Agent/step state tokens
        "step-pending": "#A3A3A3",
        "step-active": "#0A0A0A",
        "step-done": "#16A34A",
        "step-failed": "#DC2626",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
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
