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
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      borderRadius: {
        xl: "12px",
        "2xl": "16px",
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
      },
      animation: {
        "cursor-blink": "cursor-blink 0.7s ease-in-out infinite",
        shimmer: "shimmer 2s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
