import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/components/AuthProvider";
import { Toaster } from "sonner";

export const metadata: Metadata = {
  title: "DocQuery: Intelligent Document Q&A",
  description:
    "Ask questions across your documents. Multi-source answers with source citations and confidence scoring.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body>
        <AuthProvider>
          {children}
          <Toaster
            position="bottom-right"
            theme="light"
            toastOptions={{
              style: {
                background: "#FFFFFF",
                border: "1px solid #E5E5E5",
                color: "#0A0A0A",
                boxShadow: "0 4px 16px rgba(0,0,0,0.08)",
                borderRadius: "12px",
              },
            }}
          />
        </AuthProvider>
      </body>
    </html>
  );
}
