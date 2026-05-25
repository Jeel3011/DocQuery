"use client";

// app/test/page.tsx
// Foundation checkpoint — verifies auth + API + streaming work end-to-end.
// DELETE THIS FILE once Phase 2B UI is built.

import { useEffect, useState } from "react";
import { useAuthStore } from "@/stores/auth.store";
import { supabase } from "@/lib/supabase";
import { listDocuments } from "@/lib/api";
import { streamQuery } from "@/lib/streaming";

export default function TestPage() {
  const { user, token, isLoading } = useAuthStore();
  const [docs, setDocs] = useState<string>("...");
  const [streamOut, setStreamOut] = useState<string>("");
  const [streaming, setStreaming] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authMsg, setAuthMsg] = useState("");

  async function handleLogin() {
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    setAuthMsg(error ? `❌ ${error.message}` : "✅ Logged in");
  }

  async function handleLogout() {
    await supabase.auth.signOut();
    setAuthMsg("Logged out");
  }

  async function testDocuments() {
    if (!token) return setDocs("No token");
    try {
      const d = await listDocuments(token);
      setDocs(JSON.stringify(d.slice(0, 2), null, 2));
    } catch (e: unknown) {
      setDocs(`Error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function testStream() {
    if (!token) return;
    setStreaming(true);
    setStreamOut("");
    await streamQuery(
      token,
      { question: "What is this document about?" },
      {
        onSources: (s) => setStreamOut((p) => p + `\n[SOURCES: ${s.length}]\n`),
        onToken: (t) => setStreamOut((p) => p + t),
        onDone: () => setStreaming(false),
        onError: (e) => { setStreamOut((p) => p + `\n[ERROR: ${e}]`); setStreaming(false); },
      }
    );
  }

  return (
    <div style={{ padding: 32, fontFamily: "monospace", background: "#050811", color: "#F1F5F9", minHeight: "100vh" }}>
      <h1 style={{ color: "#6366F1" }}>🧪 DocQuery Foundation Checkpoint</h1>

      <section style={{ marginTop: 24, padding: 16, border: "1px solid #333", borderRadius: 8 }}>
        <h2>1. Auth State</h2>
        {isLoading ? <p>Loading…</p> : user ? (
          <div>
            <p>✅ Logged in as: <strong>{user.email}</strong></p>
            <p>User ID: {user.id}</p>
            <p>Token: {token?.slice(0, 40)}…</p>
            <button onClick={handleLogout} style={btnStyle}>Sign Out</button>
          </div>
        ) : (
          <div>
            <p>⚠️ Not logged in</p>
            <input placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} style={inputStyle} />
            <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} style={inputStyle} />
            <button onClick={handleLogin} style={btnStyle}>Sign In</button>
            {authMsg && <p>{authMsg}</p>}
          </div>
        )}
      </section>

      <section style={{ marginTop: 24, padding: 16, border: "1px solid #333", borderRadius: 8 }}>
        <h2>2. API — List Documents</h2>
        <button onClick={testDocuments} style={btnStyle}>GET /documents</button>
        <pre style={{ marginTop: 8, color: "#10B981", fontSize: 12 }}>{docs}</pre>
      </section>

      <section style={{ marginTop: 24, padding: 16, border: "1px solid #333", borderRadius: 8 }}>
        <h2>3. SSE Streaming</h2>
        <button onClick={testStream} disabled={streaming || !token} style={btnStyle}>
          {streaming ? "Streaming…" : "POST /query/stream"}
        </button>
        <pre style={{ marginTop: 8, color: "#F59E0B", fontSize: 12, whiteSpace: "pre-wrap" }}>
          {streamOut || "(stream output will appear here)"}
        </pre>
      </section>
    </div>
  );
}

const btnStyle = {
  marginTop: 8,
  marginRight: 8,
  padding: "6px 14px",
  background: "#6366F1",
  color: "#fff",
  border: "none",
  borderRadius: 6,
  cursor: "pointer",
};

const inputStyle = {
  display: "block",
  marginBottom: 8,
  padding: "6px 10px",
  background: "#161b22",
  border: "1px solid #333",
  borderRadius: 6,
  color: "#F1F5F9",
  width: 280,
};
