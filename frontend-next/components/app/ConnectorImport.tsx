"use client";

// ConnectorImport — the G8.6 vault connector entry point (Google Drive + email import).
// Sits beside UploadZone: a connector is just another SOURCE that lands files in this
// vault through the SAME ingestion pipeline, so once imported the docs show the identical
// pipeline tracks + fidelity dots a manual upload gets. The component:
//   1. probes GET /connectors/config — if it 404s (USE_CONNECTORS off) it renders nothing
//      (byte-identical guarantee, no dead button),
//   2. Google Drive: loads Google Identity Services on demand for the OAuth token flow +
//      the Picker, then POSTs the picked folder id + a short-lived access token,
//   3. Email: a small IMAP credential form (used per-request, never stored).
// The backend does the listing/fetching/dispatch; this is the thin connect UI.

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { HardDrive, Mail, Link2, X } from "lucide-react";
import { toast } from "sonner";
import {
  getConnectorConfig,
  importGoogleDriveFolder,
  importEmailAttachments,
  type ConnectorConfig,
  type ImportResult,
} from "@/lib/api";

interface ConnectorImportProps {
  token: string;
  onImported: () => void; // refetch the document list (drives the pipeline tracks)
}

// Minimal typings for the Google scripts we load on demand (no npm dep).
declare global {
  interface Window {
    google?: any;
    gapi?: any;
  }
}

const GIS_SRC = "https://accounts.google.com/gsi/client";
const GAPI_SRC = "https://apis.google.com/js/api.js";

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) return resolve();
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`failed to load ${src}`));
    document.head.appendChild(s);
  });
}

export function ConnectorImport({ token, onImported }: ConnectorImportProps) {
  const [config, setConfig] = useState<ConnectorConfig | null>(null);
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"drive" | "email">("drive");
  const [busy, setBusy] = useState(false);
  const accessTokenRef = useRef<string>("");

  useEffect(() => {
    let alive = true;
    getConnectorConfig(token)
      .then((c) => { if (alive) setConfig(c); })
      .catch(() => { if (alive) setConfig(null); });
    return () => { alive = false; };
  }, [token]);

  const reportResult = useCallback((r: ImportResult) => {
    if (r.queued > 0) toast.success(`${r.queued} file(s) importing from ${r.source}`);
    if (r.skipped > 0) toast.message(`${r.skipped} skipped (unsupported type)`);
    if (r.errored > 0) toast.error(`${r.errored} failed to import`);
    if (r.queued === 0 && r.skipped === 0 && r.errored === 0)
      toast.message("No supported files found.");
    onImported();
  }, [onImported]);

  // ── Google Drive: token flow → Picker → import the chosen folder ────────────────
  const startDrive = useCallback(async () => {
    if (!config?.google_drive?.client_id) {
      toast.error("Google Drive is not configured (missing client id).");
      return;
    }
    setBusy(true);
    try {
      await Promise.all([loadScript(GIS_SRC), loadScript(GAPI_SRC)]);

      // 1) Get a short-lived access token via the GIS token client.
      const accessToken: string = await new Promise((resolve, reject) => {
        const client = window.google.accounts.oauth2.initTokenClient({
          client_id: config.google_drive.client_id,
          scope: config.google_drive.scope,
          callback: (resp: any) => {
            if (resp?.access_token) resolve(resp.access_token);
            else reject(new Error("authorization cancelled"));
          },
        });
        client.requestAccessToken();
      });
      accessTokenRef.current = accessToken;

      // 2) Open the Picker scoped to folders so the user selects ONE folder to import.
      await new Promise<void>((resolve) => window.gapi.load("picker", () => resolve()));
      const view = new window.google.picker.DocsView(window.google.picker.ViewId.FOLDERS)
        .setSelectFolderEnabled(true)
        .setMimeTypes("application/vnd.google-apps.folder");
      const picker = new window.google.picker.PickerBuilder()
        .addView(view)
        .setOAuthToken(accessToken)
        .setCallback(async (data: any) => {
          if (data.action !== window.google.picker.Action.PICKED) {
            if (data.action === window.google.picker.Action.CANCEL) setBusy(false);
            return;
          }
          const doc = data.docs?.[0];
          if (!doc?.id) { setBusy(false); return; }
          try {
            const r = await importGoogleDriveFolder(token, accessTokenRef.current, doc.id, doc.name || "");
            reportResult(r);
            setOpen(false);
          } catch (e: unknown) {
            toast.error(`Drive import failed: ${e instanceof Error ? e.message : "unknown error"}`);
          } finally {
            setBusy(false);
          }
        })
        .build();
      picker.setVisible(true);
    } catch (e: unknown) {
      toast.error(`Google Drive: ${e instanceof Error ? e.message : "connection failed"}`);
      setBusy(false);
    }
  }, [config, token, reportResult]);

  // ── Email: IMAP credential form ─────────────────────────────────────────────────
  const [email, setEmail] = useState({ host: "imap.gmail.com", username: "", password: "", mailbox: "INBOX" });
  const submitEmail = useCallback(async () => {
    if (!email.host || !email.username || !email.password) {
      toast.error("Host, username and app-password are required.");
      return;
    }
    setBusy(true);
    try {
      const r = await importEmailAttachments(token, { ...email, port: 993, max_messages: 50 });
      reportResult(r);
      setOpen(false);
    } catch (e: unknown) {
      toast.error(`Email import failed: ${e instanceof Error ? e.message : "unknown error"}`);
    } finally {
      setBusy(false);
    }
  }, [email, token, reportResult]);

  // Feature off (config 404'd) → render nothing. No dead UI.
  if (!config?.enabled) return null;

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="w-full flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl text-[13px] font-medium transition-colors hover:bg-[var(--bg-hover)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
        style={{ background: "var(--surface)", border: "1px dashed var(--border-dotted)", color: "var(--ink-2)" }}
      >
        <span
          className="w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink-2)" }}
        >
          <Link2 size={13} />
        </span>
        <span>Import from a source</span>
        <span className="ml-auto text-[11px] text-[var(--text-muted)] hidden sm:inline">Drive · Email</span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[600] flex items-center justify-center p-4"
            style={{ background: "rgba(20,20,20,0.45)", backdropFilter: "blur(2px)" }}
            onClick={() => !busy && setOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.97, y: 8 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.97, y: 8 }}
              className="w-full max-w-md rounded-2xl overflow-hidden"
              style={{ background: "var(--surface)", border: "1px solid var(--line)" }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between px-5 py-4" style={{ borderBottom: "1px solid var(--line)" }}>
                <p style={{ fontFamily: "Fraunces, Georgia, serif", fontSize: 17, fontWeight: 500, color: "var(--ink)" }}>
                  Import documents
                </p>
                <button type="button" onClick={() => !busy && setOpen(false)} className="text-[var(--text-muted)] hover:text-[var(--ink)]">
                  <X size={16} />
                </button>
              </div>

              {/* Source tabs */}
              <div className="flex gap-1 px-5 pt-4">
                {([["drive", "Google Drive", HardDrive], ["email", "Email", Mail]] as const).map(([key, label, Icon]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setTab(key)}
                    disabled={key === "email" && !config.email?.available}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12.5px] font-medium transition-colors disabled:opacity-40"
                    style={tab === key
                      ? { background: "var(--ink)", color: "var(--on-ink)" }
                      : { background: "var(--surface-3)", color: "var(--ink-2)", border: "1px solid var(--line)" }}
                  >
                    <Icon size={13} /> {label}
                  </button>
                ))}
              </div>

              <div className="px-5 py-5">
                {tab === "drive" ? (
                  <div className="space-y-3">
                    <p className="text-[12.5px] text-[var(--ink-2)] leading-relaxed">
                      Connect Google Drive and pick a folder. Every supported file in it (PDF, DOCX,
                      XLSX, PPTX, TXT — Google Docs/Sheets are exported automatically) is imported into
                      this vault and processed like an upload.
                    </p>
                    <button
                      type="button"
                      onClick={startDrive}
                      disabled={busy}
                      className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-[13px] font-medium disabled:opacity-60"
                      style={{ background: "var(--ink)", color: "var(--on-ink)" }}
                    >
                      {busy ? "Connecting…" : (<><HardDrive size={14} /> Connect & pick a folder</>)}
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2.5">
                    <p className="text-[12.5px] text-[var(--ink-2)] leading-relaxed">
                      Import attachments from a mailbox over IMAP. Use an app-password — credentials are
                      used for this import only and never stored.
                    </p>
                    {([
                      ["host", "IMAP host", "imap.gmail.com", "text"],
                      ["username", "Email address", "you@example.com", "text"],
                      ["password", "App-password", "••••••••", "password"],
                      ["mailbox", "Mailbox / label", "INBOX", "text"],
                    ] as const).map(([key, label, ph, type]) => (
                      <label key={key} className="block">
                        <span className="text-[11px] text-[var(--text-muted)]">{label}</span>
                        <input
                          type={type}
                          value={(email as any)[key]}
                          placeholder={ph}
                          onChange={(e) => setEmail((s) => ({ ...s, [key]: e.target.value }))}
                          className="w-full mt-0.5 px-3 py-2 rounded-lg text-[13px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
                          style={{ background: "var(--surface-3)", border: "1px solid var(--line)", color: "var(--ink)" }}
                        />
                      </label>
                    ))}
                    <button
                      type="button"
                      onClick={submitEmail}
                      disabled={busy}
                      className="w-full flex items-center justify-center gap-2 px-4 py-2.5 mt-1 rounded-xl text-[13px] font-medium disabled:opacity-60"
                      style={{ background: "var(--ink)", color: "var(--on-ink)" }}
                    >
                      {busy ? "Importing…" : (<><Mail size={14} /> Import attachments</>)}
                    </button>
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
