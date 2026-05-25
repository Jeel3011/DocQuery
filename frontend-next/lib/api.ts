// lib/api.ts
// Typed wrappers for every FastAPI endpoint in DocQuery.
// No React imports — pure async TypeScript functions.
// Mirrors src/api/schemas.py exactly.

import axios, { AxiosError } from "axios";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_V1 = `${API_BASE}/api/v1`;

// ─── Types (mirror schemas.py) ────────────────────────────────────────────────

export interface DocumentResponse {
  id: string;
  filename: string;
  file_type: string | null;
  status: "processing" | "ready" | "failed";
  chunk_count: number;
  file_size_bytes: number | null;
  created_at: string | null;
  processing_progress?: number | null;
}

export interface ConversationResponse {
  id: string;
  title: string;
  created_at: string | null;
  updated_at: string | null;
}

export interface SourceInfo {
  source_id?: number;
  filename: string | null;
  page: number | string | null;
  chunk_type: string | null;
  content?: string;
}

export interface MessageResponse {
  id: string | null;
  role: "user" | "assistant";
  content: string;
  sources: SourceInfo[] | null;
  created_at: string | null;
}

export interface QueryRequest {
  question: string;
  filename_filter?: string | null;
  page_filter?: number | null;
  conversation_id?: string | null;
}

// ─── Error class ──────────────────────────────────────────────────────────────

export class APIError extends Error {
  constructor(
    public status: number,
    message: string,
    public detail?: string
  ) {
    super(message);
    this.name = "APIError";
  }
}

function handleAxiosError(err: unknown): never {
  if (err instanceof AxiosError) {
    const status = err.response?.status ?? 0;
    const detail = err.response?.data?.detail ?? err.message;
    throw new APIError(status, detail, detail);
  }
  throw err;
}

// ─── Axios factory (per-request token injection) ──────────────────────────────

function makeClient(token: string) {
  return axios.create({
    baseURL: API_V1,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    timeout: 30_000,
  });
}

// ─── Documents ────────────────────────────────────────────────────────────────

export async function listDocuments(
  token: string
): Promise<DocumentResponse[]> {
  try {
    const res = await makeClient(token).get<{
      documents: DocumentResponse[];
      total: number;
    }>("/documents");
    return res.data.documents;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function uploadDocument(
  token: string,
  file: File
): Promise<DocumentResponse> {
  try {
    const form = new FormData();
    form.append("file", file);
    const res = await makeClient(token).post<DocumentResponse>(
      "/documents/upload",
      form,
      {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120_000, // uploads can be slow
      }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function deleteDocument(
  token: string,
  docId: string
): Promise<void> {
  try {
    await makeClient(token).delete(`/documents/${docId}`);
  } catch (err) {
    handleAxiosError(err);
  }
}

// ─── Conversations ────────────────────────────────────────────────────────────

export async function listConversations(
  token: string
): Promise<ConversationResponse[]> {
  try {
    const res = await makeClient(token).get<{
      conversations: ConversationResponse[];
      total: number;
    }>("/conversations");
    return res.data.conversations;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function createConversation(
  token: string,
  title = "New Chat"
): Promise<ConversationResponse> {
  try {
    const res = await makeClient(token).post<ConversationResponse>(
      "/conversations",
      { title }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function deleteConversation(
  token: string,
  convId: string
): Promise<void> {
  try {
    await makeClient(token).delete(`/conversations/${convId}`);
  } catch (err) {
    handleAxiosError(err);
  }
}

// NOTE: renameConversation requires a PATCH /conversations/:id endpoint
// in FastAPI (not yet implemented). Tracked in deferred items.
export async function renameConversation(
  _token: string,
  _convId: string,
  _title: string
): Promise<void> {
  throw new APIError(501, "Rename not yet implemented in FastAPI backend.");
}

// ─── Messages ─────────────────────────────────────────────────────────────────

export async function getMessages(
  token: string,
  convId: string
): Promise<MessageResponse[]> {
  try {
    const res = await makeClient(token).get<{
      messages: MessageResponse[];
      conversation_id: string;
    }>(`/conversations/${convId}/messages`);
    return res.data.messages;
  } catch (err) {
    handleAxiosError(err);
  }
}

// ─── Health ───────────────────────────────────────────────────────────────────

export async function getHealth(): Promise<{
  status: string;
  dependencies: Record<string, string>;
  circuit_breakers?: Record<string, unknown>;
}> {
  try {
    const res = await axios.get(`${API_V1}/health`, { timeout: 5_000 });
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}
