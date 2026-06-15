// lib/api.ts
// Typed wrappers for every FastAPI endpoint in DocQuery.
// No React imports — pure async TypeScript functions.
// Mirrors src/api/schemas.py exactly.

import axios, { AxiosError } from "axios";
import { API_BASE } from "./config";

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
  // G2 Step F: G1d structural class + coarse extraction-fidelity grade, persisted at
  // ingest. null = unknown/legacy → the UI shows a neutral chip/dot.
  doc_type?: "legal_contract" | "financial_filing" | "mixed" | "generic" | null;
  fidelity?: "good" | "partial" | null;
  // G3 Step C: structurally-derived fiscal year. null = unknown → FY filter doesn't exclude.
  fiscal_year?: number | null;
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
  content?: string | null;  // snippet emitted by generate_stream and agentic_stream
  chunk_id?: string | null; // optional, emitted by generate_stream
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
  collection_id?: string | null;
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
    // Delete is idempotent: a 404 means the document is already gone, which is
    // exactly the goal state — treat it as success, not a failure. (Otherwise the
    // optimistic UI removal gets reverted and the row reappears even though it was
    // deleted by an earlier successful call.)
    if (axios.isAxiosError(err) && err.response?.status === 404) {
      return;
    }
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

export async function renameConversation(token: string, convId: string, title: string): Promise<void> {
  try {
    await makeClient(token).patch(`/conversations/${convId}`, { title });
  } catch (err) {
    handleAxiosError(err);
  }
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

// ─── Collections ──────────────────────────────────────────────────────────────

export interface CollectionResponse {
  id: string;
  name: string;
  description: string | null;
  document_count: number;
  created_at: string | null;
  updated_at: string | null;
}

export async function listCollections(
  token: string
): Promise<CollectionResponse[]> {
  try {
    const res = await makeClient(token).get<{
      collections: CollectionResponse[];
      total: number;
    }>("/collections");
    return res.data.collections;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function createCollection(
  token: string,
  name: string,
  description?: string
): Promise<CollectionResponse> {
  try {
    const res = await makeClient(token).post<CollectionResponse>(
      "/collections",
      { name, description }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function deleteCollection(
  token: string,
  collectionId: string
): Promise<void> {
  try {
    await makeClient(token).delete(`/collections/${collectionId}`);
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function renameCollection(
  token: string,
  collectionId: string,
  name: string
): Promise<void> {
  try {
    await makeClient(token).patch(`/collections/${collectionId}`, { name });
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function addDocToCollection(
  token: string,
  collectionId: string,
  documentId: string
): Promise<void> {
  try {
    await makeClient(token).post(`/collections/${collectionId}/documents`, {
      document_id: documentId,
    });
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function removeDocFromCollection(
  token: string,
  collectionId: string,
  documentId: string
): Promise<void> {
  try {
    await makeClient(token).delete(
      `/collections/${collectionId}/documents/${documentId}`
    );
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function getCollectionDocuments(
  token: string,
  collectionId: string
): Promise<DocumentResponse[]> {
  try {
    const res = await makeClient(token).get<{
      documents: DocumentResponse[];
      total: number;
    }>(`/collections/${collectionId}/documents`);
    return res.data.documents;
  } catch (err) {
    handleAxiosError(err);
  }
}

// ─── Export ───────────────────────────────────────────────────────────────────

export async function exportConversation(
  token: string,
  conversationId: string,
  format: "md" | "pdf" = "md"
): Promise<Blob> {
  try {
    const res = await makeClient(token).get(
      `/conversations/${conversationId}/export`,
      {
        params: { format },
        responseType: "blob",
        timeout: 60_000,
      }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// G6.1: export a gated cited deliverable (memo/summary/draft) as a .docx. The markdown
// supplied ALREADY passed the output gate — export preserves the citation contract (no
// re-gate, no added number). `includeCitations` toggles numbered endnotes vs a clean
// client copy (markers stripped at export only).
export async function exportDraftDocx(
  token: string,
  title: string,
  markdown: string,
  includeCitations = true
): Promise<Blob> {
  try {
    const res = await makeClient(token).post(
      `/export/docx`,
      { title, markdown, include_citations: includeCitations },
      { responseType: "blob", timeout: 60_000 }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// ─── Analytics ────────────────────────────────────────────────────────────────

export interface DailyQueryCount {
  date: string;
  count: number;
}

export interface AnalyticsSummary {
  total_queries: number;
  avg_latency_ms: number | null;
  cache_hit_rate: number | null;
  agentic_query_rate: number | null;
  web_search_rate: number | null;
  queries_today: number;
  queries_this_week: number;
  top_queries: { query: string; count: number }[];
  daily_queries: DailyQueryCount[];
}

export interface UsageSummary {
  documents_count: number;
  total_chunks: number;
  collections_count: number;
  conversations_count: number;
  total_messages: number;
}

export async function getAnalyticsSummary(
  token: string,
  days = 30
): Promise<AnalyticsSummary> {
  try {
    const res = await makeClient(token).get<AnalyticsSummary>(
      "/analytics/summary",
      { params: { days } }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function getUsageSummary(
  token: string
): Promise<UsageSummary> {
  try {
    const res = await makeClient(token).get<UsageSummary>("/analytics/usage");
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// ─── Audit Log ───────────────────────────────────────────────────────────────

export interface AuditEntry {
  id: string;
  action: string;
  resource_type: string | null;
  resource_id: string | null;
  metadata: Record<string, unknown> | null;
  ip_address: string | null;
  created_at: string | null;
}

export interface AuditLogResponse {
  entries: AuditEntry[];
  total: number;
  page: number;
  per_page: number;
}

export async function getAuditLog(
  token: string,
  page = 1,
  perPage = 50,
  days = 30
): Promise<AuditLogResponse> {
  try {
    const res = await makeClient(token).get<AuditLogResponse>("/audit/log", {
      params: { page, per_page: perPage, days },
    });
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// ─── Document Comparison ─────────────────────────────────────────────────────

export interface ComparisonResult {
  document_a: string;
  document_b: string;
  similarities: string[];
  differences: string[];
  summary: string;
  focus_area: string | null;
}

export async function compareDocuments(
  token: string,
  documentIdA: string,
  documentIdB: string,
  focus?: string
): Promise<ComparisonResult> {
  try {
    const res = await makeClient(token).post<ComparisonResult>(
      "/documents/compare",
      { document_id_a: documentIdA, document_id_b: documentIdB, focus: focus || null }
    );
    return res.data;
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

// ─── Profile / preferences ─────────────────────────────────────────────────────

export interface MeResponse {
  user_id: string;
  email: string;
  preferred_name: string | null;
}

export async function getMe(token: string): Promise<MeResponse> {
  try {
    const res = await makeClient(token).get<MeResponse>("/auth/me");
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// Update the name the assistant addresses the user by. The server sanitises and
// scopes the write to the authenticated user; pass null to clear it.
export async function updatePreferredName(
  token: string,
  preferredName: string | null
): Promise<MeResponse> {
  try {
    const res = await makeClient(token).patch<MeResponse>("/auth/me/preferences", {
      preferred_name: preferredName,
    });
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}
