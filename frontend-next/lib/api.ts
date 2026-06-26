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
  // F1e privilege firewall: true = attorney-client / work-product. Excluded from shared /
  // cross-vault surfaces (F6) and watermarked in exports. Legacy/null reads as false.
  privileged?: boolean | null;
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
  file: File,
  collectionId?: string | null
): Promise<DocumentResponse> {
  try {
    const form = new FormData();
    form.append("file", file);
    // F2m (D0): when uploading INTO a vault, pass the collection so the backend can stamp the
    // matter OWNER (a staffed paralegal's upload lands in the shared matter, not orphaned) and
    // auto-link it. The separate addDocToCollection below is then a no-op-safe redundancy.
    if (collectionId) form.append("collection_id", collectionId);
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

// F1e: mark/unmark a document as privileged (attorney-client / work-product). A privileged
// doc is excluded from shared / cross-vault surfaces (F6) and watermarked in exports — it is
// NOT hidden from its own vault. Returns the updated row so the caller can reconcile state.
export async function updateDocument(
  token: string,
  docId: string,
  patch: { privileged: boolean }
): Promise<DocumentResponse | undefined> {
  try {
    const res = await makeClient(token).patch<DocumentResponse>(
      `/documents/${docId}`,
      patch
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

// ─── Connectors (G8.6) ──────────────────────────────────────────────────────────
// A connector is a SOURCE: it pulls files from Drive/email and routes them into the SAME
// ingestion pipeline a manual upload uses. All endpoints 404 when USE_CONNECTORS is off
// (byte-identical guarantee) — the UI hides the entry point when getConnectorConfig 404s.

export interface ConnectorConfig {
  enabled: boolean;
  google_drive: { client_id: string; scope: string };
  email: { available: boolean; default_port?: number };
}

export interface ImportFileResult {
  name: string;
  status: "queued" | "skipped" | "error";
  doc_id: string;
  reason: string;
}

export interface ImportResult {
  source: string;
  folder: string;
  queued: number;
  skipped: number;
  errored: number;
  files: ImportFileResult[];
  message: string;
}

export async function getConnectorConfig(token: string): Promise<ConnectorConfig | null> {
  try {
    const res = await makeClient(token).get<ConnectorConfig>("/connectors/config");
    return res.data;
  } catch (err) {
    // 404 = feature off → caller hides the connector UI; not an error to surface.
    if (axios.isAxiosError(err) && err.response?.status === 404) return null;
    handleAxiosError(err);
  }
}

export async function importGoogleDriveFolder(
  token: string,
  accessToken: string,
  folderId: string,
  folderName: string
): Promise<ImportResult> {
  try {
    const res = await makeClient(token).post<ImportResult>(
      "/connectors/google-drive/import",
      { access_token: accessToken, folder_id: folderId, folder_name: folderName },
      { timeout: 120_000 } // listing + fetching a folder can be slow
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function importEmailAttachments(
  token: string,
  body: { host: string; username: string; password: string; mailbox?: string; port?: number; max_messages?: number }
): Promise<ImportResult> {
  try {
    const res = await makeClient(token).post<ImportResult>(
      "/connectors/email/import",
      body,
      { timeout: 120_000 }
    );
    return res.data;
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

// F1a/F1c matter typing. A party on a matter (for the conflict scan).
export interface MatterParty {
  name: string;
  role?: string | null;
}

// F1c: one conflict-scan finding — metadata only (party names + matter labels, never content).
export interface ConflictFinding {
  party: string;
  matched_party: string;
  collection_id: string | null;
  matter_name: string | null;
  matter_kind: string | null;
  severity: "adverse" | "same_party";
  new_side?: string | null;
  existing_side?: string | null;
}

export interface CollectionResponse {
  id: string;
  name: string;
  description: string | null;
  document_count: number;
  created_at: string | null;
  updated_at: string | null;
  // F1a matter fields (legacy rows: matter_kind/firm_id null, status 'active', parties []).
  matter_kind?: string | null;
  status?: string | null;
  parties?: MatterParty[] | null;
  firm_id?: string | null;
  // F1c: the conflict-scan result, present ONLY on create-with-parties.
  conflicts?: ConflictFinding[] | null;
  has_adverse?: boolean | null;
}

// F1c: a practice template — the starting config a matter_kind suggests. All DEFAULTS.
export interface TemplateColumn {
  key: string;
  label: string;
  prompt: string;
  kind: string;
  risk_rubric?: string | null;
}
export interface PracticeTemplate {
  matter_kind: string | null;
  label: string;
  grid_columns: TemplateColumn[];
  kb_scope: string[]; // subset of {"vault","statutes","caselaw"}
  flagship: string;
  summary?: string | null;
}

// The 9 matter kinds (lockstep with src/api/schemas.py MATTER_KINDS). Labels are display-only.
export const MATTER_KINDS: { value: string; label: string }[] = [
  { value: "litigation", label: "Litigation" },
  { value: "m&a", label: "M&A" },
  { value: "lending", label: "Lending" },
  { value: "arbitration", label: "Arbitration" },
  { value: "ip", label: "IP" },
  { value: "regulatory", label: "Regulatory" },
  { value: "employment", label: "Employment" },
  { value: "advisory", label: "Advisory" },
  { value: "compliance", label: "Compliance" },
];

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
  description?: string,
  // F1a/F1c: optional matter typing. Omitting them is the legacy create (byte-identical).
  opts?: { matter_kind?: string | null; parties?: MatterParty[] | null }
): Promise<CollectionResponse> {
  try {
    const body: Record<string, unknown> = { name, description };
    if (opts?.matter_kind) body.matter_kind = opts.matter_kind;
    if (opts?.parties && opts.parties.length) body.parties = opts.parties;
    const res = await makeClient(token).post<CollectionResponse>("/collections", body);
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// F1c: the practice template a matter_kind suggests (grid columns + KB scope + flagship pin).
// Static/$0 on the server. Pass "generic" (or omit-handling) for the neutral fall-back.
export async function getPracticeTemplate(
  token: string,
  matterKind: string
): Promise<PracticeTemplate> {
  try {
    const res = await makeClient(token).get<PracticeTemplate>(
      `/collections/practice-template/${encodeURIComponent(matterKind || "generic")}`
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// F1c: pre-create ethical-wall conflict scan — metadata only, never document content.
export async function scanConflicts(
  token: string,
  parties: MatterParty[],
  excludeCollectionId?: string
): Promise<{ conflicts: ConflictFinding[]; has_adverse: boolean }> {
  try {
    const res = await makeClient(token).post<{
      conflicts: ConflictFinding[];
      has_adverse: boolean;
    }>("/collections/scan-conflicts", {
      parties,
      exclude_collection_id: excludeCollectionId ?? null,
    });
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

// F1e: general vault PATCH — rename and/or change matter typing/lifecycle. Each field is
// optional; only the supplied keys are sent (a name-only call is byte-identical to the old
// renameCollection, and the backend validates matter_kind/status against its CHECK sets).
export async function updateCollection(
  token: string,
  collectionId: string,
  patch: {
    name?: string;
    description?: string;
    matter_kind?: string;
    status?: string;
    parties?: MatterParty[];
  }
): Promise<CollectionResponse | undefined> {
  try {
    const res = await makeClient(token).patch<CollectionResponse>(
      `/collections/${collectionId}`,
      patch
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// Thin compatibility wrapper over updateCollection for a rename-only change.
export async function renameCollection(
  token: string,
  collectionId: string,
  name: string
): Promise<void> {
  await updateCollection(token, collectionId, { name });
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

export async function exportDraftPdf(
  token: string,
  title: string,
  markdown: string,
  includeCitations = true
): Promise<Blob> {
  try {
    const res = await makeClient(token).post(
      `/export/pdf`,
      { title, markdown, include_citations: includeCitations },
      { responseType: "blob", timeout: 60_000 }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// G6.3: export a completed redline (the streamed findings) as a tracked-changes .docx.
export async function exportRedlineDocx(
  token: string,
  title: string,
  docName: string,
  findings: Array<Record<string, unknown>>
): Promise<Blob> {
  try {
    const res = await makeClient(token).post(
      `/export/redline-docx`,
      { title, doc_name: docName, findings },
      { responseType: "blob", timeout: 60_000 }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// ─── Playbook (G6.2) ──────────────────────────────────────────────────────────

export interface PlaybookRow {
  id: string;
  user_id: string;
  collection_id: string | null;
  clause_topic: string;
  standard_position: string;
  fallback_position: string | null;
  notes: string | null;
  is_seed: boolean;
  created_at: string;
  updated_at: string;
}

export interface PlaybookRowInput {
  collection_id?: string | null;
  clause_topic: string;
  standard_position: string;
  fallback_position?: string | null;
  notes?: string | null;
}

export async function listPlaybooks(token: string, collectionId?: string | null): Promise<PlaybookRow[]> {
  try {
    const params = collectionId ? { collection_id: collectionId } : {};
    const res = await makeClient(token).get(`/playbooks`, { params });
    return res.data;
  } catch (err) { handleAxiosError(err); }
}

export async function createPlaybookRow(token: string, body: PlaybookRowInput): Promise<PlaybookRow> {
  try {
    const res = await makeClient(token).post(`/playbooks`, body);
    return res.data;
  } catch (err) { handleAxiosError(err); }
}

export async function updatePlaybookRow(token: string, id: string, body: PlaybookRowInput): Promise<PlaybookRow> {
  try {
    const res = await makeClient(token).put(`/playbooks/${id}`, body);
    return res.data;
  } catch (err) { handleAxiosError(err); }
}

export async function deletePlaybookRow(token: string, id: string): Promise<void> {
  try {
    await makeClient(token).delete(`/playbooks/${id}`);
  } catch (err) { handleAxiosError(err); }
}

export async function seedPlaybook(token: string, collectionId?: string | null): Promise<{ inserted: number; skipped: number }> {
  try {
    const params = collectionId ? { collection_id: collectionId } : {};
    const res = await makeClient(token).post(`/playbooks/seed`, null, { params });
    return res.data;
  } catch (err) { handleAxiosError(err); }
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

// ─── Workflows (Phase G7) ───────────────────────────────────────────────────────
// A workflow is an authored template the agent runs over a vault. The gallery lists
// cards (params_schema → the run form); the RUN is a stream (see streamWorkflowRun in
// lib/streaming.ts). Flag-gated on the backend (USE_AGENT_CORE) — a 404 means it's off.

export interface WorkflowParamSpec {
  name: string;
  label: string;
  type: string;             // "text" | "textarea" | "doc_multiselect" | "multiselect"
  required?: boolean;
  help?: string;
  options?: string[];       // for a multiselect
}

export interface WorkflowCard {
  id: string;
  title: string;
  practice_area: string;    // "Litigation" | "Transactional" | "Financial services" | "Compliance (India)" | "Cross-cutting"
  description: string;
  shape: string;            // "grid" | "report" | "output"
  output_type: string;      // the Harvey tag: "Review" | "Draft" | "Output"
  step_count: number;
  params_schema: WorkflowParamSpec[];
}

export async function listWorkflows(token: string): Promise<WorkflowCard[]> {
  try {
    const res = await makeClient(token).get<{ templates: WorkflowCard[] }>(`/workflows`);
    return res.data.templates ?? [];
  } catch (err) {
    // Flag off ⇒ 404 (the gallery doesn't exist). Treat as "no workflows", not an error.
    if ((err as AxiosError)?.response?.status === 404) return [];
    handleAxiosError(err);
  }
}

// LEGAL_TASK_CATALOG §2.3: the DRAFT card picker, generated from the catalog table.
// A card launches the EXISTING draft route (mode="draft") with its `id` as doc_type;
// the server expands the catalog id into the India-correct structure + cite-or-bracket
// instructions via render_draft_request.
export interface DocTypeCard {
  id: string;
  title: string;
  practice_area: string;    // "Litigation" | "Transactional" | "Financial services" | "Compliance (India)"
  jurisdiction: string;     // "IN" — the vocabulary moat
  description: string;
  required_inputs: string[];
  verb: string;             // "Draft" — the §0 primitive verb this card launches
}

export async function listDocTypes(token: string): Promise<DocTypeCard[]> {
  try {
    const res = await makeClient(token).get<{ doc_types: DocTypeCard[] }>(`/doc-types`);
    return res.data.doc_types ?? [];
  } catch (err) {
    // Flag off ⇒ 404 (the catalog doesn't exist). Treat as "no catalog", not an error —
    // the draft page falls back to its generic doc-type list.
    if ((err as AxiosError)?.response?.status === 404) return [];
    handleAxiosError(err);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// F2 — Firm Console (RBAC · ethical walls · delegation · review chain · override)
// Each wrapper mirrors a cap-gated route in routes/{auth,admin,matters}.py EXACTLY.
// The server is ALWAYS the boundary: these wrappers never decide authorization — they
// surface the server's decision (a 403 carries the reason in APIError.detail). The caps
// payload (getCapabilities) decides what RENDERS; the route guard decides what's ALLOWED.
// ═══════════════════════════════════════════════════════════════════════════════

// ── Capabilities (surface 10 — caps source of truth) ──────────────────────────
// Mirrors CapabilitiesResponse. The verbs match authz.py CAPABILITIES exactly.
export type Capability =
  | "create_vault" | "ingest" | "ask" | "draft" | "run_workflow" | "grids"
  | "send_for_review" | "release_external" | "manage_matter_team" | "manage_members"
  | "view_billing" | "delete" | "sign_certificate" | "edit_playbooks"
  | "publish_to_firm_brain" | "run_sentinel" | "override_abstain";

export type FirmRole =
  | "managing_partner" | "senior_partner" | "partner" | "senior_associate"
  | "associate" | "paralegal" | "assistant" | "client" | "guest";

export interface CapabilitiesResponse {
  caps: Capability[];
  role: FirmRole | null;
  firm_id: string | null;
  is_external: boolean;
  delegated_verbs: Capability[];
}

export async function getCapabilities(token: string): Promise<CapabilitiesResponse> {
  try {
    const res = await makeClient(token).get<CapabilitiesResponse>("/auth/capabilities");
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// ── App bootstrap (latency) — one round-trip for everything the shell needs on mount ──────────
export interface BootstrapResponse {
  user: MeResponse;
  capabilities: CapabilitiesResponse;
  firm: Firm | null;
  collections: CollectionResponse[];
  conversations: ConversationResponse[];
}

export async function getBootstrap(token: string): Promise<BootstrapResponse> {
  try {
    const res = await makeClient(token).get<BootstrapResponse>("/auth/bootstrap");
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// ── Firm + onboarding (F2a) ───────────────────────────────────────────────────
export interface Firm {
  id: string;
  name: string;
  role: FirmRole | null;
}

export async function getFirm(token: string): Promise<Firm | null> {
  try {
    const res = await makeClient(token).get<Firm>("/auth/firm");
    return res.data;
  } catch (err) {
    // 404 = firm-less / legacy solo user — not an error, just "no firm yet".
    if ((err as AxiosError)?.response?.status === 404) return null;
    handleAxiosError(err);
  }
}

export async function renameFirm(token: string, _firmId: string, name: string): Promise<Firm> {
  // The firm is resolved server-side (T3) — _firmId is accepted for caller clarity but not sent.
  try {
    const res = await makeClient(token).patch<Firm>("/auth/firm", { name });
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export interface MemberResponse {
  user_id: string;
  firm_id: string;
  role: FirmRole;
  email: string | null;
  created_at: string | null;
}

export async function listMembers(token: string): Promise<MemberResponse[]> {
  try {
    const res = await makeClient(token).get<{ members: MemberResponse[] }>("/admin/firm/members");
    return res.data.members ?? [];
  } catch (err) {
    handleAxiosError(err);
  }
}

export interface InviteResponse {
  id: string;
  firm_id: string;
  email: string;
  role: FirmRole;
  expires_at: string | null;
  accepted_at: string | null;
  created_at: string | null;
  token?: string | null;   // returned ONCE on create — deliver it to the invitee
}

export async function inviteMember(
  token: string, email: string, role: FirmRole
): Promise<InviteResponse> {
  try {
    const res = await makeClient(token).post<InviteResponse>(
      "/admin/firm/invites", { email, role }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function listInvites(token: string): Promise<InviteResponse[]> {
  try {
    const res = await makeClient(token).get<{ invites: InviteResponse[] }>("/admin/firm/invites");
    return res.data.invites ?? [];
  } catch (err) {
    handleAxiosError(err);
  }
}

// Rotate a pending invite's token and return the FRESH one-time token (the "resend / copy link"
// recovery — the original is hash-stored and can't be re-shown). The prior link dies on rotation.
export async function resendInvite(token: string, inviteId: string): Promise<InviteResponse> {
  try {
    const res = await makeClient(token).post<InviteResponse>(`/admin/firm/invites/${inviteId}/resend`);
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function revokeInvite(token: string, inviteId: string): Promise<void> {
  try {
    await makeClient(token).delete(`/admin/firm/invites/${inviteId}`);
  } catch (err) {
    handleAxiosError(err);
  }
}

// The full, ready-to-deliver invite link for a one-time token. No email server is wired yet
// (F2j deferred), so the admin copies this and sends it out-of-band. The invitee logs in and
// lands here; their verified email must match the invite (T4, server-enforced).
export function acceptInviteUrl(rawToken: string): string {
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  // Public landing (/invite, NOT /app/*) so a logged-OUT invitee isn't bounced to login and lose
  // the token. /invite stashes the token, then routes to login/signup; the token is applied after
  // they authenticate as the invited email.
  return `${origin}/invite?token=${encodeURIComponent(rawToken)}`;
}

// Accept EITHER a bare invite token OR a full pasted invite LINK, and return the bare token.
// People naturally paste the whole link they were sent; pulling ?token= out of it (and trimming)
// means "Join with invite" works whether they paste the link or the token. Falls back to the
// trimmed input if it isn't a URL.
export function extractInviteToken(input: string): string {
  const raw = (input || "").trim();
  if (!raw) return "";
  // Try to parse a token= query param out of a URL (handles full links + bare ?token=… fragments).
  const m = raw.match(/[?&]token=([^&\s]+)/);
  if (m) return decodeURIComponent(m[1]);
  return raw;
}

export async function acceptInvite(token: string, inviteToken: string): Promise<Firm> {
  try {
    const res = await makeClient(token).post<Firm>("/auth/accept-invite", { token: inviteToken });
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// ── Member lifecycle (F2d) — promote/demote + offboard ────────────────────────
export interface LifecycleResponse {
  user_id: string;
  firm_id: string;
  role: FirmRole | null;
  removed: boolean | null;
  matters_reassigned: number | null;
  delegations_revoked: number | null;
}

export async function setRole(
  token: string, userId: string, role: FirmRole
): Promise<LifecycleResponse> {
  try {
    const res = await makeClient(token).patch<LifecycleResponse>(
      `/admin/firm/members/${userId}`, { role }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function removeMember(token: string, userId: string): Promise<LifecycleResponse> {
  try {
    const res = await makeClient(token).delete<LifecycleResponse>(`/admin/firm/members/${userId}`);
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// ── Ethical walls (F2c) ───────────────────────────────────────────────────────
export interface ScreenResponse {
  id: string;
  firm_id: string;
  user_id: string;
  vault_id: string;
  reason: string;
  created_by: string | null;
  created_at: string | null;
  removed_at: string | null;
}

export async function listScreens(token: string): Promise<ScreenResponse[]> {
  try {
    const res = await makeClient(token).get<{ screens: ScreenResponse[] }>("/admin/firm/screens");
    return res.data.screens ?? [];
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function createScreen(
  token: string, userId: string, vaultId: string, reason: string
): Promise<ScreenResponse> {
  try {
    const res = await makeClient(token).post<ScreenResponse>(
      "/admin/firm/screens", { user_id: userId, vault_id: vaultId, reason }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function removeScreen(token: string, screenId: string): Promise<void> {
  try {
    await makeClient(token).delete(`/admin/firm/screens/${screenId}`);
  } catch (err) {
    handleAxiosError(err);
  }
}

// ── Delegation / PA (F2d/D6) ──────────────────────────────────────────────────
export interface DelegationResponse {
  id: string;
  firm_id: string;
  delegator_id: string;
  delegate_id: string;
  verbs: Capability[];
  expires_at: string | null;
  revoked_at: string | null;
  created_at: string | null;
}

export async function grantAuthority(
  token: string, delegateId: string, verbs: Capability[], expiresAt: string
): Promise<DelegationResponse> {
  try {
    const res = await makeClient(token).post<DelegationResponse>(
      "/admin/firm/delegations",
      { delegate_id: delegateId, verbs, expires_at: expiresAt }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function revokeAuthority(token: string, delegationId: string): Promise<void> {
  try {
    await makeClient(token).delete(`/admin/firm/delegations/${delegationId}`);
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function listDelegations(token: string): Promise<DelegationResponse[]> {
  try {
    const res = await makeClient(token).get<{ delegations: DelegationResponse[] }>(
      "/admin/firm/delegations"
    );
    return res.data.delegations ?? [];
  } catch (err) {
    handleAxiosError(err);
  }
}

// ── Matter team (F2e/D3) — staffing a matter ──────────────────────────────────
export interface MatterTeamMember {
  user_id: string;
  role: FirmRole | null;
  email: string | null;
  added_by: string | null;
  created_at: string | null;
}

export async function getMatterTeam(token: string, vaultId: string): Promise<MatterTeamMember[]> {
  try {
    const res = await makeClient(token).get<{ vault_id: string; members: MatterTeamMember[] }>(
      `/matters/${vaultId}/team`
    );
    return res.data.members ?? [];
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function addMatterTeam(
  token: string, vaultId: string, userId: string
): Promise<MatterTeamMember[]> {
  try {
    const res = await makeClient(token).post<{ vault_id: string; members: MatterTeamMember[] }>(
      `/matters/${vaultId}/team`, { user_id: userId }
    );
    return res.data.members ?? [];
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function removeMatterTeam(
  token: string, vaultId: string, userId: string
): Promise<void> {
  try {
    await makeClient(token).delete(`/matters/${vaultId}/team/${userId}`);
  } catch (err) {
    handleAxiosError(err);
  }
}

// ── Review chain (F2e/D5) — submit · approve · request changes · release · queue ──
export interface ReviewRequest {
  id: string;
  firm_id: string;
  vault_id: string;
  artifact_ref: string;
  submitted_by: string;
  status: "pending" | "approved" | "changes_requested" | "released";
  current_owner: string | null;
  chain: string[];
  created_at: string | null;
  decided_at: string | null;
}

// F2j.1 — the submitted work behind a review request (so a reviewer can SEE & verify it).
export interface ReviewArtifact {
  available: boolean;
  title: string | null;
  question: string | null;
  answer_preview: string | null;
  answer_full: string | null;
  conversation_id: string | null;
  submitter_id: string | null;
  created_at: string | null;
}

export interface ReviewThreadMessage {
  role: string;
  content: string;
  sources: unknown[] | null;
  created_at: string | null;
}

export interface ReviewThread {
  available: boolean;
  title: string | null;
  conversation_id: string | null;
  messages: ReviewThreadMessage[];
}

// The card preview. A 404 means the caller can't review this request — degrade to unavailable.
export async function getReviewArtifact(token: string, requestId: string): Promise<ReviewArtifact> {
  try {
    const res = await makeClient(token).get<ReviewArtifact>(`/review/${requestId}/artifact`);
    return res.data;
  } catch {
    return { available: false, title: null, question: null, answer_preview: null,
             answer_full: null, conversation_id: null, submitter_id: null, created_at: null };
  }
}

// The full read-only thread ("View full work").
export async function getReviewThread(token: string, requestId: string): Promise<ReviewThread> {
  try {
    const res = await makeClient(token).get<ReviewThread>(`/review/${requestId}/thread`);
    return res.data;
  } catch {
    return { available: false, title: null, conversation_id: null, messages: [] };
  }
}

export async function submitForReview(
  token: string, collectionId: string, artifactRef: string
): Promise<ReviewRequest> {
  try {
    const res = await makeClient(token).post<ReviewRequest>(
      "/review", { collection_id: collectionId, artifact_ref: artifactRef }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function approveReview(
  token: string, requestId: string, note?: string
): Promise<ReviewRequest> {
  try {
    const res = await makeClient(token).post<ReviewRequest>(
      `/review/${requestId}/approve`, { note: note ?? null }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function requestChanges(
  token: string, requestId: string, note?: string
): Promise<ReviewRequest> {
  try {
    const res = await makeClient(token).post<ReviewRequest>(
      `/review/${requestId}/changes`, { note: note ?? null }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function releaseExternal(
  token: string, requestId: string, note?: string
): Promise<ReviewRequest> {
  try {
    const res = await makeClient(token).post<ReviewRequest>(
      `/review/${requestId}/release`, { note: note ?? null }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function getReviewQueue(token: string): Promise<ReviewRequest[]> {
  try {
    const res = await makeClient(token).get<{ requests: ReviewRequest[] }>("/review/queue");
    return res.data.requests ?? [];
  } catch (err) {
    handleAxiosError(err);
  }
}

// Custom review chain (surface 6 — "Customize review chain"). null/[] clears → rank default.
export async function setReviewChain(
  token: string, vaultId: string, chain: string[] | null
): Promise<{ vault_id: string; chain: string[] | null }> {
  try {
    const res = await makeClient(token).put<{ vault_id: string; chain: string[] | null }>(
      `/matters/${vaultId}/review-chain`, { chain }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// ── Abstain override (F2d/T6 — the high-trust moment) ─────────────────────────
export interface OverrideAbstainResponse {
  answer_ref: string;
  collection_id: string;
  status: string;            // "overridden"
  overridden_by: string;
  reason: string;
  created_at: string | null;
}

export async function overrideAbstain(
  token: string, answerRef: string, collectionId: string, reason: string, gateObjection?: string
): Promise<OverrideAbstainResponse> {
  try {
    const res = await makeClient(token).post<OverrideAbstainResponse>(
      "/admin/firm/answers/override",
      { answer_ref: answerRef, collection_id: collectionId, reason, gate_objection: gateObjection ?? null }
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

// ── F2j NOTIFICATIONS (the in-app inbox + anti-nag preferences) ───────────────
// The inbox is the recipient's OWN — every endpoint is server-scoped to the caller
// (a user can only read / mark / configure their own notifications, T2/T8).
export interface AppNotification {
  id: string;
  event: string;
  category: string;        // review | matter | governance
  title: string | null;
  body: string | null;
  resource_type: string | null;
  resource_id: string | null;
  vault_id: string | null;
  read: boolean;
  created_at: string | null;
}

export interface NotificationList {
  notifications: AppNotification[];
  unread: number;
}

export interface NotificationPreferences {
  muted_categories: string[];
  quiet_start: number | null;
  quiet_end: number | null;
  digest_mode: boolean;
}

export async function getNotifications(
  token: string, unreadOnly = false, limit = 50
): Promise<NotificationList> {
  try {
    const res = await makeClient(token).get<NotificationList>(
      `/notifications?unread_only=${unreadOnly}&limit=${limit}`
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function getUnreadCount(token: string): Promise<number> {
  try {
    const res = await makeClient(token).get<{ count: number }>("/notifications/unread-count");
    return res.data.count ?? 0;
  } catch {
    // The bell is non-critical chrome — a failure degrades to 0, never a broken UI.
    return 0;
  }
}

export async function markNotificationsRead(
  token: string, ids?: string[]
): Promise<number> {
  try {
    const res = await makeClient(token).post<{ updated: number }>(
      "/notifications/read", { ids: ids ?? null }
    );
    return res.data.updated ?? 0;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function getNotificationPreferences(token: string): Promise<NotificationPreferences> {
  try {
    const res = await makeClient(token).get<NotificationPreferences>("/notifications/preferences");
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}

export async function setNotificationPreferences(
  token: string, prefs: Partial<NotificationPreferences>
): Promise<NotificationPreferences> {
  try {
    const res = await makeClient(token).put<NotificationPreferences>(
      "/notifications/preferences", prefs
    );
    return res.data;
  } catch (err) {
    handleAxiosError(err);
  }
}
