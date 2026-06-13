from dataclasses import dataclass, field
from typing import Literal
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

# Project root = two levels up from this file (src/components/config.py -> project root)
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)

@dataclass
class Config:
    """Centralized configuration"""

    # Embeddings and LLM
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    LLM_MODEL_NAME: str = "gpt-4o-mini"

    # Chunking params
    CHUNK_SIZE: int = 3000
    NEW_AFTER_N_CHARS: int = 2400
    COMBINE_TEXT_UNDER_N_CHARS: int = 500
    CHUNK_OVERLAP: int = 500

    # GRAND_PLAN G1a — boilerplate strip. Drop repeated page chrome (Header/Footer/
    # PageNumber element categories + standalone "Page N of M"/URL/print-timestamp
    # lines) BEFORE text chunking so it stops dominating clause embeddings. Helps
    # both prose contracts AND financial filings (cleaner chunks). Pure-text path;
    # the finance TABLE pass (table_extraction reads PDF geometry) is untouched.
    # Off ⇒ byte-identical to pre-G1a chunking.
    STRIP_BOILERPLATE: bool = os.getenv("STRIP_BOILERPLATE", "true").lower() != "false"

    # GRAND_PLAN G1b/c/d — classify the doc at ingest (financial_filing | legal_contract
    # | mixed | generic) and run the matching extraction path: a legal contract gets
    # clause-aware chunking (G1b) + NO financial-table pass (G1c); everything else keeps
    # the proven finance path (geometry table moat untouched). Structural classifier
    # ($0, no LLM in v1). Off ⇒ byte-identical to pre-G1bcd (always chunk_by_title +
    # always run the table pass). G1a (STRIP_BOILERPLATE) is independent of this.
    CLASSIFY_DOCS: bool = os.getenv("CLASSIFY_DOCS", "true").lower() != "false"

    # ── Document Processing ──
    # PDF_STRATEGY: used for explicit overrides or non-PDF files.
    # For PDFs, _detect_strategy() in data_ingestion.py auto-selects based on page count.
    #   "fast"   — no OCR/layout model, 5-10x faster, text-heavy PDFs ≤ PDF_FAST_THRESHOLD_PAGES
    #   "auto"   — unstructured auto-detects, medium PDFs
    #   "hi_res" — full layout + OCR, scanned/large PDFs > PDF_MEDIUM_THRESHOLD_PAGES
    PDF_STRATEGY: str = "auto"          # fallback for non-PDFs and force-override
    EXTRACT_IMAGES: bool = True          # False = skip image extraction (faster)
    PARALLEL_PDF_PAGES: bool = True      # Process PDF page-ranges in parallel (3-5x faster)
    # A1: match parallelism to the box's vCPUs to avoid oversubscription thrashing.
    # 4 unstructured/YOLOX processes on a 2-vCPU worker fight over cores. Default to
    # the container's CPU count; override per-deploy via PDF_PARALLEL_WORKERS env var.
    PDF_PARALLEL_WORKERS: int = int(os.getenv("PDF_PARALLEL_WORKERS", os.cpu_count() or 2))

    # ── Phase 3: Tiered PDF strategy thresholds ──
    # ≤ PDF_FAST_THRESHOLD_PAGES   → strategy="fast" (no OCR, ~0.5s/doc)
    # ≤ PDF_MEDIUM_THRESHOLD_PAGES → strategy="auto" (~2-5s/doc)
    # > PDF_MEDIUM_THRESHOLD_PAGES → strategy="hi_res" (full OCR, ~15-30s/doc)
    PDF_FAST_THRESHOLD_PAGES: int = 10   # Raised from 5 — more PDFs avoid model loading entirely
    PDF_MEDIUM_THRESHOLD_PAGES: int = 30
    # A5: above MEDIUM, only OCR (hi_res) when the avg extractable text/page is below
    # this — i.e. the PDF is genuinely scanned. Born-digital long PDFs use "auto".
    PDF_TEXT_LAYER_MIN_CHARS_PER_PAGE: int = 100

    # Retrieval params
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.30   # Noise floor — reranker handles precision; 0.45 was too aggressive for text-embedding-3-small

    # Hybrid search: BM25 (local) + Dense (Pinecone) merged via Reciprocal Rank Fusion
    # Set to True to activate HybridRetriever in retrieval.py.
    USE_HYBRID_SEARCH: bool = False
    # How many candidates to over-fetch from Pinecone for BM25 to rank over.
    # Larger = better recall for BM25 at the cost of slightly more Pinecone latency.
    HYBRID_FETCH_K: int = 25

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY")
    # B1: local JWT verification avoids an auth.get_user() round-trip per request.
    # Modern Supabase tokens (ES256/RS256) are verified via the project's JWKS
    # endpoint using SUPABASE_URL alone — no secret needed. This is ONLY the legacy
    # HS256 shared secret (Dashboard → Settings → API → JWT Keys → Legacy JWT Secret),
    # used to verify any still-valid legacy tokens. Optional.
    SUPABASE_JWT_SECRET: str = os.getenv("SUPABASE_JWT_SECRET", "")

    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "docquery")
    PINECONE_NAMESPACE: str = ""  # B5: Must be set per-user — empty default prevents silent fallback

    # UPLOAD_DIR is used as a temp directory when downloading files from Supabase
    # for processing. Files are NOT stored here permanently anymore.
    UPLOAD_DIR: str = os.path.join(_PROJECT_ROOT, "tmp_uploads")
    SUPPORTED_FILE_TYPES: tuple = (
        "pdf",
        "docx",
        "pptx",
        "txt",
        "xlsx",
    )

    # ── Reranker ──
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_INITIAL_K: int = 10   # Lowered from 15 — 10 gives good recall with less Pinecone/reranker work
    RERANK_TOP_K: int = 5        # Final docs after reranking

    # ── Multi-Query Retrieval ──
    USE_MULTI_QUERY: bool = True
    MULTI_QUERY_COUNT: int = 2   # Lowered from 3 — 2 variants + original = 3 Pinecone calls (was 4)

    # ── Input limits ──
    MAX_QUERY_LENGTH: int = 2000   # Max characters for user question

    # ── Phase 2: Web Search Fallback ──
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    USE_WEB_FALLBACK: bool = True    # Enable web search when docs return no results
    WEB_SEARCH_MAX_RESULTS: int = 3  # Max web results to merge with doc results

    # ── Phase 0 / Brain safety invariants ──
    # Invariant R1: per-file fan-out only runs when the collection is this small.
    # Above this, Stage-1 routing must first narrow the corpus; fan-out then runs
    # over the routed top-N, never the raw collection. Prevents 200 Pinecone
    # round-trips and sequential reranker passes on large collections.
    ROUTING_MAX_FANOUT: int = int(os.getenv("ROUTING_MAX_FANOUT", "8"))

    # Invariant R2: hard token ceiling for the single-call generation path.
    # If context exceeds this after reranking, chunks are trimmed furthest-first.
    # When the map-reduce Brain (Phase 4) exists it takes over above this threshold
    # instead of silently truncating.  ~12k tokens ≈ comfortable gpt-4o-mini window.
    CONTEXT_TOKEN_BUDGET: int = int(os.getenv("CONTEXT_TOKEN_BUDGET", "12000"))

    # ── Stage-1 Document Router (Phase 3) ──
    # Top-N most-relevant documents the router returns for a collection query,
    # ordered most-relevant first. The Brain (Stage-2) reads all N. The legacy
    # per-file fan-out path (retrieve_across_files) additionally caps at
    # ROUTING_MAX_FANOUT, keeping the most-relevant docs since the list is ordered.
    ROUTING_TOP_N: int = int(os.getenv("ROUTING_TOP_N", "12"))

    # Diversity-aware routing (MMR). Plain top-N cosine over-concentrates on
    # near-duplicate docs (e.g. 3 years of one company), crowding out the docs a
    # cross-entity question needs. ROUTING_MMR_LAMBDA balances relevance vs.
    # diversity when selecting: 1.0 = pure relevance (old behaviour), lower = more
    # diverse. 0.7 keeps relevance dominant while breaking up redundant clusters.
    ROUTING_MMR_LAMBDA: float = float(os.getenv("ROUTING_MMR_LAMBDA", "0.7"))

    # ── Stage-2 Brain (Phase 4) ──
    # Opt-in: set USE_BRAIN=true to route synthesis/collection queries through
    # the map-reduce Brain instead of the single-call fast path.
    USE_BRAIN: bool = os.getenv("USE_BRAIN", "false").lower() == "true"
    # REDUCE uses a stronger model; MAP uses the cheap LLM_MODEL_NAME.
    REDUCE_LLM_MODEL: str = os.getenv("REDUCE_LLM_MODEL", "gpt-4o")
    # VERIFY uses a different model from REDUCE to de-correlate errors (§4a.3).
    VERIFY_LLM_MODEL: str = os.getenv("VERIFY_LLM_MODEL", "gpt-4o-mini")
    # How many chunks the Brain reads per document in MAP. Higher = better recall on
    # large filings (a 5-chunk read misses needles in a 300-chunk 10-K), at more cost.
    BRAIN_CHUNKS_PER_DOC: int = int(os.getenv("BRAIN_CHUNKS_PER_DOC", "15"))

    # Per-document retrieval timeout (seconds) for the Brain — fail fast on a network/
    # DNS blip instead of hanging on Pinecone retries for minutes.
    BRAIN_RETRIEVE_TIMEOUT_S: float = float(os.getenv("BRAIN_RETRIEVE_TIMEOUT_S", "20"))

    # ── Stage-2 Table intelligence (Phase 4.3) ──
    # Bounded parallelism for the per-table LLM summary generated at ingest (the
    # discriminative caption that lets the Analyst pick the right grid among near-twins).
    # These are network-bound OpenAI calls, not RAM/CPU work, so this does NOT re-arm the
    # local PDF-pool OOM. Measured: sequential ≈79s/doc, max_workers=5 ≈14s/doc. Laptop
    # default 5; raise via env on the cloud.
    TABLE_SUMMARY_WORKERS: int = int(os.getenv("TABLE_SUMMARY_WORKERS", "5"))

    # ── Agent Core (AGENT_CORE_PLAN Phase A, §3.1) ──
    # Opt-in: set USE_AGENT_CORE=true to expose POST /query/agentcore/stream (A4). A
    # frontier model orchestrates by calling our deterministic machinery as TOOLS,
    # behind non-bypassable output gates. OFF by default — flag off = the route 404s
    # and every existing Brain/Spine path is byte-identical (the plan's prime directive).
    USE_AGENT_CORE: bool = os.getenv("USE_AGENT_CORE", "false").lower() == "true"

    # Orchestrator model policy is CONFIG, not code (§3.1) — multi-vendor from day one.
    # Jeel's call (2026-06-10): Opus 4.8 for BOTH standard and deep — max quality
    # everywhere, overriding the plan's Sonnet-for-standard default. Env-overridable so
    # the cloud/cost tradeoff can change without touching code. The classifier is the one
    # cheap call (mode dispatch), so it stays on a mini model.
    AGENT_MODEL_STANDARD: str = os.getenv("AGENT_MODEL_STANDARD", "claude-opus-4-8")
    AGENT_MODEL_DEEP: str = os.getenv("AGENT_MODEL_DEEP", "claude-opus-4-8")
    AGENT_MODEL_CLASSIFIER: str = os.getenv("AGENT_MODEL_CLASSIFIER", "gpt-4o-mini")

    # Per-mode budgets (§3.1), enforced IN CODE by the loop runner (not advisory). A run
    # that exhausts a budget wraps up with whatever is gated + an explicit abstain for the
    # rest — it never silently truncates or overspends.
    AGENT_STD_MAX_STEPS: int = int(os.getenv("AGENT_STD_MAX_STEPS", "16"))
    AGENT_STD_WALL_S: float = float(os.getenv("AGENT_STD_WALL_S", "150"))
    # 220k (was 120k→60k): a multi-ENTITY question (e.g. "which of GOOG/MSFT/AMZN has the
    # highest R&D-to-revenue ratio") legitimately needs ~12 reads + 3 computes across 3
    # filings; offline the model computed 2 of 3 ratios correctly and hit 120k mid-3rd
    # (139k used). The orchestrator emits reasoning tokens + accumulates tool-result
    # context per step, so cross-entity work needs real headroom. Grids enter context
    # only via read_document (now page-scoped), so this is bounded, not runaway.
    AGENT_STD_TOKEN_BUDGET: int = int(os.getenv("AGENT_STD_TOKEN_BUDGET", "220000"))
    AGENT_DEEP_MAX_STEPS: int = int(os.getenv("AGENT_DEEP_MAX_STEPS", "40"))
    AGENT_DEEP_WALL_S: float = float(os.getenv("AGENT_DEEP_WALL_S", "480"))
    AGENT_DEEP_TOKEN_BUDGET: int = int(os.getenv("AGENT_DEEP_TOKEN_BUDGET", "250000"))

    # Anthropic SDK key (primary orchestrator vendor). Empty until set; no live agent
    # call is possible without it (the offline gates use a mocked model, so they need
    # neither this nor the network).
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")