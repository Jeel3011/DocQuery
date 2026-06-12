# DocQuery — Project Memory for Claude Code

This file is the durable, in-repo context for any Claude (CLI/Desktop/other) working
on DocQuery. It records what's been achieved, what is NOT done, the critical failures
we hit and their lessons, and the operational gotchas. Read it first, then
`plans/BRAIN_REASONING_PLAN.md` for the deep architecture.

> Maintainer: Jeel (solo, bootstrap posture, no infra budget). India-first target
> (finance now, law later). Local dev is RAM-constrained (a MacBook); cloud is the
> quality ceiling, local throttles are NOT the quality bar.

---

## What DocQuery is
A RAG + reasoning system over financial/legal documents (PDFs → text + table chunks
→ Pinecone/Supabase). A two-stage **Brain** (map-reduce over text passages with
claim-level VERIFY) answers most questions. An **executive Spine** (neuro-symbolic,
deterministic) computes numeric/bridge answers over **table grids** and is meant to
drive confidently-WRONG answers to ~0. Streaming "Trust UI" is the differentiator.

Stack: FastAPI (port 8000) + Celery worker (`--pool=solo` on macOS) + Redis +
Supabase (Postgres + Storage) + Pinecone, Next.js frontend (port 3000).

---

## Run it (3 processes)
| # | Process | Command | Port |
|---|---|---|---|
| 1 | API | `uvicorn src.api.server:app --host 0.0.0.0 --port 8000` | 8000 |
| 2 | Worker | `celery -A src.worker.celery_app worker --loglevel=info --pool=solo` | — |
| 3 | Frontend | `cd frontend-next && npm run dev` | 3000 |

- **Open http://localhost:3000** ("give me the link" → that).
- macOS: Celery MUST use `--pool=solo` (prefork SIGSEGVs on native ML model loading).
- API has NO `/health` route — check `/docs` (200) for liveness.

---

## ⚠️ Critical operational gotchas (read before running anything heavy)
- **`PDF_PARALLEL_WORKERS=1` ALWAYS, locally.** `config.py` defaults it to
  `os.cpu_count()` (=8). Worker startup pre-warms that many PDF processes, each
  loading heavy `unstructured` + YOLOX models → **8 ML procs spike at once → OOM'd
  Jeel's MacBook** (2026-05-30). Always `export PDF_PARALLEL_WORKERS=1` before
  starting the worker/API. Get consent before any heavy local process.
- **`USE_EXEC_SPINE` is OFF by default** (`config.py`). Flag off = Brain path is
  BYTE-IDENTICAL to before the Spine existed. Turning it on routes numeric/pivot
  questions through the executive Spine.
- **Permission prompts are NOT hangs.** "No response" usually = a command awaiting
  approval, not slowness/OOM. Keep commands single & allowlisted.
- **Commit on `main`, not feature branches.** All Phase 4.6 work lives on main; a
  stray branch confused Jeel once. Also: **Jeel often commits staged work himself
  between edits** — check `git log`/`status` before committing to avoid dupes.
- **`plans/` is GITIGNORED** (local-only). So is anything you want shared must go in
  tracked files (code, `eval/`, this `CLAUDE.md`). `eval/*.json` results ARE tracked.
- **Run evals with `python -u`** (unbuffered) — else `print()` block-buffers and a
  run looks hung.
- **ChatOpenAI needs `request_timeout`** — a hung LLM call blocks the whole run.
- **API-burning evals run ON-DEMAND ONLY**, when Jeel explicitly asks. Do NOT run
  `brain_solo_eval` / live evals unprompted (burns money; earlier mistake).

---

## ✅ What's ACHIEVED (verified)

### Extraction foundation — measured & fixed (2026-06-09/10, HARDENED 2026-06-12)
- **Honest framing of "100%": it was SAMPLED-cell accuracy** (112/112 ground-truth
  cells), NOT row-completeness. On 2026-06-12 a new ground-truth-FREE completeness
  gate (`eval/test_extraction_completeness.py` — cross-checks grids vs the PDF's own
  text layer) found **32 silently dropped data lines** hiding behind that 100%:
  missing rows on EVERY issuer's balance sheet (AR, PP&E, common stock) + MSFT FY23
  income-statement R&D (27,195; label/values split 3.0008pt — 0.0008pt past y_tol).
  All fixed STRUCTURALLY in `_read_geometry_lines`/`_segment_table_spans` (typography
  rules, zero doc-specific code): values = TRAILING run of value tokens (label-
  embedded numbers stay label text); pitch-relative split-row merge; x1-anchored
  over-width fold (right-aligned columns share right edges; unplaceable cells are
  DROPPED, never misaligned — a fixture caught my own fold mis-binding GOOG p65
  before it shipped); wrapped-label fragment join. Result: **643/643 text-layer data
  lines covered · benchmark still 112/112 · sections 83%→93.2% · cells gate 9/9**.
  Side effect: prose pseudo-rows die at the source (~40% fewer gated tables = junk
  gone; MD&A tables now extract as REAL tables with pct_change columns excluded).
- **The new-docs answer is a MECHANISM, not a claim:** the same completeness check
  runs at INGESTION for every doc (`src/components/extraction_fidelity.py`, hooked
  in `data_ingestion.py` after extract) — a new doc self-reports per-page fidelity
  at upload (log-only, never blocks). Breakage on unseen layouts is expected; it
  becomes a visible flag + kernel abstain, never a silent wrong answer.
- **⚠️ Live grids are STALE again** until Jeel re-ingests the collection with the
  hardened extractor (same situation as 2026-06-09; re-ingest before live tests).
- Built `eval/extraction_benchmark.py` + `eval/extraction_ground_truth.json`
  (ground truth transcribed VISUALLY from rendered PDFs = non-circular w.r.t. the
  pdfplumber pipeline). Run: `python -u eval/extraction_benchmark.py`
  (`--doc <file>`, `--cells`). Corpus in `test docs/` (gitignored): Amazon FY22/23,
  Google FY21/22/23, Microsoft FY21/22/23. **`0000950170-23-035122.pdf` IS Microsoft
  FY2023** (June-30 fiscal year). Covers ascending (AMZN/GOOG) AND descending (MSFT)
  period order, 2- vs 3-column tables, negatives, varied section styles.
- **1 real extractor bug found + fixed:** Google FY23 balance sheet dropped the
  "Total assets" row. Root cause: `_read_geometry_lines` split a subtotal row across
  two `round(top/3.0)` y-bands (label nval=0 in one band, values label-less in the
  next, 0.73pt apart) → the value band was dropped by `if not label: continue`. Fix
  = keep label-less value lines, then merge a label-only line with the adjacent
  (|Δtop|≤y_tol) value line. Surgical; benchmark went 98.2%→100%, no regression.
  Gates green: `test_table_extraction` 7/7, `test_geometry_lines` all pass.
- **Live collection re-ingested (2026-06-09/10)** with the fixed extractor — all 8
  docs, 0 failures, ~3000 chunks total. Re-ingest calls the SAME
  `extract_tables_from_pdf` the benchmark scored (`data_ingestion.py:747`), and the
  worker logs showed matching gated-table counts → live grids == benchmark grids.
  Old/new data mismatch RESOLVED for these docs.
- **Ingestion latency breakdown** (amzn-20221231, 122pg, 328 chunks, ~91s): table
  SUMMARIES ~37s (100 LLM calls/5 workers, ~45%) + vector UPSERT ~29s (~35%)
  DOMINATE; **extraction itself only ~7.5s**, parse ~4.3s. The latency lever is the
  summary step, NOT extraction. (Jeel chose to KEEP summaries — they're the §4b
  retrieval signal that separates near-identical twin grids; $ table vs %-change
  twin.)

### Kernel/grounding precision filters (2026-06-12) — abstains→resolves, 0 WRONG
- Two structural filters in `analyst.resolve()`/`build_series` hosts + grounding
  `_collect`, applied WITHIN each doc only (never mask cross-doc ambiguity):
  **exact-label > contains** ('R&D' no longer collides with 'Capitalized R&D';
  the exact 'R&D credit' row is no longer shadowed — that shadowing was a LIVE
  kernel confident-wrong: asking for the credit returned the expense) and
  **currency > percent at equal specificity** (`row_value_kind`: '%' in cells, or
  percent/growth/rate section + %-scale values — a $ query never binds the 14.9
  '% of net sales' twin; %-only metrics stay bindable; `parse_cell` now strips a
  trailing '%' so % rows actually parse). Prose guard now applied in grounding
  too (the live `=1` Kingdom-row bug class). `resolve()` no-match errors carry
  did-you-mean closest line-items (difflib over real labels — one-step self-heal).
- **Cross-doc bind matrix `test_kernel_crossdoc`: 16C/3A/0W → 19/19 CORRECT,
  0 ABSTAIN, 0 WRONG.** Suite: analyst 31/31 · grounding 8 cases 0-wrong ·
  tools/loop/gates/selection/descending/executor/planner/meta/verifier/
  enforcement/comprehension ALL green (16 suites re-run 2026-06-12).
- Gold fix: `eval_questions_multihop.json` MSFT-largest-acquisition golds were
  WRONG-from-memory (Activision "closed FY23" — it closed Oct-2023 = MSFT FY24 and
  is only PENDING in the FY23 filing). Corrected to Nuance/FY2022 (revenue
  198,270; AMZN bridge 513,983). The agent's live "Nuance" answer had been right.

### Delete robustness — fixed (2026-06-09)
- "Won't delete from UI" had TWO causes:
  1. **Backend** (`src/api/routes/documents.py::delete_document`): a transient
     Supabase Storage HTTP/2 `ConnectionTerminated` on the storage-blob delete threw
     and aborted the whole fail-fast sequence → doc half-deleted (DB row gone, blob
     orphaned), 500 to client. Fixed: each cleanup step is independent best-effort;
     only 500 if the AUTHORITATIVE `record` delete fails; orphans logged not fatal.
  2. **Frontend** (`frontend-next/lib/api.ts::deleteDocument`): threw on the
     resulting 404, and `app/app/layout.tsx::delDoc`'s catch RESTORED the
     optimistically-removed row + toasted "Failed to delete" → row reappears →
     infinite "won't delete" loop. Fixed: treat 404 as success (delete is
     idempotent; goal state = gone).
- Verified end-to-end: all 7 files deleted + re-ingested cleanly, no 404 loops.

### Executive Spine (Phase 4.6, C1→C6) — BUILT & wired, offline-green
- `perception/grounding.py` (C1), `comprehension/query_ir.py` (C2, 93% type-acc),
  `executive/planner.py` (C3, deterministic DAG, 27/27), `executive/workspace.py` +
  `executor.py` (C4, 10/10), `monitoring/reasoning_verifier.py` + `invariants.py`
  (C5, binary WRONG→ABSTAIN, 8/8), `meta_reasoner.run_executive_spine` (C6
  coordinator, 8/8). Wired into `chat.py` `/query/brain/stream` behind
  `USE_EXEC_SPINE`. All gates green OFFLINE (mocked LLM, real grids).

---

## ❌ What's NOT done / OPEN

- **THE BIG OUTSTANDING ITEM: the Spine has NEVER been measured on live data.**
  Everything is gate-green offline, but the real headline — does `USE_EXEC_SPINE=true`
  drive the confidently-WRONG rate to ~0 end-to-end — is UNPROVEN. This is now
  finally testable because the grids are clean & consistent (they weren't before).
  Next step (when Jeel asks): run `brain_solo_eval` with the flag set. API-burning.
- **Spine monitor not yet wired into `_reduce` MIN_CONFIDENCE** — the block carries
  an abstain signal but the reduce hedge isn't swapped for binary abstain yet.
- **Section coverage is 83%, not 100%** (extraction). Two known causes: (a)
  fragmentation drops the section header on split-off fragments (equity/financing
  tails); (b) **page timestamp/filename line ("30/05/2026, 16:03 goog-20231231") is
  mis-assigned as a section header** by `_assign_sections`. Cosmetic — does NOT
  affect any numeric cell. Not fixed.
- **No regression gate committed** for the new extraction benchmark or the delete-404
  fix; no automated test for the delete path.
- **Orphan reconciliation**: earlier failed deletes may have left orphaned storage
  blobs (e.g. an old goog-20211231 blob). No sweep exists.
- Baseline Brain-alone WRONG-rate (pre-Spine, n=27) = 13 CORRECT / 9 ABSTAIN /
  5 WRONG (19%); WRONG concentrates in extremum_pivot + lookup_pivot (wrong-year
  binding) — exactly what the Spine is meant to fix.

---

## 🔥 Critical failures we hit (and the lessons — don't repeat)

1. **Measurement first, always.** The whole "extraction is 44–61% broken" premise
   was a MEASUREMENT ARTIFACT. The first benchmark run read 78.6%/0% — every miss was
   a BENCHMARK bug (page-blind locator grabbing the cash-flow table because "Net
   income" recurs on 4 pages; statements split into same-page fragments; trailing-
   colon section mismatch), not an extractor bug. **Validating on ONE doc before
   trusting the aggregate number caught all three.** Build ground truth from the
   rendered source, never from memory (a past eval's "gold" AWS-$20B pivot year was
   wrong from memory and inverted a conclusion).
2. **Watch the RIGHT log.** I reported "delete works" while watching the WORKER
   (ingestion) log; the delete failures were in the API log. Jeel caught it twice.
   When the user says "the UI isn't doing X", inspect the UI + API-log behavior
   FIRST, not just the backend happy path.
3. **Don't over-engineer a verification.** When asked to confirm live grids hold
   post-reingest, I started writing a new live-Supabase query script (and churned on
   wrong class names). The right move was the SIMPLE one already used this session:
   re-run the benchmark (same `extract_tables_from_pdf`, same files). Reuse the
   proven path; don't invent.
4. **The unfixable-by-guard lesson.** A wrong compute SPEC can be string-identical to
   a correct one (e.g. AWS÷Consolidated margin). No downstream guard can catch it;
   the fix is UPSTREAM comprehension (C2) deriving the right denominator. Don't pile
   on ratio guards.
5. **Generic ≠ branching.** The system must generalize across 100s of question types
   AND finance+law. The answer is a small recombinable sub-goal BASIS + many
   SKILLS/MEMORY/KNOWLEDGE, NOT per-question handlers. See the cognitive-architecture
   rewrite in `plans/BRAIN_REASONING_PLAN.md` (10 organs).

---

## Where to look
- Deep architecture / Phase 4.6 plan: `plans/BRAIN_REASONING_PLAN.md` (gitignored
  but present locally).
- Extraction: `src/components/table_extraction.py` (geometry reader + gate),
  `src/components/data_ingestion.py` (ingest → chunks).
- Brain/Spine: `src/components/brain/` (analyst, meta_reasoner), `executive/`,
  `monitoring/`, `comprehension/`, `perception/`.
- Evals/gates: `eval/` (`extraction_benchmark.py`, `test_*.py`, `brain_solo_eval.py`).
- Richer running notes (the author's shorthand, lots of [[links]]): the agent memory
  at `~/.claude/projects/-Users-jeelthummar-Desktop-DocQuery/memory/` (NOT auto-loaded
  by Claude Code — read explicitly if you want the full backstory).
