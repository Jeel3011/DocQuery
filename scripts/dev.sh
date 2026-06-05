#!/usr/bin/env bash
# DocQuery — one-command local dev launcher.
#
#   ./scripts/dev.sh         # clean restart of the whole stack
#   ./scripts/dev.sh stop    # kill everything
#
# Fixed ports (never drift):  API=8000  FRONTEND=3000  (redis=6379)
# Order: redis -> API -> worker -> frontend.
# Guarantees the CURRENT code is what's running (kills every stale process first).

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
API_PORT=8000
WEB_PORT=3000
LOG_DIR="$ROOT/.devlogs"
mkdir -p "$LOG_DIR"

# ── Local memory safety ────────────────────────────────────────────────
# This is a dev laptop sharing RAM with the editor + browser, NOT a server.
# Cap the PDF parsing pool to 1 process so the worker doesn't pre-warm 8 heavy
# ML (unstructured + YOLOX) workers at once — that burst OOM'd the machine on
# 2026-05-30. Honors an already-set value, so prod/CI can override via env.
export PDF_PARALLEL_WORKERS="${PDF_PARALLEL_WORKERS:-1}"

say(){ printf "\033[1;36m▶ %s\033[0m\n" "$*"; }
ok(){  printf "\033[1;32m✓ %s\033[0m\n" "$*"; }
err(){ printf "\033[1;31m✗ %s\033[0m\n" "$*"; }

kill_all() {
  say "Killing any stale DocQuery processes..."
  pkill -9 -f "uvicorn src.api.server"            2>/dev/null
  pkill -9 -f "celery -A src.worker.celery_app"   2>/dev/null
  pkill -9 -f "next dev"                          2>/dev/null
  # sweep orphaned multiprocessing children (semaphore/spawn leaks from torch +
  # PDF parsing — these accumulate and OOM the laptop if left around)
  pkill -9 -f "multiprocessing.spawn"             2>/dev/null
  pkill -9 -f "multiprocessing.resource_tracker"  2>/dev/null
  # free the fixed ports no matter what holds them
  lsof -ti:$API_PORT 2>/dev/null | xargs kill -9 2>/dev/null
  lsof -ti:$WEB_PORT 2>/dev/null | xargs kill -9 2>/dev/null
  sleep 2
}

if [[ "${1:-}" == "stop" ]]; then
  kill_all
  ok "Stopped. Ports $API_PORT and $WEB_PORT are free."
  exit 0
fi

# ── 0. Redis must be up ──────────────────────────────────────────────
if ! redis-cli ping >/dev/null 2>&1; then
  err "Redis is not running. Start it (e.g. 'brew services start redis') and retry."
  exit 1
fi
ok "Redis up"

kill_all

# ── 1. API (port 8000) ───────────────────────────────────────────────
# --reload is OPT-IN. On this RAM-constrained laptop the API loads heavy ML
# (torch reranker + embeddings) into the process; with --reload, editing any
# src/ file makes uvicorn tear down + RE-IMPORT that whole ML stack on every
# save, leaking macOS semaphores + orphaned multiprocessing children that pile
# up until the machine OOMs/hangs (happened 2026-06-05 while live-editing).
# So reload is off unless you explicitly want it: API_RELOAD=1 ./scripts/dev.sh
# (only safe if you are NOT editing src/ while the server runs).
RELOAD_FLAG=""
if [ "${API_RELOAD:-0}" = "1" ]; then
  RELOAD_FLAG="--reload --reload-dir src"
  say "API --reload ENABLED (avoid editing src/ while running, or it will thrash)"
fi
say "Starting API on :$API_PORT..."
nohup python -m uvicorn src.api.server:app --host 0.0.0.0 --port $API_PORT \
  $RELOAD_FLAG > "$LOG_DIR/api.log" 2>&1 &
for i in $(seq 1 20); do
  curl -sf "http://localhost:$API_PORT/api/v1/health" >/dev/null 2>&1 && break
  sleep 1
done
if ! curl -sf "http://localhost:$API_PORT/api/v1/health" >/dev/null 2>&1; then
  err "API failed to start — see $LOG_DIR/api.log"; tail -15 "$LOG_DIR/api.log"; exit 1
fi
ok "API healthy on http://localhost:$API_PORT"

# ── 2. Celery worker (current code) ──────────────────────────────────
# --pool=solo: run tasks in the worker's main process, NOT a billiard fork.
# The PDF table/layout model (unstructured hi_res, detectron/onnx) segfaults
# (SIGSEGV) when loaded inside a forked child on macOS. A non-forking pool lets
# hi_res table extraction run safely (keeps §4b table fidelity). Local dev is
# single-doc-at-a-time anyway; production (Linux/ECS) uses prefork.
say "Starting Celery worker (pool=solo, fork-safe for hi_res tables)..."
nohup celery -A src.worker.celery_app worker --loglevel=info --pool=solo \
  -Q documents.fast,documents.normal,documents.heavy > "$LOG_DIR/worker.log" 2>&1 &
for i in $(seq 1 40); do
  grep -q "PDF pool ready\|celery@.*ready" "$LOG_DIR/worker.log" 2>/dev/null && break
  sleep 1
done
ok "Worker started (log: $LOG_DIR/worker.log)"

# ── 3. Frontend (port 3000, fixed) ───────────────────────────────────
say "Starting Next.js frontend on :$WEB_PORT..."
( cd frontend-next && nohup npx next dev -p $WEB_PORT > "$LOG_DIR/web.log" 2>&1 & )
for i in $(seq 1 40); do
  curl -sf "http://localhost:$WEB_PORT" >/dev/null 2>&1 && break
  sleep 1
done
if ! curl -sf "http://localhost:$WEB_PORT" >/dev/null 2>&1; then
  err "Frontend failed to start — see $LOG_DIR/web.log"; tail -15 "$LOG_DIR/web.log"; exit 1
fi
ok "Frontend up on http://localhost:$WEB_PORT"

echo
printf "\033[1;32m========================================\033[0m\n"
printf "  Open:  \033[1;34mhttp://localhost:%s\033[0m\n" "$WEB_PORT"
printf "  API:   http://localhost:%s\n" "$API_PORT"
printf "  Logs:  %s/{api,worker,web}.log\n" "$LOG_DIR"
printf "\033[1;32m========================================\033[0m\n"
