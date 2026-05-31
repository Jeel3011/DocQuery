#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# DocQuery — Local Development Launcher
#
# Runs FastAPI, Celery worker, and Streamlit NATIVELY using your venv.
# Only Redis runs in Docker (tiny, ~5MB image).
#
# Usage:
#   ./start_local.sh          # Start all services
#   ./start_local.sh stop     # Stop all services
#   ./start_local.sh restart  # Restart all services
#
# Benefits over `docker-compose up --build`:
#   ✅ No 5-10 min Docker rebuild on every change
#   ✅ No laptop overheating (no PyTorch/unstructured rebuild)
#   ✅ Instant hot-reload on code changes (uvicorn --reload)
#   ✅ Test locally BEFORE deploying to AWS
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/venv/bin"
LOG_DIR="$PROJECT_DIR/logs"
PID_DIR="$PROJECT_DIR/.pids"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

mkdir -p "$LOG_DIR" "$PID_DIR"

# ── Helper functions ──────────────────────────────────────────────────

log_info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $1"; }

kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti ":$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log_warn "Port $port in use — killing PIDs: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

stop_services() {
    log_info "Stopping all DocQuery services..."
    for pidfile in "$PID_DIR"/*.pid; do
        [ -f "$pidfile" ] || continue
        local pid
        pid=$(cat "$pidfile")
        local name
        name=$(basename "$pidfile" .pid)
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            log_ok "Stopped $name (PID $pid)"
        fi
        rm -f "$pidfile"
    done
    # Clean up any stragglers on our ports
    kill_port 8000
    kill_port 8501
    log_ok "All services stopped."
}

check_redis() {
    if redis-cli ping &>/dev/null; then
        log_ok "Redis is running (local)"
        return 0
    fi
    # Try starting Redis via Docker
    log_info "Starting Redis via Docker..."
    docker run -d --name docquery-redis -p 6379:6379 redis:7-alpine \
        redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru \
        &>/dev/null 2>&1 || true
    sleep 2
    if redis-cli ping &>/dev/null; then
        log_ok "Redis started via Docker"
        return 0
    fi
    log_err "Redis is not available. Please install Redis or start Docker."
    exit 1
}

start_services() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║       📄 DocQuery — Local Dev Environment       ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
    echo ""

    # ── 0. Preflight checks ──
    if [ ! -f "$VENV/python" ]; then
        log_err "Virtual environment not found at $VENV"
        log_err "Run: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
        exit 1
    fi

    if [ ! -f "$PROJECT_DIR/.env" ]; then
        log_err ".env file not found. Copy .env.example to .env and fill in your keys."
        exit 1
    fi

    # ── 1. Redis ──
    check_redis

    # ── 2. Free ports ──
    kill_port 8000
    kill_port 8501

    # ── 3. FastAPI Backend (with hot-reload) ──
    log_info "Starting FastAPI backend on http://localhost:8000 ..."
    cd "$PROJECT_DIR"
    "$VENV/uvicorn" src.api.server:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --reload-dir src/api \
        > "$LOG_DIR/api.log" 2>&1 &
    echo $! > "$PID_DIR/api.pid"
    log_ok "FastAPI started (PID $(cat "$PID_DIR/api.pid")) — logs: logs/api.log"

    # ── 4. Celery Worker ──
    log_info "Starting Celery worker..."
    cd "$PROJECT_DIR"
    "$VENV/celery" -A src.worker.celery_app worker \
        --loglevel=info \
        --concurrency=2 \
        -Q documents.fast,documents.normal,documents.heavy \
        > "$LOG_DIR/worker.log" 2>&1 &
    echo $! > "$PID_DIR/worker.pid"
    log_ok "Celery worker started (PID $(cat "$PID_DIR/worker.pid")) — logs: logs/worker.log"

    # ── 5. Streamlit Frontend ──
    log_info "Starting Streamlit frontend on http://localhost:8501 ..."
    cd "$PROJECT_DIR"
    "$VENV/streamlit" run frontend/chat.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true \
        > "$LOG_DIR/frontend.log" 2>&1 &
    echo $! > "$PID_DIR/frontend.pid"
    log_ok "Streamlit started (PID $(cat "$PID_DIR/frontend.pid")) — logs: logs/frontend.log"

    # ── 6. Summary ──
    sleep 2
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║            ✅ All services running!              ║${NC}"
    echo -e "${GREEN}╠══════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║  🌐 Frontend:  http://localhost:8501             ║${NC}"
    echo -e "${GREEN}║  🔧 API Docs:  http://localhost:8000/docs        ║${NC}"
    echo -e "${GREEN}║  📊 Metrics:   http://localhost:8000/metrics      ║${NC}"
    echo -e "${GREEN}║  🗄️  Redis:     localhost:6379                    ║${NC}"
    echo -e "${GREEN}╠══════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║  📝 Logs:  tail -f logs/api.log                  ║${NC}"
    echo -e "${GREEN}║           tail -f logs/worker.log                ║${NC}"
    echo -e "${GREEN}║           tail -f logs/frontend.log              ║${NC}"
    echo -e "${GREEN}║                                                  ║${NC}"
    echo -e "${GREEN}║  🛑 Stop:  ./start_local.sh stop                 ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}💡 Edit any file in src/ — FastAPI will auto-reload instantly!${NC}"
    echo -e "${YELLOW}💡 No Docker rebuild needed. No laptop heating.${NC}"
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────

case "${1:-start}" in
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 1
        start_services
        ;;
    start|"")
        start_services
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac
