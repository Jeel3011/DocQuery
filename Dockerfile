FROM python:3.11-slim

# ── Layer 1: System packages (very stable, rebuilds ~never) ─────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer 2: Heavy Python deps (~800MB, rebuilds ~never) ────────────────────
# torch, sentence-transformers, unstructured, numpy, pandas, etc.
# This layer takes 10-15 min from scratch but is cached across deploys.
# Only rebuilds if you modify requirements-base.txt (rare).
COPY requirements-base.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-base.txt

# ── Layer 3: App Python deps (~50MB, rebuilds on dep version bumps) ─────────
# langchain, fastapi, celery, redis, etc.
# pip skips packages already installed in Layer 2.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 4: Source code (rebuilds on every code change — fast, ~seconds) ───
COPY . .
RUN pip install --no-deps -e .

# ── Layer 5: Security (stable) ──────────────────────────────────────────────
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# Default start command — override with your platform's start command config
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
