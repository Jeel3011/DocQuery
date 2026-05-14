FROM python:3.11-slim

# System deps for Unstructured (poppler for PDF, tesseract for OCR, libmagic for MIME)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (separate layer — only rebuilds on requirements change)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .
RUN pip install -e .

# Non-root user for security
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# Default start command — override with your platform's start command config
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
