FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

WORKDIR /app

# System deps kept minimal.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Pre-normalize catalog + warm the embedding cache at build time so the first
# request isn't penalized by a cold model download. Failures here are non-fatal
# (the app degrades to lexical retrieval at runtime).
RUN python -m scripts.build_catalog || echo "build_catalog step skipped (will run lazily at startup)"

EXPOSE 7860

# HF Spaces uses 7860; other platforms inject $PORT.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
