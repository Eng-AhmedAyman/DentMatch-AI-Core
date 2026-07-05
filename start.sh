#!/bin/bash
# ==============================================================================
# FILE: start.sh
# Entrypoint for DentMatch AI on Hugging Face Spaces
#
# HF Spaces exposes ONE port (7860). Architecture:
#   - Streamlit  → port 7860  (public-facing, what HF exposes)
#   - FastAPI    → port 8000  (internal only, Streamlit talks to it internally)
# ==============================================================================

set -e

# Validate required environment variables
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ ERROR: GEMINI_API_KEY is not set. Add it in HF Space secrets."
    exit 1
fi

echo "=============================================="
echo "  DentMatch AI — Healthy Smile Core Engine"
echo "  Starting on Hugging Face Spaces"
echo "=============================================="

# 1. Start FastAPI backend on internal port 8000
echo "[1/2] Starting FastAPI backend (internal port 8000)..."
uvicorn api:app --host 127.0.0.1 --port 8000 &

# 2. Wait until FastAPI is actually ready instead of a fixed sleep.
#    Loading 3 models (MobileNet + HuggingFace ViT + EfficientNetB4) can take
#    well over 5 seconds on HF Spaces' free CPU tier, so a fixed sleep either
#    wastes time or — worse — lets Streamlit start before the backend can
#    answer requests. Poll the health endpoint instead, bounded so a broken
#    backend can't hang the container forever.
echo "      Waiting for FastAPI backend to become healthy..."
MAX_WAIT=180
WAITED=0
until python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=2)" > /dev/null 2>&1; do
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "⚠️  FastAPI did not respond within ${MAX_WAIT}s — starting Streamlit anyway."
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo "      ... still loading models (${WAITED}s elapsed)"
done
echo "✅ FastAPI backend check finished after ${WAITED}s."

# 3. Start Streamlit on port 7860 (the port HF Spaces exposes publicly)
echo "[2/2] Starting Streamlit dashboard (public port 7860)..."
streamlit run app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none \
    --server.enableCORS false \
    --server.enableXsrfProtection false