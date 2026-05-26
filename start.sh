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

# 2. Give FastAPI time to load all 3 AI models before Streamlit starts
echo "      Waiting for models to load..."
sleep 5

# 3. Start Streamlit on port 7860 (the port HF Spaces exposes publicly)
echo "[2/2] Starting Streamlit dashboard (public port 7860)..."
streamlit run app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none \
    --server.enableCORS false \
    --server.enableXsrfProtection false