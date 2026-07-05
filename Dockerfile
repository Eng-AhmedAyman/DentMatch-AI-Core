# ==============================================================================
# Dockerfile — DentMatch AI (Healthy Smile Core)
# Optimised for Hugging Face Spaces (free tier)
# HF Spaces requires the app to listen on port 7860
# ==============================================================================

FROM python:3.10-slim

# HF Spaces metadata — tells the platform which port to expose
LABEL org.opencontainers.image.description="DentMatch AI — Healthy Smile Core Engine"

WORKDIR /app

# Install system dependencies needed by OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Make entrypoint executable (kept for local/dev convenience — not used by
# this image's CMD, since the deployed Space serves the FastAPI backend
# directly to external callers, not the Streamlit dashboard)
RUN chmod +x start.sh

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Run FastAPI directly on the public port — colleagues call /analyze/,
# /triage-symptoms/, and /docs on this Space's public URL directly, so the
# API must be reachable externally, not just inside the container.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]