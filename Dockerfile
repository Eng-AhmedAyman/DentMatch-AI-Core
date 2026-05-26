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

# Make entrypoint executable
RUN chmod +x start.sh

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Run FastAPI directly on the public port
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]