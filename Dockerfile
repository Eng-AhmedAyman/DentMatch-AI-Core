# ==============================================================================
# Dockerfile for Healthy Smile AI Engine
# ==============================================================================

# 1. Use an official, lightweight Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install Python dependencies securely without caching to save space
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the project source code into the container
COPY . .

# 6. Expose ports for both Streamlit (UI) and FastAPI (Backend)
EXPOSE 8501
EXPOSE 8000

# 7. Command to run the Streamlit Dashboard by default
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]