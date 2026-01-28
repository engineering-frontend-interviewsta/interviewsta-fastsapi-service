# FastAPI Interview Service Dockerfile
FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONIOENCODING=utf-8

# Set work directory
WORKDIR /app

# Install system dependencies with retry and better error handling
# Includes audio processing, OCR, PDF handling, and AWS SDK requirements
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with timeout and retries
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.txt

# Copy application code
COPY . .

# Create directory for temporary files (audio processing)
RUN mkdir -p /tmp/audio_processing

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
