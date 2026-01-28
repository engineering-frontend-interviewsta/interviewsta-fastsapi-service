# FastAPI Interview Service Dockerfile
FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set work directory
WORKDIR /app

# Install system dependencies with retry and better error handling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
