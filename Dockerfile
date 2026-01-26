# Simple backend-only deployment
FROM python:3.10-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Copy frontend as static files
COPY frontend/ ./static/

# Environment
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

EXPOSE $PORT

# Direct command - no script needed
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "$PORT"]