# Simple backend-only deployment
FROM python:3.10-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL backend files to /app directory
COPY backend/ ./

# Copy frontend as static files
COPY frontend/ ./static/

# Verify app.py exists and show directory contents
RUN ls -la /app/ && test -f /app/app.py || (echo "ERROR: app.py not found!" && exit 1)

# Environment
ENV PYTHONPATH=/app
ENV PORT=10000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE $PORT

# Use shell form to properly expand PORT variable
CMD python -m uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1