FROM python:3.10-slim
WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Verify critical files exist (will fail build if app.py is missing)
RUN test -f /app/app.py || (echo "ERROR: app.py not found in /app!" && ls -la /app/ && exit 1)

# Health check for Sevalla
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PORT=8000
ENV PYTHONPATH=/app
EXPOSE ${PORT}

# Run uvicorn with explicit app reference
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
