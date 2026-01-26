FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY backend/ /app/

# Copy frontend files as static
COPY frontend/ /app/static/

# Debug: Show what's in the container
RUN echo "=== Container contents ===" && ls -la /app/ && echo "=== End ===" 

# Set environment
ENV PYTHONPATH=/app
ENV PORT=10000

# Expose port
EXPOSE $PORT

# Simple start command
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT}"]