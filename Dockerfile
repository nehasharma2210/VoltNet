# Multi-stage build for Render deployment
FROM python:3.10-slim as backend-build

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy frontend files
COPY frontend/ ./frontend/

# Configure nginx for frontend
COPY frontend/nginx.conf /etc/nginx/sites-available/default

# Create startup script
RUN echo '#!/bin/bash\n\
# Start nginx for frontend\n\
nginx -g "daemon on;"\n\
\n\
# Start backend API\n\
cd /app/backend\n\
exec python -m uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1\n\
' > /start.sh && chmod +x /start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

EXPOSE $PORT

CMD ["/start.sh"]