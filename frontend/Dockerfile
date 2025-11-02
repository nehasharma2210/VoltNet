FROM nginx:alpine

# Install wget for health checks and envsubst for environment variable substitution
RUN apk add --no-cache wget gettext

# Copy nginx config first
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy static files to nginx html directory (exclude nginx.conf and Dockerfile)
COPY *.html *.js *.css /usr/share/nginx/html/
COPY static/ /usr/share/nginx/html/static/

# Copy config template
COPY config.js.template /usr/share/nginx/html/config.js.template

# Create startup script to inject BACKEND_URL environment variable
RUN echo '#!/bin/sh' > /docker-entrypoint.sh && \
    echo 'export BACKEND_URL=${BACKEND_URL:-https://voltnet.onrender.com}' >> /docker-entrypoint.sh && \
    echo 'envsubst < /usr/share/nginx/html/config.js.template > /usr/share/nginx/html/config.js' >> /docker-entrypoint.sh && \
    echo 'exec nginx -g "daemon off;"' >> /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh

# Health check for Sevalla deployment
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8501/health || exit 1

EXPOSE 8501

CMD ["/docker-entrypoint.sh"]

