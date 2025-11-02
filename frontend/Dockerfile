FROM nginx:alpine

# Install wget for health checks
RUN apk add --no-cache wget

# Copy nginx config first
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy static files to nginx html directory (exclude nginx.conf and Dockerfile)
COPY *.html *.js *.css /usr/share/nginx/html/
COPY static/ /usr/share/nginx/html/static/

# Health check for Sevalla deployment
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8501/health || exit 1

EXPOSE 8501

CMD ["nginx", "-g", "daemon off;"]

