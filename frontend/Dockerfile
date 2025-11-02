FROM nginx:alpine

# Copy nginx config first
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy static files to nginx html directory (exclude nginx.conf and Dockerfile)
COPY *.html *.js *.css /usr/share/nginx/html/
COPY static/ /usr/share/nginx/html/static/

EXPOSE 8501

CMD ["nginx", "-g", "daemon off;"]

