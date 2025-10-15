# Use nginx lightweight base image
FROM nginx:alpine

# Copy your HTML file to the default Nginx web directory
COPY index.html /usr/share/nginx/html/

# Expose port 80 (default web port)
EXPOSE 80

# Default command to run Nginx
CMD ["nginx", "-g", "daemon off;"]