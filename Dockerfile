# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 (change if your app uses a different port)
EXPOSE 5000

# Command to run your app
CMD ["python", "app.py"]
