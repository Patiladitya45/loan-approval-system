# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (change 5000 if your app uses another)
EXPOSE 5000

# Run the Python app
CMD ["python", "app.py"]
