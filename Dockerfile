# Use Python 3.12.8 as base image
FROM python:3.12.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the application files
COPY models.py .
COPY prompts.py .
COPY server.py .

# Remove any env files 
RUN find . -name "*.env*" -type f -delete

# Copy the .env file
# COPY .env .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"] 