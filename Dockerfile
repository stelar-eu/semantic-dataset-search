# Use Python 3.12.8 as base image
FROM python:3.13.7-slim

# Set working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv using pip
RUN pip install uv

# Verify uv installation
RUN uv --version

# Copy dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN uv sync --frozen

# Copy the application files
COPY src/ ./src/

# Remove any env files 
RUN find . -name "*.env*" -type f -delete

# Copy the .env file
# COPY .env .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"] 