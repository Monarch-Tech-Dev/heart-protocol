FROM python:3.11-slim

LABEL maintainer="Heart Protocol Community"
LABEL description="Heart Protocol: Algorithms for Human Flourishing"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY heart_protocol/ ./heart_protocol/
COPY *.py ./

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash monarch && \
    chown -R monarch:monarch /app
USER monarch

# Expose ports
EXPOSE 8080 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "-m", "heart_protocol.main"]