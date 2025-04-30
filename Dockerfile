# Use Python 3.9 as base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    # Disable Chrome download for sentence-transformers (not needed for inference)
    SENTENCE_TRANSFORMERS_HOME=/app/models \
    # Set a specific port
    PORT=5000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -r appuser && \
    mkdir -p /app/uploads /app/chroma_db /app/models && \
    chown -R appuser:appuser /app

# Copy requirements first to leverage Docker cache
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Create .gitkeep files for empty directories
RUN touch uploads/.gitkeep chroma_db/.gitkeep

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Run the application with gunicorn
CMD ["python", "app.py"] 