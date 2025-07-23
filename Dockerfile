# Simplified, robust Dockerfile for PDF Outline Extractor v2.0
FROM python:3.11-slim

# Set platform explicitly
ARG TARGETPLATFORM=linux/amd64

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libfontconfig1 \
    libfreetype6 \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python packages with retry mechanism
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt || \
    (echo "Retrying pip install..." && pip install --no-cache-dir -r requirements.txt)

# Copy source code
COPY src/ ./src/
COPY training/ ./training/
COPY run.sh ./

# Set up Python environment
ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Make scripts executable
RUN chmod +x run.sh

# Create directories
RUN mkdir -p /app/input /app/output /app/models

# Download NLTK data (with fallback)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" 2>/dev/null || echo "NLTK download skipped"

# Try to generate lightweight model (optional)
RUN cd training && \
    python generate_training_data.py 2>/dev/null && \
    python train_classifier.py --lightweight --data-type synthetic --output ../models/heading_classifier.pkl 2>/dev/null || \
    echo "Model training skipped - will use rule-based detection"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); from pdf_processor import AdvancedPDFProcessor; print('OK')" || exit 1

# Default command
CMD ["./run.sh"]