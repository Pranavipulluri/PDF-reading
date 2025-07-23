# Enhanced Dockerfile for PDF Outline Extractor v2.0
FROM python:3.11-slim

# Set platform explicitly
ARG TARGETPLATFORM=linux/amd64

# Install system dependencies including PyMuPDF requirements
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libfontconfig1 \
    libfreetype6 \
    libxml2 \
    libxslt1.1 \
    libmupdf-dev \
    libopenjp2-7 \
    libjpeg62-turbo \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .

# Install Python packages with verbose output for debugging
RUN pip install --no-cache-dir --upgrade pip && \
    echo "Installing PyMuPDF..." && \
    pip install --no-cache-dir --verbose PyMuPDF && \
    echo "Installing requirements..." && \
    pip install --no-cache-dir --verbose -r requirements.txt && \
    echo "Verifying installations..." && \
    python -c "import fitz; print(f'PyMuPDF version: {fitz.__version__}')" && \
    python -c "import pdfplumber; print(f'pdfplumber version: {pdfplumber.__version__}')" && \
    echo "All packages installed:" && \
    pip list

# Copy source code
COPY src/ ./src/
COPY training/ ./training/

# Apply code fixes during build
RUN sed -i 's/class EnhancedSemanticTextAnalyzer:/class SemanticTextAnalyzer:/' src/text_analyzer.py && \
    echo -e "\n# Alias for compatibility\nTitleExtractor = SmartTitleExtractor" >> src/title_extractor.py && \
    sed -i '1i\import re' src/heading_detector.py && \
    for file in src/text_analyzer.py src/title_extractor.py src/rule_engine.py; do \
        if ! grep -q "^import re" "$file"; then \
            sed -i '1i\import re' "$file"; \
        fi; \
    done

# Copy run script
COPY run.sh ./
RUN chmod +x run.sh

# Set up Python environment
ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create necessary directories
RUN mkdir -p /app/input /app/output /app/models

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" 2>/dev/null || echo "NLTK download skipped"

# Generate lightweight model
RUN cd training && \
    python generate_training_data.py 2>/dev/null && \
    python train_classifier.py --lightweight --data-type synthetic --output ../models/heading_classifier.pkl 2>/dev/null || \
    echo "Model training skipped - will use rule-based detection"

# Verify PyMuPDF installation
RUN python -c "import fitz; print('PyMuPDF successfully imported')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); from pdf_processor import AdvancedPDFProcessor; print('OK')" || exit 1

# Default command
CMD ["./run.sh"]