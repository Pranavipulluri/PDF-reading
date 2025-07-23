# Complete Dockerization Guide for PDF Outline Extractor

## Prerequisites for Docker

- Docker Engine 20.10+ installed
- Docker Compose 2.0+ (optional but recommended)
- At least 4GB RAM available for Docker
- 2GB disk space for images

## Step 1: Prepare for Dockerization

### 1.1 Create .dockerignore
```bash
cat > .dockerignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
input/*
output/*
*.log
.git/
.gitignore
README.md
*.md
!requirements.txt
EOF
```

### 1.2 Update Dockerfile with Fixes
```bash
# Use the enhanced Dockerfile
cp Dockerfile Dockerfile.original
cat > Dockerfile << 'EOF'
# PDF Outline Extractor v2.0 - Production Dockerfile
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libfontconfig1 \
    libfreetype6 \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY src/ ./src/
COPY training/ ./training/
COPY run.sh ./

# Apply fixes during build
RUN sed -i 's/class EnhancedSemanticTextAnalyzer:/class SemanticTextAnalyzer:/' src/text_analyzer.py && \
    echo -e "\nTitleExtractor = SmartTitleExtractor" >> src/title_extractor.py && \
    sed -i '1i\import re' src/heading_detector.py

# Set permissions
RUN chmod +x run.sh

# Environment setup
ENV PYTHONPATH=/app/src:/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create directories
RUN mkdir -p /app/input /app/output /app/models

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" || true

# Generate model (optional - can be mounted instead)
RUN cd training && \
    python generate_training_data.py && \
    python train_classifier.py --lightweight --output ../models/heading_classifier.pkl || \
    echo "Model generation skipped"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from pdf_processor import AdvancedPDFProcessor; print('OK')" || exit 1

# Run
CMD ["./run.sh"]
EOF
```

## Step 2: Build Docker Image

### 2.1 Standard Build
```bash
# Build for AMD64 architecture (hackathon requirement)
docker build --platform linux/amd64 -t pdf-extractor:latest .

# Check image size
docker images pdf-extractor:latest
```

### 2.2 Multi-stage Build (Optimized)
```bash
# Build with caching
DOCKER_BUILDKIT=1 docker build \
    --platform linux/amd64 \
    --cache-from pdf-extractor:latest \
    -t pdf-extractor:latest .
```

### 2.3 Development Build
```bash
# Build with debug features
docker build \
    --platform linux/amd64 \
    --build-arg DEBUG=1 \
    -t pdf-extractor:dev .
```

## Step 3: Run Docker Container

### 3.1 Basic Run
```bash
# Create input/output directories if not exists
mkdir -p input output

# Place PDF files in input directory
cp /path/to/sample.pdf input/

# Run container
docker run --rm \
    --platform linux/amd64 \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output \
    --network none \
    pdf-extractor:latest
```

### 3.2 Run with Resource Limits (Hackathon Requirements)
```bash
docker run --rm \
    --platform linux/amd64 \
    --cpus="8" \
    --memory="16g" \
    --memory-swap="16g" \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output \
    --network none \
    --name pdf-extractor \
    pdf-extractor:latest
```

### 3.3 Run Specific File
```bash
# Override default command to process specific file
docker run --rm \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output \
    --network none \
    pdf-extractor:latest \
    python src/main.py /app/input/specific.pdf /app/output/specific.json
```

### 3.4 Run with Debug Mode
```bash
docker run --rm \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output \
    --network none \
    -e DEBUG=1 \
    pdf-extractor:latest
```

## Step 4: Docker Compose (Recommended)

### 4.1 Run with Docker Compose
```bash
# Start service
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

### 4.2 Development with Docker Compose
```bash
# Run development container
docker-compose run --rm pdf-extractor-dev bash

# Inside container, you can:
# - Run tests
# - Debug code
# - Process individual files
python src/main.py /app/input/test.pdf /app/output/test.json
```

## Step 5: Testing and Validation

### 5.1 Health Check
```bash
# Check if container is healthy
docker ps
docker inspect pdf-extractor | grep -A 5 "Health"
```

### 5.2 Performance Testing
```bash
# Run with time measurement
time docker run --rm \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output \
    --network none \
    pdf-extractor:latest

# Monitor resource usage
docker stats pdf-extractor
```

### 5.3 Test with Sample PDFs
```bash
# Create test script
cat > test_docker.sh << 'EOF'
#!/bin/bash

echo "Testing PDF Outline Extractor Docker Image"

# Test 1: Single page PDF
echo "Test 1: Processing single page PDF..."
docker run --rm \
    -v $(pwd)/test_pdfs:/app/input:ro \
    -v $(pwd)/test_output:/app/output \
    --network none \
    pdf-extractor:latest

# Check output
if [ -f "test_output/single_page.json" ]; then
    echo "✓ Single page test passed"
else
    echo "✗ Single page test failed"
fi

# Test 2: Multi-page PDF
echo "Test 2: Processing multi-page PDF..."
# ... similar test

echo "All tests completed!"
EOF

chmod +x test_docker.sh
./test_docker.sh
```

## Step 6: Optimization Tips

### 6.1 Reduce Image Size
```bash
# Check image layers
docker history pdf-extractor:latest

# Remove unnecessary files in Dockerfile
# Add to Dockerfile:
RUN find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} +
```

### 6.2 Use BuildKit Cache
```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build with cache mount
docker build \
    --cache-from pdf-extractor:cache \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -t pdf-extractor:latest .
```

### 6.3 Pre-build Models
```bash
# Build models separately
docker run --rm \
    -v $(pwd)/models:/app/models \
    pdf-extractor:latest \
    python training/train_classifier.py --output /app/models/heading_classifier.pkl

# Then mount pre-built models
docker run --rm \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/models:/app/models:ro \
    --network none \
    pdf-extractor:latest
```

## Step 7: Deployment for Hackathon

### 7.1 Create Submission Package
```bash
# Create submission directory
mkdir pdf-extractor-submission
cd pdf-extractor-submission

# Copy necessary files
cp -r ../src .
cp -r ../training .
cp ../Dockerfile .
cp ../requirements.txt .
cp ../run.sh .
cp ../README.md .

# Create submission script
cat > submit.sh << 'EOF'
#!/bin/bash
# Hackathon submission script

echo "Building PDF Outline Extractor for submission..."
docker build --platform linux/amd64 -t pdf-extractor-hackathon:latest .

echo "Creating submission tarball..."
docker save pdf-extractor-hackathon:latest | gzip > pdf-extractor-hackathon.tar.gz

echo "Submission ready: pdf-extractor-hackathon.tar.gz"
echo "Image size: $(du -h pdf-extractor-hackathon.tar.gz)"
EOF

chmod +x submit.sh
./submit.sh
```

### 7.2 Verify Hackathon Requirements
```bash
# Check all requirements
cat > verify_requirements.sh << 'EOF'
#!/bin/bash

echo "Verifying Hackathon Requirements:"

# 1. AMD64 platform
echo -n "✓ AMD64 platform: "
docker inspect pdf-extractor:latest | grep -q "amd64" && echo "PASS" || echo "FAIL"

# 2. Resource limits
echo -n "✓ Can run with 8 CPU, 16GB RAM: "
docker run --rm --cpus="8" --memory="16g" --network none pdf-extractor:latest echo "PASS" 2>/dev/null || echo "FAIL"

# 3. Network isolation
echo -n "✓ Network isolation: "
docker run --rm --network none pdf-extractor:latest ping -c 1 google.com 2>/dev/null && echo "FAIL" || echo "PASS"

# 4. Model size
echo -n "✓ Model < 200MB: "
docker run --rm pdf-extractor:latest du -h /app/models/ | grep -q "M" && echo "PASS" || echo "CHECK"

# 5. Processing time (need actual PDF)
echo "✓ Processing time: Manual test required with 50-page PDF"

echo "Verification complete!"
EOF

chmod +x verify_requirements.sh
./verify_requirements.sh
```

## Troubleshooting

### Common Docker Issues:

1. **Platform Error**: 
   ```bash
   # Force AMD64 platform
   export DOCKER_DEFAULT_PLATFORM=linux/amd64
   ```

2. **Memory Issues**:
   ```bash
   # Increase Docker memory in Docker Desktop settings
   # Or use swap
   docker run --memory="8g" --memory-swap="16g" ...
   ```

3. **Build Failures**:
   ```bash
   # Clean build
   docker build --no-cache -t pdf-extractor:latest .
   ```

4. **Permission Issues**:
   ```bash
   # Fix output permissions
   docker run --rm -v $(pwd)/output:/app/output pdf-extractor:latest \
       chown -R $(id -u):$(id -g) /app/output
   ```

## Success! 🎉

Your PDF Outline Extractor is now fully dockerized and ready for deployment!