#!/bin/bash

# Multi-mode runner - supports both web and CLI modes
# Usage: ./run_web.sh [web|cli]

MODE=${1:-web}  # Default to web mode

echo "🚀 PDF Outline Extractor v2.0"
echo "=============================="

case $MODE in
    "web")
        echo "🌐 Starting Web Interface..."
        echo "Access at: http://localhost:5000"
        echo "Press Ctrl+C to stop"
        echo ""
        
        # Check if running in Docker
        if [ -f "/.dockerenv" ]; then
            echo "Running in Docker container..."
            python web_server.py
        else
            echo "Running locally..."
            # Try to use Docker first
            if docker --version >/dev/null 2>&1; then
                echo "Using Docker..."
                docker run -p 5000:5000 pdf-extractor-web:latest
            else
                echo "Docker not available, running Python directly..."
                python web_server.py
            fi
        fi
        ;;
        
    "cli")
        echo "💻 Starting Command-Line Processing..."
        echo "Processing PDFs from input/ directory..."
        echo ""
        
        # Check if running in Docker
        if [ -f "/.dockerenv" ]; then
            echo "Running in Docker container..."
            find /app/input -name "*.pdf" -type f | while read pdf_file; do
                echo "Processing: $(basename "$pdf_file")"
                base_name=$(basename "$pdf_file" .pdf)
                output_file="/app/output/${base_name}.json"
                python /app/src/main.py "$pdf_file" "$output_file"
            done
        else
            echo "Running locally..."
            # Try to use Docker first
            if docker --version >/dev/null 2>&1; then
                echo "Using Docker..."
                docker run --rm \
                    -v $(pwd)/input:/app/input \
                    -v $(pwd)/output:/app/output \
                    --network none \
                    pdf-extractor-web:latest \
                    ./run.sh
            else
                echo "Docker not available, processing locally..."
                mkdir -p output
                find input -name "*.pdf" -type f | while read pdf_file; do
                    echo "Processing: $(basename "$pdf_file")"
                    base_name=$(basename "$pdf_file" .pdf)
                    output_file="output/${base_name}.json"
                    python src/main.py "$pdf_file" "$output_file"
                done
            fi
        fi
        echo "✅ Processing complete!"
        ;;
        
    *)
        echo "Usage: $0 [web|cli]"
        echo ""
        echo "Modes:"
        echo "  web  - Start web interface (default)"
        echo "  cli  - Run command-line processing"
        echo ""
        echo "Examples:"
        echo "  $0        # Start web interface"
        echo "  $0 web    # Start web interface"
        echo "  $0 cli    # Process PDFs from input/ directory"
        exit 1
        ;;
esac