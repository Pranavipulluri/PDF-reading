#!/bin/bash

echo "🚀 Ultra-Simple PDF Outline Extractor"
echo "====================================="

# Check input directory
if [ ! -d "/app/input" ]; then
    echo "❌ Input directory not found"
    exit 1
fi

# Count PDF files
pdf_count=$(find /app/input -name "*.pdf" -type f | wc -l)
echo "📄 Found $pdf_count PDF file(s) to process"

if [ $pdf_count -eq 0 ]; then
    echo "⚠️  No PDF files found in /app/input"
    echo "   Add PDF files to the input directory and try again"
    exit 0
fi

# Process each PDF
find /app/input -name "*.pdf" -type f | while read pdf_file; do
    filename=$(basename "$pdf_file" .pdf)
    output_file="/app/output/${filename}.json"
    
    echo ""
    echo "📖 Processing: $filename"
    echo "   Input: $pdf_file"
    echo "   Output: $output_file"
    
    python /app/main.py "$pdf_file" "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully processed: $filename"
    else
        echo "❌ Failed to process: $filename"
    fi
done

echo ""
echo "🎉 Processing complete!"
echo "📁 Check /app/output for results"