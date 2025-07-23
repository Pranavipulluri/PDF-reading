#!/bin/bash

# PDF Outline Extractor Runner
# This script processes all PDF files in /app/input and outputs JSON to /app/output

set -e

echo "Starting PDF Outline Extractor..."
echo "Input directory: /app/input"
echo "Output directory: /app/output"

# Check if input directory exists and has files
if [ ! -d "/app/input" ]; then
    echo "Error: Input directory /app/input does not exist"
    exit 1
fi

# Count PDF files
pdf_count=$(find /app/input -name "*.pdf" -type f | wc -l)
echo "Found $pdf_count PDF file(s) to process"

if [ $pdf_count -eq 0 ]; then
    echo "Warning: No PDF files found in input directory"
    exit 0
fi

# Process each PDF file
find /app/input -name "*.pdf" -type f | while read pdf_file; do
    echo "Processing: $(basename "$pdf_file")"
    
    # Get filename without extension for output
    base_name=$(basename "$pdf_file" .pdf)
    output_file="/app/output/${base_name}.json"
    
    # Run the main Python script
    python /app/src/main.py "$pdf_file" "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "Successfully processed: $(basename "$pdf_file")"
    else
        echo "Error processing: $(basename "$pdf_file")"
    fi
done

echo "PDF processing complete!"