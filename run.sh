#!/bin/bash
echo "Processing PDFs from input directory..."
find /app/input -name "*.pdf" -type f | while read pdf_file; do
    echo "Processing: $^(basename "$pdf_file"^)"
    base_name=$(basename "$pdf_file" .pdf)
    output_file="/app/output/${base_name}.json"
    python /app/src/main.py "$pdf_file" "$output_file"
done
echo "Processing complete!"
