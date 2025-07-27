#!/bin/bash
echo "🔍 Advanced PDF Processing from input directory..."
find /app/input -name "*.pdf" -type f | while read pdf_file; do
    echo "🤖 Processing with advanced algorithms: $^(basename "$pdf_file"^)"
    base_name=$(basename "$pdf_file" .pdf)
    output_file="/app/output/${base_name}_advanced.json"
    python /app/src/main.py "$pdf_file" "$output_file"
done
echo "✅ Advanced processing complete!"
