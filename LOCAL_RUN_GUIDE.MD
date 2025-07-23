# Complete Guide: Running PDF Outline Extractor Locally

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- 4GB+ RAM available
- Linux/macOS/Windows with WSL

## Step-by-Step Local Setup

### 1. Clone/Download the Project
```bash
# If you have the files, create project directory
mkdir pdf-outline-extractor
cd pdf-outline-extractor
# Copy all files here
```

### 2. Fix Code Issues

Create and run this fix script:

```bash
cat > apply_fixes.sh << 'EOF'
#!/bin/bash

# Fix text_analyzer.py class name
sed -i.bak 's/class EnhancedSemanticTextAnalyzer:/class SemanticTextAnalyzer:/' src/text_analyzer.py

# Add TitleExtractor alias
echo -e "\n# Alias for compatibility\nTitleExtractor = SmartTitleExtractor" >> src/title_extractor.py

# Fix missing re import in heading_detector.py
sed -i.bak '1i\
import re' src/heading_detector.py

# Fix missing re imports in other files
for file in src/text_analyzer.py src/title_extractor.py src/rule_engine.py; do
    if ! grep -q "^import re" "$file"; then
        sed -i.bak '1i\
import re' "$file"
    fi
done

echo "Fixes applied successfully!"
EOF

chmod +x apply_fixes.sh
./apply_fixes.sh
```

### 3. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 4. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 5. Download NLTK Data
```bash
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
"
```

### 6. Create Directory Structure
```bash
mkdir -p input output models training/data
```

### 7. Generate and Train Model (Optional but Recommended)
```bash
# Generate training data
cd training
python generate_training_data.py

# Train lightweight model
python train_classifier.py --lightweight --data-type synthetic --output ../models/heading_classifier.pkl
cd ..
```

### 8. Test the System
```bash
# Create a test run script
cat > test_run.py << 'EOF'
import sys
sys.path.append('src')
from pdf_processor import AdvancedPDFProcessor
from heading_detector import AdvancedHeadingDetector
from title_extractor import TitleExtractor

print("All modules loaded successfully!")
print("System is ready to process PDFs.")
EOF

python test_run.py
```

## Running the Extractor

### Option 1: Process Single PDF
```bash
# Make sure you're in the project root with venv activated
python src/main.py input/sample.pdf output/sample.json
```

### Option 2: Process All PDFs in Input Directory
```bash
# Make run.sh executable
chmod +x run.sh

# Run it
./run.sh
```

### Option 3: Run with Debug Mode
```bash
# Set DEBUG environment variable
DEBUG=1 python src/main.py input/sample.pdf output/sample.json
```

## Testing with Sample PDF

Create a simple test:
```bash
# Download a sample PDF (or use your own)
# Place it in the input directory
cp /path/to/your/test.pdf input/test.pdf

# Run the extractor
python src/main.py input/test.pdf output/test.json

# Check the output
cat output/test.json
```

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**: Make sure virtual environment is activated
   ```bash
   source venv/bin/activate
   ```

2. **NLTK Data Error**: Re-download NLTK data
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

3. **Memory Error**: For large PDFs, you might need to increase available memory
   ```bash
   # Check available memory
   free -h
   ```

4. **Permission Error**: Make sure directories are writable
   ```bash
   chmod -R 755 input output models
   ```

## Performance Optimization

For better performance:
```bash
# Run with performance monitoring
time python src/main.py input/large_document.pdf output/large_document.json

# Monitor memory usage
# In another terminal:
watch -n 1 'ps aux | grep python'
```

## Next Steps

Once everything works locally, you can proceed to dockerize the application!