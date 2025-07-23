#!/bin/bash

# PDF Outline Extractor - Local Setup Script

echo "=== Setting up PDF Outline Extractor locally ==="

# Step 1: Create virtual environment
echo "1. Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate || . venv/bin/activate

# Step 2: Upgrade pip
echo "2. Upgrading pip..."
pip install --upgrade pip

# Step 3: Install dependencies
echo "3. Installing dependencies..."
pip install -r requirements.txt

# Step 4: Download NLTK data
echo "4. Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print('NLTK data downloaded successfully')
"

# Step 5: Create necessary directories
echo "5. Creating necessary directories..."
mkdir -p input output models training/data

# Step 6: Apply fixes
echo "6. Applying code fixes..."

# Fix text_analyzer.py class name
sed -i.bak 's/class EnhancedSemanticTextAnalyzer:/class SemanticTextAnalyzer:/' src/text_analyzer.py

# Add TitleExtractor alias to title_extractor.py
echo -e "\n# Alias for compatibility\nTitleExtractor = SmartTitleExtractor" >> src/title_extractor.py

# Fix missing imports in heading_detector.py
sed -i.bak '1i\
import re' src/heading_detector.py

# Step 7: Generate training data and train model (optional)
echo "7. Generating training data and training model..."
cd training
python generate_training_data.py
python train_classifier.py --lightweight --data-type synthetic --output ../models/heading_classifier.pkl
cd ..

echo "=== Setup complete! ==="
echo ""
echo "To run the extractor:"
echo "1. Place PDF files in the 'input' directory"
echo "2. Run: python src/main.py input/your_file.pdf output/your_file.json"
echo ""
echo "Or to process all PDFs in input directory:"
echo "3. Run: ./run.sh"