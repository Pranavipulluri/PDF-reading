"""
Setup Script - Initialize and configure the PDF Outline Extractor
Downloads required models and data for offline operation
"""

import os
import sys
import logging
from pathlib import Path

def setup_nltk():
    """Download required NLTK data"""
    try:
        import nltk
        
        # Set NLTK data path
        nltk_data_path = Path.home() / 'nltk_data'
        nltk_data_path.mkdir(exist_ok=True)
        
        # Download required datasets
        downloads = [
            ('punkt', 'tokenizers/punkt'),
            ('stopwords', 'corpora/stopwords'),
            ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')
        ]
        
        for name, check_path in downloads:
            try:
                nltk.data.find(check_path)
                print(f"✓ NLTK {name} already available")
            except LookupError:
                print(f"Downloading NLTK {name}...")
                nltk.download(name, quiet=True)
                print(f"✓ NLTK {name} downloaded")
        
        return True
        
    except Exception as e:
        print(f"⚠ NLTK setup failed: {e}")
        return False

def setup_sklearn():
    """Initialize scikit-learn components"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Test basic functionality
        rf = RandomForestClassifier(n_estimators=10)
        scaler = StandardScaler()
        vectorizer = TfidfVectorizer(max_features=10)
        
        print("✓ scikit-learn components verified")
        return True
        
    except Exception as e:
        print(f"⚠ scikit-learn setup failed: {e}")
        return False

def setup_pdf_libraries():
    """Verify PDF processing libraries"""
    try:
        # Test PyMuPDF
        import fitz
        print(f"✓ PyMuPDF {fitz.__version__} available")
        
        # Test PDFPlumber
        import pdfplumber
        print(f"✓ PDFPlumber {pdfplumber.__version__} available")
        
        return True
        
    except Exception as e:
        print(f"⚠ PDF libraries setup failed: {e}")
        return False

def generate_lightweight_model():
    """Generate a lightweight ML model for deployment"""
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / 'src'))
        
        from training.generate_training_data import TrainingDataGenerator
        from training.train_classifier import ClassifierTrainer
        
        print("Generating lightweight ML model...")
        
        # Create training data
        generator = TrainingDataGenerator()
        blocks, labels = generator.generate_synthetic_training_data(500)
        
        # Add multilingual samples
        multi_blocks, multi_labels = generator.generate_multilingual_samples(100)
        blocks.extend(multi_blocks)
        labels.extend(multi_labels)
        
        # Train lightweight model
        trainer = ClassifierTrainer()
        model_path = trainer.create_lightweight_model(blocks, labels, max_model_size_mb=50)
        
        # Verify model size
        if Path(model_path).exists():
            size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            print(f"✓ Lightweight model created: {size_mb:.2f}MB")
            return True
        else:
            print("⚠ Model file not created")
            return False
            
    except Exception as e:
        print(f"⚠ Model generation failed: {e}")
        print("Continuing without ML model (rule-based detection will be used)")
        return False

def verify_installation():
    """Verify complete installation"""
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / 'src'))
        
        # Test core components
        from pdf_processor import AdvancedPDFProcessor
        from heading_detector import AdvancedHeadingDetector
        from title_extractor import TitleExtractor
        
        # Initialize components
        pdf_processor = AdvancedPDFProcessor()
        heading_detector = AdvancedHeadingDetector()
        title_extractor = TitleExtractor()
        
        print("✓ Core components initialized successfully")
        
        # Test feature extraction
        from feature_extractor import ComprehensiveFeatureExtractor
        feature_extractor = ComprehensiveFeatureExtractor()
        print("✓ Feature extraction system available")
        
        # Test text analysis
        from text_analyzer import SemanticTextAnalyzer
        text_analyzer = SemanticTextAnalyzer()
        print("✓ Semantic text analysis system available")
        
        return True
        
    except Exception as e:
        print(f"⚠ Installation verification failed: {e}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("PDF Outline Extractor v2.0 - Setup & Configuration")
    print("="*60)
    
    success_count = 0
    total_tests = 5
    
    # Setup NLTK
    print("\n1. Setting up NLTK...")
    if setup_nltk():
        success_count += 1
    
    # Setup scikit-learn
    print("\n2. Verifying scikit-learn...")
    if setup_sklearn():
        success_count += 1
    
    # Setup PDF libraries
    print("\n3. Verifying PDF libraries...")
    if setup_pdf_libraries():
        success_count += 1
    
    # Generate ML model
    print("\n4. Generating ML model...")
    if generate_lightweight_model():
        success_count += 1
    
    # Verify installation
    print("\n5. Verifying installation...")
    if verify_installation():
        success_count += 1
    
    # Final report
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print(f"Success: {success_count}/{total_tests} components initialized")
    
    if success_count == total_tests:
        print("✓ All systems operational - Ready for PDF processing!")
        return 0
    elif success_count >= 3:
        print("⚠ Partial setup - Basic functionality available")
        return 0
    else:
        print("✗ Setup failed - Check error messages above")
        return 1

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run setup
    exit_code = main()
    sys.exit(exit_code)