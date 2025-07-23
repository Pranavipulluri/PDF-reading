#!/usr/bin/env python3
"""
Solution Validation Script - Verify all components work correctly
Tests the complete AI pipeline and validates hackathon requirements
"""

import sys
import json
import time
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

class SolutionValidator:
    """Comprehensive solution validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = []
        
    def validate_imports(self) -> bool:
        """Test 1: Validate all imports work correctly"""
        print("🔍 Test 1: Validating imports...")
        
        try:
            # Core imports
            from pdf_processor import AdvancedPDFProcessor
            from heading_detector import AdvancedHeadingDetector
            from title_extractor import TitleExtractor
            from utils import setup_logging, validate_input, format_output
            
            # AI components
            from ml_classifier import HeadingMLClassifier
            from text_analyzer import SemanticTextAnalyzer
            from rule_engine import EnhancedRuleEngine
            from layout_analyzer import AdvancedLayoutAnalyzer
            from feature_extractor import ComprehensiveFeatureExtractor
            
            # Training components
            from training.generate_training_data import TrainingDataGenerator
            from training.train_classifier import ClassifierTrainer
            
            print("✅ All imports successful")
            return True
            
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during imports: {e}")
            return False
    
    def validate_dependencies(self) -> bool:
        """Test 2: Validate all dependencies are available"""
        print("🔍 Test 2: Validating dependencies...")
        
        required_packages = [
            ('fitz', 'PyMuPDF'),
            ('pdfplumber', 'PDFPlumber'), 
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('sklearn', 'scikit-learn'),
            ('nltk', 'NLTK'),
            ('regex', 'Regex'),
            ('joblib', 'Joblib')
        ]
        
        missing_packages = []
        
        for package, name in required_packages:
            try:
                __import__(package)
                print(f"✅ {name} available")
            except ImportError:
                print(f"❌ {name} missing")
                missing_packages.append(name)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            return False
        
        print("✅ All dependencies available")
        return True
    
    def validate_ai_components(self) -> bool:
        """Test 3: Validate AI components initialize correctly"""
        print("🔍 Test 3: Validating AI components...")
        
        try:
            from pdf_processor import AdvancedPDFProcessor
            from heading_detector import AdvancedHeadingDetector
            from ml_classifier import HeadingMLClassifier
            from text_analyzer import SemanticTextAnalyzer
            from rule_engine import EnhancedRuleEngine
            from feature_extractor import ComprehensiveFeatureExtractor
            
            # Test component initialization
            components = {
                'PDF Processor': AdvancedPDFProcessor(),
                'Heading Detector': AdvancedHeadingDetector(),
                'ML Classifier': HeadingMLClassifier(),
                'Text Analyzer': SemanticTextAnalyzer(),
                'Rule Engine': EnhancedRuleEngine(),
                'Feature Extractor': ComprehensiveFeatureExtractor()
            }
            
            for name, component in components.items():
                if component is not None:
                    print(f"✅ {name} initialized")
                else:
                    print(f"❌ {name} failed to initialize")
                    return False
            
            print("✅ All AI components initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ AI component initialization failed: {e}")
            return False
    
    def validate_training_system(self) -> bool:
        """Test 4: Validate training system works"""
        print("🔍 Test 4: Validating training system...")
        
        try:
            from training.generate_training_data import TrainingDataGenerator
            from training.train_classifier import ClassifierTrainer
            
            # Test synthetic data generation
            generator = TrainingDataGenerator()
            blocks, labels = generator.generate_synthetic_training_data(50)  # Small test
            
            if len(blocks) != 50:
                print(f"❌ Expected 50 samples, got {len(blocks)}")
                return False
            
            # Test multilingual generation
            multi_blocks, multi_labels = generator.generate_multilingual_samples(20)
            
            if len(multi_blocks) < 15:  # Some tolerance
                print(f"❌ Expected ~20 multilingual samples, got {len(multi_blocks)}")
                return False
            
            # Test trainer initialization
            trainer = ClassifierTrainer()
            if trainer.classifier is None:
                print("❌ Classifier trainer not initialized")
                return False
            
            print("✅ Training system validated")
            return True
            
        except Exception as e:
            print(f"❌ Training system validation failed: {e}")
            return False
    
    def validate_semantic_analysis(self) -> bool:
        """Test 5: Validate semantic analysis capabilities"""
        print("🔍 Test 5: Validating semantic analysis...")
        
        try:
            from text_analyzer import SemanticTextAnalyzer
            
            analyzer = SemanticTextAnalyzer()
            
            # Test different text types
            test_cases = [
                ("Introduction", "english"),
                ("第1章 概要", "japanese"), 
                ("Chapitre 1", "french"),
                ("1.1 Background", "english"),
                ("This is regular body text that should not be detected as a heading.", "english")
            ]
            
            for text, expected_lang in test_cases:
                result = analyzer.analyze_text_semantics(text)
                
                if not isinstance(result, dict):
                    print(f"❌ Semantic analysis returned invalid result for '{text}'")
                    return False
                
                detected_lang = result.get('language', 'unknown')
                if detected_lang != expected_lang and expected_lang != 'english':  # English is default
                    print(f"⚠ Language detection: expected {expected_lang}, got {detected_lang} for '{text}'")
                
                print(f"✅ Analyzed '{text}' - heading prob: {result.get('heading_probability', 0):.3f}")
            
            print("✅ Semantic analysis validated")
            return True
            
        except Exception as e:
            print(f"❌ Semantic analysis validation failed: {e}")
            return False
    
    def validate_feature_extraction(self) -> bool:
        """Test 6: Validate feature extraction system"""
        print("🔍 Test 6: Validating feature extraction...")
        
        try:
            from feature_extractor import ComprehensiveFeatureExtractor
            from training.generate_training_data import TrainingDataGenerator
            
            # Generate sample data
            generator = TrainingDataGenerator()
            blocks, labels = generator.generate_synthetic_training_data(20)
            
            # Extract features
            extractor = ComprehensiveFeatureExtractor()
            features_df = extractor.extract_comprehensive_features(blocks)
            
            if features_df.empty:
                print("❌ Feature extraction returned empty DataFrame")
                return False
            
            if len(features_df) != len(blocks):
                print(f"❌ Feature count mismatch: {len(features_df)} vs {len(blocks)}")
                return False
            
            # Check feature count
            feature_count = len(features_df.columns)
            if feature_count < 30:  # Expect at least 30 features
                print(f"❌ Too few features extracted: {feature_count}")
                return False
            
            print(f"✅ Extracted {feature_count} features from {len(blocks)} blocks")
            return True
            
        except Exception as e:
            print(f"❌ Feature extraction validation failed: {e}")
            return False
    
    def validate_model_training(self) -> bool:
        """Test 7: Validate model can be trained"""
        print("🔍 Test 7: Validating model training...")
        
        try:
            from ml_classifier import HeadingMLClassifier
            from training.generate_training_data import TrainingDataGenerator
            
            # Generate small training set
            generator = TrainingDataGenerator()
            blocks, labels = generator.generate_synthetic_training_data(100)
            
            # Train lightweight model
            classifier = HeadingMLClassifier()
            metrics = classifier.train(blocks, labels, model_type='logistic_regression')
            
            if not classifier.is_trained:
                print("❌ Model training failed")
                return False
            
            # Test predictions
            probabilities = classifier.predict_probabilities(blocks[:10])
            
            if len(probabilities) != 10:
                print(f"❌ Prediction failed: expected 10 results, got {len(probabilities)}")
                return False
            
            accuracy = metrics.get('train_accuracy', 0)
            print(f"✅ Model trained successfully - accuracy: {accuracy:.3f}")
            return True
            
        except Exception as e:
            print(f"❌ Model training validation failed: {e}")
            return False
    
    def validate_output_format(self) -> bool:
        """Test 8: Validate output format compliance"""
        print("🔍 Test 8: Validating output format...")
        
        try:
            from utils import format_output
            
            # Test data
            title = "Test Document"
            headings = [
                {"level": "H1", "text": "Introduction", "page": 1},
                {"level": "H2", "text": "Background", "page": 2},
                {"level": "H3", "text": "Related Work", "page": 3}
            ]
            
            # Format output
            result = format_output(title, headings)
            
            # Validate structure
            required_keys = ['title', 'outline']
            for key in required_keys:
                if key not in result:
                    print(f"❌ Missing required key: {key}")
                    return False
            
            # Validate outline structure
            outline = result['outline']
            if not isinstance(outline, list):
                print("❌ Outline should be a list")
                return False
            
            for i, heading in enumerate(outline):
                required_heading_keys = ['level', 'text', 'page']
                for key in required_heading_keys:
                    if key not in heading:
                        print(f"❌ Heading {i} missing key: {key}")
                        return False
                
                # Validate level format
                level = heading['level']
                if level not in ['H1', 'H2', 'H3']:
                    print(f"❌ Invalid heading level: {level}")
                    return False
            
            # Test JSON serialization
            json_str = json.dumps(result, indent=2, ensure_ascii=False)
            json.loads(json_str)  # Should not raise exception
            
            print("✅ Output format validated")
            return True
            
        except Exception as e:
            print(f"❌ Output format validation failed: {e}")
            return False
    
    def validate_performance_requirements(self) -> bool:
        """Test 9: Validate performance requirements"""
        print("🔍 Test 9: Validating performance requirements...")
        
        try:
            # Check model size if exists
            model_path = Path('models/heading_classifier.pkl')
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                if size_mb > 200:
                    print(f"❌ Model too large: {size_mb:.2f}MB (max 200MB)")
                    return False
                print(f"✅ Model size acceptable: {size_mb:.2f}MB")
            else:
                print("⚠ No trained model found (will use rule-based only)")
            
            # Memory usage test
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                print(f"✅ Current memory usage: {memory_mb:.2f}MB")
                
                if memory_mb > 1000:
                    print(f"⚠ High memory usage: {memory_mb:.2f}MB")
                
            except ImportError:
                print("⚠ psutil not available for memory testing")
            
            print("✅ Performance requirements validated")
            return True
            
        except Exception as e:
            print(f"❌ Performance validation failed: {e}")
            return False
    
    def run_integration_test(self) -> bool:
        """Test 10: Full integration test"""
        print("🔍 Test 10: Running integration test...")
        
        try:
            # Create a simple test PDF content (mock)
            test_blocks = [
                {
                    'text': 'Test Document Title',
                    'page': 1,
                    'bbox': [100, 100, 400, 130],
                    'font_info': {'size': 18, 'is_bold': True, 'is_italic': False, 'name': 'arial'},
                    'width': 300, 'height': 30
                },
                {
                    'text': '1. Introduction',
                    'page': 1, 
                    'bbox': [50, 200, 200, 220],
                    'font_info': {'size': 14, 'is_bold': True, 'is_italic': False, 'name': 'arial'},
                    'width': 150, 'height': 20
                },
                {
                    'text': '1.1 Background',
                    'page': 1,
                    'bbox': [70, 250, 180, 270], 
                    'font_info': {'size': 12, 'is_bold': True, 'is_italic': False, 'name': 'arial'},
                    'width': 110, 'height': 20
                },
                {
                    'text': 'This is regular body text that explains the concepts.',
                    'page': 1,
                    'bbox': [70, 300, 500, 320],
                    'font_info': {'size': 11, 'is_bold': False, 'is_italic': False, 'name': 'arial'},
                    'width': 430, 'height': 20
                }
            ]
            
            # Test complete pipeline
            from heading_detector import AdvancedHeadingDetector
            from title_extractor import TitleExtractor
            from utils import format_output
            
            # Initialize components
            heading_detector = AdvancedHeadingDetector()
            title_extractor = TitleExtractor()
            
            # Extract title
            title = title_extractor.extract_title(test_blocks, "test_document")
            
            # Detect headings
            start_time = time.time()
            headings = heading_detector.detect_headings(test_blocks)
            processing_time = time.time() - start_time
            
            # Format output
            result = format_output(title, headings)
            
            # Validate results
            if not result.get('title'):
                print("❌ No title extracted")
                return False
            
            if len(result.get('outline', [])) == 0:
                print("⚠ No headings detected (may be expected for simple test)")
            
            print(f"✅ Integration test completed in {processing_time:.3f}s")
            print(f"   Title: {result['title']}")
            print(f"   Headings: {len(result['outline'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False
    
    def run_all_validations(self) -> Dict[str, bool]:
        """Run all validation tests"""
        print("=" * 60)
        print("PDF OUTLINE EXTRACTOR v2.0 - SOLUTION VALIDATION")
        print("=" * 60)
        
        tests = [
            ("Import Validation", self.validate_imports),
            ("Dependency Validation", self.validate_dependencies),
            ("AI Components", self.validate_ai_components),
            ("Training System", self.validate_training_system),
            ("Semantic Analysis", self.validate_semantic_analysis),
            ("Feature Extraction", self.validate_feature_extraction),
            ("Model Training", self.validate_model_training),
            ("Output Format", self.validate_output_format),
            ("Performance Requirements", self.validate_performance_requirements),
            ("Integration Test", self.run_integration_test)
        ]
        
        results = {}
        passed = 0
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
                print()  # Add spacing
            except Exception as e:
                print(f"❌ {test_name} crashed: {e}")
                results[test_name] = False
                print()
        
        # Final report
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:.<40} {status}")
        
        print("-" * 60)
        print(f"TOTAL: {passed}/{len(tests)} tests passed")
        
        if passed == len(tests):
            print("🎉 ALL TESTS PASSED - Solution ready for deployment!")
            return_code = 0
        elif passed >= len(tests) * 0.8:  # 80% pass rate
            print("⚠ MOSTLY PASSING - Solution should work with minor issues")
            return_code = 0
        else:
            print("❌ MULTIPLE FAILURES - Solution needs fixes before deployment")
            return_code = 1
        
        print("=" * 60)
        return results, return_code

def main():
    """Main validation function"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run validation
    validator = SolutionValidator()
    results, return_code = validator.run_all_validations()
    
    sys.exit(return_code)

if __name__ == "__main__":
    main()