#!/usr/bin/env python3
"""
Usage Examples - Demonstrates different ways to use the PDF Outline Extractor
Shows basic usage, advanced configuration, and API-style integration
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def example_basic_usage():
    """Example 1: Basic PDF outline extraction"""
    print("🔥 Example 1: Basic Usage")
    print("-" * 40)
    
    # Simple one-line extraction
    from src import extract_pdf_outline
    
    # This would work if you have a real PDF
    # result = extract_pdf_outline('sample.pdf', 'output.json')
    
    # For demo, we'll simulate with synthetic data
    print("# Basic usage (if you had a real PDF):")
    print("from src import extract_pdf_outline")
    print("result = extract_pdf_outline('document.pdf', 'outline.json')")
    print("print(f'Found {len(result[\"outline\"])} headings')")
    print()
    
def example_advanced_configuration():
    """Example 2: Advanced configuration and customization"""
    print("🚀 Example 2: Advanced Configuration")
    print("-" * 40)
    
    from heading_detector import AdvancedHeadingDetector
    from pdf_processor import AdvancedPDFProcessor
    from title_extractor import TitleExtractor
    
    # Initialize with custom configuration
    heading_detector = AdvancedHeadingDetector()
    
    # Customize detection strategy weights
    heading_detector.strategy_weights = {
        'rule_based': 0.4,        # Increase rule-based weight
        'ml_classification': 0.2, # Decrease ML weight
        'semantic_analysis': 0.25, # Increase semantic weight
        'layout_analysis': 0.1,   # Decrease layout weight
        'consensus': 0.05         # Keep consensus bonus
    }
    
    print("# Custom strategy weights:")
    for method, weight in heading_detector.strategy_weights.items():
        print(f"#   {method}: {weight}")
    
    print("\n# Process with custom configuration:")
    print("pdf_processor = AdvancedPDFProcessor()")
    print("document_data = pdf_processor.extract_text_with_metadata('document.pdf')")
    print("headings = heading_detector.detect_headings(document_data)")
    print("print(f'Detected {len(headings)} headings with custom weights')")
    print()

def example_ml_training():
    """Example 3: Train custom ML model"""
    print("🤖 Example 3: Train Custom ML Model")
    print("-" * 40)
    
    from training.train_classifier import ClassifierTrainer
    from training.generate_training_data import TrainingDataGenerator
    
    print("# Generate training data:")
    print("generator = TrainingDataGenerator()")
    print("blocks, labels = generator.generate_synthetic_training_data(1000)")
    print()
    
    print("# Add multilingual data:")
    print("multi_blocks, multi_labels = generator.generate_multilingual_samples(200)")
    print("blocks.extend(multi_blocks)")
    print("labels.extend(multi_labels)")
    print()
    
    print("# Train custom model:")
    print("trainer = ClassifierTrainer()")
    print("metrics = trainer.classifier.train(blocks, labels, model_type='gradient_boosting')")
    print("trainer.classifier.save_model('custom_model.pkl')")
    print("print(f'Model trained with {metrics[\"cv_mean\"]:.3f} CV accuracy')")
    print()

def example_batch_processing():
    """Example 4: Batch processing multiple PDFs"""
    print("📁 Example 4: Batch Processing")
    print("-" * 40)
    
    print("# Process all PDFs in a directory:")
    print("""
import glob
from pathlib import Path
from src import extract_pdf_outline

input_dir = Path('input_pdfs')
output_dir = Path('output_json')
output_dir.mkdir(exist_ok=True)

results_summary = []

for pdf_file in glob.glob(str(input_dir / '*.pdf')):
    pdf_path = Path(pdf_file)
    output_path = output_dir / f'{pdf_path.stem}.json'
    
    try:
        result = extract_pdf_outline(str(pdf_path), str(output_path))
        
        summary = {
            'file': pdf_path.name,
            'title': result['title'],
            'heading_count': len(result['outline']),
            'status': 'success'
        }
        results_summary.append(summary)
        
        print(f'✅ {pdf_path.name}: {len(result["outline"])} headings')
        
    except Exception as e:
        summary = {
            'file': pdf_path.name,
            'error': str(e),
            'status': 'failed'
        }
        results_summary.append(summary)
        
        print(f'❌ {pdf_path.name}: {e}')

# Save batch processing summary
with open(output_dir / 'batch_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
    
print(f'Processed {len(results_summary)} files')
""")

def example_api_integration():
    """Example 5: API-style integration"""
    print("🌐 Example 5: API-Style Integration")
    print("-" * 40)
    
    print("# Create a simple API wrapper class:")
    print("""
class PDFOutlineAPI:
    def __init__(self):
        from pdf_processor import AdvancedPDFProcessor
        from heading_detector import AdvancedHeadingDetector
        from title_extractor import TitleExtractor
        
        self.pdf_processor = AdvancedPDFProcessor()
        self.heading_detector = AdvancedHeadingDetector()
        self.title_extractor = TitleExtractor()
    
    def extract_outline(self, pdf_path: str, options: dict = None) -> dict:
        \"\"\"Extract outline with optional configuration\"\"\"
        options = options or {}
        
        # Configure based on options
        if options.get('multilingual_focus'):
            # Increase semantic analysis weight for multilingual
            self.heading_detector.strategy_weights['semantic_analysis'] = 0.3
        
        if options.get('technical_document'):
            # Increase pattern matching for technical docs
            self.heading_detector.strategy_weights['rule_based'] = 0.4
        
        # Extract
        document_data = self.pdf_processor.extract_text_with_metadata(pdf_path)
        title = self.title_extractor.extract_title(document_data, Path(pdf_path).stem)
        headings = self.heading_detector.detect_headings(document_data, pdf_path)
        
        return {
            'title': title,
            'outline': headings,
            'metadata': {
                'total_blocks': len(document_data),
                'processing_options': options
            }
        }

# Usage:
api = PDFOutlineAPI()

# Standard extraction
result = api.extract_outline('document.pdf')

# Technical document with custom options
tech_result = api.extract_outline('technical_manual.pdf', {
    'technical_document': True,
    'multilingual_focus': False
})

# Multilingual document
multi_result = api.extract_outline('multilingual_doc.pdf', {
    'multilingual_focus': True
})
""")

def example_performance_monitoring():
    """Example 6: Performance monitoring and debugging"""
    print("📊 Example 6: Performance Monitoring")
    print("-" * 40)
    
    print("# Monitor processing performance:")
    print("""
import time
import psutil
import os
from src import extract_pdf_outline

def extract_with_monitoring(pdf_path: str, output_path: str = None):
    \"\"\"Extract PDF outline with performance monitoring\"\"\"
    
    # Monitor system resources
    process = psutil.Process(os.getpid())
    
    # Baseline measurements
    start_time = time.time()
    start_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    try:
        # Extract outline
        result = extract_pdf_outline(pdf_path, output_path)
        
        # Final measurements
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Calculate metrics
        processing_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Performance report
        performance_report = {
            'processing_time_seconds': processing_time,
            'memory_used_mb': memory_used,
            'peak_memory_mb': end_memory,
            'meets_10s_requirement': processing_time <= 10.0,
            'memory_efficient': memory_used < 500,  # < 500MB
            'headings_detected': len(result['outline']),
            'headings_per_second': len(result['outline']) / processing_time
        }
        
        # Add performance data to result
        result['_performance'] = performance_report
        
        # Print summary
        print(f"📈 Performance Summary:")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Memory used: {memory_used:.2f}MB")
        print(f"   Headings detected: {len(result['outline'])}")
        print(f"   Meets requirements: {'✅' if performance_report['meets_10s_requirement'] else '❌'}")
        
        return result
        
    except Exception as e:
        # Error handling with performance context
        error_time = time.time() - start_time
        
        print(f"❌ Processing failed after {error_time:.2f}s: {e}")
        raise

# Usage:
result = extract_with_monitoring('large_document.pdf', 'output.json')
""")

def example_debugging():
    """Example 7: Debugging and troubleshooting"""
    print("🔍 Example 7: Debugging and Troubleshooting")
    print("-" * 40)
    
    print("# Enable debug mode for detailed logging:")
    print("""
import logging
from heading_detector import AdvancedHeadingDetector

# Setup detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# Create detector with debug info
detector = AdvancedHeadingDetector()

# Process with detailed feedback
document_data = []  # Your PDF data here
headings = detector.detect_headings(document_data)

# Get performance stats
stats = detector.get_performance_stats()
print("Detection method contributions:")
for method, count in stats['method_contributions'].items():
    print(f"  {method}: {count} detections")

# Analyze individual heading decisions
for heading in headings:
    if '_debug' in heading:
        debug_info = heading['_debug']
        print(f"Heading: {heading['text']}")
        print(f"  Confidence: {debug_info['confidence']:.3f}")
        print(f"  Method scores: {debug_info['method_scores']}")
        print()
""")

def example_custom_rules():
    """Example 8: Custom rule configuration"""
    print("⚙️ Example 8: Custom Rules and Patterns")
    print("-" * 40)
    
    print("# Customize detection rules for specific document types:")
    print("""
from rule_engine import EnhancedRuleEngine

# Create custom rule engine
rule_engine = EnhancedRuleEngine()

# Customize font thresholds
rule_engine.font_thresholds = {
    'h1_min_ratio': 1.6,   # Stricter H1 requirements
    'h2_min_ratio': 1.3,   # Stricter H2 requirements  
    'h3_min_ratio': 1.15,  # Slightly stricter H3
}

# Add custom patterns for your document type
custom_patterns = [
    (r'^exercise\\s+\\d+', 'H2', 0.9),        # Exercise 1, Exercise 2
    (r'^problem\\s+\\d+', 'H3', 0.85),        # Problem 1, Problem 2
    (r'^solution[:\\s]', 'H3', 0.8),          # Solution: 
    (r'^note[:\\s]', 'H3', 0.7),              # Note:
]

# Add to pattern rules
rule_engine.pattern_rules['custom_academic'] = {
    'patterns': custom_patterns,
    'weight': 0.95
}

# Use with heading detector
from heading_detector import AdvancedHeadingDetector

detector = AdvancedHeadingDetector()
detector.rule_engine = rule_engine  # Use custom rules

# Process document with custom rules
headings = detector.detect_headings(document_data)
""")

def run_all_examples():
    """Run all examples"""
    print("=" * 60)
    print("PDF OUTLINE EXTRACTOR v2.0 - USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        example_basic_usage,
        example_advanced_configuration,
        example_ml_training,
        example_batch_processing,
        example_api_integration,
        example_performance_monitoring,
        example_debugging,
        example_custom_rules
    ]
    
    for example_func in examples:
        try:
            example_func()
            print()
        except Exception as e:
            print(f"❌ Example failed: {e}")
            print()
    
    print("=" * 60)
    print("📚 More Information")
    print("=" * 60)
    print("For complete documentation, see:")
    print("  • README.md - Comprehensive guide")
    print("  • DEPLOYMENT_GUIDE.md - Step-by-step setup")
    print("  • config.yaml - Configuration options")
    print("  • validate_solution.py - System validation")
    print("  • benchmark.py - Performance testing")
    print()
    print("🏆 Ready for hackathon submission!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_examples()