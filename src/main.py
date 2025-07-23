#!/usr/bin/env python3
"""
PDF Outline Extractor - Advanced Main Entry Point
Uses multiple AI techniques: rule-based detection, ML classification, 
layout analysis, and semantic understanding for maximum accuracy
"""

import sys
import json
import time
import logging
from pathlib import Path

# Import our advanced modules
from pdf_processor import AdvancedPDFProcessor
from heading_detector import AdvancedHeadingDetector
from title_extractor import TitleExtractor
from utils import setup_logging, validate_input, format_output

def main():
    """Main execution function with advanced AI-powered detection"""
    start_time = time.time()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Validate command line arguments
    if len(sys.argv) != 3:
        logger.error("Usage: python main.py <input_pdf> <output_json>")
        sys.exit(1)
    
    input_pdf = Path(sys.argv[1])
    output_json = Path(sys.argv[2])
    
    try:
        # Validate input file
        if not validate_input(input_pdf):
            logger.error(f"Invalid input file: {input_pdf}")
            sys.exit(1)
        
        logger.info(f"Processing PDF with advanced AI techniques: {input_pdf}")
        logger.info("Initializing advanced components...")
        
        # Initialize advanced processors
        pdf_processor = AdvancedPDFProcessor()
        title_extractor = TitleExtractor()
        heading_detector = AdvancedHeadingDetector()
        
        # Display processing strategy
        logger.info("Processing strategy:")
        logger.info("  1. Multi-library PDF extraction (PyMuPDF + PDFPlumber)")
        logger.info("  2. Advanced layout analysis")
        logger.info("  3. Semantic text understanding (NLP)")
        logger.info("  4. Rule-based pattern detection")
        logger.info("  5. ML classification (if model available)")
        logger.info("  6. Ensemble decision making")
        
        # Step 1: Advanced text extraction with multiple libraries
        logger.info("Step 1/6: Extracting text with comprehensive metadata...")
        document_data = pdf_processor.extract_text_with_metadata(str(input_pdf))
        
        if not document_data:
            logger.error("Failed to extract text from PDF")
            sys.exit(1)
        
        logger.info(f"Extracted {len(document_data)} text blocks with enhanced metadata")
        
        # Step 2: Extract document title using advanced methods
        logger.info("Step 2/6: Extracting document title...")
        title = title_extractor.extract_title(document_data, input_pdf.stem)
        logger.info(f"Document title: '{title}'")
        
        # Step 3: Advanced heading detection using ensemble approach
        logger.info("Step 3/6: Detecting headings with AI ensemble...")
        
        # Pass PDF path for additional layout analysis
        headings = heading_detector.detect_headings(document_data, str(input_pdf))
        
        logger.info(f"AI ensemble detected {len(headings)} potential headings")
        
        # Step 4: Display detection method contributions
        performance_stats = heading_detector.get_performance_stats()
        logger.info("Detection method contributions:")
        for method, count in performance_stats.get('method_contributions', {}).items():
            if count > 0:
                logger.info(f"  {method}: {count} detections")
        
        # Step 5: Format and validate output
        logger.info("Step 4/6: Formatting and validating output...")
        output_data = format_output(title, headings)
        
        # Step 6: Save results with metadata
        logger.info("Step 5/6: Saving results...")
        output_json.parent.mkdir(parents=True, exist_ok=True)
        
        # Add processing metadata to output
        processing_metadata = {
            'processing_time_seconds': time.time() - start_time,
            'total_text_blocks_analyzed': len(document_data),
            'detection_methods_used': [
                'rule_based_analysis',
                'semantic_text_analysis', 
                'layout_analysis',
                'pattern_matching'
            ],
            'extraction_libraries': ['PyMuPDF', 'PDFPlumber'],
            'ai_techniques': [
                'font_analysis',
                'position_analysis', 
                'pattern_recognition',
                'semantic_nlp',
                'ensemble_decision_making'
            ]
        }
        
        # Add ML classifier info if available
        if hasattr(heading_detector, 'ml_classifier') and heading_detector.ml_classifier.is_trained:
            processing_metadata['detection_methods_used'].append('ml_classification')
            processing_metadata['ml_model_available'] = True
        else:
            processing_metadata['ml_model_available'] = False
        
        # Enhanced output format with metadata
        enhanced_output = {
            'title': output_data['title'],
            'outline': output_data['outline'],
            '_metadata': processing_metadata
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(enhanced_output, f, indent=2, ensure_ascii=False)
        
        # Performance metrics
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info("Step 6/6: Processing complete!")
        logger.info("="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Input PDF: {input_pdf.name}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Document title: {title}")
        logger.info(f"Total headings detected: {len(headings)}")
        
        # Heading level breakdown
        level_counts = {}
        for heading in headings:
            level = heading.get('level', 'Unknown')
            level_counts[level] = level_counts.get(level, 0) + 1
        
        for level in ['H1', 'H2', 'H3']:
            count = level_counts.get(level, 0)
            logger.info(f"  {level} headings: {count}")
        
        logger.info(f"Output saved to: {output_json}")
        logger.info(f"Text blocks analyzed: {len(document_data)}")
        
        # Performance assessment
        if processing_time <= 10:
            logger.info("✓ Performance: Excellent (under 10s requirement)")
        elif processing_time <= 15:
            logger.info("⚠ Performance: Good (slightly over 10s target)")
        else:
            logger.info("⚠ Performance: Needs optimization (over 15s)")
        
        # Quality indicators
        if len(headings) > 0:
            avg_confidence = sum(h.get('_debug', {}).get('confidence', 0.5) for h in headings if '_debug' in h) / len(headings)
            logger.info(f"Average detection confidence: {avg_confidence:.3f}")
        
        logger.info("="*60)
        
        # Print summary to stdout for Docker
        print(f"SUCCESS: Processed {input_pdf.name} in {processing_time:.2f}s")
        print(f"Title: {title}")
        print(f"Headings: {len(headings)} ({', '.join([f'{level_counts.get(level, 0)} {level}' for level in ['H1', 'H2', 'H3']])})")
        
        if processing_time > 10:
            print(f"WARNING: Processing time ({processing_time:.2f}s) exceeded 10s target")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"ERROR: File not found - {e}")
        sys.exit(1)
        
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        print(f"ERROR: Permission denied - {e}")
        sys.exit(1)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON encoding error: {e}")
        print(f"ERROR: JSON encoding failed - {e}")
        sys.exit(1)
        
    except MemoryError as e:
        logger.error(f"Memory error - PDF too large: {e}")
        print(f"ERROR: PDF too large for available memory")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"ERROR: {e}")
        
        # Try to provide helpful error information
        if "fitz" in str(e).lower():
            print("HINT: PDF parsing error - file may be corrupted or password-protected")
        elif "memory" in str(e).lower():
            print("HINT: Out of memory - try with a smaller PDF")
        elif "permission" in str(e).lower():
            print("HINT: File permissions issue - check file access rights")
        
        sys.exit(1)


def print_system_info():
    """Print system information for debugging"""
    import platform
    import psutil
    
    print("System Information:")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print("Installed packages:")
    
    try:
        import fitz
        print(f"  PyMuPDF: {fitz.__version__}")
    except:
        print("  PyMuPDF: Not available")
    
    try:
        import pdfplumber
        print(f"  PDFPlumber: {pdfplumber.__version__}")
    except:
        print("  PDFPlumber: Not available")
    
    try:
        import sklearn
        print(f"  scikit-learn: {sklearn.__version__}")
    except:
        print("  scikit-learn: Not available")


if __name__ == "__main__":
    # Check for special flags
    if len(sys.argv) == 2 and sys.argv[1] in ['--version', '-v']:
        print("PDF Outline Extractor v2.0 - Advanced AI Edition")
        print("Features: Multi-library extraction, ML classification, semantic analysis")
        sys.exit(0)
    
    if len(sys.argv) == 2 and sys.argv[1] in ['--system-info', '--info']:
        print_system_info()
        sys.exit(0)
    
    # Run main processing
    main()