"""
PDF Outline Extractor v2.0 - Advanced AI Edition

A state-of-the-art PDF outline extraction system that combines multiple AI techniques:
- Multi-library PDF processing (PyMuPDF + PDFPlumber)
- Machine learning classification with 60+ features
- Semantic NLP analysis with multilingual support
- Advanced layout analysis and pattern recognition
- Ensemble decision making for maximum accuracy

Author: AI-Powered PDF Processing Team
Version: 2.0.0
License: MIT
"""

__version__ = "2.0.0"
__author__ = "AI-Powered PDF Processing Team"
__description__ = "Advanced AI-powered PDF outline extractor"

# Core components
from .pdf_processor import AdvancedPDFProcessor
from .heading_detector import AdvancedHeadingDetector
from .title_extractor import TitleExtractor
from .utils import setup_logging, validate_input, format_output

# Advanced AI components
from .ml_classifier import HeadingMLClassifier
from .text_analyzer import SemanticTextAnalyzer
from .rule_engine import EnhancedRuleEngine
from .layout_analyzer import AdvancedLayoutAnalyzer
from .feature_extractor import ComprehensiveFeatureExtractor

# Main processing function
def extract_pdf_outline(pdf_path: str, output_path: str = None) -> dict:
    """
    Extract structured outline from PDF using advanced AI techniques
    
    Args:
        pdf_path: Path to input PDF file
        output_path: Optional path for JSON output
        
    Returns:
        Dictionary with extracted title and outline
    """
    import json
    from pathlib import Path
    
    # Initialize processors
    pdf_processor = AdvancedPDFProcessor()
    title_extractor = TitleExtractor()
    heading_detector = AdvancedHeadingDetector()
    
    # Extract text with metadata
    document_data = pdf_processor.extract_text_with_metadata(pdf_path)
    
    # Extract title
    title = title_extractor.extract_title(document_data, Path(pdf_path).stem)
    
    # Detect headings using AI ensemble
    headings = heading_detector.detect_headings(document_data, pdf_path)
    
    # Format output
    result = format_output(title, headings)
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result

# Version info
def get_version_info():
    """Get detailed version information"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "ai_techniques": [
            "Multi-library PDF processing",
            "Machine learning classification", 
            "Semantic NLP analysis",
            "Advanced layout analysis",
            "Ensemble decision making"
        ],
        "supported_languages": ["English", "Japanese", "Chinese", "French", "German", "Spanish"],
        "features": [
            "60+ ML features",
            "Pattern recognition",
            "Font analysis", 
            "Position analysis",
            "Multilingual support",
            "Hierarchy validation"
        ]
    }

__all__ = [
    'extract_pdf_outline',
    'get_version_info',
    'AdvancedPDFProcessor',
    'AdvancedHeadingDetector', 
    'TitleExtractor',
    'HeadingMLClassifier',
    'SemanticTextAnalyzer',
    'EnhancedRuleEngine',
    'AdvancedLayoutAnalyzer',
    'ComprehensiveFeatureExtractor',
    'setup_logging',
    'validate_input',
    'format_output'
]