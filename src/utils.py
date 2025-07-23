"""
Utility functions for PDF outline extractor
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )

def validate_input(pdf_path: Path) -> bool:
    """
    Validate input PDF file
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        True if valid, False otherwise
    """
    if not pdf_path.exists():
        return False
    
    if not pdf_path.is_file():
        return False
    
    if pdf_path.suffix.lower() != '.pdf':
        return False
    
    # Check file size (reasonable limit: 100MB)
    if pdf_path.stat().st_size > 100 * 1024 * 1024:
        return False
    
    return True

def format_output(title: str, headings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format the final output JSON structure
    
    Args:
        title: Document title
        headings: List of detected headings
        
    Returns:
        Formatted output dictionary
    """
    return {
        "title": title,
        "outline": headings
    }

def validate_output(output_data: Dict[str, Any]) -> bool:
    """
    Validate output JSON structure
    
    Args:
        output_data: Output dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check required fields
    if "title" not in output_data or "outline" not in output_data:
        return False
    
    # Validate title
    if not isinstance(output_data["title"], str):
        return False
    
    # Validate outline
    outline = output_data["outline"]
    if not isinstance(outline, list):
        return False
    
    # Validate each heading
    for heading in outline:
        if not isinstance(heading, dict):
            return False
        
        required_fields = ["level", "text", "page"]
        if not all(field in heading for field in required_fields):
            return False
        
        # Validate field types
        if not isinstance(heading["level"], str):
            return False
        
        if not isinstance(heading["text"], str):
            return False
        
        if not isinstance(heading["page"], int):
            return False
        
        # Validate level format
        if heading["level"] not in ["H1", "H2", "H3"]:
            return False
        
        # Validate page number
        if heading["page"] < 1:
            return False
    
    return True

def clean_text(text: str) -> str:
    """
    Clean text for better processing
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove control characters but keep unicode
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n')
    
    return text

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings (simple implementation)
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Simple word-based similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def is_likely_noise(text: str) -> bool:
    """
    Check if text is likely noise (headers, footers, page numbers, etc.)
    
    Args:
        text: Text string to check
        
    Returns:
        True if likely noise, False otherwise
    """
    text = text.strip()
    
    if not text:
        return True
    
    # Very short text
    if len(text) < 3:
        return True
    
    # Page numbers
    if text.isdigit() and len(text) <= 3:
        return True
    
    # Common header/footer patterns
    noise_patterns = [
        r'^\d+$',                    # Pure numbers
        r'^page \d+',                # "Page 1"
        r'^\d+ of \d+$',            # "1 of 10"
        r'^copyright',               # Copyright notices
        r'^©',                       # Copyright symbol
        r'^all rights reserved',     # Rights notices
        r'^\w+\.(com|org|net)',     # Website URLs
        r'^printed on',              # Print information
        r'^confidential',            # Confidentiality notices
    ]
    
    import re
    text_lower = text.lower()
    
    for pattern in noise_patterns:
        if re.match(pattern, text_lower):
            return True
    
    return False

def merge_consecutive_headings(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive headings that might be split across lines
    
    Args:
        headings: List of heading dictionaries
        
    Returns:
        List of merged headings
    """
    if not headings:
        return headings
    
    merged = []
    current_heading = None
    
    for heading in headings:
        if current_heading is None:
            current_heading = heading.copy()
        elif (heading["page"] == current_heading["page"] and
              heading["level"] == current_heading["level"] and
              len(current_heading["text"]) < 50):  # Only merge short headings
            
            # Merge if they seem to be continuation
            current_heading["text"] += " " + heading["text"]
        else:
            merged.append(current_heading)
            current_heading = heading.copy()
    
    if current_heading is not None:
        merged.append(current_heading)
    
    return merged

def deduplicate_headings(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate or very similar headings
    
    Args:
        headings: List of heading dictionaries
        
    Returns:
        List of deduplicated headings
    """
    if not headings:
        return headings
    
    deduplicated = []
    seen_texts = set()
    
    for heading in headings:
        text = heading["text"].strip().lower()
        
        # Skip exact duplicates
        if text in seen_texts:
            continue
        
        # Check for very similar headings
        is_similar = False
        for seen_text in seen_texts:
            similarity = calculate_text_similarity(text, seen_text)
            if similarity > 0.8:  # 80% similarity threshold
                is_similar = True
                break
        
        if not is_similar:
            deduplicated.append(heading)
            seen_texts.add(text)
    
    return deduplicated

def save_debug_info(debug_data: Dict[str, Any], output_path: Path):
    """
    Save debug information for troubleshooting
    
    Args:
        debug_data: Debug information dictionary
        output_path: Path to save debug information
    """
    debug_path = output_path.parent / (output_path.stem + "_debug.json")
    
    try:
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save debug info: {e}")

def estimate_processing_time(page_count: int) -> float:
    """
    Estimate processing time based on page count
    
    Args:
        page_count: Number of pages in PDF
        
    Returns:
        Estimated processing time in seconds
    """
    # Base processing time per page (empirically determined)
    base_time_per_page = 0.15  # seconds
    
    # Additional overhead
    base_overhead = 1.0  # seconds
    
    return base_overhead + (page_count * base_time_per_page)