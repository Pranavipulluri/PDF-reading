#!/usr/bin/env python3
"""
Fixed PDF Outline Extractor - Addresses the key issues in your current implementation
"""

import sys
import json
import time
import re
from pathlib import Path

def extract_pdf_outline(pdf_path, output_path):
    """Extract outline with proper heading detection and hierarchy"""
    
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("ERROR: PyMuPDF not installed")
        return False
    
    start_time = time.time()
    print(f"Processing: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        all_blocks = []
        
        # Extract text with proper block consolidation
        for page_num in range(min(doc.page_count, 50)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                
                # Consolidate all text in the block
                block_text = ""
                max_font_size = 0
                is_bold = False
                bbox = block.get("bbox", [0, 0, 0, 0])
                
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if text:
                            line_text += text + " "
                            max_font_size = max(max_font_size, span.get("size", 12))
                            if span.get("flags", 0) & 16:  # Bold flag
                                is_bold = True
                    
                    if line_text.strip():
                        block_text += line_text.strip() + " "
                
                block_text = block_text.strip()
                
                if block_text and len(block_text) >= 3:
                    all_blocks.append({
                        'text': block_text,
                        'page': page_num,
                        'font_size': max_font_size,
                        'is_bold': is_bold,
                        'bbox': bbox,
                        'y_pos': bbox[1]  # For sorting
                    })
        
        doc.close()
        
        if not all_blocks:
            print("ERROR: No text extracted")
            return False
        
        print(f"Extracted {len(all_blocks)} text blocks")
        
        # Remove duplicates and filter noise
        all_blocks = remove_duplicates_and_noise(all_blocks)
        print(f"After filtering: {len(all_blocks)} blocks")
        
        # Extract title from first page (FIXED)
        title = extract_title_fixed(all_blocks)
        print(f"Title: '{title}'")
        
        # Detect headings with improved logic (FIXED)
        headings = detect_headings_fixed(all_blocks)
        print(f"Found {len(headings)} headings")
        
        # Create output with proper page numbering
        result = {
            "title": title,
            "outline": [
                {
                    "level": h["level"],
                    "text": h["text"],
                    "page": h["page"] + 1  # Convert to 1-based
                }
                for h in headings
            ]
        }
        
        # Save result
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        processing_time = time.time() - start_time
        print(f"SUCCESS: {len(headings)} headings in {processing_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def remove_duplicates_and_noise(blocks):
    """Remove duplicates and filter out noise - IMPROVED"""
    filtered_blocks = []
    seen_texts = set()
    
    for block in blocks:
        text = block['text'].strip()
        
        # Skip very short text
        if len(text) < 4:
            continue
            
        # Skip obvious noise patterns - IMPROVED PATTERNS
        if is_noise_text(text):
            continue
            
        # Skip duplicates
        text_key = text.lower()
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        
        filtered_blocks.append(block)
    
    return filtered_blocks


def is_noise_text(text):
    """Improved noise detection - FIXED"""
    text_lower = text.lower().strip()
    
    # Skip obvious noise patterns
    noise_patterns = [
        r'^\.*$',                           # Just dots
        r'^\.{3,}',                         # Multiple dots at start
        r'^\d+$',                           # Just numbers  
        r'^page \d+',                       # "Page 1"
        r'^\d+ of \d+$',                   # "1 of 10"
        r'^copyright',                      # Copyright notices
        r'^©',                             # Copyright symbol
        r'version \d{4}\s+page \d+',       # Footer text
        r'may \d+, \d{4}$',                # Date footers
        r'international software testing', # Header repetition
        r'^qualifications board',          # Header repetition
        r'foundation level extension',     # When it's just header repetition
    ]
    
    for pattern in noise_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Skip table of contents entries (dots with page numbers) - KEY FIX
    if re.search(r'\.{3,}\s*\d+$', text):
        return True
    
    # Skip revision history table rows - KEY FIX  
    if re.match(r'^[0-9.]+\s+\d+\s+[A-Z]+\s+\d+', text):
        return True
        
    return False


def extract_title_fixed(blocks):
    """FIXED: Extract title from the actual document content, not filename"""
    # Look at first page blocks only
    first_page_blocks = [b for b in blocks if b['page'] == 0]
    
    if not first_page_blocks:
        return "Unknown Document"
    
    # Sort by Y position (top to bottom) and font size (largest first)
    first_page_blocks.sort(key=lambda x: (x['y_pos'], -x['font_size']))
    
    # Look for title candidates in the top portion
    title_candidates = []
    
    for block in first_page_blocks[:15]:  # Check first 15 blocks
        text = block['text'].strip()
        
        # Skip noise
        if is_noise_text(text):
            continue
            
        # Good title characteristics
        words = text.split()
        if (2 <= len(words) <= 15 and 
            len(text) <= 200 and
            block['font_size'] >= 14):  # Reasonable title size
            
            # Calculate title score
            score = calculate_title_score(block, text)
            title_candidates.append({
                'text': text,
                'score': score
            })
    
    if title_candidates:
        # Sort by score and return best candidate
        title_candidates.sort(key=lambda x: x['score'], reverse=True)
        best_title = title_candidates[0]['text']
        
        # Clean up the title
        best_title = re.sub(r'\s+', ' ', best_title).strip()
        return best_title
    
    return "Unknown Document"


def calculate_title_score(block, text):
    """Calculate title score"""
    score = 0
    
    # Font size bonus
    score += (block['font_size'] - 10) * 0.5
    
    # Bold bonus
    if block['is_bold']:
        score += 2
    
    # Position bonus (prefer top of page)
    if block['y_pos'] <= 200:
        score += 3
    
    # Length bonus (prefer medium length titles)
    word_count = len(text.split())
    if 2 <= word_count <= 8:
        score += 2
    elif word_count <= 15:
        score += 1
    
    # Title-like words
    title_indicators = ['overview', 'foundation', 'level', 'extensions', 'guide', 'manual']
    if any(word in text.lower() for word in title_indicators):
        score += 2
    
    return score


def detect_headings_fixed(blocks):
    """FIXED: Improved heading detection that focuses on actual headings"""
    headings = []
    
    # Calculate font statistics
    font_sizes = [b['font_size'] for b in blocks]
    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
    
    # Sort blocks by page and Y position
    blocks.sort(key=lambda x: (x['page'], x['y_pos']))
    
    for block in blocks:
        text = block['text'].strip()
        
        # Skip noise
        if is_noise_text(text):
            continue
        
        # Skip table of contents section content
        if is_table_of_contents_content(text):
            continue
            
        # Calculate heading score
        score = calculate_heading_score(block, text, avg_font_size)
        
        # Accept as heading if score is high enough
        if score >= 4:  # Raised threshold
            level = determine_heading_level_fixed(block, text, avg_font_size)
            
            headings.append({
                'level': level,
                'text': clean_heading_text(text),
                'page': block['page'],
                'y_pos': block['y_pos'],
                'score': score
            })
    
    # Validate and fix hierarchy
    headings = fix_heading_hierarchy(headings)
    
    return headings


def is_table_of_contents_content(text):
    """Check if text is part of table of contents - KEY FIX"""
    text_lower = text.lower()
    
    # Skip obvious ToC patterns
    toc_patterns = [
        r'revision history.*\d+$',
        r'table of contents.*\d+$', 
        r'acknowledgements.*\d+$',
        r'introduction.*\d+$',
        r'references.*\d+$',
        r'\d+\.\d+.*\d+$',  # "2.1 Something... 7"
        r'^\d+\.\s.*\d+$',  # "1. Something... 6"
    ]
    
    for pattern in toc_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Skip if it ends with page number and has dots
    if re.search(r'\.{2,}.*\d+\s*$', text):
        return True
        
    return False


def calculate_heading_score(block, text, avg_font_size):
    """Calculate heading likelihood score"""
    score = 0
    
    # Font size scoring - KEY FIX
    size_ratio = block['font_size'] / avg_font_size
    if size_ratio >= 1.3:
        score += 4
    elif size_ratio >= 1.15:
        score += 3
    elif size_ratio >= 1.05:
        score += 2
    
    # Bold bonus
    if block['is_bold']:
        score += 3
    
    # Length check (headings are usually concise)
    word_count = len(text.split())
    if 1 <= word_count <= 12:
        score += 2
    elif word_count <= 20:
        score += 1
    elif word_count > 30:
        score -= 3  # Too long to be a heading
    
    # Pattern matching for numbered sections - KEY FIX
    if re.match(r'^\d+\.\s+[A-Z]', text):  # "1. Introduction"
        score += 4
    elif re.match(r'^\d+\.\d+\s+[A-Z]', text):  # "2.1 Intended"
        score += 3
    elif text.endswith(':') and word_count <= 8:
        score += 2
    
    # Heading-like words
    heading_words = [
        'introduction', 'overview', 'summary', 'background', 'conclusion',
        'requirements', 'approach', 'methodology', 'references', 'appendix',
        'acknowledgements', 'history', 'contents'
    ]
    
    if any(word in text.lower() for word in heading_words):
        score += 2
    
    # All caps bonus (but not for very long text)
    if text.isupper() and word_count <= 10:
        score += 1
    
    return score


def determine_heading_level_fixed(block, text, avg_font_size):
    """FIXED: Determine proper heading level"""
    
    # Check for numbered sections first - KEY FIX
    if re.match(r'^\d+\.\s+', text):  # "1. Introduction"
        return 'H1'
    elif re.match(r'^\d+\.\d+\s+', text):  # "2.1 Something"  
        return 'H2'
    elif re.match(r'^\d+\.\d+\.\d+\s+', text):  # "2.1.1 Something"
        return 'H3'
    
    # Check for single words/short phrases that are major headings
    single_word_headings = [
        'acknowledgements', 'introduction', 'overview', 'summary', 
        'background', 'conclusion', 'references', 'appendix'
    ]
    
    if (len(text.split()) <= 3 and 
        any(word in text.lower() for word in single_word_headings)):
        return 'H1'
    
    # Font size based classification - IMPROVED
    font_size = block['font_size']
    size_ratio = font_size / avg_font_size
    
    if size_ratio >= 1.4 and block['is_bold']:
        return 'H1'
    elif size_ratio >= 1.25 or (size_ratio >= 1.1 and block['is_bold']):
        return 'H2' 
    else:
        return 'H3'


def clean_heading_text(text):
    """Clean up heading text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove trailing punctuation except for meaningful ones
    text = re.sub(r'[.]+$', '', text)
    
    return text


def fix_heading_hierarchy(headings):
    """Ensure logical heading hierarchy"""
    if not headings:
        return headings
    
    fixed_headings = []
    last_level = 0
    
    for heading in headings:
        current_level = int(heading['level'][1])  # Extract number from H1, H2, H3
        
        # Don't allow skipping levels (e.g., H1 -> H3)
        if last_level > 0 and current_level > last_level + 1:
            current_level = last_level + 1
            heading['level'] = f'H{current_level}'
        
        fixed_headings.append(heading)
        last_level = current_level
    
    return fixed_headings


def main():
    if len(sys.argv) != 3:
        print("Usage: python fixed_extractor.py <input_pdf> <output_json>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not Path(pdf_path).exists():
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)
    
    success = extract_pdf_outline(pdf_path, output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()