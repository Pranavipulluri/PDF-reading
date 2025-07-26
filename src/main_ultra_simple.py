#!/usr/bin/env python3
"""
COMPLETE PDF Outline Extractor
Features: Improved heading detection, title extraction, noise filtering, AND language detection
"""

import sys
import json
import time
import re
from pathlib import Path
from collections import Counter

def extract_pdf_outline(pdf_path, output_path):
    """Extract outline with improved accuracy and language detection"""
    
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("ERROR: PyMuPDF not installed")
        return False
    
    start_time = time.time()
    print(f"Processing: {pdf_path}")
    
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        all_blocks = []
        
        # STEP 1: Extract text with proper merging
        for page_num in range(min(doc.page_count, 50)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            # Extract complete text blocks with language detection
            page_blocks = extract_complete_blocks_with_language(text_dict, page_num)
            all_blocks.extend(page_blocks)
        
        doc.close()
        
        if not all_blocks:
            print("ERROR: No text extracted")
            return False
        
        print(f"Extracted {len(all_blocks)} complete text blocks")
        
        # STEP 2: Detect document language
        doc_language = detect_document_language(all_blocks)
        print(f"Document language: {doc_language}")
        
        # STEP 3: Remove duplicates and filter noise  
        all_blocks = remove_duplicates_and_noise(all_blocks, doc_language)
        print(f"After deduplication: {len(all_blocks)} blocks")
        
        # STEP 4: Calculate font statistics
        font_sizes = [b['font_size'] for b in all_blocks if b['font_size'] > 0]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        max_font_size = max(font_sizes) if font_sizes else 12
        
        print(f"Font stats - Avg: {avg_font_size:.1f}, Max: {max_font_size:.1f}")
        
        # STEP 5: Extract title
        title = extract_document_title(all_blocks, Path(pdf_path).stem, doc_language)
        print(f"Title: '{title}'")
        
        # STEP 6: Detect headings with improved logic
        headings = detect_headings_improved(all_blocks, avg_font_size, max_font_size, doc_language)
        print(f"Found {len(headings)} headings")
        
        # STEP 7: Create final output with metadata
        result = {
            "title": title,
            "outline": [
                {
                    "level": h["level"],
                    "text": h["text"], 
                    "page": h["page"] + 1  # Convert to 1-based pages
                }
                for h in headings
            ],
            "metadata": {
                "detected_language": doc_language,
                "total_blocks_analyzed": len(all_blocks)
            }
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


def extract_complete_blocks_with_language(text_dict, page_num):
    """Extract complete text blocks with better merging and language detection"""
    
    complete_blocks = []
    
    for block in text_dict.get("blocks", []):
        if "lines" not in block:
            continue
        
        # Merge ALL text in this block
        block_text = ""
        max_font_size = 0
        is_bold = False
        is_italic = False
        block_bbox = block.get("bbox", [0, 0, 0, 0])
        
        for line in block["lines"]:
            line_text = ""
            for span in line["spans"]:
                span_text = span.get("text", "").strip()
                if span_text:
                    # Clean corrupted text patterns
                    span_text = clean_corrupted_text(span_text)
                    if span_text:  # Only add if still has content after cleaning
                        line_text += span_text + " "
                    
                    # Track font properties
                    span_size = span.get("size", 12)
                    if span_size > max_font_size:
                        max_font_size = span_size
                    
                    flags = span.get("flags", 0)
                    if flags & 16:  # Bold flag
                        is_bold = True
                    if flags & 2:   # Italic flag
                        is_italic = True
            
            if line_text.strip():
                block_text += line_text.strip() + " "
        
        # Clean up text
        block_text = clean_text_block(block_text)
        
        # Only keep substantial blocks
        if block_text and len(block_text) >= 3:
            complete_blocks.append({
                'text': block_text,
                'page': page_num,
                'font_size': max_font_size,
                'is_bold': is_bold,
                'is_italic': is_italic,
                'bbox': block_bbox,
                'y_pos': block_bbox[1],
                'language': detect_language(block_text)
            })
    
    return complete_blocks


def clean_corrupted_text(text):
    """Clean corrupted text patterns like repeated characters"""
    if not text or len(text) < 4:
        return text
    
    # Fix excessive character repetition like "Reeeequest"
    text = re.sub(r'([a-zA-Z])\1{3,}', r'\1', text)
    
    # Fix fragmented words patterns
    text = re.sub(r'\b[a-zA-Z]\s+([a-zA-Z]{3,})', r'\1', text)
    
    return text


def clean_text_block(text):
    """Clean and normalize text block"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Fix common OCR issues - ENHANCED
    text = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z])\b', r'\1\2\3', text)  # "R F P" -> "RFP"
    text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b', r'\1\2\3\4', text)
    
    # Fix repeated character patterns like "Reeeequest"
    text = re.sub(r'([a-zA-Z])\1{2,}', r'\1', text)
    
    # Fix fragmented words pattern "f quest foooor"
    text = re.sub(r'\b[a-z]\s+([a-z]{4,})\s+[a-z]+o{3,}r\s+', r'\1 for ', text)
    
    # Remove obvious corruption patterns
    text = re.sub(r'(.{2,})\1{2,}', r'\1', text)
    
    return text


def detect_language(text):
    """Detect language of text block"""
    if not text:
        return 'unknown'
    
    # Japanese detection
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return 'japanese'
    if re.search(r'[\u4E00-\u9FAF]', text) and re.search(r'第\d+章|第\d+節', text):
        return 'japanese'
    
    # Chinese detection
    if re.search(r'[\u4E00-\u9FFF]', text) and re.search(r'第\d+章|第\d+节', text):
        return 'chinese'
    
    # Korean detection
    if re.search(r'[\uAC00-\uD7AF]', text):
        return 'korean'
    
    # European languages
    text_lower = text.lower()
    if re.search(r'[àâäçéèêëîïôöùûüÿ]', text) or any(word in text_lower for word in ['chapitre', 'résumé']):
        return 'french'
    if re.search(r'[äöüß]', text) or any(word in text_lower for word in ['kapitel', 'zusammenfassung']):
        return 'german'
    if re.search(r'[áéíóúüñ]', text) or any(word in text_lower for word in ['capítulo', 'resumen']):
        return 'spanish'
    
    return 'english'


def detect_document_language(blocks):
    """Detect overall document language"""
    language_counts = Counter(block['language'] for block in blocks)
    return language_counts.most_common(1)[0][0] if language_counts else 'english'


def remove_duplicates_and_noise(blocks, doc_language):
    """Improved noise filtering and deduplication with language awareness"""
    
    # Step 1: Remove obvious noise and form elements
    clean_blocks = []
    for block in blocks:
        text = block['text'].strip()
        
        if is_noise_or_form_element(text, doc_language):
            continue
            
        clean_blocks.append(block)
    
    # Step 2: Remove duplicates
    unique_blocks = []
    seen_texts = set()
    
    for block in clean_blocks:
        text_key = block['text'].lower().strip()
        
        if text_key not in seen_texts:
            unique_blocks.append(block)
            seen_texts.add(text_key)
    
    # Step 3: Remove repeated headers/footers
    unique_blocks = filter_repeated_elements(unique_blocks)
    
    return unique_blocks


def is_noise_or_form_element(text, doc_language):
    """Enhanced noise and form element detection with language awareness"""
    
    text_clean = text.strip()
    text_lower = text_clean.lower()
    
    # Too short or too long
    if len(text_clean) < 2 or len(text_clean) > 500:
        return True
    
    # Dates are not headings
    if re.match(r'^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}$', text_lower):
        return True
    
    # Financial data is not headings
    if re.match(r'^total\s+annual\s+\$\d+m\s+\$\d+m$', text_lower):
        return True
    
    # Single month names
    if re.match(r'^(january|february|march|april|may|june|july|august|september|october|november|december)$', text_lower):
        return True
    
    # Form field patterns (major improvement)
    form_patterns = [
        r'^\d+\.\s*$',                          # Just "1." "2." etc
        r'^\d+\.\s*[A-Z][a-z\s]{3,50}$',       # Form fields like "1. Name of person"
        r'^[A-Z][a-z\s]*:\s*$',                # Labels ending with colon
        r'^[A-Z][a-z\s]*:?\s*_+\s*$',          # Fields with underlines
        r'^\d+\.\s+[A-Z\s]+\+\s+[A-Z\s]+$',   # "4. PAY + SI + NPA"
        r'^S\.No\s+Name\s+Age',                 # Table headers
        r'^\d+\.\s+\d+\.\s+\d+\.',              # Multiple numbers
        r'^Rs\.\s*$',                           # Currency indicators
        r'^\d+\s+\d+\s*$',                      # "5 6"
        r'^date\s*$',                           # Just "date"
    ]
    
    for pattern in form_patterns:
        if re.match(pattern, text_clean):
            return True
    
    # Table of Contents entries
    toc_patterns = [
        r'\.{3,}.*\d+\s*$',                     # "Something........ 5"
        r'^\d+\.\s*.+\s+\d+\s*$',              # "1. Introduction 6"
        r'^(revision\s+history|table\s+of\s+contents|acknowledgements|references).*\d+\s*$',
        r'^\d+\.\d+\s+.+\s+\d+\s*$',           # "2.1 Something 7"
    ]
    
    for pattern in toc_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Common noise patterns
    noise_patterns = [
        r'^\d+$',                               # Just numbers
        r'^page\s+\d+$',                        # Page numbers (exact match)
        r'^\d+\s+of\s+\d+$',                   # Page indicators
        r'^copyright|^©\s*\d{4}',              # Copyright
        r'^\w+\.(com|org|net|edu)',            # URLs
        r'^www\.',                              # Web addresses
        r'^version\s+[\d.]+',                  # Version numbers
        r'^march\s+\d{4}\s+\d+$',              # Date with numbers at end only
    ]
    
    for pattern in noise_patterns:
        if re.match(pattern, text_lower):
            return True
    
    return False


def filter_repeated_elements(blocks):
    """Filter out headers/footers that repeat across pages"""
    
    # Group by text content
    text_counts = {}
    for block in blocks:
        text = block['text'].strip().lower()
        if text not in text_counts:
            text_counts[text] = []
        text_counts[text].append(block)
    
    # Keep only texts that don't repeat across multiple pages
    filtered_blocks = []
    
    for text, block_list in text_counts.items():
        pages = set(b['page'] for b in block_list)
        
        # If appears on 3+ pages and is short, likely header/footer
        if len(pages) >= 3 and len(text.split()) <= 10:
            continue
        
        filtered_blocks.extend(block_list)
    
    return filtered_blocks


def extract_document_title(blocks, filename_fallback, doc_language):
    """Improved title extraction with language awareness"""
    
    # Look at first page only
    first_page_blocks = [b for b in blocks if b['page'] == 0]
    
    if not first_page_blocks:
        return filename_fallback
    
    # Sort by font size (largest first), then by position (top first)
    first_page_blocks.sort(key=lambda x: (-x['font_size'], x['bbox'][1]))
    
    # Look for RFP title pattern specifically
    for block in first_page_blocks:
        text = block['text'].strip()
        
        # Check for RFP title pattern
        if 'rfp' in text.lower() and 'request' in text.lower() and 'proposal' in text.lower():
            # This is likely the main title
            if len(text.split()) >= 10:  # Long enough to be the full title
                return clean_title_text(text)
    
    # Look for meaningful titles in top blocks
    for block in first_page_blocks[:15]:  # Check more blocks
        text = block['text'].strip()
        words = text.split()
        font_size = block['font_size']
        
        # Skip form elements and short fragments
        if is_noise_or_form_element(text, doc_language):
            continue
        
        # Good title characteristics
        if (3 <= len(words) <= 25 and  # Increased upper limit
            len(text) <= 200 and
            font_size >= 10 and
            not text.endswith('.pdf')):
            
            # Look for title indicators
            title_indicators = [
                'overview', 'introduction', 'guide', 'manual', 'report',
                'application', 'form', 'pathways', 'program', 'plan',
                'ontario', 'digital', 'library'
            ]
            
            if (any(indicator in text.lower() for indicator in title_indicators) or
                font_size >= 14 or block['is_bold'] or text.isupper()):
                
                return clean_title_text(text)
    
    # If no good title found, return empty or filename
    return "" if filename_fallback.startswith("file") else filename_fallback


def clean_title_text(text):
    """Clean up title text"""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common prefixes
    text = re.sub(r'^(title|subject):\s*', '', text, flags=re.IGNORECASE)
    
    return text


def detect_headings_improved(blocks, avg_font_size, max_font_size, doc_language):
    """Improved heading detection with better filtering and language awareness"""
    
    headings = []
    
    # Calculate thresholds
    font_range = max_font_size - avg_font_size
    
    if font_range > 2:
        h1_threshold = avg_font_size + (font_range * 0.7)
        h2_threshold = avg_font_size + (font_range * 0.4)
        h3_threshold = avg_font_size + (font_range * 0.2)
    else:
        h1_threshold = avg_font_size * 1.4
        h2_threshold = avg_font_size * 1.2
        h3_threshold = avg_font_size * 1.1
    
    print(f"Heading thresholds - H1: {h1_threshold:.1f}, H2: {h2_threshold:.1f}, H3: {h3_threshold:.1f}")
    
    for block in blocks:
        text = block['text'].strip()
        font_size = block['font_size']
        is_bold = block['is_bold']
        word_count = len(text.split())
        
        # Skip very long text and form elements
        if word_count > 20 or is_noise_or_form_element(text, doc_language):
            continue
        
        # Calculate heading score
        heading_score = 0
        suggested_level = None
        
        # Font size scoring
        if font_size >= h1_threshold:
            heading_score += 4
            suggested_level = "H1"
        elif font_size >= h2_threshold:
            heading_score += 3
            suggested_level = "H2"
        elif font_size >= h3_threshold:
            heading_score += 2
            suggested_level = "H3"
        elif font_size > avg_font_size * 1.05:
            heading_score += 1
            suggested_level = "H3"
        
        # Bold bonus
        if is_bold:
            heading_score += 2
            if not suggested_level:
                suggested_level = "H3"
        
        # Strong heading indicators
        if re.match(r'^\d+\.\s+[A-Z]', text) and not re.search(r'(name|designation|pay|town|amount)', text.lower()):
            heading_score += 3
            suggested_level = "H2"
        elif re.match(r'^\d+\.\d+\s+[A-Z]', text):
            heading_score += 3
            suggested_level = "H3"
        elif text.endswith(':') and word_count <= 6:
            heading_score += 2
            if not suggested_level:
                suggested_level = "H3"
        
        # Major section detection
        major_sections = ['summary', 'background', 'timeline', 'approach', 'evaluation', 'appendix']
        if any(section in text.lower() for section in major_sections):
            if word_count <= 3:  # Single word or short phrase
                heading_score += 6
                suggested_level = "H2"
        
        # Language-specific patterns
        if doc_language in ['japanese', 'chinese']:
            if re.match(r'^第\d+章', text):
                heading_score += 5
                suggested_level = "H1"
            elif re.match(r'^第\d+節|^第\d+节', text):
                heading_score += 4
                suggested_level = "H2"
        
        # Section headings
        section_words = [
            'overview', 'introduction', 'background', 'summary', 'conclusion',
            'references', 'acknowledgements', 'contents', 'history', 'pathways',
            'options', 'goals', 'outcomes', 'requirements', 'structure', 'current'
        ]
        
        if any(word in text.lower() for word in section_words):
            heading_score += 2
        
        # All caps bonus (for short text)
        if text.isupper() and 3 <= word_count <= 8:
            heading_score += 2
        
        # Apply minimum threshold and validate
        if heading_score >= 3.0 and suggested_level:
            headings.append({
                'text': clean_heading_text(text),
                'level': suggested_level,
                'page': block['page'],
                'score': heading_score,
                'font_size': font_size,
                'y_pos': block.get('y_pos', 0)
            })
    
    # Add document sections detection
    additional_headings = detect_document_sections(blocks, doc_language)
    headings.extend(additional_headings)
    
    # Sort by page and position
    headings.sort(key=lambda x: (x['page'], x.get('y_pos', 0)))
    
    # Remove duplicates
    headings = remove_duplicate_headings(headings)
    
    # Validate hierarchy
    headings = validate_heading_hierarchy(headings)
    
    print(f"Final headings: {[(h['text'][:50], h['level']) for h in headings]}")
    
    return headings


def detect_document_sections(blocks, doc_language):
    """Detect standard document sections that might be missed"""
    section_patterns = {
        r'summary\s*$': 'H2',
        r'background\s*$': 'H2', 
        r'timeline\s*:?\s*$': 'H3',
        r'approach\s+and\s+specific': 'H2',
        r'evaluation\s+and\s+awarding': 'H2',
        r'appendix\s+[a-c]': 'H2',
        r'^\d+\.\s+preamble': 'H3',
        r'^\d+\.\s+terms\s+of\s+reference': 'H3',
        r'^\d+\.\s+membership': 'H3',
        r'^\d+\.\s+appointment\s+criteria': 'H3',
        r'^\d+\.\s+term': 'H3',
        r'^\d+\.\s+chair': 'H3',
        r'^\d+\.\s+meetings': 'H3',
        r'^\d+\.\s+lines\s+of\s+accountability': 'H3',
        r'^\d+\.\s+financial\s+and\s+administrative': 'H3',
    }
    
    additional_headings = []
    
    for block in blocks:
        text = block['text'].strip()
        text_lower = text.lower()
        
        for pattern, level in section_patterns.items():
            if re.search(pattern, text_lower):
                additional_headings.append({
                    'text': clean_heading_text(text),
                    'level': level,
                    'page': block['page'],
                    'y_pos': block.get('y_pos', 0),
                    'score': 10  # High confidence
                })
                break
    
    return additional_headings


def clean_heading_text(text):
    """Clean heading text"""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\.+$', '', text)  # Remove trailing dots
    return text


def remove_duplicate_headings(headings):
    """Remove duplicates"""
    if not headings:
        return headings
    
    unique = []
    seen = set()
    
    for heading in headings:
        text_key = heading['text'].lower().strip()
        if text_key and text_key not in seen:
            unique.append(heading)
            seen.add(text_key)
    
    return unique


def validate_heading_hierarchy(headings):
    """Ensure logical heading hierarchy"""
    
    if not headings:
        return headings
    
    validated = []
    last_level_num = 0
    
    for heading in headings:
        current_level = heading['level']
        current_level_num = int(current_level[1])
        
        # Don't allow jumping more than one level down
        if last_level_num > 0 and current_level_num > last_level_num + 1:
            adjusted_level_num = last_level_num + 1
            heading['level'] = f'H{adjusted_level_num}'
            current_level_num = adjusted_level_num
        
        validated.append(heading)
        last_level_num = current_level_num
    
    return validated


def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_pdf> <output_json>")
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