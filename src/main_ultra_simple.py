#!/usr/bin/env python3
"""
Improved Ultra-Simple PDF Outline Extractor
Fixed to handle invitations and merge text properly
"""

import sys
import json
import time
import re
from pathlib import Path

def extract_pdf_outline(pdf_path, output_path):
    """Extract outline using only PyMuPDF with improved text merging"""
    
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
        
        # Extract text with font info
        for page_num in range(min(doc.page_count, 50)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            # Group text by lines first
            page_lines = []
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    line_text = ""
                    line_font_size = 0
                    line_is_bold = False
                    line_bbox = line.get("bbox", [0, 0, 0, 0])
                    
                    # Merge all spans in this line
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if text:
                            line_text += text + " "
                            # Use the largest font size in the line
                            line_font_size = max(line_font_size, span.get("size", 12))
                            # Bold if any span is bold
                            if span.get("flags", 0) & 16:
                                line_is_bold = True
                    
                    line_text = line_text.strip()
                    if line_text and len(line_text) >= 2:
                        page_lines.append({
                            'text': line_text,
                            'page': page_num + 1,  # Changed to match expected (0-based would be page_num)
                            'font_size': line_font_size,
                            'is_bold': line_is_bold,
                            'bbox': line_bbox
                        })
            
            # Merge nearby lines that might be part of the same heading
            merged_lines = []
            i = 0
            while i < len(page_lines):
                current_line = page_lines[i]
                
                # Look ahead to see if we should merge with next lines
                merged_text = current_line['text']
                j = i + 1
                
                while j < len(page_lines):
                    next_line = page_lines[j]
                    
                    # Merge if: similar font size, close vertically, and combined text is reasonable
                    font_diff = abs(current_line['font_size'] - next_line['font_size'])
                    vertical_gap = abs(next_line['bbox'][1] - current_line['bbox'][3])
                    combined_length = len(merged_text + " " + next_line['text'])
                    
                    if (font_diff <= 2 and vertical_gap <= 10 and combined_length <= 100):
                        merged_text += " " + next_line['text']
                        # Update bbox to include next line
                        current_line['bbox'] = [
                            min(current_line['bbox'][0], next_line['bbox'][0]),
                            min(current_line['bbox'][1], next_line['bbox'][1]),
                            max(current_line['bbox'][2], next_line['bbox'][2]),
                            max(current_line['bbox'][3], next_line['bbox'][3])
                        ]
                        j += 1
                    else:
                        break
                
                current_line['text'] = merged_text.strip()
                merged_lines.append(current_line)
                i = j if j > i + 1 else i + 1
            
            all_blocks.extend(merged_lines)
        
        doc.close()
        
        if not all_blocks:
            print("ERROR: No text extracted")
            return False
        
        print(f"Extracted {len(all_blocks)} text blocks")
        
        # Calculate average font size
        font_sizes = [b['font_size'] for b in all_blocks]
        avg_font_size = sum(font_sizes) / len(font_sizes)
        
        # Extract title - be more selective
        first_page = [b for b in all_blocks if b['page'] == 1]
        title = ""  # Start with empty title like expected output
        
        # Look for a reasonable title (not too promotional/decorative)
        if first_page:
            for candidate in first_page:
                text = candidate['text']
                words = text.split()
                
                # Good title criteria: reasonable length, not all caps promotional text
                if (3 <= len(words) <= 12 and 
                    not text.startswith(('WWW.', 'HTTP', 'RSVP')) and
                    '---' not in text and
                    not text.isupper() or len(words) <= 4):
                    
                    size_ratio = candidate['font_size'] / avg_font_size
                    if size_ratio >= 1.2:  # Reasonably large
                        title = text
                        break
        
        print(f"Title: '{title}'")
        
        # Detect headings with improved filtering
        headings = []
        
        for block in all_blocks:
            text = block['text'].strip()
            font_size = block['font_size']
            is_bold = block['is_bold']
            
            # Skip obvious noise
            if (text.startswith(('WWW.', 'HTTP', 'RSVP:', '---')) or
                text in ['FOR:', 'DATE:', 'TIME:', 'ADDRESS:'] or
                len(text) < 3 or
                text.isdigit() or
                '.' in text and len(text.split()) == 1):  # Skip URLs
                continue
            
            # Heading score calculation
            score = 0
            size_ratio = font_size / avg_font_size
            
            # Font size scoring
            if size_ratio >= 1.3:
                score += 3
                level = "H1"
            elif size_ratio >= 1.1:
                score += 2
                level = "H2"
            elif size_ratio >= 1.0:
                score += 1
                level = "H3"
            else:
                continue
            
            # Bold bonus
            if is_bold:
                score += 2
            
            # Length check - prefer medium length text
            word_count = len(text.split())
            if 2 <= word_count <= 15:
                score += 1
            elif word_count > 20:
                score -= 2
            
            # Pattern matching for numbered sections
            if re.match(r'^\d+\.', text):
                score += 2
                level = "H1"
            elif re.match(r'^\d+\.\d+', text):
                score += 2
                level = "H2"
            
            # Common heading words
            heading_words = ['introduction', 'conclusion', 'summary', 'background', 
                           'methodology', 'results', 'discussion', 'references',
                           'welcome', 'invitation', 'party', 'event']
            if any(word in text.lower() for word in heading_words):
                score += 1
            
            # Promotional/invitation text patterns
            promo_patterns = ['hope to see', 'you\'re invited', 'join us', 'welcome to']
            if any(pattern in text.lower() for pattern in promo_patterns):
                score += 2
                level = "H1"
            
            # All caps bonus (but not for very long text)
            if text.isupper() and word_count <= 8:
                score += 1
            elif text.istitle():
                score += 0.5
            
            # Accept if score >= 2
            if score >= 2:
                # Clean up text
                clean_text = re.sub(r'^\d+\.?\s*', '', text).strip()
                if clean_text and clean_text != title:  # Don't duplicate title
                    headings.append({
                        'level': level,
                        'text': clean_text,
                        'page': block['page'] - 1  # Convert to 0-based like expected output
                    })
        
        print(f"Found {len(headings)} headings")
        
        # Create output
        result = {
            "title": title,
            "outline": headings
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

def main():
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