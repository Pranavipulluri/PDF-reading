#!/usr/bin/env python3
"""Ultra-simple PDF processor that works reliably"""
import sys, json, fitz, re
from pathlib import Path

def extract_pdf_outline(pdf_path, output_path):
    try:
        doc = fitz.open(pdf_path)
        blocks = []
        for page_num in range(min(doc.page_count, 50)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if "lines" not in block: continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if not text or len(text) < 2: continue
                        blocks.append({
                            'text': text, 'page': page_num + 1,
                            'font_size': span.get("size", 12),
                            'is_bold': bool(span.get("flags", 0) & 2**4),
                            'bbox': span.get("bbox", [0,0,0,0])
                        })
        if blocks:
            avg_size = sum(b['font_size'] for b in blocks) / len(blocks)
            headings = []
            for block in blocks:
                text, size, bold = block['text'], block['font_size'], block['is_bold']
                ratio, words = size / avg_size, len(text.split())
                score = 0
                if ratio >= 1.3 or bold: score += 2
                if 2 <= words <= 15: score += 1
                if re.match(r'\d+\.', text): score += 2
                if score >= 2:
                    level = "H1" if ratio >= 1.4 else "H2" if ratio >= 1.2 else "H3"
                    clean_text = re.sub(r'\d+\.?\s*', '', text)
                    headings.append({'level': level, 'text': clean_text, 'page': block['page']})
            title = Path(pdf_path).stem.replace('_', ' ').title()
            result = {"title": title, "outline": headings[:20]}
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            return True
        doc.close()
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py input.pdf output.json")
        sys.exit(1)
    success = extract_pdf_outline(sys.argv[1], sys.argv[2])
    if success:
        print("SUCCESS: PDF processed successfully")
    else:
        print("ERROR: Failed to process PDF")
        sys.exit(1)
