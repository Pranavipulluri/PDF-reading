"""
PDF Processor - Multi-library PDF text extraction with comprehensive metadata
Uses PyMuPDF, PDFPlumber, and PDFMiner for maximum compatibility and accuracy
"""

import fitz  # PyMuPDF
import pdfplumber
import logging
from typing import List, Dict, Any, Optional
import numpy as np

class AdvancedPDFProcessor:
    """Advanced PDF processor using multiple libraries for robust extraction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_text_with_metadata(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text blocks with comprehensive metadata using multiple PDF libraries
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of text blocks with detailed metadata
        """
        try:
            self.logger.info(f"Starting advanced PDF processing: {pdf_path}")
            
            # Method 1: PyMuPDF for fast extraction with font details
            pymupdf_blocks = self._extract_with_pymupdf(pdf_path)
            
            # Method 2: PDFPlumber for layout-aware extraction
            pdfplumber_blocks = self._extract_with_pdfplumber(pdf_path)
            
            # Method 3: Combine and enhance results
            combined_blocks = self._combine_extraction_results(pymupdf_blocks, pdfplumber_blocks)
            
            # Method 4: Post-process and clean
            processed_blocks = self._post_process_blocks(combined_blocks)
            
            self.logger.info(f"Extracted {len(processed_blocks)} text blocks with enhanced metadata")
            return processed_blocks
            
        except Exception as e:
            self.logger.error(f"Advanced PDF processing failed: {e}")
            # Fallback to basic extraction
            return self._fallback_extraction(pdf_path)
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract using PyMuPDF with detailed font information"""
        blocks = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(min(doc.page_count, 50)):
                page = doc[page_num]
                
                # Extract text blocks with font information
                text_dict = page.get_text("dict")
                
                for block in text_dict.get("blocks", []):
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            if not text or len(text) < 2:
                                continue
                            
                            # Enhanced font information extraction
                            font_info = self._extract_enhanced_font_info(span)
                            
                            # Position information
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            
                            block_data = {
                                'text': text,
                                'page': page_num + 1,
                                'bbox': bbox,
                                'font_info': font_info,
                                'extraction_method': 'pymupdf',
                                'line_bbox': line.get("bbox", bbox),
                                'block_bbox': block.get("bbox", bbox),
                                'width': bbox[2] - bbox[0],
                                'height': bbox[3] - bbox[1]
                            }
                            
                            blocks.append(block_data)
            
            doc.close()
            self.logger.info(f"PyMuPDF extracted {len(blocks)} text spans")
            
        except Exception as e:
            self.logger.error(f"PyMuPDF extraction failed: {e}")
        
        return blocks
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract using PDFPlumber with layout analysis"""
        blocks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages[:50]):
                    
                    # Extract words with position information
                    words = page.extract_words()
                    
                    for word in words:
                        text = word.get('text', '').strip()
                        if not text:
                            continue
                        
                        # Enhanced position and font information
                        block_data = {
                            'text': text,
                            'page': page_num + 1,
                            'bbox': [
                                word.get('x0', 0),
                                word.get('top', 0),
                                word.get('x1', 0),
                                word.get('bottom', 0)
                            ],
                            'font_info': {
                                'size': word.get('size', 12),
                                'name': word.get('fontname', '').lower(),
                                'is_bold': 'bold' in word.get('fontname', '').lower(),
                                'is_italic': 'italic' in word.get('fontname', '').lower(),
                                'color': word.get('stroking_color', None)
                            },
                            'extraction_method': 'pdfplumber',
                            'width': word.get('x1', 0) - word.get('x0', 0),
                            'height': word.get('bottom', 0) - word.get('top', 0),
                            'char_width': word.get('width', 0),
                            'direction': word.get('direction', 0)
                        }
                        
                        blocks.append(block_data)
            
            self.logger.info(f"PDFPlumber extracted {len(blocks)} words")
            
        except Exception as e:
            self.logger.error(f"PDFPlumber extraction failed: {e}")
        
        return blocks
    
    def _extract_enhanced_font_info(self, span: Dict) -> Dict[str, Any]:
        """Extract comprehensive font information from PyMuPDF span"""
        font_size = span.get("size", 12.0)
        font_flags = span.get("flags", 0)
        font_name = span.get("font", "").lower()
        
        # Enhanced flag analysis
        font_info = {
            'size': float(font_size),
            'name': font_name,
            'flags': int(font_flags),
            'is_superscript': bool(font_flags & 2**0),
            'is_italic': bool(font_flags & 2**1),
            'is_serifed': bool(font_flags & 2**2),
            'is_monospaced': bool(font_flags & 2**3),
            'is_bold': bool(font_flags & 2**4),
            'color': span.get("color", 0),
            'ascender': span.get("ascender", 0),
            'descender': span.get("descender", 0),
            'origin': span.get("origin", [0, 0])
        }
        
        # Font family classification
        font_info['is_sans_serif'] = any(term in font_name for term in [
            'arial', 'helvetica', 'calibri', 'verdana', 'tahoma', 'trebuchet'
        ])
        font_info['is_serif'] = any(term in font_name for term in [
            'times', 'georgia', 'garamond', 'minion', 'palatino'
        ])
        font_info['is_display_font'] = font_size > 18
        font_info['is_title_font'] = font_size > 14 and font_info['is_bold']
        
        # Additional style detection from font name
        if not font_info['is_bold'] and any(term in font_name for term in ['bold', 'heavy', 'black']):
            font_info['is_bold'] = True
        
        if not font_info['is_italic'] and any(term in font_name for term in ['italic', 'oblique']):
            font_info['is_italic'] = True
        
        return font_info
    
    def _combine_extraction_results(self, pymupdf_blocks: List[Dict[str, Any]], 
                                   pdfplumber_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine results from different extraction methods"""
        
        # Use PyMuPDF as primary source (better font metadata)
        # Enhance with PDFPlumber data where available
        combined_blocks = []
        
        for pymupdf_block in pymupdf_blocks:
            enhanced_block = pymupdf_block.copy()
            
            # Find matching PDFPlumber block
            matching_pdfplumber = self._find_matching_block(
                pymupdf_block, pdfplumber_blocks
            )
            
            if matching_pdfplumber:
                # Enhance with PDFPlumber information
                enhanced_block = self._enhance_block_with_pdfplumber(
                    enhanced_block, matching_pdfplumber
                )
            
            combined_blocks.append(enhanced_block)
        
        # Add unique PDFPlumber blocks that weren't matched
        for pdfplumber_block in pdfplumber_blocks:
            if not self._find_matching_block(pdfplumber_block, combined_blocks):
                combined_blocks.append(pdfplumber_block)
        
        return combined_blocks
    
    def _find_matching_block(self, target_block: Dict[str, Any], 
                            candidate_blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find matching block based on text similarity and position"""
        
        target_text = target_block.get('text', '').strip().lower()
        target_bbox = target_block.get('bbox', [0, 0, 0, 0])
        target_page = target_block.get('page', 1)
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidate_blocks:
            candidate_text = candidate.get('text', '').strip().lower()
            candidate_bbox = candidate.get('bbox', [0, 0, 0, 0])
            candidate_page = candidate.get('page', 1)
            
            # Must be on same page
            if candidate_page != target_page:
                continue
            
            # Calculate similarity scores
            text_similarity = self._calculate_text_similarity(target_text, candidate_text)
            position_similarity = self._calculate_bbox_overlap(target_bbox, candidate_bbox)
            
            # Combined score (weighted towards text similarity)
            combined_score = (text_similarity * 0.7) + (position_similarity * 0.3)
            
            if combined_score > best_score and combined_score > 0.6:
                best_score = combined_score
                best_match = candidate
        
        return best_match
    
    def _enhance_block_with_pdfplumber(self, pymupdf_block: Dict[str, Any], 
                                     pdfplumber_block: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance PyMuPDF block with PDFPlumber information"""
        
        enhanced_block = pymupdf_block.copy()
        
        # Add PDFPlumber-specific information
        pdfplumber_font = pdfplumber_block.get('font_info', {})
        
        # Enhance font information
        enhanced_block['font_info']['pdfplumber_size'] = pdfplumber_font.get('size', 0)
        enhanced_block['font_info']['color_info'] = pdfplumber_font.get('color', None)
        enhanced_block['font_info']['char_width'] = pdfplumber_block.get('char_width', 0)
        enhanced_block['font_info']['direction'] = pdfplumber_block.get('direction', 0)
        
        # Cross-validation of font properties
        enhanced_block['font_validation'] = {
            'size_consistent': abs(
                enhanced_block['font_info']['size'] - pdfplumber_font.get('size', 0)
            ) < 2,
            'bold_consistent': (
                enhanced_block['font_info']['is_bold'] == 
                pdfplumber_font.get('is_bold', False)
            ),
            'extraction_agreement': True
        }
        
        return enhanced_block
    
    def _post_process_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process and clean extracted blocks"""
        if not blocks:
            return blocks
        
        # Step 1: Calculate document statistics
        doc_stats = self._calculate_document_stats(blocks)
        
        # Step 2: Add statistical features to each block
        enhanced_blocks = []
        for block in blocks:
            enhanced_block = self._add_statistical_features(block, doc_stats)
            enhanced_blocks.append(enhanced_block)
        
        # Step 3: Group nearby blocks into lines
        line_grouped_blocks = self._group_blocks_into_lines(enhanced_blocks)
        
        # Step 4: Filter out noise and duplicates
        filtered_blocks = self._filter_noise_blocks(line_grouped_blocks)
        
        # Step 5: Sort by page and position
        filtered_blocks.sort(key=lambda x: (x.get('page', 1), x.get('bbox', [0, 0, 0, 0])[1], x.get('bbox', [0, 0, 0, 0])[0]))
        
        return filtered_blocks
    
    def _calculate_document_stats(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate document-level statistics"""
        
        font_sizes = []
        x_positions = []
        y_positions = []
        text_lengths = []
        
        for block in blocks:
            font_info = block.get('font_info', {})
            if font_info.get('size'):
                font_sizes.append(font_info['size'])
            
            bbox = block.get('bbox', [0, 0, 0, 0])
            x_positions.append(bbox[0])
            y_positions.append(bbox[1])
            
            text_lengths.append(len(block.get('text', '')))
        
        stats = {
            'font_stats': {
                'mean': np.mean(font_sizes) if font_sizes else 12,
                'std': np.std(font_sizes) if font_sizes else 2,
                'percentiles': {
                    25: np.percentile(font_sizes, 25) if font_sizes else 10,
                    50: np.percentile(font_sizes, 50) if font_sizes else 12,
                    75: np.percentile(font_sizes, 75) if font_sizes else 14,
                    90: np.percentile(font_sizes, 90) if font_sizes else 16,
                    95: np.percentile(font_sizes, 95) if font_sizes else 18
                }
            },
            'position_stats': {
                'x_mean': np.mean(x_positions) if x_positions else 72,
                'x_std': np.std(x_positions) if x_positions else 20,
                'left_margin': np.min(x_positions) if x_positions else 0,
                'right_margin': np.max([b[2] for b in [block.get('bbox', [0, 0, 500, 0]) for block in blocks]]) if blocks else 612
            },
            'text_stats': {
                'mean_length': np.mean(text_lengths) if text_lengths else 20,
                'total_blocks': len(blocks)
            }
        }
        
        return stats
    
    def _add_statistical_features(self, block: Dict[str, Any], doc_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Add statistical features to block based on document statistics"""
        
        enhanced_block = block.copy()
        font_info = block.get('font_info', {})
        font_size = font_info.get('size', 12)
        
        font_stats = doc_stats.get('font_stats', {})
        
        # Add normalized features
        enhanced_block['statistical_features'] = {
            'font_size_ratio': font_size / font_stats.get('mean', 12),
            'font_size_zscore': (font_size - font_stats.get('mean', 12)) / max(font_stats.get('std', 1), 0.1),
            'font_size_percentile': self._get_font_percentile(font_size, font_stats),
            'is_large_font': font_size > font_stats.get('percentiles', {}).get(75, 14),
            'is_very_large_font': font_size > font_stats.get('percentiles', {}).get(90, 16),
            'avg_font_size': font_stats.get('mean', 12)
        }
        
        return enhanced_block
    
    def _get_font_percentile(self, font_size: float, font_stats: Dict[str, Any]) -> float:
        """Calculate font size percentile"""
        percentiles = font_stats.get('percentiles', {})
        
        if font_size <= percentiles.get(25, 10):
            return 0.25
        elif font_size <= percentiles.get(50, 12):
            return 0.5
        elif font_size <= percentiles.get(75, 14):
            return 0.75
        elif font_size <= percentiles.get(90, 16):
            return 0.9
        else:
            return 0.95
    
    def _group_blocks_into_lines(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group nearby blocks into text lines"""
        if not blocks:
            return blocks
        
        # Group by page first
        pages = {}
        for block in blocks:
            page_num = block.get('page', 1)
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(block)
        
        # Process each page
        grouped_blocks = []
        for page_num, page_blocks in pages.items():
            # Sort by position
            page_blocks.sort(key=lambda x: (x.get('bbox', [0, 0, 0, 0])[1], x.get('bbox', [0, 0, 0, 0])[0]))
            
            # Group into lines based on y-position similarity
            current_line = []
            current_y = None
            y_tolerance = 3  # pixels
            
            for block in page_blocks:
                bbox = block.get('bbox', [0, 0, 0, 0])
                block_y = bbox[1]
                
                if current_y is None or abs(block_y - current_y) <= y_tolerance:
                    current_line.append(block)
                    current_y = block_y
                else:
                    # Process current line
                    if current_line:
                        line_block = self._merge_line_blocks(current_line)
                        grouped_blocks.append(line_block)
                    
                    # Start new line
                    current_line = [block]
                    current_y = block_y
            
            # Process final line
            if current_line:
                line_block = self._merge_line_blocks(current_line)
                grouped_blocks.append(line_block)
        
        return grouped_blocks
    
    def _merge_line_blocks(self, line_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge blocks that are on the same line"""
        if len(line_blocks) == 1:
            return line_blocks[0]
        
        # Sort blocks by x position
        line_blocks.sort(key=lambda x: x.get('bbox', [0, 0, 0, 0])[0])
        
        # Merge text with spaces
        merged_text = ' '.join(block.get('text', '') for block in line_blocks)
        
        # Calculate merged bounding box
        all_bboxes = [block.get('bbox', [0, 0, 0, 0]) for block in line_blocks]
        merged_bbox = [
            min(bbox[0] for bbox in all_bboxes),  # min x
            min(bbox[1] for bbox in all_bboxes),  # min y
            max(bbox[2] for bbox in all_bboxes),  # max x
            max(bbox[3] for bbox in all_bboxes)   # max y
        ]
        
        # Use font info from largest block or first block
        primary_block = max(line_blocks, key=lambda x: x.get('font_info', {}).get('size', 0))
        
        merged_block = primary_block.copy()
        merged_block.update({
            'text': merged_text.strip(),
            'bbox': merged_bbox,
            'width': merged_bbox[2] - merged_bbox[0],
            'height': merged_bbox[3] - merged_bbox[1],
            'merged_from_blocks': len(line_blocks),
            'extraction_methods': list(set(block.get('extraction_method', 'unknown') for block in line_blocks))
        })
        
        return merged_block
    
    def _filter_noise_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out noise blocks and obvious non-content"""
        filtered = []
        
        for block in blocks:
            text = block.get('text', '').strip()
            
            # Skip empty or very short text
            if len(text) < 2:
                continue
            
            # Skip obvious noise patterns
            if self._is_noise_text(text):
                continue
            
            # Skip very small fonts (likely footnotes/metadata)
            font_size = block.get('font_info', {}).get('size', 12)
            if font_size < 8:
                continue
            
            # Skip blocks with excessive special characters
            special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
            if special_char_ratio > 0.5:
                continue
            
            filtered.append(block)
        
        return filtered
    
    def _is_noise_text(self, text: str) -> bool:
        """Check if text is likely noise"""
        text_lower = text.lower().strip()
        
        # Common noise patterns
        noise_patterns = [
            r'^\d+$',                  # Only digits
            r'^\W+$',                  # Only non-alphanumeric characters
            r'^[a-z]$',                # Single lowercase letter
            r'^\d+\.\d+$',             # Decimal numbers
            r'^(lorem|ipsum|test|dummy)$',  # Common placeholder words
            r'^[x]{2,}$',              # Repeated 'x' characters like 'xxx'
        ]

        for pattern in noise_patterns:
            if re.fullmatch(pattern, text_lower):
                return True
        return False
    
    def _extract_page_blocks(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract all text blocks from a single page with metadata"""
        blocks = []
        
        try:
            # Get text blocks with font information
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:  # Skip image blocks
                    continue
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        
                        if not text or len(text) < 2:  # Skip empty or very short text
                            continue
                        
                        # Extract font metadata
                        font_info = self._extract_font_info(span)
                        
                        # Extract position metadata
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        
                        block_data = {
                            "text": text,
                            "page": page_num,
                            "font_size": font_info["size"],
                            "font_flags": font_info["flags"],
                            "font_name": font_info["name"],
                            "x0": bbox[0],
                            "y0": bbox[1],
                            "x1": bbox[2],
                            "y1": bbox[3],
                            "width": bbox[2] - bbox[0],
                            "height": bbox[3] - bbox[1],
                            "is_bold": font_info["is_bold"],
                            "is_italic": font_info["is_italic"]
                        }
                        
                        blocks.append(block_data)
        
        except Exception as e:
            self.logger.error(f"Error extracting blocks from page {page_num}: {e}")
        
        return blocks
    
    def _extract_font_info(self, span: Dict) -> Dict[str, Any]:
        """Extract font information from a text span"""
        font_size = span.get("size", 12.0)
        font_flags = span.get("flags", 0)
        font_name = span.get("font", "").lower()
        
        # Detect bold and italic from flags
        is_bold = bool(font_flags & 2**4)  # Bold flag
        is_italic = bool(font_flags & 2**0)  # Italic flag
        
        # Additional bold detection from font name
        if not is_bold and any(keyword in font_name for keyword in ["bold", "heavy", "black"]):
            is_bold = True
        
        return {
            "size": float(font_size),
            "flags": int(font_flags),
            "name": font_name,
            "is_bold": is_bold,
            "is_italic": is_italic
        }
    
    def _post_process_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process extracted blocks for better quality"""
        if not blocks:
            return blocks
        
        # Calculate font size statistics
        font_sizes = [block["font_size"] for block in blocks]
        avg_font_size = np.mean(font_sizes)
        std_font_size = np.std(font_sizes)
        
        # Add statistical information to each block
        for block in blocks:
            block["avg_font_size"] = avg_font_size
            block["font_size_ratio"] = block["font_size"] / avg_font_size if avg_font_size > 0 else 1.0
            block["font_size_zscore"] = (block["font_size"] - avg_font_size) / std_font_size if std_font_size > 0 else 0
        
        # Remove duplicates and merge close blocks
        processed_blocks = self._merge_close_blocks(blocks)
        
        # Filter out headers/footers and page numbers
        filtered_blocks = self._filter_headers_footers(processed_blocks)
        
        # Sort by page and position
        filtered_blocks.sort(key=lambda x: (x["page"], x["y0"], x["x0"]))
        
        return filtered_blocks
    
    def _merge_close_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge text blocks that are very close to each other"""
        if not blocks:
            return blocks
        
        merged = []
        current_block = None
        
        # Sort by page and position
        blocks.sort(key=lambda x: (x["page"], x["y0"], x["x0"]))
        
        for block in blocks:
            if current_block is None:
                current_block = block.copy()
            elif (block["page"] == current_block["page"] and
                  abs(block["y0"] - current_block["y1"]) < 5 and  # Close vertically
                  abs(block["font_size"] - current_block["font_size"]) < 2):  # Similar font size
                
                # Merge blocks
                current_block["text"] += " " + block["text"]
                current_block["x1"] = max(current_block["x1"], block["x1"])
                current_block["y1"] = block["y1"]
                current_block["width"] = current_block["x1"] - current_block["x0"]
                current_block["height"] = current_block["y1"] - current_block["y0"]
            else:
                merged.append(current_block)
                current_block = block.copy()
        
        if current_block is not None:
            merged.append(current_block)
        
        return merged
    


    def _filter_headers_footers(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out headers, footers, and page numbers."""
        if not blocks:
            return blocks

        filtered = []
        pages = {}

        # Group blocks by page number
        for block in blocks:
            page_num = block["page"]
            pages.setdefault(page_num, []).append(block)

        # Regex for identifying noise (page numbers, etc.)
        noise_patterns = [
            r'^\d+$',               # Just numbers
            r'^page \d+$',          # e.g., "Page 1"
            r'^\d+ of \d+$',        # e.g., "1 of 5"
        ]

        # Process each page separately
        for page_num, page_blocks in pages.items():
            if not page_blocks:
                continue

            # Sort blocks vertically
            page_blocks.sort(key=lambda x: x["y0"])

            # Define header/footer thresholds (top/bottom 10% of the page)
            page_height = max(block["y1"] for block in page_blocks)
            header_threshold = page_height * 0.1
            footer_threshold = page_height * 0.9

            for block in page_blocks:
                text = block["text"].strip().lower()

                # Skip very short text
                if len(text) < 3:
                    continue

                # Skip known noise patterns
                if any(re.fullmatch(pattern, text) for pattern in noise_patterns):
                    continue

                # Skip repetitive headers/footers that appear on multiple pages
                is_repeated = sum(
                    1 for other_page in pages.values()
                    for other_block in other_page
                    if other_block["text"].strip().lower() == text
                ) > 1

                if is_repeated and (block["y0"] < header_threshold or block["y0"] > footer_threshold):
                    continue

                filtered.append(block)

        return filtered

    
    def _extract_page_blocks(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract all text blocks from a single page with metadata"""
        blocks = []
        
        try:
            # Get text blocks with font information
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:  # Skip image blocks
                    continue
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        
                        if not text or len(text) < 2:  # Skip empty or very short text
                            continue
                        
                        # Extract font metadata
                        font_info = self._extract_font_info(span)
                        
                        # Extract position metadata
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        
                        block_data = {
                            "text": text,
                            "page": page_num,
                            "font_size": font_info["size"],
                            "font_flags": font_info["flags"],
                            "font_name": font_info["name"],
                            "x0": bbox[0],
                            "y0": bbox[1],
                            "x1": bbox[2],
                            "y1": bbox[3],
                            "width": bbox[2] - bbox[0],
                            "height": bbox[3] - bbox[1],
                            "is_bold": font_info["is_bold"],
                            "is_italic": font_info["is_italic"]
                        }
                        
                        blocks.append(block_data)
        
        except Exception as e:
            self.logger.error(f"Error extracting blocks from page {page_num}: {e}")
        
        return blocks
    
    def _extract_font_info(self, span: Dict) -> Dict[str, Any]:
        """Extract font information from a text span"""
        font_size = span.get("size", 12.0)
        font_flags = span.get("flags", 0)
        font_name = span.get("font", "").lower()
        
        # Detect bold and italic from flags
        is_bold = bool(font_flags & 2**4)  # Bold flag
        is_italic = bool(font_flags & 2**0)  # Italic flag
        
        # Additional bold detection from font name
        if not is_bold and any(keyword in font_name for keyword in ["bold", "heavy", "black"]):
            is_bold = True
        
        return {
            "size": float(font_size),
            "flags": int(font_flags),
            "name": font_name,
            "is_bold": is_bold,
            "is_italic": is_italic
        }
    
    def _post_process_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process extracted blocks for better quality"""
        if not blocks:
            return blocks
        
        # Calculate font size statistics
        font_sizes = [block["font_size"] for block in blocks]
        avg_font_size = np.mean(font_sizes)
        std_font_size = np.std(font_sizes)
        
        # Add statistical information to each block
        for block in blocks:
            block["avg_font_size"] = avg_font_size
            block["font_size_ratio"] = block["font_size"] / avg_font_size if avg_font_size > 0 else 1.0
            block["font_size_zscore"] = (block["font_size"] - avg_font_size) / std_font_size if std_font_size > 0 else 0
        
        # Remove duplicates and merge close blocks
        processed_blocks = self._merge_close_blocks(blocks)
        
        # Filter out headers/footers and page numbers
        filtered_blocks = self._filter_headers_footers(processed_blocks)
        
        # Sort by page and position
        filtered_blocks.sort(key=lambda x: (x["page"], x["y0"], x["x0"]))
        
        return filtered_blocks
    
    def _merge_close_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge text blocks that are very close to each other"""
        if not blocks:
            return blocks
        
        merged = []
        current_block = None
        
        # Sort by page and position
        blocks.sort(key=lambda x: (x["page"], x["y0"], x["x0"]))
        
        for block in blocks:
            if current_block is None:
                current_block = block.copy()
            elif (block["page"] == current_block["page"] and
                  abs(block["y0"] - current_block["y1"]) < 5 and  # Close vertically
                  abs(block["font_size"] - current_block["font_size"]) < 2):  # Similar font size
                
                # Merge blocks
                current_block["text"] += " " + block["text"]
                current_block["x1"] = max(current_block["x1"], block["x1"])
                current_block["y1"] = block["y1"]
                current_block["width"] = current_block["x1"] - current_block["x0"]
                current_block["height"] = current_block["y1"] - current_block["y0"]
            else:
                merged.append(current_block)
                current_block = block.copy()
        
        if current_block is not None:
            merged.append(current_block)
        
        return merged


    def _filter_headers_footers(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out headers, footers, page numbers, and noise text."""
        if not blocks:
            return blocks

        noise_patterns = [
            r'^\d+$',               # Page numbers: just numbers
            r'^page \d+',           # "Page 1"
            r'^\d+ of \d+',         # "1 of 5"
            r'^copyright',          # Copyright text
            r'^©',                  # Copyright symbol
            r'^\w+\.(com|org|net)', # URLs
            r'^www\.',              # Web addresses
            r'^printed ',           # Printed info
            r'^confidential',       # Confidential
            r'^draft',              # Draft notice
        ]

        filtered = []
        pages = {}

        # Group blocks by page
        for block in blocks:
            page_num = block["page"]
            pages.setdefault(page_num, []).append(block)

        # Process each page
        for page_num, page_blocks in pages.items():
            if not page_blocks:
                continue

            # Sort by y position
            page_blocks.sort(key=lambda x: x["y0"])

            # Header/footer thresholds
            page_height = max(block["y1"] for block in page_blocks)
            header_threshold = page_height * 0.1
            footer_threshold = page_height * 0.9

            for block in page_blocks:
                text = block["text"].strip()
                text_lower = text.lower()

                # Skip very short text
                if len(text) < 3:
                    continue

                # Skip noise patterns
                if any(re.match(pattern, text_lower) for pattern in noise_patterns):
                    continue

                # Skip repeated headers/footers
                is_repeated = sum(
                    1 for other_page in pages.values()
                    for other_block in other_page
                    if other_block["text"].strip().lower() == text_lower
                ) > 1

                if is_repeated and (block["y0"] < header_threshold or block["y0"] > footer_threshold):
                    continue

                filtered.append(block)

        return filtered

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity ratio"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate bounding box overlap ratio"""
        # Calculate intersection
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        if area1 == 0 or area2 == 0:
            return 0.0
        
        # Use intersection over minimum area
        min_area = min(area1, area2)
        return intersection_area / min_area
    
    def _fallback_extraction(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Fallback extraction method if advanced methods fail"""
        self.logger.warning("Using fallback extraction method")
        
        try:
            doc = fitz.open(pdf_path)
            blocks = []
            
            for page_num in range(min(doc.page_count, 50)):
                page = doc[page_num]
                text_dict = page.get_text("dict")
                
                for block in text_dict.get("blocks", []):
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            if text and len(text) >= 2:
                                
                                bbox = span.get("bbox", [0, 0, 0, 0])
                                
                                block_data = {
                                    'text': text,
                                    'page': page_num + 1,
                                    'bbox': bbox,
                                    'font_info': {
                                        'size': span.get("size", 12),
                                        'is_bold': bool(span.get("flags", 0) & 2**4),
                                        'is_italic': bool(span.get("flags", 0) & 2**1),
                                        'name': span.get("font", "").lower()
                                    },
                                    'width': bbox[2] - bbox[0],
                                    'height': bbox[3] - bbox[1],
                                    'extraction_method': 'fallback'
                                }
                                
                                blocks.append(block_data)
            
            doc.close()
            return blocks
            
        except Exception as e:
            self.logger.error(f"Fallback extraction also failed: {e}")
            return []
    
    def _extract_page_blocks(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract all text blocks from a single page with metadata"""
        blocks = []
        
        try:
            # Get text blocks with font information
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:  # Skip image blocks
                    continue
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        
                        if not text or len(text) < 2:  # Skip empty or very short text
                            continue
                        
                        # Extract font metadata
                        font_info = self._extract_font_info(span)
                        
                        # Extract position metadata
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        
                        block_data = {
                            "text": text,
                            "page": page_num,
                            "font_size": font_info["size"],
                            "font_flags": font_info["flags"],
                            "font_name": font_info["name"],
                            "x0": bbox[0],
                            "y0": bbox[1],
                            "x1": bbox[2],
                            "y1": bbox[3],
                            "width": bbox[2] - bbox[0],
                            "height": bbox[3] - bbox[1],
                            "is_bold": font_info["is_bold"],
                            "is_italic": font_info["is_italic"]
                        }
                        
                        blocks.append(block_data)
        
        except Exception as e:
            self.logger.error(f"Error extracting blocks from page {page_num}: {e}")
        
        return blocks
    
    def _extract_font_info(self, span: Dict) -> Dict[str, Any]:
        """Extract font information from a text span"""
        font_size = span.get("size", 12.0)
        font_flags = span.get("flags", 0)
        font_name = span.get("font", "").lower()
        
        # Detect bold and italic from flags
        is_bold = bool(font_flags & 2**4)  # Bold flag
        is_italic = bool(font_flags & 2**0)  # Italic flag
        
        # Additional bold detection from font name
        if not is_bold and any(keyword in font_name for keyword in ["bold", "heavy", "black"]):
            is_bold = True
        
        return {
            "size": float(font_size),
            "flags": int(font_flags),
            "name": font_name,
            "is_bold": is_bold,
            "is_italic": is_italic
        }
    
    def _post_process_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process extracted blocks for better quality"""
        if not blocks:
            return blocks
        
        # Calculate font size statistics
        font_sizes = [block["font_size"] for block in blocks]
        avg_font_size = np.mean(font_sizes)
        std_font_size = np.std(font_sizes)
        
        # Add statistical information to each block
        for block in blocks:
            block["avg_font_size"] = avg_font_size
            block["font_size_ratio"] = block["font_size"] / avg_font_size if avg_font_size > 0 else 1.0
            block["font_size_zscore"] = (block["font_size"] - avg_font_size) / std_font_size if std_font_size > 0 else 0
        
        # Remove duplicates and merge close blocks
        processed_blocks = self._merge_close_blocks(blocks)
        
        # Filter out headers/footers and page numbers
        filtered_blocks = self._filter_headers_footers(processed_blocks)
        
        # Sort by page and position
        filtered_blocks.sort(key=lambda x: (x["page"], x["y0"], x["x0"]))
        
        return filtered_blocks
    
    def _merge_close_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge text blocks that are very close to each other"""
        if not blocks:
            return blocks
        
        merged = []
        current_block = None
        
        # Sort by page and position
        blocks.sort(key=lambda x: (x["page"], x["y0"], x["x0"]))
        
        for block in blocks:
            if current_block is None:
                current_block = block.copy()
            elif (block["page"] == current_block["page"] and
                  abs(block["y0"] - current_block["y1"]) < 5 and  # Close vertically
                  abs(block["font_size"] - current_block["font_size"]) < 2):  # Similar font size
                
                # Merge blocks
                current_block["text"] += " " + block["text"]
                current_block["x1"] = max(current_block["x1"], block["x1"])
                current_block["y1"] = block["y1"]
                current_block["width"] = current_block["x1"] - current_block["x0"]
                current_block["height"] = current_block["y1"] - current_block["y0"]
            else:
                merged.append(current_block)
                current_block = block.copy()
        
        if current_block is not None:
            merged.append(current_block)
        
        return merged
    
    def _filter_headers_footers(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out headers, footers, and page numbers"""
        if not blocks:
            return blocks
        
        filtered = []
        pages = {}
        
        # Group blocks by page
        for block in blocks:
            page_num = block["page"]
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(block)
        
        # Process each page
        for page_num, page_blocks in pages.items():
            if not page_blocks:
                continue
            
            # Sort by y position
            page_blocks.sort(key=lambda x: x["y0"])
            
            # Identify potential headers (top 10% of page)
            # Identify potential footers (bottom 10% of page)
            page_height = max(block["y1"] for block in page_blocks)
            header_threshold = page_height * 0.1
            footer_threshold = page_height * 0.9
            
            for block in page_blocks:
                text = block["text"].strip()
                
                # Skip very short text
                if len(text) < 3:
                    continue
                
                # Skip obvious page numbers
                if text.isdigit() and len(text) <= 3:
                    continue
                
                # Skip repetitive headers/footers (appears on multiple pages)
                is_repeated = sum(1 for other_page in pages.values() 
                                for other_block in other_page 
                                if other_block["text"].strip() == text) > 1
                
                if (is_repeated and 
                    (block["y0"] < header_threshold or block["y0"] > footer_threshold)):
                    continue
                
                filtered.append(block)
        
        return filtered