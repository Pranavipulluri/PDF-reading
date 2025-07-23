"""
Layout Analyzer - Advanced PDF layout analysis using multiple libraries
Combines PyMuPDF, PDFPlumber, and PDFMiner for comprehensive layout understanding
"""

import fitz  # PyMuPDF
import pdfplumber
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

class AdvancedLayoutAnalyzer:
    """Advanced PDF layout analysis using multiple parsing libraries"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.page_layouts = {}
        
    def analyze_document_layout(self, pdf_path: str) -> Dict[str, Any]:
        """
        Comprehensive layout analysis using multiple PDF libraries
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Complete layout analysis results
        """
        self.logger.info("Starting comprehensive layout analysis...")
        
        # Method 1: PyMuPDF for fast text extraction with font metadata
        pymupdf_data = self._analyze_with_pymupdf(pdf_path)
        
        # Method 2: PDFPlumber for precise layout and table detection
        pdfplumber_data = self._analyze_with_pdfplumber(pdf_path)
        
        # Method 3: Combine and enhance results
        combined_analysis = self._combine_analyses(pymupdf_data, pdfplumber_data)
        
        # Method 4: Advanced layout features
        enhanced_analysis = self._extract_advanced_features(combined_analysis)
        
        self.logger.info("Layout analysis completed")
        return enhanced_analysis
    
    def _analyze_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze using PyMuPDF for detailed font information"""
        try:
            doc = fitz.open(pdf_path)
            pages_data = []
            
            for page_num in range(min(doc.page_count, 50)):
                page = doc[page_num]
                page_data = {
                    'page_number': page_num + 1,
                    'page_size': page.rect,
                    'blocks': []
                }
                
                # Get text blocks with detailed font information
                text_dict = page.get_text("dict")
                
                for block in text_dict.get("blocks", []):
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            if not text:
                                continue
                            
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            font_info = self._extract_detailed_font_info(span)
                            
                            block_data = {
                                'text': text,
                                'bbox': bbox,
                                'font_info': font_info,
                                'line_bbox': line.get("bbox", bbox),
                                'block_bbox': block.get("bbox", bbox)
                            }
                            
                            page_data['blocks'].append(block_data)
                
                pages_data.append(page_data)
            
            doc.close()
            return {'method': 'pymupdf', 'pages': pages_data}
            
        except Exception as e:
            self.logger.error(f"PyMuPDF analysis failed: {e}")
            return {'method': 'pymupdf', 'pages': []}
    
    def _analyze_with_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze using PDFPlumber for layout-aware extraction"""
        try:
            pages_data = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages[:50]):  # Limit to 50 pages
                    page_data = {
                        'page_number': page_num + 1,
                        'page_size': {'width': page.width, 'height': page.height},
                        'chars': [],
                        'lines': [],
                        'layout_objects': []
                    }
                    
                    # Extract character-level information
                    chars = page.chars
                    for char in chars:
                        char_data = {
                            'text': char.get('text', ''),
                            'bbox': [char.get('x0', 0), char.get('top', 0), 
                                   char.get('x1', 0), char.get('bottom', 0)],
                            'font_name': char.get('fontname', ''),
                            'font_size': char.get('size', 0),
                            'color': char.get('stroking_color', None),
                            'width': char.get('width', 0),
                            'height': char.get('height', 0)
                        }
                        page_data['chars'].append(char_data)
                    
                    # Extract lines and group characters
                    lines = self._group_chars_into_lines(chars)
                    page_data['lines'] = lines
                    
                    # Detect layout objects (tables, images, etc.)
                    layout_objects = self._detect_layout_objects(page)
                    page_data['layout_objects'] = layout_objects
                    
                    pages_data.append(page_data)
            
            return {'method': 'pdfplumber', 'pages': pages_data}
            
        except Exception as e:
            self.logger.error(f"PDFPlumber analysis failed: {e}")
            return {'method': 'pdfplumber', 'pages': []}
    
    def _extract_detailed_font_info(self, span: Dict) -> Dict[str, Any]:
        """Extract comprehensive font information from PyMuPDF span"""
        font_size = span.get("size", 12.0)
        font_flags = span.get("flags", 0)
        font_name = span.get("font", "").lower()
        
        # Detailed flag analysis
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
            'descender': span.get("descender", 0)
        }
        
        # Additional font analysis
        font_info['is_sans_serif'] = any(term in font_name for term in ['arial', 'helvetica', 'calibri', 'verdana'])
        font_info['is_serif'] = any(term in font_name for term in ['times', 'georgia', 'garamond'])
        font_info['is_display_font'] = font_size > 18
        font_info['is_title_font'] = font_size > 14 and font_info['is_bold']
        
        return font_info
    
    def _group_chars_into_lines(self, chars: List[Dict]) -> List[Dict[str, Any]]:
        """Group characters into text lines based on position"""
        if not chars:
            return []
        
        # Sort characters by position
        chars_sorted = sorted(chars, key=lambda c: (c.get('top', 0), c.get('x0', 0)))
        
        lines = []
        current_line = []
        current_y = None
        y_tolerance = 2  # pixels
        
        for char in chars_sorted:
            char_y = char.get('top', 0)
            
            if current_y is None or abs(char_y - current_y) <= y_tolerance:
                current_line.append(char)
                current_y = char_y
            else:
                if current_line:
                    lines.append(self._create_line_from_chars(current_line))
                current_line = [char]
                current_y = char_y
        
        if current_line:
            lines.append(self._create_line_from_chars(current_line))
        
        return lines
    
    def _create_line_from_chars(self, chars: List[Dict]) -> Dict[str, Any]:
        """Create line object from character list"""
        if not chars:
            return {}
        
        # Sort characters by x position
        chars = sorted(chars, key=lambda c: c.get('x0', 0))
        
        # Combine text
        text = ''.join(c.get('text', '') for c in chars)
        
        # Calculate bounding box
        x0 = min(c.get('x0', 0) for c in chars)
        y0 = min(c.get('top', 0) for c in chars)
        x1 = max(c.get('x1', 0) for c in chars)
        y1 = max(c.get('bottom', 0) for c in chars)
        
        # Analyze font consistency
        font_sizes = [c.get('size', 0) for c in chars if c.get('size', 0) > 0]
        font_names = [c.get('fontname', '') for c in chars if c.get('fontname')]
        
        line_data = {
            'text': text.strip(),
            'bbox': [x0, y0, x1, y1],
            'char_count': len(chars),
            'font_size': np.median(font_sizes) if font_sizes else 0,
            'font_name': max(set(font_names), key=font_names.count) if font_names else '',
            'font_consistency': len(set(font_names)) <= 2 if font_names else True,
            'chars': chars
        }
        
        return line_data
    
    def _detect_layout_objects(self, page) -> List[Dict[str, Any]]:
        """Detect tables, images, and other layout objects"""
        layout_objects = []
        
        try:
            # Detect tables
            tables = page.find_tables()
            for i, table in enumerate(tables):
                layout_objects.append({
                    'type': 'table',
                    'bbox': table.bbox,
                    'id': f'table_{i}',
                    'rows': len(table.rows) if hasattr(table, 'rows') else 0,
                    'cols': len(table.rows[0]) if hasattr(table, 'rows') and table.rows else 0
                })
            
            # Detect potential images/figures (rectangles with no text)
            rects = page.rects
            for i, rect in enumerate(rects):
                # Check if rectangle is large enough to be significant
                width = rect.get('x1', 0) - rect.get('x0', 0)
                height = rect.get('y1', 0) - rect.get('y0', 0)
                
                if width > 50 and height > 30:  # Minimum size threshold
                    layout_objects.append({
                        'type': 'rectangle',
                        'bbox': [rect.get('x0', 0), rect.get('top', 0),
                                rect.get('x1', 0), rect.get('bottom', 0)],
                        'id': f'rect_{i}',
                        'width': width,
                        'height': height
                    })
        
        except Exception as e:
            self.logger.warning(f"Layout object detection failed: {e}")
        
        return layout_objects
    
    def _combine_analyses(self, pymupdf_data: Dict, pdfplumber_data: Dict) -> Dict[str, Any]:
        """Combine results from different PDF libraries"""
        combined_pages = []
        
        pymupdf_pages = pymupdf_data.get('pages', [])
        pdfplumber_pages = pdfplumber_data.get('pages', [])
        
        max_pages = max(len(pymupdf_pages), len(pdfplumber_pages))
        
        for page_num in range(max_pages):
            combined_page = {'page_number': page_num + 1, 'blocks': []}
            
            # Get PyMuPDF data for this page
            pymupdf_page = pymupdf_pages[page_num] if page_num < len(pymupdf_pages) else None
            pdfplumber_page = pdfplumber_pages[page_num] if page_num < len(pdfplumber_pages) else None
            
            # Combine text blocks
            blocks = []
            
            if pymupdf_page:
                # Use PyMuPDF blocks as primary source (better font info)
                for block in pymupdf_page.get('blocks', []):
                    enhanced_block = self._enhance_block_with_layout_info(
                        block, pdfplumber_page
                    )
                    blocks.append(enhanced_block)
            
            # Add layout information
            combined_page['blocks'] = blocks
            combined_page['page_info'] = self._extract_page_info(pymupdf_page, pdfplumber_page)
            
            combined_pages.append(combined_page)
        
        return {'pages': combined_pages, 'analysis_methods': ['pymupdf', 'pdfplumber']}
    
    def _enhance_block_with_layout_info(self, block: Dict, pdfplumber_page: Optional[Dict]) -> Dict:
        """Enhance PyMuPDF block with PDFPlumber layout information"""
        enhanced_block = block.copy()
        
        if not pdfplumber_page:
            return enhanced_block
        
        # Find corresponding text in PDFPlumber data
        block_text = block.get('text', '').strip()
        block_bbox = block.get('bbox', [0, 0, 0, 0])
        
        # Look for matching lines in PDFPlumber data
        matching_lines = []
        for line in pdfplumber_page.get('lines', []):
            line_text = line.get('text', '').strip()
            line_bbox = line.get('bbox', [0, 0, 0, 0])
            
            # Check for text similarity and bbox overlap
            if (self._calculate_text_similarity(block_text, line_text) > 0.7 or
                self._calculate_bbox_overlap(block_bbox, line_bbox) > 0.5):
                matching_lines.append(line)
        
        # Enhance block with layout information
        if matching_lines:
            enhanced_block['layout_info'] = {
                'line_spacing': self._calculate_line_spacing(matching_lines),
                'character_spacing': self._calculate_character_spacing(matching_lines[0]),
                'text_alignment': self._detect_text_alignment(matching_lines, pdfplumber_page),
                'indentation': self._calculate_indentation(matching_lines[0], pdfplumber_page)
            }
        
        return enhanced_block
    
    def _extract_advanced_features(self, combined_analysis: Dict) -> Dict[str, Any]:
        """Extract advanced layout features for heading detection"""
        enhanced_analysis = combined_analysis.copy()
        
        for page in enhanced_analysis.get('pages', []):
            blocks = page.get('blocks', [])
            
            # Calculate page-level statistics
            page_stats = self._calculate_page_statistics(blocks)
            
            # Enhance each block with advanced features
            for block in blocks:
                block.update(self._extract_block_layout_features(block, blocks, page_stats))
        
        return enhanced_analysis
    
    def _calculate_page_statistics(self, blocks: List[Dict]) -> Dict[str, Any]:
        """Calculate page-level layout statistics"""
        if not blocks:
            return {}
        
        font_sizes = [b.get('font_info', {}).get('size', 12) for b in blocks]
        y_positions = [b.get('bbox', [0, 0, 0, 0])[1] for b in blocks]
        x_positions = [b.get('bbox', [0, 0, 0, 0])[0] for b in blocks]
        
        return {
            'avg_font_size': np.mean(font_sizes),
            'std_font_size': np.std(font_sizes),
            'font_size_percentiles': {
                '25': np.percentile(font_sizes, 25),
                '50': np.percentile(font_sizes, 50),
                '75': np.percentile(font_sizes, 75),
                '90': np.percentile(font_sizes, 90),
                '95': np.percentile(font_sizes, 95)
            },
            'left_margin': np.min(x_positions),
            'avg_x_position': np.mean(x_positions),
            'std_x_position': np.std(x_positions)
        }
    
    def _extract_block_layout_features(self, block: Dict, all_blocks: List[Dict], 
                                     page_stats: Dict) -> Dict[str, Any]:
        """Extract advanced layout features for a single block"""
        features = {}
        
        bbox = block.get('bbox', [0, 0, 0, 0])
        font_info = block.get('font_info', {})
        layout_info = block.get('layout_info', {})
        
        # Font-based features
        features['font_size_z_score'] = ((font_info.get('size', 12) - page_stats.get('avg_font_size', 12)) / 
                                        max(page_stats.get('std_font_size', 1), 0.1))
        features['font_size_percentile'] = self._get_percentile_rank(
            font_info.get('size', 12), 
            [b.get('font_info', {}).get('size', 12) for b in all_blocks]
        )
        
        # Position-based features
        features['distance_from_left_margin'] = bbox[0] - page_stats.get('left_margin', 0)
        features['x_position_z_score'] = ((bbox[0] - page_stats.get('avg_x_position', 0)) / 
                                         max(page_stats.get('std_x_position', 1), 0.1))
        
        # Layout-based features
        features['line_spacing'] = layout_info.get('line_spacing', 0)
        features['character_spacing'] = layout_info.get('character_spacing', 0)
        features['indentation_level'] = layout_info.get('indentation', 0)
        features['text_alignment'] = layout_info.get('text_alignment', 'left')
        
        # Context-based features
        features.update(self._calculate_context_features(block, all_blocks))
        
        return {'advanced_features': features}
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
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
        """Calculate overlap ratio between two bounding boxes"""
        # Calculate intersection
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _get_percentile_rank(self, value: float, values: List[float]) -> float:
        """Get percentile rank of value in list"""
        if not values:
            return 0.5
        return sum(v <= value for v in values) / len(values)
    
    # Additional helper methods for layout analysis would go here...
    def _calculate_line_spacing(self, lines: List[Dict]) -> float:
        """Calculate average line spacing"""
        if len(lines) < 2:
            return 0.0
        
        spacings = []
        for i in range(1, len(lines)):
            prev_line = lines[i-1]
            curr_line = lines[i]
            spacing = curr_line.get('bbox', [0, 0, 0, 0])[1] - prev_line.get('bbox', [0, 0, 0, 0])[3]
            spacings.append(spacing)
        
        return np.mean(spacings) if spacings else 0.0
    
    def _calculate_character_spacing(self, line: Dict) -> float:
        """Calculate average character spacing in a line"""
        chars = line.get('chars', [])
        if len(chars) < 2:
            return 0.0
        
        spacings = []
        for i in range(1, len(chars)):
            prev_char = chars[i-1]
            curr_char = chars[i]
            spacing = curr_char.get('x0', 0) - prev_char.get('x1', 0)
            spacings.append(spacing)
        
        return np.mean(spacings) if spacings else 0.0
    
    def _detect_text_alignment(self, lines: List[Dict], page: Dict) -> str:
        """Detect text alignment (left, center, right)"""
        if not lines:
            return 'left'
        
        page_width = page.get('page_info', {}).get('width', 612)  # Default letter size
        left_positions = [line.get('bbox', [0, 0, 0, 0])[0] for line in lines]
        right_positions = [line.get('bbox', [0, 0, 0, 0])[2] for line in lines]
        
        avg_left = np.mean(left_positions)
        avg_right = np.mean(right_positions)
        center = page_width / 2
        
        # Calculate distances from alignment points
        left_variance = np.var(left_positions)
        right_variance = np.var(right_positions)
        center_distance = abs((avg_left + avg_right) / 2 - center)
        
        # Determine alignment
        if left_variance < 10:  # Low variance in left positions
            return 'left'
        elif right_variance < 10:  # Low variance in right positions
            return 'right'
        elif center_distance < page_width * 0.1:  # Close to center
            return 'center'
        else:
            return 'justified'
    
    def _calculate_indentation(self, line: Dict, page: Dict) -> float:
        """Calculate indentation level relative to left margin"""
        left_margin = page.get('page_info', {}).get('left_margin', 0)
        line_left = line.get('bbox', [0, 0, 0, 0])[0]
        return max(0, line_left - left_margin)
    
    def _calculate_context_features(self, block: Dict, all_blocks: List[Dict]) -> Dict[str, Any]:
        """Calculate contextual features based on surrounding blocks"""
        # This would include features like:
        # - Distance to previous/next heading
        # - Whitespace before/after
        # - Font size relative to neighbors
        # etc.
        return {
            'context_calculated': True,
            # Implement specific context calculations here
        }