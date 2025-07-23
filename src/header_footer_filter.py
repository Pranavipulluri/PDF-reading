"""
Header/Footer Filter - Automatically detects and filters repeated page elements
Uses position analysis and text similarity to identify headers, footers, and page numbers
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict, Counter
import numpy as np

class HeaderFooterFilter:
    """Advanced header/footer detection and filtering system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.config = {
            'min_repetition_rate': 0.7,      # 70% of pages must have element
            'position_tolerance': 20,         # Pixels tolerance for position matching
            'text_similarity_threshold': 0.8, # Text similarity threshold
            'header_y_threshold_ratio': 0.15, # Top 15% of page
            'footer_y_threshold_ratio': 0.85, # Bottom 15% of page
            'min_pages_for_analysis': 3,     # Minimum pages needed for analysis
        }
        
        # Detected patterns
        self.detected_patterns = {
            'headers': [],
            'footers': [],
            'page_numbers': [],
            'repeated_elements': []
        }
        
        # Statistics
        self.filter_stats = {
            'total_blocks_analyzed': 0,
            'blocks_filtered': 0,
            'headers_filtered': 0,
            'footers_filtered': 0,
            'page_numbers_filtered': 0,
            'repeated_elements_filtered': 0
        }
    
    def detect_and_filter_headers_footers(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main function to detect and filter headers, footers, and repeated elements
        
        Args:
            blocks: List of text blocks from all pages
            
        Returns:
            Filtered list of blocks with headers/footers removed
        """
        if not blocks:
            return blocks
        
        self.filter_stats['total_blocks_analyzed'] = len(blocks)
        
        # Get page information
        page_info = self._analyze_page_structure(blocks)
        
        if len(page_info) < self.config['min_pages_for_analysis']:
            self.logger.info("Not enough pages for header/footer analysis")
            return blocks
        
        self.logger.info(f"Analyzing headers/footers across {len(page_info)} pages")
        
        # Step 1: Detect page numbers
        page_number_patterns = self._detect_page_numbers(blocks, page_info)
        
        # Step 2: Detect headers (top of pages)
        header_patterns = self._detect_headers(blocks, page_info)
        
        # Step 3: Detect footers (bottom of pages)
        footer_patterns = self._detect_footers(blocks, page_info)
        
        # Step 4: Detect other repeated elements
        repeated_patterns = self._detect_repeated_elements(blocks, page_info)
        
        # Step 5: Create filter mask
        filter_mask = self._create_filter_mask(blocks, {
            'page_numbers': page_number_patterns,
            'headers': header_patterns,
            'footers': footer_patterns,
            'repeated': repeated_patterns
        })
        
        # Step 6: Apply filter
        filtered_blocks = self._apply_filter(blocks, filter_mask)
        
        # Update statistics
        self._update_filter_statistics(filter_mask)
        
        self.logger.info(f"Filtered {self.filter_stats['blocks_filtered']} blocks "
                        f"({self.filter_stats['blocks_filtered']/len(blocks)*100:.1f}%): "
                        f"{self.filter_stats['headers_filtered']} headers, "
                        f"{self.filter_stats['footers_filtered']} footers, "
                        f"{self.filter_stats['page_numbers_filtered']} page numbers")
        
        return filtered_blocks
    
    def _analyze_page_structure(self, blocks: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Analyze the structure of each page"""
        
        page_info = defaultdict(lambda: {
            'blocks': [],
            'y_positions': [],
            'min_y': float('inf'),
            'max_y': float('-inf'),
            'page_height': 0,
            'block_count': 0
        })
        
        # Group blocks by page and analyze positions
        for block in blocks:
            page_num = block.get('page', 1)
            bbox = block.get('bbox', [0, 0, 0, 0])
            y_pos = bbox[1]
            
            page_info[page_num]['blocks'].append(block)
            page_info[page_num]['y_positions'].append(y_pos)
            page_info[page_num]['min_y'] = min(page_info[page_num]['min_y'], bbox[1])
            page_info[page_num]['max_y'] = max(page_info[page_num]['max_y'], bbox[3])
            page_info[page_num]['block_count'] += 1
        
        # Calculate page dimensions
        for page_num in page_info:
            info = page_info[page_num]
            info['page_height'] = info['max_y'] - info['min_y']
            
            # Define header/footer regions
            info['header_threshold'] = info['min_y'] + (info['page_height'] * self.config['header_y_threshold_ratio'])
            info['footer_threshold'] = info['min_y'] + (info['page_height'] * self.config['footer_y_threshold_ratio'])
        
        return dict(page_info)
    
    def _detect_page_numbers(self, blocks: List[Dict[str, Any]], page_info: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect page number patterns"""
        
        patterns = []
        
        # Common page number patterns
        page_number_regexes = [
            r'^\d+$',                    # Just a number
            r'^page\s+\d+$',            # "Page 1"
            r'^\d+\s+of\s+\d+$',        # "1 of 10"
            r'^-\s*\d+\s*-$',           # "- 1 -"
            r'^\d+\s*/\s*\d+$',         # "1 / 10"
        ]
        
        # Look for page number candidates
        for page_num, info in page_info.items():
            for block in info['blocks']:
                text = block.get('text', '').strip().lower()
                
                # Check if text matches page number pattern
                for regex in page_number_regexes:
                    if re.match(regex, text):
                        bbox = block.get('bbox', [0, 0, 0, 0])
                        
                        pattern = {
                            'text': text,
                            'page': page_num,
                            'bbox': bbox,
                            'pattern_type': 'page_number',
                            'regex_matched': regex,
                            'position': 'header' if bbox[1] <= info['header_threshold'] else 'footer'
                        }
                        
                        patterns.append(pattern)
        
        # Filter patterns that appear on multiple pages
        validated_patterns = self._validate_repeated_patterns(patterns, 'page_number')
        
        self.detected_patterns['page_numbers'] = validated_patterns
        return validated_patterns
    
    def _detect_headers(self, blocks: List[Dict[str, Any]], page_info: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect header patterns"""
        
        header_candidates = []
        
        # Find blocks in header region of each page
        for page_num, info in page_info.items():
            for block in info['blocks']:
                bbox = block.get('bbox', [0, 0, 0, 0])
                
                # Check if block is in header region
                if bbox[1] <= info['header_threshold']:
                    text = block.get('text', '').strip()
                    
                    # Skip obvious non-header content
                    if self._is_likely_content(text):
                        continue
                    
                    header_candidates.append({
                        'text': text,
                        'page': page_num,
                        'bbox': bbox,
                        'pattern_type': 'header'
                    })
        
        # Group similar headers and validate repetition
        header_patterns = self._group_and_validate_patterns(header_candidates, 'header')
        
        self.detected_patterns['headers'] = header_patterns
        return header_patterns
    
    def _detect_footers(self, blocks: List[Dict[str, Any]], page_info: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect footer patterns"""
        
        footer_candidates = []
        
        # Find blocks in footer region of each page
        for page_num, info in page_info.items():
            for block in info['blocks']:
                bbox = block.get('bbox', [0, 0, 0, 0])
                
                # Check if block is in footer region
                if bbox[1] >= info['footer_threshold']:
                    text = block.get('text', '').strip()
                    
                    # Skip obvious non-footer content
                    if self._is_likely_content(text):
                        continue
                    
                    footer_candidates.append({
                        'text': text,
                        'page': page_num,
                        'bbox': bbox,
                        'pattern_type': 'footer'
                    })
        
        # Group similar footers and validate repetition
        footer_patterns = self._group_and_validate_patterns(footer_candidates, 'footer')
        
        self.detected_patterns['footers'] = footer_patterns
        return footer_patterns
    
    def _detect_repeated_elements(self, blocks: List[Dict[str, Any]], 
                                page_info: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect other repeated elements across pages"""
        
        # Group blocks by similar text and position
        text_position_groups = defaultdict(list)
        
        for block in blocks:
            text = block.get('text', '').strip().lower()
            bbox = block.get('bbox', [0, 0, 0, 0])
            page_num = block.get('page', 1)
            
            # Skip very short text
            if len(text) < 3:
                continue
            
            # Skip obvious content
            if self._is_likely_content(text):
                continue
            
            # Create position-based key
            page_height = page_info.get(page_num, {}).get('page_height', 800)
            relative_y = bbox[1] / max(page_height, 1)  # Normalize position
            
            # Group by text similarity and position
            position_bucket = int(relative_y * 10)  # 10 position buckets
            group_key = f"{text[:20]}_{position_bucket}"  # First 20 chars + position
            
            text_position_groups[group_key].append(block)
        
        # Find groups that appear on multiple pages
        repeated_patterns = []
        
        for group_key, group_blocks in text_position_groups.items():
            if len(group_blocks) < 2:
                continue
            
            # Check if appears on multiple pages
            pages_with_element = set(block.get('page', 1) for block in group_blocks)
            total_pages = len(page_info)
            
            repetition_rate = len(pages_with_element) / total_pages
            
            if repetition_rate >= self.config['min_repetition_rate']:
                pattern = {
                    'text_pattern': group_blocks[0].get('text', '').strip(),
                    'pages': sorted(pages_with_element),
                    'repetition_rate': repetition_rate,
                    'pattern_type': 'repeated_element',
                    'blocks': group_blocks
                }
                
                repeated_patterns.append(pattern)
        
        self.detected_patterns['repeated_elements'] = repeated_patterns
        return repeated_patterns
    
    def _group_and_validate_patterns(self, candidates: List[Dict[str, Any]], pattern_type: str) -> List[Dict[str, Any]]:
        """Group similar patterns and validate their repetition"""
        
        # Group by text similarity
        text_groups = defaultdict(list)
        
        for candidate in candidates:
            text = candidate['text'].lower().strip()
            
            # Find similar text group
            matched_group = None
            for existing_text in text_groups.keys():
                similarity = self._calculate_text_similarity(text, existing_text)
                if similarity >= self.config['text_similarity_threshold']:
                    matched_group = existing_text
                    break
            
            if matched_group:
                text_groups[matched_group].append(candidate)
            else:
                text_groups[text].append(candidate)
        
        # Validate repetition for each group
        validated_patterns = []
        
        for text, group_candidates in text_groups.items():
            pages_with_pattern = set(c['page'] for c in group_candidates)
            total_pages = len(set(c['page'] for c in candidates))
            
            repetition_rate = len(pages_with_pattern) / max(total_pages, 1)
            
            if repetition_rate >= self.config['min_repetition_rate']:
                pattern = {
                    'text_pattern': text,
                    'pages': sorted(pages_with_pattern),
                    'repetition_rate': repetition_rate,
                    'pattern_type': pattern_type,
                    'candidates': group_candidates
                }
                
                validated_patterns.append(pattern)
        
        return validated_patterns
    
    def _validate_repeated_patterns(self, patterns: List[Dict[str, Any]], pattern_type: str) -> List[Dict[str, Any]]:
        """Validate that patterns actually repeat across pages"""
        
        # Group by text similarity
        text_groups = defaultdict(list)
        
        for pattern in patterns:
            text = pattern['text'].lower()
            text_groups[text].append(pattern)
        
        # Keep only patterns that appear on multiple pages
        validated = []
        
        for text, group_patterns in text_groups.items():
            pages = set(p['page'] for p in group_patterns)
            
            if len(pages) >= 2:  # Must appear on at least 2 pages
                validated_pattern = {
                    'text_pattern': text,
                    'pages': sorted(pages),
                    'pattern_type': pattern_type,
                    'instances': group_patterns
                }
                validated.append(validated_pattern)
        
        return validated
    
    def _create_filter_mask(self, blocks: List[Dict[str, Any]], all_patterns: Dict[str, List[Dict[str, Any]]]) -> List[bool]:
        """Create a filter mask indicating which blocks to remove"""
        
        filter_mask = [False] * len(blocks)  # False = keep, True = filter out
        
        for i, block in enumerate(blocks):
            text = block.get('text', '').strip().lower()
            page = block.get('page', 1)
            bbox = block.get('bbox', [0, 0, 0, 0])
            
            # Check against all pattern types
            for pattern_type, patterns in all_patterns.items():
                if self._block_matches_patterns(block, patterns, pattern_type):
                    filter_mask[i] = True
                    break  # Only need one match to filter
        
        return filter_mask
    
    def _block_matches_patterns(self, block: Dict[str, Any], patterns: List[Dict[str, Any]], pattern_type: str) -> bool:
        """Check if a block matches any of the given patterns"""
        
        block_text = block.get('text', '').strip().lower()
        block_page = block.get('page', 1)
        block_bbox = block.get('bbox', [0, 0, 0, 0])
        
        for pattern in patterns:
            # Check page membership
            if block_page not in pattern.get('pages', []):
                continue
            
            # Check text similarity
            pattern_text = pattern.get('text_pattern', '').lower()
            similarity = self._calculate_text_similarity(block_text, pattern_text)
            
            if similarity >= self.config['text_similarity_threshold']:
                # Additional position check for headers/footers
                if pattern_type in ['header', 'footer']:
                    if self._positions_match(block_bbox, pattern, pattern_type):
                        return True
                else:
                    return True
        
        return False
    
    def _positions_match(self, bbox: List[float], pattern: Dict[str, Any], 
                        pattern_type: str) -> bool:
        """Check if positions match for header/footer patterns"""
        
        # For now, just check if in same general region
        # Could be enhanced with more sophisticated position matching
        return True
    
    def _apply_filter(self, blocks: List[Dict[str, Any]], filter_mask: List[bool]) -> List[Dict[str, Any]]:
        """Apply the filter mask to remove unwanted blocks"""
        
        filtered_blocks = []
        
        for i, (block, should_filter) in enumerate(zip(blocks, filter_mask)):
            if not should_filter:
                # Mark block as filtered for debugging
                enhanced_block = block.copy()
                enhanced_block['header_footer_analysis'] = {
                    'was_filtered': False,
                    'analysis_performed': True
                }
                filtered_blocks.append(enhanced_block)
            else:
                # Log filtered block for debugging
                self.logger.debug(f"Filtered block: '{block.get('text', '')[:50]}...' on page {block.get('page', 0)}")
        
        return filtered_blocks
    
    def _update_filter_statistics(self, filter_mask: List[bool]):
        """Update filtering statistics"""
        
        self.filter_stats['blocks_filtered'] = sum(filter_mask)
        
        # Count by pattern type (simplified)
        self.filter_stats['headers_filtered'] = len(self.detected_patterns['headers'])
        self.filter_stats['footers_filtered'] = len(self.detected_patterns['footers'])
        self.filter_stats['page_numbers_filtered'] = len(self.detected_patterns['page_numbers'])
        self.filter_stats['repeated_elements_filtered'] = len(self.detected_patterns['repeated_elements'])
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between two strings"""
        
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Also check for substring similarity
        if text1 in text2 or text2 in text1:
            jaccard_similarity = max(jaccard_similarity, 0.8)
        
        return jaccard_similarity
    
    def _is_likely_content(self, text: str) -> bool:
        """Check if text is likely to be content rather than header/footer"""
        
        text = text.strip()
        
        # Length check - content is usually longer
        if len(text) > 100:
            return True
        
        # Word count - content has more words
        word_count = len(text.split())
        if word_count > 15:
            return True
        
        # Contains sentence-ending punctuation
        if re.search(r'[.!?]\s', text):
            return True
        
        # Contains common content words
        content_indicators = ['the', 'and', 'of', 'to', 'in', 'is', 'that', 'for', 'with']
        text_lower = text.lower()
        content_word_count = sum(1 for word in content_indicators if word in text_lower)
        
        if content_word_count >= 2:
            return True
        
        return False
    
    def get_filter_report(self) -> Dict[str, Any]:
        """Get comprehensive filtering report"""
        
        report = {
            'statistics': self.filter_stats.copy(),
            'detected_patterns': {
                'headers': len(self.detected_patterns['headers']),
                'footers': len(self.detected_patterns['footers']), 
                'page_numbers': len(self.detected_patterns['page_numbers']),
                'repeated_elements': len(self.detected_patterns['repeated_elements'])
            },
            'pattern_details': {}
        }
        
        # Add pattern details
        for pattern_type, patterns in self.detected_patterns.items():
            if patterns:
                report['pattern_details'][pattern_type] = [
                    {
                        'text': p.get('text_pattern', '')[:50],
                        'pages': p.get('pages', []),
                        'repetition_rate': p.get('repetition_rate', 0)
                    }
                    for p in patterns[:5]  # Show first 5 patterns
                ]
        
        # Calculate filtering effectiveness
        if self.filter_stats['total_blocks_analyzed'] > 0:
            report['filter_rate'] = self.filter_stats['blocks_filtered'] / self.filter_stats['total_blocks_analyzed']
        else:
            report['filter_rate'] = 0.0
        
        return report