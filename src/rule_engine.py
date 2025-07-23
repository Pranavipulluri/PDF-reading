"""
Rule Engine - Enhanced rule-based heading detection system
Combines font analysis, layout analysis, pattern matching, and semantic rules
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import re

class EnhancedRuleEngine:
    """Advanced rule-based system for heading detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Rule weights for different detection methods
        self.rule_weights = {
            'font_size': 0.25,
            'font_style': 0.15,
            'position': 0.20,
            'pattern': 0.20,
            'semantic': 0.15,
            'context': 0.05
        }
        
        # Font size thresholds (will be calculated dynamically)
        self.font_thresholds = {
            'h1_min_ratio': 1.4,   # Minimum ratio vs average font size for H1
            'h2_min_ratio': 1.2,   # Minimum ratio vs average font size for H2
            'h3_min_ratio': 1.1,   # Minimum ratio vs average font size for H3
        }
        
        # Position-based rules
        self.position_rules = {
            'left_margin_tolerance': 20,     # Pixels from left margin
            'center_tolerance_ratio': 0.15,  # Ratio of page width for centering
            'whitespace_threshold': 15,      # Minimum vertical whitespace before heading
            'top_page_ratio': 0.3,          # Top 30% of page more likely to have headings
        }
        
        # Pattern-based rules with confidence scores
        self.pattern_rules = {
            'numbered_section': {
                'patterns': [
                    (r'^\s*\d+\.\s+', 'H1', 0.8),
                    (r'^\s*\d+\.\d+\.\s+', 'H2', 0.9),
                    (r'^\s*\d+\.\d+\.\d+\.\s+', 'H3', 0.95),
                    (r'^\s*\d+\)\s+', 'H2', 0.7),
                    (r'^\s*[A-Z]\.\s+', 'H2', 0.6),
                    (r'^\s*\([a-z]\)\s+', 'H3', 0.6),
                ],
                'weight': 0.9
            },
            'chapter_markers': {
                'patterns': [
                    (r'^\s*chapter\s+\d+', 'H1', 0.95),
                    (r'^\s*section\s+\d+', 'H1', 0.85),
                    (r'^\s*part\s+\d+', 'H1', 0.9),
                    (r'^\s*appendix\s*[a-z]?', 'H1', 0.8),
                ],
                'weight': 1.0
            },
            'content_markers': {
                'patterns': [
                    (r'^\s*(introduction|overview|summary|conclusion)$', 'H1', 0.8),
                    (r'^\s*(background|methodology|results|discussion)$', 'H1', 0.7),
                    (r'^\s*(references|bibliography|acknowledgments)$', 'H1', 0.9),
                    (r'^\s*(table of contents|index|glossary)$', 'H1', 0.9),
                ],
                'weight': 0.8
            }
        }
        
        # Style-based rules
        self.style_rules = {
            'all_caps': {'min_confidence': 0.7, 'max_words': 8, 'level_preference': 'H1'},
            'title_case': {'min_confidence': 0.5, 'max_words': 12, 'level_preference': 'H2'},
            'bold': {'confidence_boost': 0.3},
            'italic': {'confidence_boost': 0.1},
            'underline': {'confidence_boost': 0.2}
        }
        
    def apply_rules(self, blocks: List[Dict[str, Any]], 
                   document_stats: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Apply comprehensive rule-based detection to text blocks
        
        Args:
            blocks: List of text blocks with metadata
            document_stats: Document-level statistics
            
        Returns:
            List of blocks with rule-based analysis results
        """
        if not blocks:
            return []
        
        self.logger.info(f"Applying rules to {len(blocks)} text blocks")
        
        # Calculate document statistics if not provided
        if document_stats is None:
            document_stats = self._calculate_document_stats(blocks)
        
        # Apply rules to each block
        analyzed_blocks = []
        for block in blocks:
            rule_analysis = self._analyze_block_with_rules(block, blocks, document_stats)
            
            # Add rule analysis to block
            enhanced_block = block.copy()
            enhanced_block['rule_analysis'] = rule_analysis
            analyzed_blocks.append(enhanced_block)
        
        # Apply post-processing rules
        post_processed_blocks = self._apply_post_processing_rules(analyzed_blocks)
        
        return post_processed_blocks
    
    def _calculate_document_stats(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate document-level statistics for rule application"""
        if not blocks:
            return {}
        
        # Font size statistics
        font_sizes = []
        x_positions = []
        y_positions = []
        page_widths = []
        
        for block in blocks:
            font_info = block.get('font_info', {})
            if font_info.get('size'):
                font_sizes.append(font_info['size'])
            
            bbox = block.get('bbox', [0, 0, 0, 0])
            x_positions.append(bbox[0])
            y_positions.append(bbox[1])
            
            # Estimate page width from rightmost position
            if len(bbox) >= 3:
                page_widths.append(bbox[2])
        
        stats = {
            'font_stats': {
                'mean': np.mean(font_sizes) if font_sizes else 12,
                'std': np.std(font_sizes) if font_sizes else 2,
                'percentiles': {
                    50: np.percentile(font_sizes, 50) if font_sizes else 12,
                    75: np.percentile(font_sizes, 75) if font_sizes else 14,
                    90: np.percentile(font_sizes, 90) if font_sizes else 16,
                    95: np.percentile(font_sizes, 95) if font_sizes else 18
                }
            },
            'layout_stats': {
                'left_margin': np.min(x_positions) if x_positions else 0,
                'typical_x': np.median(x_positions) if x_positions else 0,
                'page_width': np.max(page_widths) if page_widths else 612,
                'page_height': np.max(y_positions) if y_positions else 792
            },
            'block_count': len(blocks),
            'pages': len(set(block.get('page', 1) for block in blocks))
        }
        
        return stats
    
    def _analyze_block_with_rules(self, block: Dict[str, Any], 
                                all_blocks: List[Dict[str, Any]], 
                                document_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all rule categories to a single block"""
        text = block.get('text', '').strip()
        if not text:
            return {'is_heading': False, 'confidence': 0.0, 'suggested_level': None}
        
        rule_scores = {}
        
        # Font-based rules
        font_score, font_level = self._apply_font_rules(block, document_stats)
        rule_scores['font'] = {
            'score': font_score,
            'suggested_level': font_level,
            'weight': self.rule_weights['font_size']
        }
        
        # Style-based rules
        style_score, style_boost = self._apply_style_rules(block)
        rule_scores['style'] = {
            'score': style_score,
            'confidence_boost': style_boost,
            'weight': self.rule_weights['font_style']
        }
        
        # Position-based rules
        position_score, position_confidence = self._apply_position_rules(block, all_blocks, document_stats)
        rule_scores['position'] = {
            'score': position_score,
            'confidence': position_confidence,
            'weight': self.rule_weights['position']
        }
        
        # Pattern-based rules
        pattern_score, pattern_level, pattern_confidence = self._apply_pattern_rules(text)
        rule_scores['pattern'] = {
            'score': pattern_score,
            'suggested_level': pattern_level,
            'confidence': pattern_confidence,
            'weight': self.rule_weights['pattern']
        }
        
        # Context-based rules
        context_score, context_factors = self._apply_context_rules(block, all_blocks)
        rule_scores['context'] = {
            'score': context_score,
            'factors': context_factors,
            'weight': self.rule_weights['context']
        }
        
        # Combine all rule scores
        final_analysis = self._combine_rule_scores(rule_scores, text)
        final_analysis['detailed_scores'] = rule_scores
        
        return final_analysis
    
    def _apply_font_rules(self, block: Dict[str, Any], 
                         document_stats: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Apply font-based rules"""
        font_info = block.get('font_info', {})
        font_size = font_info.get('size', 12)
        
        font_stats = document_stats.get('font_stats', {})
        mean_size = font_stats.get('mean', 12)
        std_size = font_stats.get('std', 2)
        
        # Calculate size ratio and z-score
        size_ratio = font_size / mean_size if mean_size > 0 else 1.0
        z_score = (font_size - mean_size) / std_size if std_size > 0 else 0
        
        # Determine score and level based on size
        if size_ratio >= self.font_thresholds['h1_min_ratio'] or z_score >= 2.0:
            return 0.8, 'H1'
        elif size_ratio >= self.font_thresholds['h2_min_ratio'] or z_score >= 1.5:
            return 0.6, 'H2'
        elif size_ratio >= self.font_thresholds['h3_min_ratio'] or z_score >= 1.0:
            return 0.4, 'H3'
        else:
            return 0.0, None
    
    def _apply_style_rules(self, block: Dict[str, Any]) -> Tuple[float, float]:
        """Apply style-based rules (bold, italic, etc.)"""
        font_info = block.get('font_info', {})
        text = block.get('text', '').strip()
        word_count = len(text.split())
        
        score = 0.0
        confidence_boost = 0.0
        
        # Bold text
        if font_info.get('is_bold', False):
            score += 0.4
            confidence_boost += self.style_rules['bold']['confidence_boost']
        
        # Italic text
        if font_info.get('is_italic', False):
            score += 0.2
            confidence_boost += self.style_rules['italic']['confidence_boost']
        
        # All caps
        if text.isupper() and word_count <= self.style_rules['all_caps']['max_words']:
            score += 0.6
            confidence_boost += 0.2
        
        # Title case
        elif text.istitle() and word_count <= self.style_rules['title_case']['max_words']:
            score += 0.3
            confidence_boost += 0.1
        
        return score, confidence_boost
    
    def _apply_position_rules(self, block: Dict[str, Any], 
                            all_blocks: List[Dict[str, Any]], 
                            document_stats: Dict[str, Any]) -> Tuple[float, float]:
        """Apply position-based rules"""
        bbox = block.get('bbox', [0, 0, 0, 0])
        page = block.get('page', 1)
        
        layout_stats = document_stats.get('layout_stats', {})
        left_margin = layout_stats.get('left_margin', 0)
        page_width = layout_stats.get('page_width', 612)
        page_height = layout_stats.get('page_height', 792)
        
        score = 0.0
        confidence_factors = []
        
        # Left margin alignment
        distance_from_margin = abs(bbox[0] - left_margin)
        if distance_from_margin <= self.position_rules['left_margin_tolerance']:
            score += 0.3
            confidence_factors.append('left_aligned')
        
        # Centering
        block_center = (bbox[0] + bbox[2]) / 2
        page_center = page_width / 2
        center_distance = abs(block_center - page_center)
        center_tolerance = page_width * self.position_rules['center_tolerance_ratio']
        
        if center_distance <= center_tolerance:
            score += 0.4
            confidence_factors.append('centered')
        
        # Whitespace before block
        whitespace_before = self._calculate_whitespace_before(block, all_blocks)
        if whitespace_before >= self.position_rules['whitespace_threshold']:
            score += 0.2
            confidence_factors.append('whitespace_before')
        
        # Position on page (top sections more likely to be headings)
        relative_y = bbox[1] / page_height if page_height > 0 else 0.5
        if relative_y <= self.position_rules['top_page_ratio']:
            score += 0.1
            confidence_factors.append('top_of_page')
        
        confidence = len(confidence_factors) * 0.2
        return score, confidence
    
    def _apply_pattern_rules(self, text: str) -> Tuple[float, Optional[str], float]:
        """Apply pattern-based rules"""
        text_lower = text.lower().strip()
        best_score = 0.0
        best_level = None
        best_confidence = 0.0
        
        # Check all pattern categories
        for category, rule_set in self.pattern_rules.items():
            patterns = rule_set['patterns']
            weight = rule_set['weight']
            
            for pattern, level, confidence in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    weighted_score = confidence * weight
                    if weighted_score > best_score:
                        best_score = weighted_score
                        best_level = level
                        best_confidence = confidence
        
        return best_score, best_level, best_confidence
    
    def _apply_context_rules(self, block: Dict[str, Any], 
                           all_blocks: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
        """Apply context-based rules"""
        score = 0.0
        factors = []
        
        # Find blocks on same page
        page = block.get('page', 1)
        page_blocks = [b for b in all_blocks if b.get('page') == page]
        page_blocks.sort(key=lambda x: x.get('bbox', [0, 0, 0, 0])[1])  # Sort by y position
        
        # Find current block index
        current_bbox = block.get('bbox', [0, 0, 0, 0])
        block_index = -1
        for i, b in enumerate(page_blocks):
            b_bbox = b.get('bbox', [0, 0, 0, 0])
            if (abs(b_bbox[0] - current_bbox[0]) < 5 and 
                abs(b_bbox[1] - current_bbox[1]) < 5):
                block_index = i
                break
        
        if block_index >= 0:
            # Check for consistent indentation with other potential headings
            similar_x_blocks = [
                b for b in page_blocks 
                if abs(b.get('bbox', [0, 0, 0, 0])[0] - current_bbox[0]) < 10
            ]
            
            if len(similar_x_blocks) >= 2:
                score += 0.2
                factors.append('consistent_indentation')
            
            # Check if followed by indented text
            if block_index < len(page_blocks) - 1:
                next_block = page_blocks[block_index + 1]
                next_bbox = next_block.get('bbox', [0, 0, 0, 0])
                
                if next_bbox[0] > current_bbox[0] + 10:  # Next block is indented
                    score += 0.3
                    factors.append('followed_by_indented_text')
        
        return score, factors
    
    def _combine_rule_scores(self, rule_scores: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Combine all rule scores into final decision"""
        # Calculate weighted score
        total_score = 0.0
        total_weight = 0.0
        confidence_boosts = 0.0
        suggested_levels = []
        
        for rule_type, rule_data in rule_scores.items():
            score = rule_data.get('score', 0.0)
            weight = rule_data.get('weight', 0.0)
            
            total_score += score * weight
            total_weight += weight
            
            # Collect confidence boosts
            if 'confidence_boost' in rule_data:
                confidence_boosts += rule_data['confidence_boost']
            if 'confidence' in rule_data:
                confidence_boosts += rule_data['confidence']
            
            # Collect level suggestions
            if rule_data.get('suggested_level'):
                suggested_levels.append(rule_data['suggested_level'])
        
        # Normalize score
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Determine if it's a heading
        is_heading = final_score >= 0.3  # Threshold for heading detection
        
        # Determine suggested level
        suggested_level = None
        if suggested_levels:
            # Use most common suggestion, preferring pattern-based suggestions
            pattern_level = rule_scores.get('pattern', {}).get('suggested_level')
            if pattern_level:
                suggested_level = pattern_level
            else:
                # Use most common level
                level_counts = {}
                for level in suggested_levels:
                    level_counts[level] = level_counts.get(level, 0) + 1
                suggested_level = max(level_counts, key=level_counts.get)
        
        # Calculate final confidence
        base_confidence = min(1.0, final_score + confidence_boosts)
        
        # Adjust confidence based on text characteristics
        word_count = len(text.split())
        if 2 <= word_count <= 12:
            base_confidence += 0.1  # Good length for headings
        elif word_count > 20:
            base_confidence -= 0.2  # Too long for typical heading
        
        final_confidence = max(0.0, min(1.0, base_confidence))
        
        return {
            'is_heading': is_heading,
            'confidence': final_confidence,
            'suggested_level': suggested_level,
            'raw_score': final_score,
            'contributing_rules': [
                rule_type for rule_type, data in rule_scores.items() 
                if data.get('score', 0) > 0.1
            ]
        }
    
    def _calculate_whitespace_before(self, block: Dict[str, Any], 
                                   all_blocks: List[Dict[str, Any]]) -> float:
        """Calculate whitespace before current block"""
        current_bbox = block.get('bbox', [0, 0, 0, 0])
        page = block.get('page', 1)
        
        # Find previous block on same page
        page_blocks = [b for b in all_blocks if b.get('page') == page]
        page_blocks.sort(key=lambda x: x.get('bbox', [0, 0, 0, 0])[1])  # Sort by y position
        
        for i, b in enumerate(page_blocks):
            b_bbox = b.get('bbox', [0, 0, 0, 0])
            if (abs(b_bbox[0] - current_bbox[0]) < 5 and 
                abs(b_bbox[1] - current_bbox[1]) < 5):
                if i > 0:
                    prev_block = page_blocks[i - 1]
                    prev_bbox = prev_block.get('bbox', [0, 0, 0, 0])
                    return current_bbox[1] - prev_bbox[3]  # Gap between blocks
                break
        
        return 0.0
    
    def _apply_post_processing_rules(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply post-processing rules to ensure consistency"""
        if not blocks:
            return blocks
        
        # Sort blocks by page and position
        blocks.sort(key=lambda x: (x.get('page', 1), x.get('bbox', [0, 0, 0, 0])[1]))
        
        # Apply hierarchy consistency rules
        blocks = self._enforce_hierarchy_consistency(blocks)
        
        # Apply minimum heading requirements
        blocks = self._apply_minimum_heading_rules(blocks)
        
        # Remove unlikely headings
        blocks = self._filter_unlikely_headings(blocks)
        
        return blocks
    
    def _enforce_hierarchy_consistency(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure heading hierarchy is logical"""
        last_level = 0
        
        for block in blocks:
            rule_analysis = block.get('rule_analysis', {})
            if not rule_analysis.get('is_heading', False):
                continue
            
            suggested_level = rule_analysis.get('suggested_level')
            if not suggested_level:
                continue
            
            current_level = int(suggested_level[1])  # Extract number from 'H1', 'H2', etc.
            
            # Don't allow jumping more than one level
            if last_level > 0 and current_level > last_level + 1:
                # Adjust level to maintain hierarchy
                adjusted_level = last_level + 1
                rule_analysis['suggested_level'] = f'H{adjusted_level}'
                rule_analysis['hierarchy_adjusted'] = True
                current_level = adjusted_level
            
            last_level = current_level
        
        return blocks
    
    def _apply_minimum_heading_rules(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply minimum requirements for heading detection"""
        for block in blocks:
            rule_analysis = block.get('rule_analysis', {})
            if not rule_analysis.get('is_heading', False):
                continue
            
            text = block.get('text', '').strip()
            word_count = len(text.split())
            
            # Minimum word count
            if word_count < 1:
                rule_analysis['is_heading'] = False
                rule_analysis['rejection_reason'] = 'too_short'
            
            # Maximum word count
            elif word_count > 25:
                rule_analysis['is_heading'] = False  
                rule_analysis['rejection_reason'] = 'too_long'
            
            # Minimum confidence threshold
            elif rule_analysis.get('confidence', 0) < 0.3:
                rule_analysis['is_heading'] = False
                rule_analysis['rejection_reason'] = 'low_confidence'
        
        return blocks
    
    def _filter_unlikely_headings(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out blocks that are unlikely to be headings"""
        for block in blocks:
            rule_analysis = block.get('rule_analysis', {})
            if not rule_analysis.get('is_heading', False):
                continue
            
            text = block.get('text', '').strip().lower()
            
            # Skip common non-heading patterns
            unlikely_patterns = [
                r'^\d+$',  # Just numbers
                r'^page \d+',  # Page numbers
                r'^\d+ of \d+$',  # Page indicators
                r'^copyright',  # Copyright notices
                r'^©',  # Copyright symbol
                r'^\w+\.(com|org|net)',  # URLs
            ]
            
            for pattern in unlikely_patterns:
                if re.match(pattern, text):
                    rule_analysis['is_heading'] = False
                    rule_analysis['rejection_reason'] = 'unlikely_pattern'
                    break
        
        return blocks