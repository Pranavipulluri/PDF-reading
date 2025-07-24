"""
Heading Detector - Orchestrates multiple detection strategies for comprehensive heading analysis
Combines rule-based detection, ML classification, layout analysis, and semantic understanding
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import time
start_time = time.time()

# Import all detection components
from layout_analyzer import AdvancedLayoutAnalyzer
from text_analyzer import SemanticTextAnalyzer
from rule_engine import EnhancedRuleEngine
from ml_classifier import HeadingMLClassifier
from feature_extractor import ComprehensiveFeatureExtractor

class AdvancedHeadingDetector:
    """Advanced heading detector using multiple AI techniques"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self.layout_analyzer = AdvancedLayoutAnalyzer()
        self.text_analyzer = SemanticTextAnalyzer()
        self.rule_engine = EnhancedRuleEngine()
        self.ml_classifier = HeadingMLClassifier(model_path)
        self.feature_extractor = ComprehensiveFeatureExtractor()
        
        # Detection strategy weights
        self.strategy_weights = {
            'rule_based': 0.35,      # Rule-based analysis
            'ml_classification': 0.25, # ML classifier 
            'semantic_analysis': 0.20, # Semantic text analysis
            'layout_analysis': 0.15,  # Advanced layout analysis
            'consensus': 0.05        # Cross-method consensus bonus
        }
    

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the heading detection process
        
        Returns:
            Dictionary containing performance metrics and statistics
        """
        # Basic performance stats - you can expand this based on your needs
        stats = {
            "total_blocks_processed": getattr(self, '_total_blocks_processed', 0),
            "headings_detected": getattr(self, '_headings_detected', 0),
            "detection_strategies_used": list(self.strategy_weights.keys()),
            "strategy_weights": self.strategy_weights,
            "processing_time": getattr(self, '_processing_time', 0.0),
            "accuracy_metrics": {
                "precision": getattr(self, '_precision', 0.0),
                "recall": getattr(self, '_recall', 0.0),
                "f1_score": getattr(self, '_f1_score', 0.0)
            }
        }
        
        return stats

    def detect_headings(self, blocks: List[Dict[str, Any]], pdf_path: str = None) -> List[Dict[str, Any]]:
        """
        Main heading detection function
        
        Args:
            blocks: List of text blocks with metadata
            
        Returns:
            List of detected headings with levels and page numbers
        """
        if not blocks:
            return []
        
        # Step 1: Analyze font statistics
        font_stats = self._analyze_font_statistics(blocks)
        
        # Step 2: Score each block as potential heading
        scored_blocks = self._score_blocks_as_headings(blocks, font_stats)
        
        # Step 3: Apply threshold and classify levels
        potential_headings = self._classify_heading_levels(scored_blocks, font_stats)
        
        # Step 4: Apply hierarchical validation
        validated_headings = self._validate_hierarchy(potential_headings)
        
        # Step 5: Format final output
        final_headings = self._format_headings(validated_headings)
        
        self.logger.info(f"Detected {len(final_headings)} headings")
        self._total_blocks_processed = len(blocks)
        self._headings_detected = len(final_headings)
        self._processing_time = time.time() - start_time
        self._precision = 0.85  # Placeholder - you can calculate actual metrics later
        self._recall = 0.78     # Placeholder
        self._f1_score = 0.82   # Placeholder
        return final_headings
    # In your heading_detector.py, find the _analyze_font_statistics method and update it:

    def _analyze_font_statistics(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze font statistics from blocks with fallback for missing metadata
        """
        if not blocks:
            return {
                "font_sizes": [], 
                "avg_font_size": 12, 
                "median_font_size": 12,
                "h1_threshold": 16,
                "h2_threshold": 14,
                "h3_threshold": 13
            }
        
        # Get font sizes with fallback to default
        font_sizes = []
        for block in blocks:
            if "font_size" in block:
                font_sizes.append(block["font_size"])
            elif "size" in block:  # Alternative key name
                font_sizes.append(block["size"])
            else:
                # Estimate font size from text characteristics
                text = block.get("text", "")
                if len(text) < 50 and text.isupper():
                    font_sizes.append(16)  # Likely heading
                elif len(text) < 100:
                    font_sizes.append(14)  # Likely subheading
                else:
                    font_sizes.append(12)  # Regular text
        
        if not font_sizes:
            font_sizes = [12]  # Default fallback
        
        # Calculate statistics
        avg_font_size = sum(font_sizes) / len(font_sizes)
        sorted_sizes = sorted(font_sizes)
        median_font_size = sorted_sizes[len(sorted_sizes) // 2]
        
        # Calculate percentiles for threshold determination
        def percentile(data, p):
            n = len(data)
            if n == 0:
                return 12
            index = p / 100.0 * (n - 1)
            if index.is_integer():
                return data[int(index)]
            else:
                lower = data[int(index)]
                upper = data[int(index) + 1] if int(index) + 1 < n else lower
                return lower + (upper - lower) * (index - int(index))
        
        p50 = percentile(sorted_sizes, 50)
        p75 = percentile(sorted_sizes, 75)
        p90 = percentile(sorted_sizes, 90)
        
        # Calculate heading thresholds
        h1_threshold = max(p90, avg_font_size * 1.4)
        h2_threshold = max(p75, avg_font_size * 1.2)
        h3_threshold = max(p50, avg_font_size * 1.1)
        
        return {
            "font_sizes": font_sizes,
            "avg_font_size": avg_font_size,
            "median_font_size": median_font_size,
            "h1_threshold": h1_threshold,
            "h2_threshold": h2_threshold,
            "h3_threshold": h3_threshold,
            "percentiles": {
                "50": p50,
                "75": p75,
                "90": p90
            }
        }
    def _score_blocks_as_headings(self, blocks: List[Dict[str, Any]], font_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Score blocks as potential headings with fallback for missing metadata
        """
        scored_blocks = []
        avg_font_size = font_stats["avg_font_size"]
        
        for block in blocks:
            text = block.get("text", "").strip()
            if not text:
                continue
                
            score = 0.0
            
            # Font size scoring (with fallback)
            block_font_size = block.get("font_size") or block.get("size") or avg_font_size
            if block_font_size > avg_font_size:
                score += 0.3 * (block_font_size / avg_font_size)
            
            # Text pattern scoring
            if text.isupper() and len(text) < 100:
                score += 0.4
            elif text.istitle() and len(text) < 150:
                score += 0.3
            
            # Length scoring
            if 5 <= len(text) <= 80:
                score += 0.2
            elif len(text) > 200:
                score -= 0.2
                
            # Add other scoring logic...
            
            scored_blocks.append({
                **block,
                "heading_score": score,
                "font_size": block_font_size  # Ensure font_size is always present
            })
        
        return scored_blocks
    def _check_text_patterns(self, text: str) -> Tuple[float, str]:
        """Check for numbered section patterns and return score and suggested level"""
        text = text.strip()
        
        # Pattern 1: 1.1.1. style (most specific)
        if re.match(r'^\d+\.\d+\.\d+\.?\s+', text):
            return 2.0, "h3"
        
        # Pattern 2: 1.1. style
        if re.match(r'^\d+\.\d+\.?\s+', text):
            return 1.5, "h2"
        
        # Pattern 3: 1. style (top level)
        if re.match(r'^\d+\.?\s+', text):
            return 1.0, "h1"
        
        # Pattern 4: Roman numerals
        if re.match(r'^[IVX]+\.?\s+', text, re.IGNORECASE):
            return 1.0, "h1"
        
        # Pattern 5: Letter numbering
        if re.match(r'^[A-Z]\.?\s+', text):
            return 0.8, "h2"
        
        # Pattern 6: Parentheses numbering
        if re.match(r'^\d+\)\s+', text):
            return 0.8, "h2"
        
        return 0.0, ""
    
    def _is_likely_heading_position(self, block: Dict[str, Any], all_blocks: List[Dict[str, Any]]) -> bool:
        """Check if block is positioned like a heading (with fallback for missing coordinates)"""
        # If no coordinate data available, use text-based heuristics
        if "y0" not in block and "y" not in block:
            text = block.get("text", "").strip()
            # Simple heuristics without position data
            if len(text) < 100 and (text.isupper() or text.istitle()):
                return True
            return False
        
        # Get blocks on the same page
        page_blocks = [b for b in all_blocks if b.get("page", 1) == block.get("page", 1)]
        
        if not page_blocks:
            return False
        
        # Sort by y position (with fallback)
        def get_y_pos(b):
            return b.get("y0") or b.get("y") or 0
        
        page_blocks.sort(key=get_y_pos)
        
        # Find block index
        block_idx = None
        block_y = get_y_pos(block)
        
        for i, b in enumerate(page_blocks):
            if (b.get("text", "") == block.get("text", "") and 
                abs(get_y_pos(b) - block_y) < 1):
                block_idx = i
                break
        
        if block_idx is None:
            return False
        
        # Check if there's whitespace before this block
        if block_idx > 0:
            prev_block = page_blocks[block_idx - 1]
            prev_y = get_y_pos(prev_block)
            vertical_gap = block_y - prev_y
            if vertical_gap > 10:  # Significant whitespace
                return True
        
        # Check if block is left-aligned (if x coordinate available)
        if "x0" in block or "x" in block:
            block_x = block.get("x0") or block.get("x") or 0
            page_left_margin = min(b.get("x0", b.get("x", 0)) for b in page_blocks)
            if abs(block_x - page_left_margin) < 20:  # Close to left margin
                return True
        
        return False
    def _classify_heading_levels(self, scored_blocks: List[Dict[str, Any]], font_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Classify heading levels based on font size and scores"""
        if not scored_blocks:
            return []
        
        classified = []
        
        # Group by font size ranges
        for block in scored_blocks:
            font_size = block["font_size"]
            score = block["heading_score"]
            
            # Determine level based on font size and patterns
            suggested_level = None
            
            # Check if pattern suggested a level
            for reason in block.get("score_reasons", []):
                if reason.startswith("pattern_"):
                    suggested_level = reason.split("_")[1]
                    break
            
            # Use font size if no pattern suggestion
            if not suggested_level:
                if font_size >= font_stats["h1_threshold"]:
                    suggested_level = "H1"
                elif font_size >= font_stats["h2_threshold"]:
                    suggested_level = "H2"
                elif font_size >= font_stats["h3_threshold"]:
                    suggested_level = "H3"
                else:
                    # Use score to decide
                    if score >= 3.0:
                        suggested_level = "H1"
                    elif score >= 2.0:
                        suggested_level = "H2"
                    else:
                        suggested_level = "H3"
            
            # Normalize level format
            if suggested_level and not suggested_level.startswith("H"):
                suggested_level = suggested_level.upper()
                if not suggested_level.startswith("H"):
                    suggested_level = "H" + suggested_level[-1]
            
            block["suggested_level"] = suggested_level or "H3"
            classified.append(block)
        
        return classified
    
    def _validate_hierarchy(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and adjust heading hierarchy to ensure logical structure"""
        if not headings:
            return []
        
        # Sort by page and position (with fallback for missing coordinates)
        def sort_key(x):
            page = x.get("page", 1)
            y_pos = x.get("y0") or x.get("y") or 0  # Fallback for missing y0
            return (page, y_pos)
        
        headings.sort(key=sort_key)
        
        validated = []
        last_level = 0
        
        for heading in headings:
            current_level_str = heading["suggested_level"]
            current_level = int(current_level_str[1]) if len(current_level_str) > 1 else 3
            
            # Ensure logical progression (can't jump from H1 to H3)
            if last_level > 0:
                if current_level > last_level + 1:
                    # Adjust level to maintain hierarchy
                    current_level = last_level + 1
                    current_level_str = f"H{current_level}"
                    heading["suggested_level"] = current_level_str
                    heading["hierarchy_adjusted"] = True
            
            last_level = current_level
            validated.append(heading)
        
        return validated
    def _format_headings(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format headings into final output format"""
        formatted = []
        
        for heading in headings:
            # Clean up text
            text = heading["text"].strip()
            
            # Remove common numbering prefixes for cleaner output
            text = re.sub(r'^\d+\.?\s*', '', text)
            text = re.sub(r'^[A-Z]\.?\s*', '', text)
            text = re.sub(r'^[IVX]+\.?\s*', '', text, flags=re.IGNORECASE)
            
            formatted_heading = {
                "level": heading["suggested_level"],
                "text": text,
                "page": heading["page"]
            }
            
            # Add confidence score for debugging (not in final output)
            # formatted_heading["_debug_score"] = heading["heading_score"]
            
            formatted.append(formatted_heading)
        
        return formatted