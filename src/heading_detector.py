"""
Heading Detector - Orchestrates multiple detection strategies for comprehensive heading analysis
Combines rule-based detection, ML classification, layout analysis, and semantic understanding
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

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
    
    def detect_headings(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        return final_headings
    
    def _analyze_font_statistics(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze font size distribution and calculate thresholds"""
        font_sizes = [block["font_size"] for block in blocks]
        
        if not font_sizes:
            return {"avg_size": 12.0, "std_size": 0, "percentiles": {}}
        
        stats = {
            "avg_size": np.mean(font_sizes),
            "std_size": np.std(font_sizes),
            "min_size": np.min(font_sizes),
            "max_size": np.max(font_sizes),
            "percentiles": {
                "50": np.percentile(font_sizes, 50),
                "75": np.percentile(font_sizes, 75),
                "90": np.percentile(font_sizes, 90),
                "95": np.percentile(font_sizes, 95)
            }
        }
        
        # Calculate heading thresholds based on font size distribution
        stats["h1_threshold"] = max(stats["percentiles"]["90"], stats["avg_size"] * 1.4)
        stats["h2_threshold"] = max(stats["percentiles"]["75"], stats["avg_size"] * 1.2)
        stats["h3_threshold"] = max(stats["percentiles"]["50"], stats["avg_size"] * 1.1)
        
        return stats
    
    def _score_blocks_as_headings(self, blocks: List[Dict[str, Any]], 
                                font_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Score each text block based on heading likelihood"""
        scored_blocks = []
        
        for block in blocks:
            score = 0.0
            reasons = []
            
            text = block["text"].strip()
            font_size = block["font_size"]
            is_bold = block["is_bold"]
            is_italic = block["is_italic"]
            
            # Skip very long text (likely paragraphs)
            if len(text) > 200:
                continue
            
            # Font size scoring
            if font_size >= font_stats["h1_threshold"]:
                score += 3.0
                reasons.append("large_font")
            elif font_size >= font_stats["h2_threshold"]:
                score += 2.0
                reasons.append("medium_font")
            elif font_size >= font_stats["h3_threshold"]:
                score += 1.0
                reasons.append("above_avg_font")
            
            # Bold text bonus
            if is_bold:
                score += 1.5
                reasons.append("bold")
            
            # Pattern matching bonuses
            pattern_score, pattern_level = self._check_text_patterns(text)
            score += pattern_score
            if pattern_score > 0:
                reasons.append(f"pattern_{pattern_level}")
            
            # Length penalty/bonus
            word_count = len(text.split())
            if 2 <= word_count <= 10:  # Ideal heading length
                score += 0.5
                reasons.append("good_length")
            elif word_count > 20:  # Too long for heading
                score -= 1.0
                reasons.append("too_long")
            
            # Position bonuses
            if self._is_likely_heading_position(block, blocks):
                score += 0.5
                reasons.append("good_position")
            
            # Capitalization patterns
            if text.isupper() and word_count <= 8:
                score += 1.0
                reasons.append("all_caps")
            elif text.istitle():
                score += 0.3
                reasons.append("title_case")
            
            # Content-based scoring
            if any(indicator in text.lower() for indicator in self.heading_indicators):
                score += 0.5
                reasons.append("heading_keyword")
            
            # Add to results if score is above threshold
            if score >= 1.0:  # Minimum threshold for consideration
                block_copy = block.copy()
                block_copy["heading_score"] = score
                block_copy["score_reasons"] = reasons
                scored_blocks.append(block_copy)
        
        # Sort by score descending
        scored_blocks.sort(key=lambda x: x["heading_score"], reverse=True)
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
        """Check if block is positioned like a heading"""
        # Get blocks on the same page
        page_blocks = [b for b in all_blocks if b["page"] == block["page"]]
        
        if not page_blocks:
            return False
        
        # Sort by y position
        page_blocks.sort(key=lambda x: x["y0"])
        
        # Find block index
        block_idx = None
        for i, b in enumerate(page_blocks):
            if (b["text"] == block["text"] and 
                abs(b["y0"] - block["y0"]) < 1):
                block_idx = i
                break
        
        if block_idx is None:
            return False
        
        # Check if there's whitespace before this block
        if block_idx > 0:
            prev_block = page_blocks[block_idx - 1]
            vertical_gap = block["y0"] - prev_block["y1"]
            if vertical_gap > 10:  # Significant whitespace
                return True
        
        # Check if block is left-aligned (not indented much)
        page_left_margin = min(b["x0"] for b in page_blocks)
        if abs(block["x0"] - page_left_margin) < 20:  # Close to left margin
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
        
        # Sort by page and position
        headings.sort(key=lambda x: (x["page"], x["y0"]))
        
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