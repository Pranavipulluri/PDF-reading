"""
Enhanced Title Extractor - Smart multi-candidate title detection with advanced scoring
Uses position analysis, font scoring, and content analysis for robust title extraction
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class SmartTitleExtractor:
    """Enhanced title extractor with multi-candidate scoring and smart positioning"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Enhanced title patterns with confidence scores
        self.title_patterns = [
            (r'^title[:\s]+(.+)', 0.9),
            (r'^(.+)\s*[-–—]\s*(?:report|study|analysis|guide|manual)', 0.8),
            (r'^(.+?)\s*\n\s*(?:a\s+)?(?:report|study|analysis)', 0.7),
            (r'^(.+?)\s*:\s*(?:an?\s+)?(?:overview|introduction|guide)', 0.6),
        ]
        
        # Title quality indicators
        self.title_quality_indicators = {
            'good_words': {
                'analysis', 'study', 'report', 'guide', 'manual', 'handbook',
                'overview', 'introduction', 'survey', 'review', 'proceedings',
                'annual', 'quarterly', 'white paper', 'technical', 'research',
                'comprehensive', 'complete', 'advanced', 'practical', 'essential'
            },
            'avoid_words': {
                'page', 'copyright', '©', 'confidential', 'draft', 'version',
                'printed', 'published', 'author', 'date', 'file', 'document'
            },
            'punctuation_penalties': {
                '.': -0.2,  # Titles usually don't end with periods
                ',': -0.1,  # Commas are less common
                ';': -0.3,  # Semicolons are rare in titles
                ':': 0.1,   # Colons can be good for subtitles
                '?': -0.1,  # Questions less common in titles
                '!': -0.2,  # Exclamations rare in formal titles
            }
        }
    
    def extract_title(self, blocks: List[Dict[str, Any]], fallback_title: str = "") -> str:
        """
        Smart title extraction using multi-candidate analysis
        
        Args:
            blocks: List of text blocks from PDF
            fallback_title: Fallback title (usually filename)
            
        Returns:
            Extracted or inferred title
        """
        if not blocks:
            return fallback_title or "Untitled Document"
        
        self.logger.info("Starting smart title extraction with multi-candidate analysis")
        
        # Step 1: Find title candidates from first 2-3 pages
        candidates = self._find_title_candidates(blocks)
        
        if not candidates:
            self.logger.warning("No title candidates found, using fallback")
            return fallback_title or "Untitled Document"
        
        # Step 2: Score each candidate
        scored_candidates = self._score_title_candidates(candidates)
        
        # Step 3: Select best candidate
        best_candidate = self._select_best_candidate(scored_candidates)
        
        # Step 4: Clean and validate final title
        final_title = self._clean_and_validate_title(best_candidate['text'])
        
        self.logger.info(f"Selected title: '{final_title}' (score: {best_candidate['total_score']:.3f}, method: {best_candidate['source']})")
        
        return final_title
    
    def _find_title_candidates(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find potential title candidates from first few pages"""
        
        # Only look at first 2-3 pages for titles
        first_pages = [b for b in blocks if b.get('page', 1) <= 3]
        
        if not first_pages:
            return []
        
        candidates = []
        
        # Strategy 1: Largest font size on first page
        page_1_blocks = [b for b in first_pages if b.get('page', 1) == 1]
        if page_1_blocks:
            candidates.extend(self._find_font_based_candidates(page_1_blocks))
        
        # Strategy 2: Centered text on first few pages
        candidates.extend(self._find_position_based_candidates(first_pages))
        
        # Strategy 3: Pattern-based detection
        candidates.extend(self._find_pattern_based_candidates(first_pages))
        
        # Strategy 4: Content analysis
        candidates.extend(self._find_content_based_candidates(first_pages))
        
        # Strategy 5: Top-of-page candidates
        candidates.extend(self._find_top_position_candidates(first_pages))
        
        # Remove duplicates and sort by page/position
        candidates = self._deduplicate_candidates(candidates)
        
        self.logger.info(f"Found {len(candidates)} title candidates")
        return candidates
    
    def _find_font_based_candidates(self, page_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find candidates based on font size analysis"""
        
        if not page_blocks:
            return []
        
        # Get font sizes
        font_sizes = [b.get('font_info', {}).get('size', 12) for b in page_blocks]
        max_font_size = max(font_sizes) if font_sizes else 12
        
        # Find blocks with largest fonts (within tolerance)
        size_tolerance = 1.0  # Allow slight variations
        large_font_blocks = [
            b for b in page_blocks 
            if b.get('font_info', {}).get('size', 12) >= max_font_size - size_tolerance
        ]
        
        candidates = []
        for block in large_font_blocks:
            text = block.get('text', '').strip()
            word_count = len(text.split())
            
            # Filter for reasonable title characteristics
            if 2 <= word_count <= 20 and len(text) <= 200:
                candidate = {
                    'text': text,
                    'source': 'font_size',
                    'page': block.get('page', 1),
                    'bbox': block.get('bbox', [0, 0, 0, 0]),
                    'font_info': block.get('font_info', {}),
                    'metadata': {
                        'font_size': block.get('font_info', {}).get('size', 12),
                        'max_font_size': max_font_size,
                        'word_count': word_count
                    }
                }
                candidates.append(candidate)
        
        return candidates
    
    def _find_position_based_candidates(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find candidates based on position analysis"""
        
        candidates = []
        
        # Group by page for position analysis
        pages = {}
        for block in blocks:
            page_num = block.get('page', 1)
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(block)
        
        for page_num, page_blocks in pages.items():
            if page_num > 2:  # Only first 2 pages
                continue
            
            # Calculate page dimensions
            if not page_blocks:
                continue
            
            x_positions = [b.get('bbox', [0, 0, 0, 0])[0] for b in page_blocks]
            x_max_positions = [b.get('bbox', [0, 0, 0, 0])[2] for b in page_blocks]
            y_positions = [b.get('bbox', [0, 0, 0, 0])[1] for b in page_blocks]
            
            if not x_positions:
                continue
                
            min_x = min(x_positions)
            max_x = max(x_max_positions)
            page_width = max_x - min_x
            page_center = (min_x + max_x) / 2
            
            # Find centered candidates
            for block in page_blocks:
                bbox = block.get('bbox', [0, 0, 0, 0])
                block_center = (bbox[0] + bbox[2]) / 2
                distance_from_center = abs(block_center - page_center)
                
                # Consider "centered" if within 25% of page width from center
                if distance_from_center < page_width * 0.25:
                    text = block.get('text', '').strip()
                    word_count = len(text.split())
                    
                    if 2 <= word_count <= 15 and len(text) <= 150:
                        candidate = {
                            'text': text,
                            'source': 'centered_position',
                            'page': page_num,
                            'bbox': bbox,
                            'font_info': block.get('font_info', {}),
                            'metadata': {
                                'center_distance': distance_from_center,
                                'page_width': page_width,
                                'center_ratio': distance_from_center / page_width,
                                'word_count': word_count
                            }
                        }
                        candidates.append(candidate)
        
        return candidates
    
    def _find_pattern_based_candidates(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find candidates using explicit title patterns"""
        
        candidates = []
        
        for block in blocks:
            text = block.get('text', '').strip()
            text_lower = text.lower()
            
            # Check each title pattern
            for pattern, confidence in self.title_patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                if match:
                    extracted_title = match.group(1).strip() if match.groups() else text
                    
                    if len(extracted_title.split()) >= 2:
                        candidate = {
                            'text': extracted_title,
                            'source': 'pattern_match',
                            'page': block.get('page', 1),
                            'bbox': block.get('bbox', [0, 0, 0, 0]),
                            'font_info': block.get('font_info', {}),
                            'metadata': {
                                'pattern_confidence': confidence,
                                'matched_pattern': pattern,
                                'original_text': text
                            }
                        }
                        candidates.append(candidate)
        
        return candidates
    
    def _find_content_based_candidates(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find candidates using content analysis"""
        
        candidates = []
        
        # Sort blocks by page and position
        sorted_blocks = sorted(blocks, key=lambda x: (x.get('page', 1), x.get('bbox', [0, 0, 0, 0])[1]))
        
        # Look at first 10 blocks for content-based titles
        for block in sorted_blocks[:10]:
            text = block.get('text', '').strip()
            word_count = len(text.split())
            
            if word_count < 2 or word_count > 20 or len(text) > 200:
                continue
            
            # Content quality scoring
            content_score = self._calculate_content_quality_score(text)
            
            if content_score > 0.3:  # Threshold for content-based candidates
                candidate = {
                    'text': text,
                    'source': 'content_analysis',
                    'page': block.get('page', 1),
                    'bbox': block.get('bbox', [0, 0, 0, 0]),
                    'font_info': block.get('font_info', {}),
                    'metadata': {
                        'content_score': content_score,
                        'word_count': word_count,
                        'text_length': len(text)
                    }
                }
                candidates.append(candidate)
        
        return candidates
    
    def _find_top_position_candidates(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find candidates in top positions of pages"""
        
        candidates = []
        
        # Group by page
        pages = {}
        for block in blocks:
            page_num = block.get('page', 1)
            if page_num <= 2:  # Only first 2 pages
                if page_num not in pages:
                    pages[page_num] = []
                pages[page_num].append(block)
        
        for page_num, page_blocks in pages.items():
            # Sort by y position (top to bottom)
            page_blocks.sort(key=lambda x: x.get('bbox', [0, 0, 0, 0])[1])
            
            # Consider top 30% of page
            if page_blocks:
                max_y = max(b.get('bbox', [0, 0, 0, 0])[3] for b in page_blocks)
                top_threshold = max_y * 0.3
                
                for block in page_blocks:
                    if block.get('bbox', [0, 0, 0, 0])[1] <= top_threshold:
                        text = block.get('text', '').strip()
                        word_count = len(text.split())
                        
                        if 3 <= word_count <= 15:
                            candidate = {
                                'text': text,
                                'source': 'top_position',
                                'page': page_num,
                                'bbox': block.get('bbox', [0, 0, 0, 0]),
                                'font_info': block.get('font_info', {}),
                                'metadata': {
                                    'y_position': block.get('bbox', [0, 0, 0, 0])[1],
                                    'top_threshold': top_threshold,
                                    'word_count': word_count
                                }
                            }
                            candidates.append(candidate)
        
        return candidates
    
    def _calculate_content_quality_score(self, text: str) -> float:
        """Calculate content quality score for title candidacy"""
        
        score = 0.0
        text_lower = text.lower()
        words = text.split()
        
        # Good word bonus
        good_word_count = sum(1 for word in self.title_quality_indicators['good_words'] 
                             if word in text_lower)
        score += good_word_count * 0.2
        
        # Avoid word penalty
        avoid_word_count = sum(1 for word in self.title_quality_indicators['avoid_words'] 
                              if word in text_lower)
        score -= avoid_word_count * 0.3
        
        # Punctuation analysis
        for punct, penalty in self.title_quality_indicators['punctuation_penalties'].items():
            if punct in text:
                score += penalty
        
        # Capitalization bonus
        if text[0].isupper():
            score += 0.1
        
        if text.istitle():
            score += 0.2
        
        # Length scoring
        word_count = len(words)
        if 4 <= word_count <= 10:
            score += 0.3
        elif 2 <= word_count <= 15:
            score += 0.1
        
        return max(0.0, score)
    
    def _score_title_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score all title candidates using comprehensive criteria"""
        
        scored_candidates = []
        
        for candidate in candidates:
            scores = {}
            
            # Source-based scoring
            source_scores = {
                'pattern_match': 0.9,
                'font_size': 0.7,
                'centered_position': 0.6,
                'content_analysis': 0.5,
                'top_position': 0.4
            }
            scores['source_score'] = source_scores.get(candidate['source'], 0.3)
            
            # Position scoring
            scores['position_score'] = self._calculate_position_score(candidate)
            
            # Font scoring
            scores['font_score'] = self._calculate_font_score(candidate)
            
            # Content scoring
            scores['content_score'] = self._calculate_content_quality_score(candidate['text'])
            
            # Length scoring
            scores['length_score'] = self._calculate_length_score(candidate['text'])
            
            # Page preference (first page is better)
            page_num = candidate.get('page', 1)
            scores['page_score'] = 1.0 if page_num == 1 else 0.7 if page_num == 2 else 0.4
            
            # Calculate weighted total score
            weights = {
                'source_score': 0.25,
                'position_score': 0.20,
                'font_score': 0.20,
                'content_score': 0.15,
                'length_score': 0.10,
                'page_score': 0.10
            }
            
            total_score = sum(scores[key] * weights[key] for key in scores.keys())
            
            scored_candidate = candidate.copy()
            scored_candidate['scores'] = scores
            scored_candidate['total_score'] = total_score
            
            scored_candidates.append(scored_candidate)
        
        # Sort by total score (highest first)
        scored_candidates.sort(key=lambda x: x['total_score'], reverse=True)
        
        return scored_candidates
    
    def _calculate_position_score(self, candidate: Dict[str, Any]) -> float:
        """Calculate position-based score"""
        
        score = 0.5  # Base score
        
        bbox = candidate.get('bbox', [0, 0, 0, 0])
        metadata = candidate.get('metadata', {})
        
        # Y position bonus (higher on page is better for titles)
        y_pos = bbox[1]
        if y_pos <= 100:  # Very top of page
            score += 0.3
        elif y_pos <= 200:  # Upper portion
            score += 0.2
        
        # Center alignment bonus
        if candidate['source'] == 'centered_position':
            center_ratio = metadata.get('center_ratio', 0.5)
            score += (1.0 - center_ratio) * 0.3
        
        return min(1.0, score)
    
    def _calculate_font_score(self, candidate: Dict[str, Any]) -> float:
        """Calculate font-based score"""
        
        score = 0.5  # Base score
        
        font_info = candidate.get('font_info', {})
        font_size = font_info.get('size', 12)
        
        # Size bonus
        if font_size >= 18:
            score += 0.4
        elif font_size >= 14:
            score += 0.3
        elif font_size >= 12:
            score += 0.1
        
        # Style bonuses
        if font_info.get('is_bold'):
            score += 0.2
        
        if font_info.get('is_italic'):
            score += 0.1  # Italic can be good for titles
        
        return min(1.0, score)
    
    def _calculate_length_score(self, text: str) -> float:
        """Calculate length-based score"""
        
        word_count = len(text.split())
        char_count = len(text)
        
        # Optimal length scoring
        if 4 <= word_count <= 8:
            return 1.0
        elif 2 <= word_count <= 12:
            return 0.8
        elif word_count <= 15:
            return 0.6
        elif word_count <= 20:
            return 0.3
        else:
            return 0.1  # Too long
    
    def _select_best_candidate(self, scored_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best candidate from scored list"""
        
        if not scored_candidates:
            return {'text': 'Untitled Document', 'total_score': 0.0, 'source': 'fallback'}
        
        # Log top candidates for debugging
        self.logger.debug("Top title candidates:")
        for i, candidate in enumerate(scored_candidates[:3]):
            self.logger.debug(f"  {i+1}. '{candidate['text'][:50]}...' "
                            f"(score: {candidate['total_score']:.3f}, source: {candidate['source']})")
        
        return scored_candidates[0]
    
    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate candidates"""
        
        seen_texts = set()
        unique_candidates = []
        
        for candidate in candidates:
            text_normalized = candidate['text'].lower().strip()
            
            if text_normalized not in seen_texts:
                seen_texts.add(text_normalized)
                unique_candidates.append(candidate)
        
        return unique_candidates
    


    def _clean_and_validate_title(self, title: str) -> str:
        """Clean and validate the final title."""
        
        if not title:
            return "Untitled Document"
        
        # Remove extra whitespace
        title = ' '.join(title.split())
        
        # Remove common prefixes like 'Title:'
        title = re.sub(r'^title[:\s]+', '', title, flags=re.IGNORECASE)
        
        # Remove file extensions
        title = re.sub(r'\.(pdf|doc|docx|txt)$', '', title, flags=re.IGNORECASE)
        
        # Fallback if title is empty after cleaning
        if not title.strip():
            return "Untitled Document"
        
        return title.strip()

        
    def extract_title(self, blocks: List[Dict[str, Any]], fallback_title: str = "") -> str:
        """
        Extract document title using multiple strategies
        
        Args:
            blocks: List of text blocks from PDF
            fallback_title: Fallback title (usually filename)
            
        Returns:
            Extracted or inferred title
        """
        if not blocks:
            return fallback_title or "Untitled Document"
        
        # Strategy 1: Look for largest font on first few pages
        font_based_title = self._find_title_by_font_size(blocks)
        
        # Strategy 2: Look for centered text on first page
        position_based_title = self._find_title_by_position(blocks)
        
        # Strategy 3: Look for explicit title patterns
        pattern_based_title = self._find_title_by_patterns(blocks)
        
        # Strategy 4: Look for document metadata or first significant text
        content_based_title = self._find_title_by_content(blocks)
        
        # Choose best title from candidates
        candidates = [
            (font_based_title, "font"),
            (position_based_title, "position"), 
            (pattern_based_title, "pattern"),
            (content_based_title, "content")
        ]
        
        best_title = self._select_best_title(candidates, fallback_title)
        
        # Clean up and validate title
        final_title = self._clean_title(best_title)
        
        self.logger.info(f"Extracted title: '{final_title}'")
        return final_title
    
    def _find_title_by_font_size(self, blocks: List[Dict[str, Any]]) -> Optional[str]:
        """Find title based on largest font size on first few pages"""
        # Only look at first 3 pages
        first_pages = [b for b in blocks if b["page"] <= 3]
        
        if not first_pages:
            return None
        
        # Find blocks with largest font size
        max_font_size = max(b["font_size"] for b in first_pages)
        largest_blocks = [b for b in first_pages if b["font_size"] == max_font_size]
        
        # Filter by reasonable title characteristics
        title_candidates = []
        for block in largest_blocks:
            text = block["text"].strip()
            word_count = len(text.split())
            
            # Title should be reasonable length
            if 2 <= word_count <= 20 and len(text) <= 200:
                # Prefer text from first page
                priority = 1 if block["page"] == 1 else 2
                title_candidates.append((text, priority, block["page"]))
        
        if title_candidates:
            # Sort by priority (page number) and return best
            title_candidates.sort(key=lambda x: (x[1], x[2]))
            return title_candidates[0][0]
        
        return None
    
    def _find_title_by_position(self, blocks: List[Dict[str, Any]]) -> Optional[str]:
        """Find title based on centered position on first page"""
        first_page = [b for b in blocks if b["page"] == 1]
        
        if not first_page:
            return None
        
        # Calculate page width
        min_x = min(b["x0"] for b in first_page)
        max_x = max(b["x1"] for b in first_page)
        page_width = max_x - min_x
        page_center = (min_x + max_x) / 2
        
        # Find blocks that are approximately centered
        centered_blocks = []
        for block in first_page:
            block_center = (block["x0"] + block["x1"]) / 2
            distance_from_center = abs(block_center - page_center)
            
            # Consider "centered" if within 20% of page width from center
            if distance_from_center < page_width * 0.2:
                text = block["text"].strip()
                word_count = len(text.split())
                
                # Filter for title-like characteristics
                if (2 <= word_count <= 15 and 
                    len(text) <= 150 and
                    not text.isdigit() and
                    block["y0"] < max(b["y0"] for b in first_page) * 0.5):  # Upper half of page
                    
                    centered_blocks.append((text, block["font_size"], block["y0"]))
        
        if centered_blocks:
            # Sort by font size (desc) and position (asc - higher on page)
            centered_blocks.sort(key=lambda x: (-x[1], x[2]))
            return centered_blocks[0][0]
        
        return None
    
    def _find_title_by_patterns(self, blocks: List[Dict[str, Any]]) -> Optional[str]:
        """Find title using explicit title patterns"""
        first_pages = [b for b in blocks if b["page"] <= 2]
        
        for block in first_pages:
            text = block["text"].strip()
            text_lower = text.lower()
            
            # Check for explicit title patterns
            for pattern in self.title_patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                if match:
                    title = match.group(1).strip()
                    if len(title.split()) >= 2:
                        return title
            
            # Check for lines starting with common title indicators
            if text_lower.startswith('title:'):
                title = text[6:].strip()
                if title:
                    return title
        
        return None
    
    def _find_title_by_content(self, blocks: List[Dict[str, Any]]) -> Optional[str]:
        """Find title based on content analysis"""
        first_page = [b for b in blocks if b["page"] == 1]
        
        if not first_page:
            return None
        
        # Sort by position (top to bottom)
        first_page.sort(key=lambda x: x["y0"])
        
        # Look for first substantial text that could be a title
        for block in first_page[:10]:  # Check first 10 blocks
            text = block["text"].strip()
            word_count = len(text.split())
            
            # Skip very short or very long text
            if word_count < 2 or word_count > 20 or len(text) > 200:
                continue
            
            # Skip common non-title text
            text_lower = text.lower()
            if any(skip in text_lower for skip in ['page', 'copyright', '©', 'abstract']):
                continue
            
            # Prefer text with title-like characteristics
            has_title_keywords = any(keyword in text_lower for keyword in self.title_keywords)
            is_capitalized = text[0].isupper() if text else False
            
            if has_title_keywords or is_capitalized:
                return text
        
        # Fallback: return first non-trivial text
        for block in first_page[:5]:
            text = block["text"].strip()
            word_count = len(text.split())
            if 3 <= word_count <= 15:
                return text
        
        return None
    
    def _select_best_title(self, candidates: List[tuple], fallback: str) -> str:
        """Select the best title from multiple candidates"""
        # Filter out None candidates
        valid_candidates = [(title, source) for title, source in candidates if title]
        
        if not valid_candidates:
            return fallback or "Untitled Document"
        
        # Scoring system for different sources
        source_scores = {
            "pattern": 4,    # Explicit title patterns are best
            "position": 3,   # Centered text is good
            "font": 2,       # Large font is decent
            "content": 1     # Content analysis is fallback
        }
        
        # Score each candidate
        scored_candidates = []
        for title, source in valid_candidates:
            base_score = source_scores.get(source, 0)
            
            # Additional scoring factors
            word_count = len(title.split())
            length_score = 1 if 3 <= word_count <= 10 else 0.5
            
            # Bonus for title-like keywords
            keyword_bonus = 0.5 if any(kw in title.lower() for kw in self.title_keywords) else 0
            
            total_score = base_score + length_score + keyword_bonus
            scored_candidates.append((title, total_score, source))
        
        # Sort by score and return best
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        best_title = scored_candidates[0][0]
        best_source = scored_candidates[0][2]
        
        self.logger.debug(f"Selected title from {best_source}: '{best_title}'")
        return best_title
    
    import re

    def _clean_title(self, title: str) -> str:
        """Clean and normalize the extracted title."""
        if not title:
            return "Untitled Document"

        # Remove extra whitespace
        title = ' '.join(title.split())

        # Remove common prefixes like 'Title:'
        title = re.sub(r'^title[:\s]+', '', title, flags=re.IGNORECASE)

        # Remove common file extensions if accidentally included
        title = re.sub(r'\.(pdf|doc|docx|txt)$', '', title, flags=re.IGNORECASE)

        # Capitalize first letter if needed
        if title and title[0].islower():
            title = title[0].upper() + title[1:]

        # Length limit with smart truncation at word boundary
        if len(title) > 100:
            words = title[:97].split()
            if len(words) > 1:
                title = ' '.join(words[:-1]) + "..."
            else:
                title = title[:97] + "..."

        return title

    
    def extract_title(self, blocks: List[Dict[str, Any]], fallback_title: str = "") -> str:
        """
        Extract document title using multiple strategies
        
        Args:
            blocks: List of text blocks from PDF
            fallback_title: Fallback title (usually filename)
            
        Returns:
            Extracted or inferred title
        """
        if not blocks:
            return fallback_title or "Untitled Document"
        
        # Strategy 1: Look for largest font on first few pages
        font_based_title = self._find_title_by_font_size(blocks)
        
        # Strategy 2: Look for centered text on first page
        position_based_title = self._find_title_by_position(blocks)
        
        # Strategy 3: Look for explicit title patterns
        pattern_based_title = self._find_title_by_patterns(blocks)
        
        # Strategy 4: Look for document metadata or first significant text
        content_based_title = self._find_title_by_content(blocks)
        
        # Choose best title from candidates
        candidates = [
            (font_based_title, "font"),
            (position_based_title, "position"), 
            (pattern_based_title, "pattern"),
            (content_based_title, "content")
        ]
        
        best_title = self._select_best_title(candidates, fallback_title)
        
        # Clean up and validate title
        final_title = self._clean_title(best_title)
        
        self.logger.info(f"Extracted title: '{final_title}'")
        return final_title
    
    def _find_title_by_font_size(self, blocks: List[Dict[str, Any]]) -> Optional[str]:
        """Find title based on largest font size on first few pages"""
        # Only look at first 3 pages
        first_pages = [b for b in blocks if b["page"] <= 3]
        
        if not first_pages:
            return None
        
        # Find blocks with largest font size
        max_font_size = max(b["font_size"] for b in first_pages)
        largest_blocks = [b for b in first_pages if b["font_size"] == max_font_size]
        
        # Filter by reasonable title characteristics
        title_candidates = []
        for block in largest_blocks:
            text = block["text"].strip()
            word_count = len(text.split())
            
            # Title should be reasonable length
            if 2 <= word_count <= 20 and len(text) <= 200:
                # Prefer text from first page
                priority = 1 if block["page"] == 1 else 2
                title_candidates.append((text, priority, block["page"]))
        
        if title_candidates:
            # Sort by priority (page number) and return best
            title_candidates.sort(key=lambda x: (x[1], x[2]))
            return title_candidates[0][0]
        
        return None
    
    def _find_title_by_position(self, blocks: List[Dict[str, Any]]) -> Optional[str]:
        """Find title based on centered position on first page"""
        first_page = [b for b in blocks if b["page"] == 1]
        
        if not first_page:
            return None
        
        # Calculate page width
        min_x = min(b["x0"] for b in first_page)
        max_x = max(b["x1"] for b in first_page)
        page_width = max_x - min_x
        page_center = (min_x + max_x) / 2
        
        # Find blocks that are approximately centered
        centered_blocks = []
        for block in first_page:
            block_center = (block["x0"] + block["x1"]) / 2
            distance_from_center = abs(block_center - page_center)
            
            # Consider "centered" if within 20% of page width from center
            if distance_from_center < page_width * 0.2:
                text = block["text"].strip()
                word_count = len(text.split())
                
                # Filter for title-like characteristics
                if (2 <= word_count <= 15 and 
                    len(text) <= 150 and
                    not text.isdigit() and
                    block["y0"] < max(b["y0"] for b in first_page) * 0.5):  # Upper half of page
                    
                    centered_blocks.append((text, block["font_size"], block["y0"]))
        
        if centered_blocks:
            # Sort by font size (desc) and position (asc - higher on page)
            centered_blocks.sort(key=lambda x: (-x[1], x[2]))
            return centered_blocks[0][0]
        
        return None
    
    def _find_title_by_patterns(self, blocks: List[Dict[str, Any]]) -> Optional[str]:
        """Find title using explicit title patterns"""
        first_pages = [b for b in blocks if b["page"] <= 2]
        
        for block in first_pages:
            text = block["text"].strip()
            text_lower = text.lower()
            
            # Check for explicit title patterns
            for pattern in self.title_patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                if match:
                    title = match.group(1).strip()
                    if len(title.split()) >= 2:
                        return title
            
            # Check for lines starting with common title indicators
            if text_lower.startswith('title:'):
                title = text[6:].strip()
                if title:
                    return title
        
        return None
    
    def _find_title_by_content(self, blocks: List[Dict[str, Any]]) -> Optional[str]:
        """Find title based on content analysis"""
        first_page = [b for b in blocks if b["page"] == 1]
        
        if not first_page:
            return None
        
        # Sort by position (top to bottom)
        first_page.sort(key=lambda x: x["y0"])
        
        # Look for first substantial text that could be a title
        for block in first_page[:10]:  # Check first 10 blocks
            text = block["text"].strip()
            word_count = len(text.split())
            
            # Skip very short or very long text
            if word_count < 2 or word_count > 20 or len(text) > 200:
                continue
            
            # Skip common non-title text
            text_lower = text.lower()
            if any(skip in text_lower for skip in ['page', 'copyright', '©', 'abstract']):
                continue
            
            # Prefer text with title-like characteristics
            has_title_keywords = any(keyword in text_lower for keyword in self.title_keywords)
            is_capitalized = text[0].isupper() if text else False
            
            if has_title_keywords or is_capitalized:
                return text
        
        # Fallback: return first non-trivial text
        for block in first_page[:5]:
            text = block["text"].strip()
            word_count = len(text.split())
            if 3 <= word_count <= 15:
                return text
        
        return None
    
    def _select_best_title(self, candidates: List[tuple], fallback: str) -> str:
        """Select the best title from multiple candidates"""
        # Filter out None candidates
        valid_candidates = [(title, source) for title, source in candidates if title]
        
        if not valid_candidates:
            return fallback or "Untitled Document"
        
        # Scoring system for different sources
        source_scores = {
            "pattern": 4,    # Explicit title patterns are best
            "position": 3,   # Centered text is good
            "font": 2,       # Large font is decent
            "content": 1     # Content analysis is fallback
        }
        
        # Score each candidate
        scored_candidates = []
        for title, source in valid_candidates:
            base_score = source_scores.get(source, 0)
            
            # Additional scoring factors
            word_count = len(title.split())
            length_score = 1 if 3 <= word_count <= 10 else 0.5
            
            # Bonus for title-like keywords
            keyword_bonus = 0.5 if any(kw in title.lower() for kw in self.title_keywords) else 0
            
            total_score = base_score + length_score + keyword_bonus
            scored_candidates.append((title, total_score, source))
        
        # Sort by score and return best
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        best_title = scored_candidates[0][0]
        best_source = scored_candidates[0][2]
        
        self.logger.debug(f"Selected title from {best_source}: '{best_title}'")
        return best_title
    
    def _clean_title(self, title: str) -> str:
        """Clean and normalize the extracted title"""
        if not title:
            return "Untitled Document"
        
        # Remove extra whitespace
        title = ' '.join(title.split())
        
        # Remove common prefixes
        title = re.sub(r'^title[:\s]+', '', title, flags=re.IGNORECASE)
        
        # Remove file extensions if accidentally included
        title = re.sub(r'\.(pdf|doc|docx|txt)$', '', title, flags=re.IGNORECASE)
        
        # Capitalize first letter if not already
        if title and title[0].islower():
            title = title[0].upper() + title[1:]
        
        # Limit length
        if len(title) > 100:
            # Try to cut at word boundary
            title = title[:97] + "..."
        
        return title