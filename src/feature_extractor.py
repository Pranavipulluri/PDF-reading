"""
Feature Extractor - Extract comprehensive features for ML classification
Combines layout, semantic, and contextual features for heading detection
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple, Optional
import re
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class ComprehensiveFeatureExtractor:
    """Extract comprehensive features for ML-based heading detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.feature_columns = []
        
        # Feature categories and their weights for selection
        self.feature_categories = {
            'font_features': ['font_size', 'font_size_ratio', 'font_size_zscore', 'is_bold', 'is_italic'],
            'position_features': ['x_position', 'y_position', 'distance_from_margin', 'is_centered'],
            'text_features': ['char_count', 'word_count', 'avg_word_length', 'alpha_ratio', 'digit_ratio'],
            'pattern_features': ['has_numbering', 'section_depth', 'all_caps', 'title_case'],
            'semantic_features': ['heading_vocabulary_score', 'semantic_confidence', 'language_score'],
            'context_features': ['whitespace_before', 'whitespace_after', 'similar_blocks_nearby'],
            'layout_features': ['line_spacing', 'character_spacing', 'indentation_level']
        }
    
    def extract_comprehensive_features(self, 
                                     blocks: List[Dict[str, Any]],
                                     include_text_features: bool = True) -> pd.DataFrame:
        """
        Extract comprehensive feature set from text blocks
        
        Args:
            blocks: List of text blocks with all analysis data
            include_text_features: Whether to include TF-IDF text features
            
        Returns:
            DataFrame with extracted features
        """
        if not blocks:
            return pd.DataFrame()
        
        self.logger.info(f"Extracting comprehensive features from {len(blocks)} blocks")
        
        # Initialize feature list
        feature_list = []
        
        # Document-level statistics
        doc_stats = self._calculate_document_statistics(blocks)
        
        for i, block in enumerate(blocks):
            # Extract features for this block
            block_features = self._extract_block_features(block, blocks, doc_stats, i)
            feature_list.append(block_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_list)
        
        # Add text-based features if requested
        if include_text_features:
            text_features = self._extract_text_features(blocks)
            if not text_features.empty:
                features_df = pd.concat([features_df, text_features], axis=1)
        
        # Store feature columns for future reference
        self.feature_columns = list(features_df.columns)
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        self.logger.info(f"Extracted {len(self.feature_columns)} features")
        return features_df
    
    def _calculate_document_statistics(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate document-level statistics for normalization"""
        # Font statistics
        font_sizes = []
        x_positions = []
        y_positions = []
        text_lengths = []
        
        for block in blocks:
            # Font information
            font_info = block.get('font_info', {})
            if font_info.get('size'):
                font_sizes.append(font_info['size'])
            
            # Position information
            bbox = block.get('bbox', [0, 0, 0, 0])
            x_positions.append(bbox[0])
            y_positions.append(bbox[1])
            
            # Text statistics
            text = block.get('text', '')
            text_lengths.append(len(text))
        
        return {
            'font_stats': {
                'mean': np.mean(font_sizes) if font_sizes else 12,
                'std': np.std(font_sizes) if font_sizes else 2,
                'min': np.min(font_sizes) if font_sizes else 8,
                'max': np.max(font_sizes) if font_sizes else 24,
                'percentiles': {
                    25: np.percentile(font_sizes, 25) if font_sizes else 10,
                    50: np.percentile(font_sizes, 50) if font_sizes else 12,
                    75: np.percentile(font_sizes, 75) if font_sizes else 14,
                    90: np.percentile(font_sizes, 90) if font_sizes else 16
                }
            },
            'position_stats': {
                'x_mean': np.mean(x_positions) if x_positions else 72,
                'x_std': np.std(x_positions) if x_positions else 20,
                'x_min': np.min(x_positions) if x_positions else 0,
                'y_range': np.max(y_positions) - np.min(y_positions) if y_positions else 792
            },
            'text_stats': {
                'mean_length': np.mean(text_lengths) if text_lengths else 50,
                'std_length': np.std(text_lengths) if text_lengths else 30
            },
            'document_info': {
                'total_blocks': len(blocks),
                'pages': len(set(block.get('page', 1) for block in blocks))
            }
        }
    
    def _extract_block_features(self, block: Dict[str, Any], 
                               all_blocks: List[Dict[str, Any]],
                               doc_stats: Dict[str, Any],
                               block_index: int) -> Dict[str, Any]:
        """Extract all features for a single block"""
        features = {}
        
        # Basic block information
        text = block.get('text', '').strip()
        bbox = block.get('bbox', [0, 0, 0, 0])
        page = block.get('page', 1)
        
        # Font-based features
        features.update(self._extract_font_features(block, doc_stats))
        
        # Position-based features  
        features.update(self._extract_position_features(block, doc_stats))
        
        # Text analysis features
        features.update(self._extract_text_analysis_features(block))
        
        # Pattern matching features
        features.update(self._extract_pattern_features(text))
        
        # Semantic analysis features
        features.update(self._extract_semantic_features(block))
        
        # Layout analysis features
        features.update(self._extract_layout_features(block))
        
        # Context-based features
        features.update(self._extract_context_features(block, all_blocks, block_index))
        
        # Rule-based features (if available)
        features.update(self._extract_rule_based_features(block))
        
        # ML classifier features (if available)
        features.update(self._extract_ml_features(block))
        
        return features
    
    def _extract_font_features(self, block: Dict[str, Any], doc_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract font-related features"""
        font_info = block.get('font_info', {})
        font_stats = doc_stats.get('font_stats', {})
        
        font_size = font_info.get('size', 12)
        mean_size = font_stats.get('mean', 12)
        std_size = font_stats.get('std', 2)
        
        features = {
            'font_size': font_size,
            'font_size_ratio': font_size / mean_size if mean_size > 0 else 1.0,
            'font_size_zscore': (font_size - mean_size) / std_size if std_size > 0 else 0,
            'font_size_percentile': self._calculate_percentile(font_size, doc_stats['font_stats']),
            'is_bold': int(font_info.get('is_bold', False)),
            'is_italic': int(font_info.get('is_italic', False)),
            'is_serif': int(font_info.get('is_serif', False)),
            'is_sans_serif': int(font_info.get('is_sans_serif', False)),
            'is_monospace': int(font_info.get('is_monospaced', False)),
            'font_flags': font_info.get('flags', 0),
            'is_display_font': int(font_info.get('is_display_font', False)),
            'is_title_font': int(font_info.get('is_title_font', False))
        }
        
        return features
    
    def _extract_position_features(self, block: Dict[str, Any], doc_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract position and layout features"""
        bbox = block.get('bbox', [0, 0, 0, 0])
        position_stats = doc_stats.get('position_stats', {})
        
        x_mean = position_stats.get('x_mean', 72)
        x_std = position_stats.get('x_std', 20)
        x_min = position_stats.get('x_min', 0)
        
        # Advanced layout features from layout analyzer
        advanced_features = block.get('advanced_features', {})
        layout_info = block.get('layout_info', {})
        
        features = {
            'x_position': bbox[0],
            'y_position': bbox[1],
            'width': bbox[2] - bbox[0],
            'height': bbox[3] - bbox[1],
            'distance_from_left_margin': bbox[0] - x_min,
            'x_position_zscore': (bbox[0] - x_mean) / x_std if x_std > 0 else 0,
            'relative_x_position': (bbox[0] - x_min) / (position_stats.get('x_range', 500)) if position_stats.get('x_range', 0) > 0 else 0,
            'aspect_ratio': (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) if bbox[3] > bbox[1] else 1.0,
            
            # Layout-specific features
            'line_spacing': layout_info.get('line_spacing', 0),
            'character_spacing': layout_info.get('character_spacing', 0),
            'indentation_level': layout_info.get('indentation', 0),
            'is_centered': int(layout_info.get('text_alignment', '') == 'center'),
            'is_left_aligned': int(layout_info.get('text_alignment', 'left') == 'left'),
            'is_right_aligned': int(layout_info.get('text_alignment', '') == 'right'),
            
            # Advanced position features
            'font_size_z_score': advanced_features.get('font_size_z_score', 0),
            'font_size_percentile': advanced_features.get('font_size_percentile', 0.5),
            'x_position_z_score': advanced_features.get('x_position_z_score', 0)
        }
        
        return features
    
    def _extract_text_analysis_features(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text analysis features"""
        text = block.get('text', '').strip()
        words = text.split()
        
        # Semantic analysis results
        semantic_analysis = block.get('semantic_analysis', {})
        semantic_features = semantic_analysis.get('features', {})
        
        features = {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': semantic_features.get('sentence_count', 1),
            'avg_word_length': semantic_features.get('avg_word_length', 0),
            'alpha_ratio': semantic_features.get('alpha_ratio', 0),
            'digit_ratio': semantic_features.get('digit_ratio', 0),
            'punct_ratio': semantic_features.get('punct_ratio', 0),
            'space_ratio': semantic_features.get('space_ratio', 0),
            'upper_ratio': semantic_features.get('upper_ratio', 0),
            'stop_word_ratio': semantic_features.get('stop_word_ratio', 0),
            
            # Text patterns
            'first_word_capitalized': int(semantic_features.get('first_word_capitalized', False)),
            'all_words_capitalized': int(semantic_features.get('all_words_capitalized', False)),
            'title_case': int(semantic_features.get('title_case', False)),
            'all_caps': int(semantic_features.get('all_caps', False)),
            
            # Language detection
            'language_english': int(semantic_analysis.get('language', 'english') == 'english'),
            'language_score': 1.0 if semantic_analysis.get('language') == 'english' else 0.5
        }
        
        return features
    
    def _extract_pattern_features(self, text: str) -> Dict[str, Any]:
        """Extract pattern-based features"""
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        # Numbering patterns
        numbering_patterns = {
            'decimal_1': r'^\d+\.\s+',
            'decimal_2': r'^\d+\.\d+\.\s+', 
            'decimal_3': r'^\d+\.\d+\.\d+\.\s+',
            'roman': r'^[ivx]+\.\s+',
            'letter': r'^[a-z]\.\s+',
            'paren_number': r'^\d+\)\s+',
            'paren_letter': r'^\([a-z]\)\s+'
        }
        
        features = {}
        
        # Check each pattern
        for pattern_name, pattern in numbering_patterns.items():
            features[f'has_{pattern_name}'] = int(bool(re.match(pattern, text_clean, re.IGNORECASE)))
        
        # Overall numbering indicators
        features['has_any_numbering'] = int(any(features[f'has_{p}'] for p in numbering_patterns.keys()))
        features['section_depth'] = self._calculate_section_depth(text_clean)
        
        # Content patterns
        features['starts_with_number'] = int(bool(text_clean and text_clean[0].isdigit()))
        features['starts_with_letter'] = int(bool(text_clean and text_clean[0].isalpha()))
        features['starts_with_capital'] = int(bool(text_clean and text_clean[0].isupper()))
        features['ends_with_colon'] = int(text_clean.endswith(':'))
        features['ends_with_period'] = int(text_clean.endswith('.'))
        features['has_question_mark'] = int('?' in text_clean)
        features['has_parentheses'] = int('(' in text_clean and ')' in text_clean)
        features['has_acronyms'] = int(bool(re.search(r'\b[A-Z]{2,}\b', text_clean)))
        
        return features
    
    def _extract_semantic_features(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic analysis features"""
        semantic_analysis = block.get('semantic_analysis', {})
        
        if not semantic_analysis:
            return {
                'heading_probability': 0.0,
                'semantic_confidence': 0.0,
                'max_vocabulary_score': 0.0,
                'has_heading_keywords': 0
            }
        
        features = semantic_analysis.get('features', {})
        vocabulary_scores = features.get('vocabulary_scores', {})
        
        return {
            'heading_probability': semantic_analysis.get('heading_probability', 0.0),
            'semantic_confidence': semantic_analysis.get('confidence', 0.0),
            'max_vocabulary_score': features.get('max_vocabulary_score', 0.0),
            'academic_vocab_score': vocabulary_scores.get('academic', 0.0),
            'technical_vocab_score': vocabulary_scores.get('technical', 0.0),
            'business_vocab_score': vocabulary_scores.get('business', 0.0),
            'legal_vocab_score': vocabulary_scores.get('legal', 0.0),
            'has_heading_keywords': int(features.get('has_common_words', False)),
            'has_proper_nouns': int(features.get('has_proper_nouns', False)),
            'has_technical_terms': int(features.get('has_technical_terms', False)),
            'is_question': int(features.get('is_question', False)),
            'is_instruction': int(features.get('is_instruction', False)),
            'is_definition': int(features.get('is_definition', False))
        }
    
    def _extract_layout_features(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Extract advanced layout features"""
        advanced_features = block.get('advanced_features', {})
        
        return {
            'line_spacing': advanced_features.get('line_spacing', 0),
            'character_spacing': advanced_features.get('character_spacing', 0),
            'indentation_level': advanced_features.get('indentation_level', 0),
            'distance_from_left_margin': advanced_features.get('distance_from_left_margin', 0),
            'context_calculated': int(advanced_features.get('context_calculated', False))
        }
    
    def _extract_context_features(self, block: Dict[str, Any], 
                                 all_blocks: List[Dict[str, Any]],
                                 block_index: int) -> Dict[str, Any]:
        """Extract contextual features based on surrounding blocks"""
        features = {
            'block_index': block_index,
            'relative_position': block_index / len(all_blocks) if all_blocks else 0.5,
            'is_first_block': int(block_index == 0),
            'is_last_block': int(block_index == len(all_blocks) - 1),
            'whitespace_before': 0,
            'whitespace_after': 0,
            'similar_font_blocks_nearby': 0,
            'prev_block_font_size_diff': 0,
            'next_block_font_size_diff': 0
        }
        
        current_bbox = block.get('bbox', [0, 0, 0, 0])
        current_font_size = block.get('font_info', {}).get('size', 12)
        page = block.get('page', 1)
        
        # Find blocks on same page
        page_blocks = [b for i, b in enumerate(all_blocks) 
                      if b.get('page') == page]
        page_blocks.sort(key=lambda x: x.get('bbox', [0, 0, 0, 0])[1])
        
        # Find current block in page blocks
        current_page_index = -1
        for i, b in enumerate(page_blocks):
            b_bbox = b.get('bbox', [0, 0, 0, 0])
            if (abs(b_bbox[0] - current_bbox[0]) < 5 and 
                abs(b_bbox[1] - current_bbox[1]) < 5):
                current_page_index = i
                break
        
        if current_page_index >= 0:
            # Previous block features
            if current_page_index > 0:
                prev_block = page_blocks[current_page_index - 1]
                prev_bbox = prev_block.get('bbox', [0, 0, 0, 0])
                prev_font_size = prev_block.get('font_info', {}).get('size', 12)
                
                features['whitespace_before'] = current_bbox[1] - prev_bbox[3]
                features['prev_block_font_size_diff'] = current_font_size - prev_font_size
            
            # Next block features
            if current_page_index < len(page_blocks) - 1:
                next_block = page_blocks[current_page_index + 1]
                next_bbox = next_block.get('bbox', [0, 0, 0, 0])
                next_font_size = next_block.get('font_info', {}).get('size', 12)
                
                features['whitespace_after'] = next_bbox[1] - current_bbox[3]
                features['next_block_font_size_diff'] = next_font_size - current_font_size
            
            # Count similar font size blocks nearby
            nearby_range = 3  # Look at 3 blocks before/after
            start_idx = max(0, current_page_index - nearby_range)
            end_idx = min(len(page_blocks), current_page_index + nearby_range + 1)
            
            similar_count = 0
            for i in range(start_idx, end_idx):
                if i != current_page_index:
                    other_font_size = page_blocks[i].get('font_info', {}).get('size', 12)
                    if abs(other_font_size - current_font_size) <= 1:  # Within 1 point
                        similar_count += 1
            
            features['similar_font_blocks_nearby'] = similar_count
        
        return features
    
    def _extract_rule_based_features(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from rule-based analysis"""
        rule_analysis = block.get('rule_analysis', {})
        
        if not rule_analysis:
            return {
                'rule_is_heading': 0,
                'rule_confidence': 0.0,
                'rule_raw_score': 0.0,
                'rule_font_score': 0.0,
                'rule_pattern_score': 0.0,
                'rule_position_score': 0.0
            }
        
        detailed_scores = rule_analysis.get('detailed_scores', {})
        
        return {
            'rule_is_heading': int(rule_analysis.get('is_heading', False)),
            'rule_confidence': rule_analysis.get('confidence', 0.0),
            'rule_raw_score': rule_analysis.get('raw_score', 0.0),
            'rule_font_score': detailed_scores.get('font', {}).get('score', 0.0),
            'rule_pattern_score': detailed_scores.get('pattern', {}).get('score', 0.0),
            'rule_position_score': detailed_scores.get('position', {}).get('score', 0.0),
            'rule_style_score': detailed_scores.get('style', {}).get('score', 0.0),
            'rule_context_score': detailed_scores.get('context', {}).get('score', 0.0)
        }
    
    def _extract_ml_features(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from ML classifier if available"""
        ml_analysis = block.get('ml_analysis', {})
        
        if not ml_analysis:
            return {
                'ml_heading_probability': 0.0,
                'ml_confidence': 0.0
            }
        
        return {
            'ml_heading_probability': ml_analysis.get('heading_probability', 0.0),
            'ml_confidence': ml_analysis.get('confidence', 0.0)
        }
    
    def _extract_text_features(self, blocks: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract TF-IDF text features"""
        texts = [block.get('text', '') for block in blocks]
        
        if not any(texts):
            return pd.DataFrame()
        
        try:
            # Initialize TF-IDF vectorizer with heading-specific parameters
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=100,  # Keep feature count manageable
                    stop_words='english',
                    ngram_range=(1, 2),  # Include bigrams
                    min_df=2,  # Must appear in at least 2 documents
                    max_df=0.8,  # Remove very common terms
                    token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Alphanumeric tokens
                )
            
            # Fit and transform texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Create DataFrame with TF-IDF features
            feature_names = [f'tfidf_{name}' for name in self.tfidf_vectorizer.get_feature_names_out()]
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
            
            return tfidf_df
            
        except Exception as e:
            self.logger.warning(f"TF-IDF feature extraction failed: {e}")
            return pd.DataFrame()
    
    def _calculate_percentile(self, value: float, font_stats: Dict[str, Any]) -> float:
        """Calculate percentile of value in distribution"""
        percentiles = font_stats.get('percentiles', {})
        
        if value <= percentiles.get(25, 10):
            return 0.25
        elif value <= percentiles.get(50, 12):
            return 0.5
        elif value <= percentiles.get(75, 14):
            return 0.75
        elif value <= percentiles.get(90, 16):
            return 0.9
        else:
            return 0.95
    
    def _calculate_section_depth(self, text: str) -> int:
        """Calculate section depth from numbering pattern"""
        # Pattern: 1.2.3 -> depth 3
        match = re.match(r'^(\d+(?:\.\d+)*)', text.strip())
        if match:
            return min(match.group(1).count('.') + 1, 3)
        return 0
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature matrix"""
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Fill non-numeric columns with mode or default values
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0)
        
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Return feature groups for importance analysis"""
        feature_groups = {}
        
        for category, base_features in self.feature_categories.items():
            # Find actual features that match the base feature names
            matching_features = []
            for feature_name in self.feature_columns:
                if any(base_feature in feature_name for base_feature in base_features):
                    matching_features.append(feature_name)
            
            if matching_features:
                feature_groups[category] = matching_features
        
        return feature_groups
    
    def select_top_features(self, X: pd.DataFrame, y: np.ndarray, 
                          n_features: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """Select top features using multiple selection methods"""
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        from sklearn.ensemble import RandomForestClassifier
        
        if X.empty or len(y) == 0:
            return X, list(X.columns) if not X.empty else []
        
        selected_features = set()
        
        try:
            # Method 1: Statistical test (ANOVA F-test)
            selector_f = SelectKBest(score_func=f_classif, k=min(n_features//2, X.shape[1]))
            selector_f.fit(X, y)
            f_features = X.columns[selector_f.get_support()].tolist()
            selected_features.update(f_features[:n_features//3])
            
            # Method 2: Mutual information
            selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(n_features//2, X.shape[1]))
            selector_mi.fit(X, y)
            mi_features = X.columns[selector_mi.get_support()].tolist()
            selected_features.update(mi_features[:n_features//3])
            
            # Method 3: Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X, y)
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            rf_features = feature_importance.head(n_features//3)['feature'].tolist()
            selected_features.update(rf_features)
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}")
            # Fallback to all features or random selection
            selected_features = set(X.columns[:n_features])
        
        # Ensure we have at least some features
        if len(selected_features) < 10:
            selected_features.update(X.columns[:min(20, X.shape[1])])
        
        final_features = list(selected_features)[:n_features]
        return X[final_features], final_features