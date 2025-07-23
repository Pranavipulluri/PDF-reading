"""
ML Classifier - Machine Learning based heading detection
Uses lightweight models trained on heading vs. non-heading examples
"""

import numpy as np
import pandas as pd
import logging
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib

class HeadingMLClassifier:
    """Machine Learning classifier for heading detection"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        
        # Model hyperparameters optimized for heading detection
        self.model_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 50,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            },
            'logistic_regression': {
                'random_state': 42,
                'max_iter': 1000
            }
        }
        
        if model_path:
            self.load_model(model_path)
    
    def extract_features(self, blocks: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract features from text blocks for ML classification
        
        Args:
            blocks: List of text blocks with metadata
            
        Returns:
            DataFrame with features for each block
        """
        features = []
        
        # Calculate document-level statistics
        if blocks:
            font_sizes = [b["font_size"] for b in blocks]
            avg_font_size = np.mean(font_sizes)
            std_font_size = np.std(font_sizes)
            max_font_size = np.max(font_sizes)
            min_font_size = np.min(font_sizes)
        else:
            avg_font_size = std_font_size = max_font_size = min_font_size = 12.0
        
        for block in blocks:
            text = block["text"].strip()
            words = text.split()
            
            # Basic text features
            feature_dict = {
                # Font features
                'font_size': block["font_size"],
                'font_size_ratio': block["font_size"] / avg_font_size if avg_font_size > 0 else 1.0,
                'font_size_zscore': (block["font_size"] - avg_font_size) / std_font_size if std_font_size > 0 else 0,
                'is_bold': int(block.get("is_bold", False)),
                'is_italic': int(block.get("is_italic", False)),
                'font_flags': block.get("font_flags", 0),
                
                # Text length features
                'char_count': len(text),
                'word_count': len(words),
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                
                # Position features
                'x_position': block.get("x0", 0),
                'y_position': block.get("y0", 0),
                'width': block.get("width", 0),
                'height': block.get("height", 0),
                'page_number': block.get("page", 1),
                
                # Text pattern features
                'starts_with_number': int(bool(text and text[0].isdigit())),
                'has_colon': int(':' in text),
                'has_period': int('.' in text),
                'all_caps': int(text.isupper() and len(words) > 0),
                'title_case': int(text.istitle()),
                'ends_with_period': int(text.endswith('.')),
                'starts_with_capital': int(bool(text and text[0].isupper())),
                
                # Numeric pattern features
                'has_section_number': int(bool(self._extract_section_number(text))),
                'section_depth': self._get_section_depth(text),
                
                # Content analysis
                'has_common_words': int(self._has_common_heading_words(text)),
                'digit_ratio': sum(c.isdigit() for c in text) / len(text) if text else 0,
                'punct_ratio': sum(c in '.,;:!?' for c in text) / len(text) if text else 0,
                'space_ratio': sum(c.isspace() for c in text) / len(text) if text else 0,
                
                # Relative features (compared to document stats)
                'font_size_percentile': self._get_percentile(block["font_size"], font_sizes),
                'is_largest_font': int(block["font_size"] == max_font_size),
                'is_above_avg_font': int(block["font_size"] > avg_font_size),
            }
            
            # Add contextual features
            contextual_features = self._extract_contextual_features(block, blocks)
            feature_dict.update(contextual_features)
            
            features.append(feature_dict)
        
        df = pd.DataFrame(features)
        self.feature_names = list(df.columns)
        
        return df
    
    def _extract_section_number(self, text: str) -> str:
        """Extract section number pattern from text"""
        import re
        patterns = [
            r'^(\d+(?:\.\d+)*)',  # 1, 1.1, 1.1.1
            r'^([A-Z](?:\.\d+)*)',  # A, A.1
            r'^([IVX]+(?:\.\d+)*)',  # I, II, III
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text.strip())
            if match:
                return match.group(1)
        return ""
    
    def _get_section_depth(self, text: str) -> int:
        """Get depth of section numbering (1=H1, 2=H2, 3=H3)"""
        section_num = self._extract_section_number(text)
        if not section_num:
            return 0
        return min(section_num.count('.') + 1, 3)
    
    def _has_common_heading_words(self, text: str) -> bool:
        """Check if text contains common heading words"""
        heading_words = {
            'introduction', 'conclusion', 'summary', 'abstract', 'overview',
            'background', 'methodology', 'method', 'results', 'discussion',
            'references', 'bibliography', 'appendix', 'chapter', 'section',
            'acknowledgments', 'acknowledgements', 'preface', 'foreword',
            'glossary', 'index', 'contents', 'objectives', 'goals',
            'analysis', 'findings', 'recommendations', 'implications',
            'literature', 'review', 'related', 'work', 'future',
            'limitations', 'scope', 'definition', 'framework'
        }
        
        text_words = set(text.lower().split())
        return bool(heading_words.intersection(text_words))
    
    def _get_percentile(self, value: float, values: List[float]) -> float:
        """Get percentile rank of value in list"""
        if not values:
            return 0.5
        return sum(v <= value for v in values) / len(values)
    
    def _extract_contextual_features(self, block: Dict[str, Any], 
                                   all_blocks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract features based on surrounding context"""
        features = {}
        
        # Find blocks on same page
        page_blocks = [b for b in all_blocks if b["page"] == block["page"]]
        page_blocks.sort(key=lambda x: x["y0"])
        
        # Find current block index
        block_idx = -1
        for i, b in enumerate(page_blocks):
            if (abs(b["y0"] - block["y0"]) < 1 and 
                abs(b["x0"] - block["x0"]) < 1):
                block_idx = i
                break
        
        if block_idx >= 0:
            # Distance to previous/next blocks
            if block_idx > 0:
                prev_block = page_blocks[block_idx - 1]
                features['prev_vertical_gap'] = block["y0"] - prev_block.get("y1", block["y0"])
                features['prev_font_size_diff'] = block["font_size"] - prev_block.get("font_size", block["font_size"])
            else:
                features['prev_vertical_gap'] = 0
                features['prev_font_size_diff'] = 0
                
            if block_idx < len(page_blocks) - 1:
                next_block = page_blocks[block_idx + 1]
                features['next_vertical_gap'] = next_block.get("y0", block["y0"]) - block.get("y1", block["y0"])
                features['next_font_size_diff'] = next_block.get("font_size", block["font_size"]) - block["font_size"]
            else:
                features['next_vertical_gap'] = 0
                features['next_font_size_diff'] = 0
            
            # Position on page
            features['relative_y_position'] = block_idx / len(page_blocks) if page_blocks else 0.5
        else:
            features.update({
                'prev_vertical_gap': 0,
                'prev_font_size_diff': 0,
                'next_vertical_gap': 0,
                'next_font_size_diff': 0,
                'relative_y_position': 0.5
            })
        
        return features
    
    def train(self, training_blocks: List[Dict[str, Any]], 
              labels: List[str], model_type: str = 'random_forest') -> Dict[str, float]:
        """
        Train the ML classifier
        
        Args:
            training_blocks: List of text blocks with features
            labels: List of labels ('heading' or 'text')
            model_type: Type of model to train
            
        Returns:
            Training metrics
        """
        self.logger.info(f"Training {model_type} classifier with {len(training_blocks)} samples")
        
        # Extract features
        X = self.extract_features(training_blocks)
        y = np.array([1 if label == 'heading' else 0 for label in labels])
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.model_params['random_forest'])
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(**self.model_params['gradient_boosting'])
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(**self.model_params['logistic_regression'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate metrics
        train_score = self.model.score(X_scaled, y)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        
        metrics = {
            'train_accuracy': train_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_features': X.shape[1],
            'n_samples': len(y),
            'positive_ratio': y.mean()
        }
        
        self.logger.info(f"Training completed. CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        return metrics
    
    def predict_probabilities(self, blocks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Predict heading probabilities for blocks
        
        Args:
            blocks: List of text blocks
            
        Returns:
            Array of heading probabilities
        """
        if not self.is_trained:
            self.logger.warning("Model not trained, returning default probabilities")
            return np.full(len(blocks), 0.1)  # Low default probability
        
        X = self.extract_features(blocks)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities for positive class (heading)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return probabilities
    
    def predict_headings(self, blocks: List[Dict[str, Any]], 
                        threshold: float = 0.5) -> List[bool]:
        """
        Predict which blocks are headings
        
        Args:
            blocks: List of text blocks
            threshold: Classification threshold
            
        Returns:
            List of boolean predictions
        """
        probabilities = self.predict_probabilities(blocks)
        return (probabilities >= threshold).tolist()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"Model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False