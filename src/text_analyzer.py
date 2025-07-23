"""
Enhanced Text Analyzer - Advanced NLP and semantic analysis with junk filtering
Uses lightweight models for semantic understanding plus rule-based filtering
"""

import re
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# Download required NLTK data if not present
def ensure_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass  # Fail silently if download fails

class EnhancedSemanticTextAnalyzer:
    """Enhanced text analysis with junk filtering and multilingual detection"""
    
    def __init__(self):
        import logging
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer

        self.logger = logging.getLogger(__name__)
        ensure_nltk_data()

        try:
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
        except Exception as e:
            self.logger.warning(f"Fallback stopwords used: {e}")
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            self.stemmer = None

        self.junk_patterns = {
            'page_numbers': [r'^page\s+\d+$', r'^\d+\s+of\s+\d+$', r'^\d{1,3}$'],
            'headers_footers': [
                r'^copyright',
                r'^©\s*\d{4}',
                r'confidential',
                r'proprietary',
                r'all rights reserved',
                r'printed\s+(on|in)',
                r'published\s+by',
                r'version\s+[\d.]+',
                r'draft',
                r'^www\.',
                r'@[\w.]+\.[\w]+',
                r'https?://',
            ],
            'document_metadata': [
                r'file\s*:\s*',
                r'author\s*:\s*',
                r'date\s*:\s*',
                r'modified\s*:\s*',
                r'created\s*:\s*',
                r'last\s+updated',
                r'revision\s+\d+',
            ]
        }

        self.heading_vocabularies = {
            'academic': {
                'high_confidence': {'abstract', 'introduction', 'conclusion', 'methodology', 'results',
                                    'discussion', 'references', 'bibliography', 'acknowledgments',
                                    'literature review', 'related work', 'future work', 'limitations'},
                'medium_confidence': {'background', 'overview', 'summary', 'analysis', 'evaluation',
                                    'implementation', 'experiment', 'case study', 'findings',
                                    'implications', 'recommendations', 'scope', 'objectives'}
            },
            'technical': {
                'high_confidence': {'installation', 'configuration', 'setup', 'requirements',
                                    'architecture', 'design', 'implementation', 'api reference',
                                    'troubleshooting', 'examples', 'usage', 'getting started'},
                'medium_confidence': {'features', 'components', 'modules', 'system', 'network',
                                    'database', 'security', 'performance', 'testing', 'deployment'}
            },
            'business': {
                'high_confidence': {'executive summary', 'overview', 'objectives', 'strategy',
                                    'recommendations', 'conclusion', 'next steps', 'action plan'},
                'medium_confidence': {'analysis', 'market', 'competition', 'opportunities', 'risks',
                                    'budget', 'timeline', 'resources', 'metrics', 'performance'}
            }
        }

        self.multilingual_patterns = {
            'japanese': {
                'chapter_markers': ['第', '章', '節', '項', '編', '部'],
                'numbering': ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十'],
                'section_patterns': [r'第\d+章', r'第\d+節', r'\d+\.\d+', r'第\d+項'],
                'common_headings': ['概要', '序論', '結論', '参考文献', '謝辞', '付録']
            },
            'chinese': {
                'chapter_markers': ['第', '章', '节', '条', '篇', '部'],
                'numbering': ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十'],
                'section_patterns': [r'第\d+章', r'第\d+节', r'\d+\.\d+'],
                'common_headings': ['概述', '引言', '结论', '参考文献', '致谢', '附录']
            },
            'french': {
                'chapter_markers': ['chapitre', 'section', 'partie', 'annexe'],
                'section_patterns': [r'chapitre\s+\d+', r'section\s+\d+', r'\d+\.\d+'],
                'common_headings': ['introduction', 'conclusion', 'résumé', 'références', 'annexe']
            },
            'german': {
                'chapter_markers': ['kapitel', 'abschnitt', 'teil', 'anhang'],
                'section_patterns': [r'kapitel\s+\d+', r'abschnitt\s+\d+', r'\d+\.\d+'],
                'common_headings': ['einführung', 'zusammenfassung', 'schlussfolgerung', 'literatur', 'anhang']
            },
            'spanish': {
                'chapter_markers': ['capítulo', 'sección', 'parte', 'anexo'],
                'section_patterns': [r'capítulo\s+\d+', r'sección\s+\d+', r'\d+\.\d+'],
                'common_headings': ['introducción', 'conclusión', 'resumen', 'referencias', 'anexo']
            }
        }

        self.langdetect_available = False
        try:
            from langdetect import detect, LangDetectError
            self.langdetect = detect
            self.LangDetectError = LangDetectError
            self.langdetect_available = True
            self.logger.info("Enhanced language detection with langdetect available")
        except ImportError:
            self.logger.info("Using basic language detection (langdetect not available)")

    
    def analyze_text_semantics(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive semantic analysis of text
        
        Args:
            text: Text to analyze
            context: Additional context information
            
        Returns:
            Semantic analysis results
        """
        if not text.strip():
            return {'is_heading_likely': False, 'confidence': 0.0}
        
        analysis = {
            'text': text.strip(),
            'length': len(text),
            'word_count': len(text.split()),
            'language': self._detect_language(text),
            'heading_probability': 0.0,
            'suggested_level': None,
            'confidence': 0.0,
            'features': {}
        }
        
        # Extract linguistic features
        linguistic_features = self._extract_linguistic_features(text)
        analysis['features'].update(linguistic_features)
        
        # Pattern-based analysis
        pattern_analysis = self._analyze_patterns(text)
        analysis['features'].update(pattern_analysis)
        
        # Semantic analysis
        semantic_features = self._analyze_semantics(text)
        analysis['features'].update(semantic_features)
        
        # Multilingual analysis
        if analysis['language'] != 'english':
            multilingual_features = self._analyze_multilingual_patterns(text, analysis['language'])
            analysis['features'].update(multilingual_features)
        
        # Calculate overall heading probability
        analysis['heading_probability'] = self._calculate_heading_probability(analysis['features'])
        analysis['suggested_level'] = self._suggest_heading_level(analysis['features'])
        analysis['confidence'] = self._calculate_confidence(analysis['features'])
        
        return analysis
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        text = text.lower()
        
        # Japanese detection (Hiragana, Katakana, Kanji)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'japanese'
        
        # Chinese detection
        if re.search(r'[\u4E00-\u9FFF]', text):
            return 'chinese'
        
        # French detection
        french_chars = len(re.findall(r'[àâäçéèêëîïôöùûüÿ]', text))
        if french_chars > 0 or any(word in text for word in ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou']):
            return 'french'
        
        # German detection
        german_chars = len(re.findall(r'[äöüß]', text))
        if german_chars > 0 or any(word in text for word in ['der', 'die', 'das', 'und', 'oder', 'mit']):
            return 'german'
        
        # Spanish detection
        spanish_chars = len(re.findall(r'[áéíóúüñ]', text))
        if spanish_chars > 0 or any(word in text for word in ['el', 'la', 'los', 'las', 'y', 'o', 'con']):
            return 'spanish'
        
        return 'english'
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract basic linguistic features"""
        features = {}
        
        words = text.split()
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['sentence_count'] = len(sent_tokenize(text)) if text else 0
        
        # Character type ratios
        features['alpha_ratio'] = sum(c.isalpha() for c in text) / len(text) if text else 0
        features['digit_ratio'] = sum(c.isdigit() for c in text) / len(text) if text else 0
        features['punct_ratio'] = sum(c in '.,;:!?()[]{}"\'-' for c in text) / len(text) if text else 0
        features['space_ratio'] = sum(c.isspace() for c in text) / len(text) if text else 0
        features['upper_ratio'] = sum(c.isupper() for c in text) / len(text) if text else 0
        
        # Word-level features
        if words:
            features['first_word_capitalized'] = words[0][0].isupper() if words[0] else False
            features['all_words_capitalized'] = all(w[0].isupper() for w in words if w)
            features['title_case'] = text.istitle()
            features['all_caps'] = text.isupper()
            
            # Stop word analysis
            if self.stop_words:
                stop_word_count = sum(1 for w in words if w.lower() in self.stop_words)
                features['stop_word_ratio'] = stop_word_count / len(words)
            else:
                features['stop_word_ratio'] = 0
        
        return features
    
    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text patterns for heading indicators"""
        features = {}
        text_lower = text.lower().strip()
        
        # Numbering patterns
        features['has_number_prefix'] = bool(re.match(r'^\d+', text_lower))
        features['has_roman_numeral'] = bool(re.match(r'^[ivx]+\b', text_lower))
        features['has_letter_prefix'] = bool(re.match(r'^[a-z]\)', text_lower))
        
        # Section numbering patterns
        section_patterns = [
            (r'^\d+\.\s+', 'decimal_1'),
            (r'^\d+\.\d+\.\s+', 'decimal_2'),
            (r'^\d+\.\d+\.\d+\.\s+', 'decimal_3'),
            (r'^\d+\)\s+', 'paren_number'),
            (r'^[A-Z]\.\s+', 'letter_dot'),
            (r'^\([a-z]\)\s+', 'paren_letter'),
            (r'^[IVX]+\.\s+', 'roman_dot'),
        ]
        
        features['section_pattern'] = None
        features['section_level'] = 0
        
        for pattern, pattern_name in section_patterns:
            if re.match(pattern, text):
                features['section_pattern'] = pattern_name
                if 'decimal' in pattern_name:
                    features['section_level'] = int(pattern_name[-1])
                elif pattern_name in ['decimal_1', 'paren_number', 'letter_dot', 'roman_dot']:
                    features['section_level'] = 1
                elif pattern_name in ['paren_letter']:
                    features['section_level'] = 2
                break
        
        # Punctuation patterns
        features['ends_with_colon'] = text.endswith(':')
        features['ends_with_period'] = text.endswith('.')
        features['has_question_mark'] = '?' in text
        features['has_exclamation'] = '!' in text
        
        # Special formatting
        features['has_parentheses'] = '(' in text and ')' in text
        features['has_brackets'] = '[' in text and ']' in text
        features['has_quotes'] = '"' in text or "'" in text
        
        return features
    
    def _analyze_semantics(self, text: str) -> Dict[str, Any]:
        """Analyze semantic content for heading likelihood"""
        features = {}
        text_lower = text.lower()
        
        # Check against heading vocabularies
        vocabulary_scores = {}
        for domain, vocab in self.heading_vocabularies.items():
            words = set(text_lower.split())
            matches = words.intersection(vocab)
            vocabulary_scores[domain] = len(matches) / len(words) if words else 0
        
        features['vocabulary_scores'] = vocabulary_scores
        features['max_vocabulary_score'] = max(vocabulary_scores.values()) if vocabulary_scores else 0
        features['dominant_domain'] = max(vocabulary_scores.keys(), key=vocabulary_scores.get) if vocabulary_scores else None
        
        # Content type analysis
        content_indicators = {
            'question': ['what', 'how', 'why', 'when', 'where', 'who', 'which'],
            'instruction': ['step', 'procedure', 'process', 'method', 'approach'],
            'list': ['list', 'items', 'elements', 'components', 'parts'],
            'definition': ['definition', 'meaning', 'concept', 'term', 'explanation'],
            'example': ['example', 'instance', 'case', 'illustration', 'sample'],
            'comparison': ['versus', 'vs', 'compared', 'difference', 'similarity']
        }
        
        for content_type, indicators in content_indicators.items():
            features[f'is_{content_type}'] = any(indicator in text_lower for indicator in indicators)
        
        # Named entity-like analysis (simple version)
        features['has_proper_nouns'] = self._detect_proper_nouns(text)
        features['has_technical_terms'] = self._detect_technical_terms(text_lower)
        features['has_acronyms'] = bool(re.search(r'\b[A-Z]{2,}\b', text))
        
        return features
    
    def _analyze_multilingual_patterns(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze patterns specific to detected language"""
        features = {}
        
        if language not in self.multilingual_patterns:
            return features
        
        patterns = self.multilingual_patterns[language]
        
        # Check for language-specific chapter markers
        if 'chapter_markers' in patterns:
            features['has_chapter_marker'] = any(marker in text for marker in patterns['chapter_markers'])
        
        # Check for language-specific section patterns
        if 'section_patterns' in patterns:
            features['matches_section_pattern'] = any(
                re.search(pattern, text, re.IGNORECASE) 
                for pattern in patterns['section_patterns']
            )
        
        # Check for language-specific numbering
        if 'numbering' in patterns:
            features['has_native_numbering'] = any(num in text for num in patterns['numbering'])
        
        return features
    
    def _detect_proper_nouns(self, text: str) -> bool:
        """Simple proper noun detection"""
        words = text.split()
        if not words:
            return False
        
        # Count capitalized words (excluding first word of sentence)
        capitalized_count = sum(1 for i, word in enumerate(words) if i > 0 and word and word[0].isupper() and word.lower() not in self.stop_words)
        
        return capitalized_count > 0
    
    def _detect_technical_terms(self, text_lower: str) -> bool:
        """Detect technical terminology"""
        technical_indicators = [
            'algorithm', 'system', 'method', 'framework', 'model', 'architecture',
            'protocol', 'interface', 'implementation', 'specification', 'api',
            'database', 'server', 'client', 'network', 'security', 'encryption',
            'authentication', 'authorization', 'configuration', 'deployment'
        ]
        
        return any(term in text_lower for term in technical_indicators)
    
    def _calculate_heading_probability(self, features: Dict[str, Any]) -> float:
        """Calculate overall probability that text is a heading"""
        score = 0.0
        
        # Length factors
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 10:
            score += 0.3
        elif word_count <= 15:
            score += 0.1
        elif word_count > 20:
            score -= 0.2
        
        # Formatting factors
        if features.get('first_word_capitalized', False):
            score += 0.1
        if features.get('title_case', False):
            score += 0.2
        if features.get('all_caps', False) and word_count <= 8:
            score += 0.3
        
        # Pattern factors
        if features.get('section_pattern'):
            score += 0.4
        if features.get('has_number_prefix', False):
            score += 0.2
        
        # Content factors
        vocab_score = features.get('max_vocabulary_score', 0)
        score += vocab_score * 0.3
        
        # Multilingual factors
        if features.get('has_chapter_marker', False):
            score += 0.3
        if features.get('matches_section_pattern', False):
            score += 0.4
        
        # Punctuation factors
        if features.get('ends_with_colon', False):
            score += 0.1
        if not features.get('ends_with_period', False) and word_count > 3:
            score += 0.1  # Headings often don't end with periods
        
        # Stop word factor (headings tend to have fewer stop words)
        stop_word_ratio = features.get('stop_word_ratio', 0.5)
        if stop_word_ratio < 0.3:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _suggest_heading_level(self, features: Dict[str, Any]) -> Optional[str]:
        """Suggest heading level based on features"""
        section_level = features.get('section_level', 0)
        
        if section_level > 0:
            return f"H{min(section_level, 3)}"
        
        # Use vocabulary and pattern analysis
        vocab_scores = features.get('vocabulary_scores', {})
        max_score = features.get('max_vocabulary_score', 0)
        
        if max_score > 0.3:
            # High-level concepts suggest H1
            if any(features.get(f'is_{t}', False) for t in ['question', 'definition']):
                return "H1"
            # Implementation details suggest H3
            elif any(features.get(f'is_{t}', False) for t in ['instruction', 'example']):
                return "H3"
            else:
                return "H2"
        
        # Default based on capitalization and length
        word_count = features.get('word_count', 0)
        if features.get('all_caps', False) and word_count <= 5:
            return "H1"
        elif features.get('title_case', False) and word_count <= 8:
            return "H2"
        else:
            return "H3"
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence
        
        # Strong indicators increase confidence
        if features.get('section_pattern'):
            confidence += 0.3
        if features.get('max_vocabulary_score', 0) > 0.4:
            confidence += 0.2
        if features.get('has_chapter_marker', False):
            confidence += 0.3
        
        # Consistent formatting increases confidence
        if (features.get('title_case', False) or features.get('all_caps', False) or features.get('first_word_capitalized', False)):
            confidence += 0.1
        
        # Appropriate length increases confidence
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 12:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def analyze_document_structure(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall document structure and heading relationships"""
        if not blocks:
            return {}
        
        # Analyze each block
        analyzed_blocks = []
        for block in blocks:
            text = block.get('text', '')
            semantic_analysis = self.analyze_text_semantics(text)
            
            # Combine with existing block data
            enhanced_block = block.copy()
            enhanced_block['semantic_analysis'] = semantic_analysis
            analyzed_blocks.append(enhanced_block)
        
        # Document-level analysis
        document_analysis = {
            'total_blocks': len(analyzed_blocks),
            'potential_headings': sum(1 for b in analyzed_blocks 
                                    if b['semantic_analysis']['heading_probability'] > 0.3),
            'language_distribution': self._analyze_language_distribution(analyzed_blocks),
            'heading_hierarchy': self._analyze_heading_hierarchy(analyzed_blocks),
            'content_domains': self._analyze_content_domains(analyzed_blocks)
        }
        
        return {
            'blocks': analyzed_blocks,
            'document_analysis': document_analysis
        }
    
    def _analyze_language_distribution(self, blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze language distribution in document"""
        languages = [block['semantic_analysis']['language'] for block in blocks]
        return dict(Counter(languages))
    
    def _analyze_heading_hierarchy(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential heading hierarchy"""
        heading_blocks = [b for b in blocks if b['semantic_analysis']['heading_probability'] > 0.3]
        
        levels = [b['semantic_analysis']['suggested_level'] for b in heading_blocks if b['semantic_analysis']['suggested_level']]
        
        return {
            'level_distribution': dict(Counter(levels)),
            'total_potential_headings': len(heading_blocks),
            'hierarchy_depth': len(set(levels))
        }
    
    def _analyze_content_domains(self, blocks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze dominant content domains in document"""
        domain_scores = defaultdict(float)
        
        for block in blocks:
            vocab_scores = block['semantic_analysis']['features'].get('vocabulary_scores', {})
            for domain, score in vocab_scores.items():
                domain_scores[domain] += score
        
        total = sum(domain_scores.values())
        if total > 0:
            return {domain: score/total for domain, score in domain_scores.items()}
        
        return {}
    
    def analyze_text_semantics(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive semantic analysis of text
        
        Args:
            text: Text to analyze
            context: Additional context information
            
        Returns:
            Semantic analysis results
        """
        if not text.strip():
            return {'is_heading_likely': False, 'confidence': 0.0}
        
        analysis = {
            'text': text.strip(),
            'length': len(text),
            'word_count': len(text.split()),
            'language': self._detect_language(text),
            'heading_probability': 0.0,
            'suggested_level': None,
            'confidence': 0.0,
            'features': {}
        }
        
        # Extract linguistic features
        linguistic_features = self._extract_linguistic_features(text)
        analysis['features'].update(linguistic_features)
        
        # Pattern-based analysis
        pattern_analysis = self._analyze_patterns(text)
        analysis['features'].update(pattern_analysis)
        
        # Semantic analysis
        semantic_features = self._analyze_semantics(text)
        analysis['features'].update(semantic_features)
        
        # Multilingual analysis
        if analysis['language'] != 'english':
            multilingual_features = self._analyze_multilingual_patterns(text, analysis['language'])
            analysis['features'].update(multilingual_features)
        
        # Calculate overall heading probability
        analysis['heading_probability'] = self._calculate_heading_probability(analysis['features'])
        analysis['suggested_level'] = self._suggest_heading_level(analysis['features'])
        analysis['confidence'] = self._calculate_confidence(analysis['features'])
        
        return analysis
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        text = text.lower()
        
        # Japanese detection (Hiragana, Katakana, Kanji)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'japanese'
        
        # Chinese detection
        if re.search(r'[\u4E00-\u9FFF]', text):
            return 'chinese'
        
        # French detection
        french_chars = len(re.findall(r'[àâäçéèêëîïôöùûüÿ]', text))
        if french_chars > 0 or any(word in text for word in ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou']):
            return 'french'
        
        # German detection
        german_chars = len(re.findall(r'[äöüß]', text))
        if german_chars > 0 or any(word in text for word in ['der', 'die', 'das', 'und', 'oder', 'mit']):
            return 'german'
        
        # Spanish detection
        spanish_chars = len(re.findall(r'[áéíóúüñ]', text))
        if spanish_chars > 0 or any(word in text for word in ['el', 'la', 'los', 'las', 'y', 'o', 'con']):
            return 'spanish'
        
        return 'english'
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract basic linguistic features"""
        features = {}
        
        words = text.split()
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['sentence_count'] = len(sent_tokenize(text)) if text else 0
        
        # Character type ratios
        features['alpha_ratio'] = sum(c.isalpha() for c in text) / len(text) if text else 0
        features['digit_ratio'] = sum(c.isdigit() for c in text) / len(text) if text else 0
        features['punct_ratio'] = sum(c in '.,;:!?()[]{}"\'-' for c in text) / len(text) if text else 0
        features['space_ratio'] = sum(c.isspace() for c in text) / len(text) if text else 0
        features['upper_ratio'] = sum(c.isupper() for c in text) / len(text) if text else 0
        
        # Word-level features
        if words:
            features['first_word_capitalized'] = words[0][0].isupper() if words[0] else False
            features['all_words_capitalized'] = all(w[0].isupper() for w in words if w)
            features['title_case'] = text.istitle()
            features['all_caps'] = text.isupper()
            
            # Stop word analysis
            if self.stop_words:
                stop_word_count = sum(1 for w in words if w.lower() in self.stop_words)
                features['stop_word_ratio'] = stop_word_count / len(words)
            else:
                features['stop_word_ratio'] = 0
        
        return features
    
    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text patterns for heading indicators"""
        features = {}
        text_lower = text.lower().strip()
        
        # Numbering patterns
        features['has_number_prefix'] = bool(re.match(r'^\d+', text_lower))
        features['has_roman_numeral'] = bool(re.match(r'^[ivx]+\b', text_lower))
        features['has_letter_prefix'] = bool(re.match(r'^[a-z]\)', text_lower))
        
        # Section numbering patterns
        section_patterns = [
            (r'^\d+\.\s+', 'decimal_1'),
            (r'^\d+\.\d+\.\s+', 'decimal_2'),
            (r'^\d+\.\d+\.\d+\.\s+', 'decimal_3'),
            (r'^\d+\)\s+', 'paren_number'),
            (r'^[A-Z]\.\s+', 'letter_dot'),
            (r'^\([a-z]\)\s+', 'paren_letter'),
            (r'^[IVX]+\.\s+', 'roman_dot'),
        ]
        
        features['section_pattern'] = None
        features['section_level'] = 0
        
        for pattern, pattern_name in section_patterns:
            if re.match(pattern, text):
                features['section_pattern'] = pattern_name
                if 'decimal' in pattern_name:
                    features['section_level'] = int(pattern_name[-1])
                elif pattern_name in ['decimal_1', 'paren_number', 'letter_dot', 'roman_dot']:
                    features['section_level'] = 1
                elif pattern_name in ['paren_letter']:
                    features['section_level'] = 2
                break
        
        # Punctuation patterns
        features['ends_with_colon'] = text.endswith(':')
        features['ends_with_period'] = text.endswith('.')
        features['has_question_mark'] = '?' in text
        features['has_exclamation'] = '!' in text
        
        # Special formatting
        features['has_parentheses'] = '(' in text and ')' in text
        features['has_brackets'] = '[' in text and ']' in text
        features['has_quotes'] = '"' in text or "'" in text
        
        return features
    
    def _analyze_semantics(self, text: str) -> Dict[str, Any]:
        """Analyze semantic content for heading likelihood"""
        features = {}
        text_lower = text.lower()
        
        # Check against heading vocabularies
        vocabulary_scores = {}
        for domain, vocab in self.heading_vocabularies.items():
            words = set(text_lower.split())
            matches = words.intersection(vocab)
            vocabulary_scores[domain] = len(matches) / len(words) if words else 0
        
        features['vocabulary_scores'] = vocabulary_scores
        features['max_vocabulary_score'] = max(vocabulary_scores.values()) if vocabulary_scores else 0
        features['dominant_domain'] = max(vocabulary_scores.keys(), key=vocabulary_scores.get) if vocabulary_scores else None
        
        # Content type analysis
        content_indicators = {
            'question': ['what', 'how', 'why', 'when', 'where', 'who', 'which'],
            'instruction': ['step', 'procedure', 'process', 'method', 'approach'],
            'list': ['list', 'items', 'elements', 'components', 'parts'],
            'definition': ['definition', 'meaning', 'concept', 'term', 'explanation'],
            'example': ['example', 'instance', 'case', 'illustration', 'sample'],
            'comparison': ['versus', 'vs', 'compared', 'difference', 'similarity']
        }
        
        for content_type, indicators in content_indicators.items():
            features[f'is_{content_type}'] = any(indicator in text_lower for indicator in indicators)
        
        # Named entity-like analysis (simple version)
        features['has_proper_nouns'] = self._detect_proper_nouns(text)
        features['has_technical_terms'] = self._detect_technical_terms(text_lower)
        features['has_acronyms'] = bool(re.search(r'\b[A-Z]{2,}\b', text))
        
        return features
    
    def _analyze_multilingual_patterns(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze patterns specific to detected language"""
        features = {}
        
        if language not in self.multilingual_patterns:
            return features
        
        patterns = self.multilingual_patterns[language]
        
        # Check for language-specific chapter markers
        if 'chapter_markers' in patterns:
            features['has_chapter_marker'] = any(marker in text for marker in patterns['chapter_markers'])
        
        # Check for language-specific section patterns
        if 'section_patterns' in patterns:
            features['matches_section_pattern'] = any(
                re.search(pattern, text, re.IGNORECASE) 
                for pattern in patterns['section_patterns']
            )
        
        # Check for language-specific numbering
        if 'numbering' in patterns:
            features['has_native_numbering'] = any(num in text for num in patterns['numbering'])
        
        return features
    
    def _detect_proper_nouns(self, text: str) -> bool:
        """Simple proper noun detection"""
        words = text.split()
        if not words:
            return False
        
        # Count capitalized words (excluding first word of sentence)
        capitalized_count = sum(1 for i, word in enumerate(words) if i > 0 and word and word[0].isupper() and word.lower() not in self.stop_words)
        
        return capitalized_count > 0
    
    def _detect_technical_terms(self, text_lower: str) -> bool:
        """Detect technical terminology"""
        technical_indicators = [
            'algorithm', 'system', 'method', 'framework', 'model', 'architecture',
            'protocol', 'interface', 'implementation', 'specification', 'api',
            'database', 'server', 'client', 'network', 'security', 'encryption',
            'authentication', 'authorization', 'configuration', 'deployment'
        ]
        
        return any(term in text_lower for term in technical_indicators)
    
    def _calculate_heading_probability(self, features: Dict[str, Any]) -> float:
        """Calculate overall probability that text is a heading"""
        score = 0.0
        
        # Length factors
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 10:
            score += 0.3
        elif word_count <= 15:
            score += 0.1
        elif word_count > 20:
            score -= 0.2
        
        # Formatting factors
        if features.get('first_word_capitalized', False):
            score += 0.1
        if features.get('title_case', False):
            score += 0.2
        if features.get('all_caps', False) and word_count <= 8:
            score += 0.3
        
        # Pattern factors
        if features.get('section_pattern'):
            score += 0.4
        if features.get('has_number_prefix', False):
            score += 0.2
        
        # Content factors
        vocab_score = features.get('max_vocabulary_score', 0)
        score += vocab_score * 0.3
        
        # Multilingual factors
        if features.get('has_chapter_marker', False):
            score += 0.3
        if features.get('matches_section_pattern', False):
            score += 0.4
        
        # Punctuation factors
        if features.get('ends_with_colon', False):
            score += 0.1
        if not features.get('ends_with_period', False) and word_count > 3:
            score += 0.1  # Headings often don't end with periods
        
        # Stop word factor (headings tend to have fewer stop words)
        stop_word_ratio = features.get('stop_word_ratio', 0.5)
        if stop_word_ratio < 0.3:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _suggest_heading_level(self, features: Dict[str, Any]) -> Optional[str]:
        """Suggest heading level based on features"""
        section_level = features.get('section_level', 0)
        
        if section_level > 0:
            return f"H{min(section_level, 3)}"
        
        # Use vocabulary and pattern analysis
        vocab_scores = features.get('vocabulary_scores', {})
        max_score = features.get('max_vocabulary_score', 0)
        
        if max_score > 0.3:
            # High-level concepts suggest H1
            if any(features.get(f'is_{t}', False) for t in ['question', 'definition']):
                return "H1"
            # Implementation details suggest H3
            elif any(features.get(f'is_{t}', False) for t in ['instruction', 'example']):
                return "H3"
            else:
                return "H2"
        
        # Default based on capitalization and length
        word_count = features.get('word_count', 0)
        if features.get('all_caps', False) and word_count <= 5:
            return "H1"
        elif features.get('title_case', False) and word_count <= 8:
            return "H2"
        else:
            return "H3"
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence
        
        # Strong indicators increase confidence
        if features.get('section_pattern'):
            confidence += 0.3
        if features.get('max_vocabulary_score', 0) > 0.4:
            confidence += 0.2
        if features.get('has_chapter_marker', False):
            confidence += 0.3
        
        # Consistent formatting increases confidence
        if (features.get('title_case', False) or 
            features.get('all_caps', False) or 
            features.get('first_word_capitalized', False)):
            confidence += 0.1
        
        # Appropriate length increases confidence
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 12:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def analyze_document_structure(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall document structure and heading relationships"""
        if not blocks:
            return {}
        
        # Analyze each block
        analyzed_blocks = []
        for block in blocks:
            text = block.get('text', '')
            semantic_analysis = self.analyze_text_semantics(text)
            
            # Combine with existing block data
            enhanced_block = block.copy()
            enhanced_block['semantic_analysis'] = semantic_analysis
            analyzed_blocks.append(enhanced_block)
        
        # Document-level analysis
        document_analysis = {
            'total_blocks': len(analyzed_blocks),
            'potential_headings': sum(1 for b in analyzed_blocks 
                                    if b['semantic_analysis']['heading_probability'] > 0.3),
            'language_distribution': self._analyze_language_distribution(analyzed_blocks),
            'heading_hierarchy': self._analyze_heading_hierarchy(analyzed_blocks),
            'content_domains': self._analyze_content_domains(analyzed_blocks)
        }
        
        return {
            'blocks': analyzed_blocks,
            'document_analysis': document_analysis
        }
    
    def _analyze_language_distribution(self, blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze language distribution in document"""
        languages = [block['semantic_analysis']['language'] for block in blocks]
        return dict(Counter(languages))
    
    def _analyze_heading_hierarchy(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential heading hierarchy"""
        heading_blocks = [b for b in blocks if b['semantic_analysis']['heading_probability'] > 0.3]
        
        levels = [b['semantic_analysis']['suggested_level'] for b in heading_blocks if b['semantic_analysis']['suggested_level']]
        
        return {
            'level_distribution': dict(Counter(levels)),
            'total_potential_headings': len(heading_blocks),
            'hierarchy_depth': len(set(levels))
        }
    
    def _analyze_content_domains(self, blocks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze dominant content domains in document"""
        domain_scores = defaultdict(float)
        
        for block in blocks:
            vocab_scores = block['semantic_analysis']['features'].get('vocabulary_scores', {})
            for domain, score in vocab_scores.items():
                domain_scores[domain] += score
        
        total = sum(domain_scores.values())
        if total > 0:
            return {domain: score/total for domain, score in domain_scores.items()}
        
        return {}
    
    def analyze_text_semantics(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive semantic analysis of text
        
        Args:
            text: Text to analyze
            context: Additional context information
            
        Returns:
            Semantic analysis results
        """
        if not text.strip():
            return {'is_heading_likely': False, 'confidence': 0.0}
        
        analysis = {
            'text': text.strip(),
            'length': len(text),
            'word_count': len(text.split()),
            'language': self._detect_language(text),
            'heading_probability': 0.0,
            'suggested_level': None,
            'confidence': 0.0,
            'features': {}
        }
        
        # Extract linguistic features
        linguistic_features = self._extract_linguistic_features(text)
        analysis['features'].update(linguistic_features)
        
        # Pattern-based analysis
        pattern_analysis = self._analyze_patterns(text)
        analysis['features'].update(pattern_analysis)
        
        # Semantic analysis
        semantic_features = self._analyze_semantics(text)
        analysis['features'].update(semantic_features)
        
        # Multilingual analysis
        if analysis['language'] != 'english':
            multilingual_features = self._analyze_multilingual_patterns(text, analysis['language'])
            analysis['features'].update(multilingual_features)
        
        # Calculate overall heading probability
        analysis['heading_probability'] = self._calculate_heading_probability(analysis['features'])
        analysis['suggested_level'] = self._suggest_heading_level(analysis['features'])
        analysis['confidence'] = self._calculate_confidence(analysis['features'])
        
        return analysis
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        text = text.lower()
        
        # Japanese detection (Hiragana, Katakana, Kanji)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'japanese'
        
        # Chinese detection
        if re.search(r'[\u4E00-\u9FFF]', text):
            return 'chinese'
        
        # French detection
        french_chars = len(re.findall(r'[àâäçéèêëîïôöùûüÿ]', text))
        if french_chars > 0 or any(word in text for word in ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou']):
            return 'french'
        
        # German detection
        german_chars = len(re.findall(r'[äöüß]', text))
        if german_chars > 0 or any(word in text for word in ['der', 'die', 'das', 'und', 'oder', 'mit']):
            return 'german'
        
        # Spanish detection
        spanish_chars = len(re.findall(r'[áéíóúüñ]', text))
        if spanish_chars > 0 or any(word in text for word in ['el', 'la', 'los', 'las', 'y', 'o', 'con']):
            return 'spanish'
        
        return 'english'
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract basic linguistic features"""
        features = {}
        
        words = text.split()
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['sentence_count'] = len(sent_tokenize(text)) if text else 0
        
        # Character type ratios
        features['alpha_ratio'] = sum(c.isalpha() for c in text) / len(text) if text else 0
        features['digit_ratio'] = sum(c.isdigit() for c in text) / len(text) if text else 0
        features['punct_ratio'] = sum(c in '.,;:!?()[]{}"\'-' for c in text) / len(text) if text else 0
        features['space_ratio'] = sum(c.isspace() for c in text) / len(text) if text else 0
        features['upper_ratio'] = sum(c.isupper() for c in text) / len(text) if text else 0
        
        # Word-level features
        if words:
            features['first_word_capitalized'] = words[0][0].isupper() if words[0] else False
            features['all_words_capitalized'] = all(w[0].isupper() for w in words if w)
            features['title_case'] = text.istitle()
            features['all_caps'] = text.isupper()
            
            # Stop word analysis
            if self.stop_words:
                stop_word_count = sum(1 for w in words if w.lower() in self.stop_words)
                features['stop_word_ratio'] = stop_word_count / len(words)
            else:
                features['stop_word_ratio'] = 0
        
        return features
    
    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text patterns for heading indicators"""
        features = {}
        text_lower = text.lower().strip()
        
        # Numbering patterns
        features['has_number_prefix'] = bool(re.match(r'^\d+', text_lower))
        features['has_roman_numeral'] = bool(re.match(r'^[ivx]+\b', text_lower))
        features['has_letter_prefix'] = bool(re.match(r'^[a-z]\)', text_lower))
        
        # Section numbering patterns
        section_patterns = [
            (r'^\d+\.\s+', 'decimal_1'),
            (r'^\d+\.\d+\.\s+', 'decimal_2'),
            (r'^\d+\.\d+\.\d+\.\s+', 'decimal_3'),
            (r'^\d+\)\s+', 'paren_number'),
            (r'^[A-Z]\.\s+', 'letter_dot'),
            (r'^\([a-z]\)\s+', 'paren_letter'),
            (r'^[IVX]+\.\s+', 'roman_dot'),
        ]
        
        features['section_pattern'] = None
        features['section_level'] = 0
        
        for pattern, pattern_name in section_patterns:
            if re.match(pattern, text):
                features['section_pattern'] = pattern_name
                if 'decimal' in pattern_name:
                    features['section_level'] = int(pattern_name[-1])
                elif pattern_name in ['decimal_1', 'paren_number', 'letter_dot', 'roman_dot']:
                    features['section_level'] = 1
                elif pattern_name in ['paren_letter']:
                    features['section_level'] = 2
                break
        
        # Punctuation patterns
        features['ends_with_colon'] = text.endswith(':')
        features['ends_with_period'] = text.endswith('.')
        features['has_question_mark'] = '?' in text
        features['has_exclamation'] = '!' in text
        
        # Special formatting
        features['has_parentheses'] = '(' in text and ')' in text
        features['has_brackets'] = '[' in text and ']' in text
        features['has_quotes'] = '"' in text or "'" in text
        
        return features
    
    def _analyze_semantics(self, text: str) -> Dict[str, Any]:
        """Analyze semantic content for heading likelihood"""
        features = {}
        text_lower = text.lower()
        
        # Check against heading vocabularies
        vocabulary_scores = {}
        for domain, vocab in self.heading_vocabularies.items():
            words = set(text_lower.split())
            matches = words.intersection(vocab)
            vocabulary_scores[domain] = len(matches) / len(words) if words else 0
        
        features['vocabulary_scores'] = vocabulary_scores
        features['max_vocabulary_score'] = max(vocabulary_scores.values()) if vocabulary_scores else 0
        features['dominant_domain'] = max(vocabulary_scores.keys(), key=vocabulary_scores.get) if vocabulary_scores else None
        
        # Content type analysis
        content_indicators = {
            'question': ['what', 'how', 'why', 'when', 'where', 'who', 'which'],
            'instruction': ['step', 'procedure', 'process', 'method', 'approach'],
            'list': ['list', 'items', 'elements', 'components', 'parts'],
            'definition': ['definition', 'meaning', 'concept', 'term', 'explanation'],
            'example': ['example', 'instance', 'case', 'illustration', 'sample'],
            'comparison': ['versus', 'vs', 'compared', 'difference', 'similarity']
        }
        
        for content_type, indicators in content_indicators.items():
            features[f'is_{content_type}'] = any(indicator in text_lower for indicator in indicators)
        
        # Named entity-like analysis (simple version)
        features['has_proper_nouns'] = self._detect_proper_nouns(text)
        features['has_technical_terms'] = self._detect_technical_terms(text_lower)
        features['has_acronyms'] = bool(re.search(r'\b[A-Z]{2,}\b', text))
        
        return features
    
    def _analyze_multilingual_patterns(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze patterns specific to detected language"""
        features = {}
        
        if language not in self.multilingual_patterns:
            return features
        
        patterns = self.multilingual_patterns[language]
        
        # Check for language-specific chapter markers
        if 'chapter_markers' in patterns:
            features['has_chapter_marker'] = any(marker in text for marker in patterns['chapter_markers'])
        
        # Check for language-specific section patterns
        if 'section_patterns' in patterns:
            features['matches_section_pattern'] = any(
                re.search(pattern, text, re.IGNORECASE) 
                for pattern in patterns['section_patterns']
            )
        
        # Check for language-specific numbering
        if 'numbering' in patterns:
            features['has_native_numbering'] = any(num in text for num in patterns['numbering'])
        
        return features
    
    def _detect_proper_nouns(self, text: str) -> bool:
        """Simple proper noun detection"""
        words = text.split()
        if not words:
            return False
        
        # Count capitalized words (excluding first word of sentence)
        capitalized_count = sum(1 for i, word in enumerate(words) if i > 0 and word and word[0].isupper() and word.lower() not in self.stop_words)
        
        return capitalized_count > 0
    
    def _detect_technical_terms(self, text_lower: str) -> bool:
        """Detect technical terminology"""
        technical_indicators = [
            'algorithm', 'system', 'method', 'framework', 'model', 'architecture',
            'protocol', 'interface', 'implementation', 'specification', 'api',
            'database', 'server', 'client', 'network', 'security', 'encryption',
            'authentication', 'authorization', 'configuration', 'deployment'
        ]
        
        return any(term in text_lower for term in technical_indicators)
    
    def _calculate_heading_probability(self, features: Dict[str, Any]) -> float:
        """Calculate overall probability that text is a heading"""
        score = 0.0
        
        # Length factors
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 10:
            score += 0.3
        elif word_count <= 15:
            score += 0.1
        elif word_count > 20:
            score -= 0.2
        
        # Formatting factors
        if features.get('first_word_capitalized', False):
            score += 0.1
        if features.get('title_case', False):
            score += 0.2
        if features.get('all_caps', False) and word_count <= 8:
            score += 0.3
        
        # Pattern factors
        if features.get('section_pattern'):
            score += 0.4
        if features.get('has_number_prefix', False):
            score += 0.2
        
        # Content factors
        vocab_score = features.get('max_vocabulary_score', 0)
        score += vocab_score * 0.3
        
        # Multilingual factors
        if features.get('has_chapter_marker', False):
            score += 0.3
        if features.get('matches_section_pattern', False):
            score += 0.4
        
        # Punctuation factors
        if features.get('ends_with_colon', False):
            score += 0.1
        if not features.get('ends_with_period', False) and word_count > 3:
            score += 0.1  # Headings often don't end with periods
        
        # Stop word factor (headings tend to have fewer stop words)
        stop_word_ratio = features.get('stop_word_ratio', 0.5)
        if stop_word_ratio < 0.3:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _suggest_heading_level(self, features: Dict[str, Any]) -> Optional[str]:
        """Suggest heading level based on features"""
        section_level = features.get('section_level', 0)
        
        if section_level > 0:
            return f"H{min(section_level, 3)}"
        
        # Use vocabulary and pattern analysis
        vocab_scores = features.get('vocabulary_scores', {})
        max_score = features.get('max_vocabulary_score', 0)
        
        if max_score > 0.3:
            # High-level concepts suggest H1
            if any(features.get(f'is_{t}', False) for t in ['question', 'definition']):
                return "H1"
            # Implementation details suggest H3
            elif any(features.get(f'is_{t}', False) for t in ['instruction', 'example']):
                return "H3"
            else:
                return "H2"
        
        # Default based on capitalization and length
        word_count = features.get('word_count', 0)
        if features.get('all_caps', False) and word_count <= 5:
            return "H1"
        elif features.get('title_case', False) and word_count <= 8:
            return "H2"
        else:
            return "H3"
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence
        
        # Strong indicators increase confidence
        if features.get('section_pattern'):
            confidence += 0.3
        if features.get('max_vocabulary_score', 0) > 0.4:
            confidence += 0.2
        if features.get('has_chapter_marker', False):
            confidence += 0.3
        
        # Consistent formatting increases confidence
        if (features.get('title_case', False) or 
            features.get('all_caps', False) or 
            features.get('first_word_capitalized', False)):
            confidence += 0.1
        
        # Appropriate length increases confidence
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 12:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def analyze_document_structure(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall document structure and heading relationships"""
        if not blocks:
            return {}
        
        # Analyze each block
        analyzed_blocks = []
        for block in blocks:
            text = block.get('text', '')
            semantic_analysis = self.analyze_text_semantics(text)
            
            # Combine with existing block data
            enhanced_block = block.copy()
            enhanced_block['semantic_analysis'] = semantic_analysis
            analyzed_blocks.append(enhanced_block)
        
        # Document-level analysis
        document_analysis = {
            'total_blocks': len(analyzed_blocks),
            'potential_headings': sum(1 for b in analyzed_blocks 
                                    if b['semantic_analysis']['heading_probability'] > 0.3),
            'language_distribution': self._analyze_language_distribution(analyzed_blocks),
            'heading_hierarchy': self._analyze_heading_hierarchy(analyzed_blocks),
            'content_domains': self._analyze_content_domains(analyzed_blocks)
        }
        
        return {
            'blocks': analyzed_blocks,
            'document_analysis': document_analysis
        }
    
    def _analyze_language_distribution(self, blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze language distribution in document"""
        languages = [block['semantic_analysis']['language'] for block in blocks]
        return dict(Counter(languages))
    
    def _analyze_heading_hierarchy(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential heading hierarchy"""
        heading_blocks = [b for b in blocks if b['semantic_analysis']['heading_probability'] > 0.3]
        
        levels = [b['semantic_analysis']['suggested_level'] for b in heading_blocks if b['semantic_analysis']['suggested_level']]
        
        return {
            'level_distribution': dict(Counter(levels)),
            'total_potential_headings': len(heading_blocks),
            'hierarchy_depth': len(set(levels))
        }
    def _analyze_content_domains(self, blocks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze dominant content domains in document."""
        domain_scores = defaultdict(float)

        for block in blocks:
            vocab_scores = block.get('semantic_analysis', {}).get('features', {}).get('vocabulary_scores', {})
            for domain, score in vocab_scores.items():
                domain_scores[domain] += score

        total = sum(domain_scores.values())
        if total > 0:
            return {domain: score / total for domain, score in domain_scores.items()}

        return {}
    def enhanced_language_detection(self, text: str) -> Tuple[str, float]:
        """
        Enhanced language detection with confidence scoring
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language, confidence)
        """
        if not text.strip():
            return 'english', 0.0
        
        # Try langdetect first if available
        if self.langdetect_available:
            try:
                detected_lang = self.langdetect(text)
                # Map common language codes
                lang_mapping = {
                    'ja': 'japanese',
                    'zh': 'chinese', 
                    'zh-cn': 'chinese',
                    'fr': 'french',
                    'de': 'german',
                    'es': 'spanish',
                    'en': 'english'
                }
                return lang_mapping.get(detected_lang, 'english'), 0.8
            except self.LangDetectError:
                pass
        
        # Fallback to pattern-based detection
        return self._pattern_based_language_detection(text)
    
    def _pattern_based_language_detection(self, text: str) -> Tuple[str, float]:
        """Pattern-based language detection with confidence"""
        text_lower = text.lower()
        
        # Character-based detection
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'japanese', 0.9
        
        if re.search(r'[\u4E00-\u9FFF]', text):
            return 'chinese', 0.85
        
        # Word-based detection with confidence scoring
        language_scores = {}
        
        for lang, patterns in self.multilingual_patterns.items():
            score = 0.0
            
            # Check for language-specific markers
            if 'chapter_markers' in patterns:
                for marker in patterns['chapter_markers']:
                    if marker in text_lower:
                        score += 0.3
            
            # Check for common headings
            if 'common_headings' in patterns:
                for heading in patterns['common_headings']:
                    if heading in text_lower:
                        score += 0.2
            
            language_scores[lang] = score
        
        # Find best match
        if language_scores and max(language_scores.values()) > 0.1:
            best_lang = max(language_scores.keys(), key=language_scores.get)
            confidence = min(language_scores[best_lang], 0.9)
            return best_lang, confidence
        
        return 'english', 0.5
    
    def analyze_heading_likelihood(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced heading likelihood analysis with junk filtering
        
        Args:
            text: Text to analyze
            context: Additional context information
            
        Returns:
            Comprehensive analysis results
        """
        if not text.strip():
            return {'is_heading_likely': False, 'confidence': 0.0, 'rejection_reason': 'empty_text'}
        
        analysis = {
            'text': text.strip(),
            'length': len(text),
            'word_count': len(text.split()),
            'is_heading_likely': False,
            'confidence': 0.0,
            'features': {},
            'language_info': {},
            'junk_analysis': {}
        }
        
        # Step 1: Junk filtering
        is_junk, junk_reason = self.is_junk_text(text)
        analysis['junk_analysis'] = {
            'is_junk': is_junk,
            'reason': junk_reason
        }
        
        if is_junk:
            analysis['rejection_reason'] = f'junk_detected_{junk_reason}'
            return analysis
        
        # Step 2: Enhanced language detection
        detected_language, lang_confidence = self.enhanced_language_detection(text)
        analysis['language_info'] = {
            'detected_language': detected_language,
            'confidence': lang_confidence
        }
        
        # Step 3: Extract enhanced linguistic features
        linguistic_features = self._extract_enhanced_linguistic_features(text, detected_language)
        analysis['features'].update(linguistic_features)
        
        # Step 4: Advanced pattern analysis
        pattern_analysis = self._enhanced_pattern_analysis(text, detected_language)
        analysis['features'].update(pattern_analysis)
        
        # Step 5: Vocabulary and semantic analysis
        semantic_features = self._enhanced_semantic_analysis(text, detected_language)
        analysis['features'].update(semantic_features)
        
        # Step 6: Calculate enhanced heading probability
        analysis['heading_probability'] = self._calculate_enhanced_heading_probability(analysis['features'], detected_language)
        analysis['suggested_level'] = self._suggest_heading_level_enhanced(analysis['features'], detected_language)
        analysis['confidence'] = self._calculate_enhanced_confidence(analysis['features'], lang_confidence)
        
        # Final decision
        analysis['is_heading_likely'] = analysis['heading_probability'] >= 0.3
        
        return analysis
    
    def _extract_enhanced_linguistic_features(self, text: str, language: str) -> Dict[str, Any]:
        """Extract enhanced linguistic features with language awareness"""
        features = {}
        words = text.split()
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['sentence_count'] = len(sent_tokenize(text)) if text else 0
        
        # Character type analysis
        features['alpha_ratio'] = sum(c.isalpha() for c in text) / len(text) if text else 0
        features['digit_ratio'] = sum(c.isdigit() for c in text) / len(text) if text else 0
        features['punct_ratio'] = sum(c in '.,;:!?()[]{}"\'-' for c in text) / len(text) if text else 0
        features['upper_ratio'] = sum(c.isupper() for c in text) / len(text) if text else 0
        
        # Enhanced word-level features
        if words:
            features['first_word_capitalized'] = words[0][0].isupper() if words[0] else False
            features['all_words_capitalized'] = all(w[0].isupper() for w in words if w and w[0].isalpha())
            features['title_case'] = text.istitle()
            features['all_caps'] = text.isupper() and len(words) <= 8
            features['mixed_case'] = not text.islower() and not text.isupper()
            
            # Verb detection (simple heuristic)
            features['has_verbs'] = self._detect_verbs(words)
            
            # Stop word analysis (language-aware)
            if self.stop_words and language == 'english':
                stop_word_count = sum(1 for w in words if w.lower() in self.stop_words)
                features['stop_word_ratio'] = stop_word_count / len(words)
            else:
                features['stop_word_ratio'] = 0.3  # Default for non-English
        
        return features
    


    def _detect_verbs(self, words: List[str]) -> bool:
        """Simple verb detection heuristic based on common suffixes."""
        # Common verb suffix patterns
        verb_patterns = [
            r'.*ing$',   # running, walking
            r'.*ed$',    # played, walked
            r'.*es$',    # goes, does
            r'.*s$',     # runs, eats
        ]

        for word in words:
            word_lower = word.lower().strip()
            if any(re.match(pattern, word_lower) for pattern in verb_patterns):
                return True

        return False

    
    def analyze_text_semantics(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive semantic analysis of text
        
        Args:
            text: Text to analyze
            context: Additional context information
            
        Returns:
            Semantic analysis results
        """
        if not text.strip():
            return {'is_heading_likely': False, 'confidence': 0.0}
        
        analysis = {
            'text': text.strip(),
            'length': len(text),
            'word_count': len(text.split()),
            'language': self._detect_language(text),
            'heading_probability': 0.0,
            'suggested_level': None,
            'confidence': 0.0,
            'features': {}
        }
        
        # Extract linguistic features
        linguistic_features = self._extract_linguistic_features(text)
        analysis['features'].update(linguistic_features)
        
        # Pattern-based analysis
        pattern_analysis = self._analyze_patterns(text)
        analysis['features'].update(pattern_analysis)
        
        # Semantic analysis
        semantic_features = self._analyze_semantics(text)
        analysis['features'].update(semantic_features)
        
        # Multilingual analysis
        if analysis['language'] != 'english':
            multilingual_features = self._analyze_multilingual_patterns(text, analysis['language'])
            analysis['features'].update(multilingual_features)
        
        # Calculate overall heading probability
        analysis['heading_probability'] = self._calculate_heading_probability(analysis['features'])
        analysis['suggested_level'] = self._suggest_heading_level(analysis['features'])
        analysis['confidence'] = self._calculate_confidence(analysis['features'])
        
        return analysis
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        text = text.lower()
        
        # Japanese detection (Hiragana, Katakana, Kanji)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'japanese'
        
        # Chinese detection
        if re.search(r'[\u4E00-\u9FFF]', text):
            return 'chinese'
        
        # French detection
        french_chars = len(re.findall(r'[àâäçéèêëîïôöùûüÿ]', text))
        if french_chars > 0 or any(word in text for word in ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou']):
            return 'french'
        
        # German detection
        german_chars = len(re.findall(r'[äöüß]', text))
        if german_chars > 0 or any(word in text for word in ['der', 'die', 'das', 'und', 'oder', 'mit']):
            return 'german'
        
        # Spanish detection
        spanish_chars = len(re.findall(r'[áéíóúüñ]', text))
        if spanish_chars > 0 or any(word in text for word in ['el', 'la', 'los', 'las', 'y', 'o', 'con']):
            return 'spanish'
        
        return 'english'
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract basic linguistic features"""
        features = {}
        
        words = text.split()
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['sentence_count'] = len(sent_tokenize(text)) if text else 0
        
        # Character type ratios
        features['alpha_ratio'] = sum(c.isalpha() for c in text) / len(text) if text else 0
        features['digit_ratio'] = sum(c.isdigit() for c in text) / len(text) if text else 0
        features['punct_ratio'] = sum(c in '.,;:!?()[]{}"\'-' for c in text) / len(text) if text else 0
        features['space_ratio'] = sum(c.isspace() for c in text) / len(text) if text else 0
        features['upper_ratio'] = sum(c.isupper() for c in text) / len(text) if text else 0
        
        # Word-level features
        if words:
            features['first_word_capitalized'] = words[0][0].isupper() if words[0] else False
            features['all_words_capitalized'] = all(w[0].isupper() for w in words if w)
            features['title_case'] = text.istitle()
            features['all_caps'] = text.isupper()
            
            # Stop word analysis
            if self.stop_words:
                stop_word_count = sum(1 for w in words if w.lower() in self.stop_words)
                features['stop_word_ratio'] = stop_word_count / len(words)
            else:
                features['stop_word_ratio'] = 0
        
        return features
    
    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text patterns for heading indicators"""
        features = {}
        text_lower = text.lower().strip()
        
        # Numbering patterns
        features['has_number_prefix'] = bool(re.match(r'^\d+', text_lower))
        features['has_roman_numeral'] = bool(re.match(r'^[ivx]+\b', text_lower))
        features['has_letter_prefix'] = bool(re.match(r'^[a-z]\)', text_lower))
        
        # Section numbering patterns
        section_patterns = [
            (r'^\d+\.\s+', 'decimal_1'),
            (r'^\d+\.\d+\.\s+', 'decimal_2'),
            (r'^\d+\.\d+\.\d+\.\s+', 'decimal_3'),
            (r'^\d+\)\s+', 'paren_number'),
            (r'^[A-Z]\.\s+', 'letter_dot'),
            (r'^\([a-z]\)\s+', 'paren_letter'),
            (r'^[IVX]+\.\s+', 'roman_dot'),
        ]
        
        features['section_pattern'] = None
        features['section_level'] = 0
        
        for pattern, pattern_name in section_patterns:
            if re.match(pattern, text):
                features['section_pattern'] = pattern_name
                if 'decimal' in pattern_name:
                    features['section_level'] = int(pattern_name[-1])
                elif pattern_name in ['decimal_1', 'paren_number', 'letter_dot', 'roman_dot']:
                    features['section_level'] = 1
                elif pattern_name in ['paren_letter']:
                    features['section_level'] = 2
                break
        
        # Punctuation patterns
        features['ends_with_colon'] = text.endswith(':')
        features['ends_with_period'] = text.endswith('.')
        features['has_question_mark'] = '?' in text
        features['has_exclamation'] = '!' in text
        
        # Special formatting
        features['has_parentheses'] = '(' in text and ')' in text
        features['has_brackets'] = '[' in text and ']' in text
        features['has_quotes'] = '"' in text or "'" in text
        
        return features
    
    def _analyze_semantics(self, text: str) -> Dict[str, Any]:
        """Analyze semantic content for heading likelihood"""
        features = {}
        text_lower = text.lower()
        
        # Check against heading vocabularies
        vocabulary_scores = {}
        for domain, vocab in self.heading_vocabularies.items():
            words = set(text_lower.split())
            matches = words.intersection(vocab)
            vocabulary_scores[domain] = len(matches) / len(words) if words else 0
        
        features['vocabulary_scores'] = vocabulary_scores
        features['max_vocabulary_score'] = max(vocabulary_scores.values()) if vocabulary_scores else 0
        features['dominant_domain'] = max(vocabulary_scores.keys(), key=vocabulary_scores.get) if vocabulary_scores else None
        
        # Content type analysis
        content_indicators = {
            'question': ['what', 'how', 'why', 'when', 'where', 'who', 'which'],
            'instruction': ['step', 'procedure', 'process', 'method', 'approach'],
            'list': ['list', 'items', 'elements', 'components', 'parts'],
            'definition': ['definition', 'meaning', 'concept', 'term', 'explanation'],
            'example': ['example', 'instance', 'case', 'illustration', 'sample'],
            'comparison': ['versus', 'vs', 'compared', 'difference', 'similarity']
        }
        
        for content_type, indicators in content_indicators.items():
            features[f'is_{content_type}'] = any(indicator in text_lower for indicator in indicators)
        
        # Named entity-like analysis (simple version)
        features['has_proper_nouns'] = self._detect_proper_nouns(text)
        features['has_technical_terms'] = self._detect_technical_terms(text_lower)
        features['has_acronyms'] = bool(re.search(r'\b[A-Z]{2,}\b', text))
        
        return features
    
    def _analyze_multilingual_patterns(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze patterns specific to detected language"""
        features = {}
        
        if language not in self.multilingual_patterns:
            return features
        
        patterns = self.multilingual_patterns[language]
        
        # Check for language-specific chapter markers
        if 'chapter_markers' in patterns:
            features['has_chapter_marker'] = any(marker in text for marker in patterns['chapter_markers'])
        
        # Check for language-specific section patterns
        if 'section_patterns' in patterns:
            features['matches_section_pattern'] = any(
                re.search(pattern, text, re.IGNORECASE) 
                for pattern in patterns['section_patterns']
            )
        
        # Check for language-specific numbering
        if 'numbering' in patterns:
            features['has_native_numbering'] = any(num in text for num in patterns['numbering'])
        
        return features
    
    def _detect_proper_nouns(self, text: str) -> bool:
        """Simple proper noun detection"""
        words = text.split()
        if not words:
            return False
        
        # Count capitalized words (excluding first word of sentence)
        capitalized_count = sum(1 for i, word in enumerate(words) 
                              if i > 0 and word and word[0].isupper() and word.lower() not in self.stop_words)
        
        return capitalized_count > 0
    
    def _detect_technical_terms(self, text_lower: str) -> bool:
        """Detect technical terminology"""
        technical_indicators = [
            'algorithm', 'system', 'method', 'framework', 'model', 'architecture',
            'protocol', 'interface', 'implementation', 'specification', 'api',
            'database', 'server', 'client', 'network', 'security', 'encryption',
            'authentication', 'authorization', 'configuration', 'deployment'
        ]
        
        return any(term in text_lower for term in technical_indicators)
    
    def _calculate_heading_probability(self, features: Dict[str, Any]) -> float:
        """Calculate overall probability that text is a heading"""
        score = 0.0
        
        # Length factors
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 10:
            score += 0.3
        elif word_count <= 15:
            score += 0.1
        elif word_count > 20:
            score -= 0.2
        
        # Formatting factors
        if features.get('first_word_capitalized', False):
            score += 0.1
        if features.get('title_case', False):
            score += 0.2
        if features.get('all_caps', False) and word_count <= 8:
            score += 0.3
        
        # Pattern factors
        if features.get('section_pattern'):
            score += 0.4
        if features.get('has_number_prefix', False):
            score += 0.2
        
        # Content factors
        vocab_score = features.get('max_vocabulary_score', 0)
        score += vocab_score * 0.3
        
        # Multilingual factors
        if features.get('has_chapter_marker', False):
            score += 0.3
        if features.get('matches_section_pattern', False):
            score += 0.4
        
        # Punctuation factors
        if features.get('ends_with_colon', False):
            score += 0.1
        if not features.get('ends_with_period', False) and word_count > 3:
            score += 0.1  # Headings often don't end with periods
        
        # Stop word factor (headings tend to have fewer stop words)
        stop_word_ratio = features.get('stop_word_ratio', 0.5)
        if stop_word_ratio < 0.3:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _suggest_heading_level(self, features: Dict[str, Any]) -> Optional[str]:
        """Suggest heading level based on features"""
        section_level = features.get('section_level', 0)
        
        if section_level > 0:
            return f"H{min(section_level, 3)}"
        
        # Use vocabulary and pattern analysis
        vocab_scores = features.get('vocabulary_scores', {})
        max_score = features.get('max_vocabulary_score', 0)
        
        if max_score > 0.3:
            # High-level concepts suggest H1
            if any(features.get(f'is_{t}', False) for t in ['question', 'definition']):
                return "H1"
            # Implementation details suggest H3
            elif any(features.get(f'is_{t}', False) for t in ['instruction', 'example']):
                return "H3"
            else:
                return "H2"
        
        # Default based on capitalization and length
        word_count = features.get('word_count', 0)
        if features.get('all_caps', False) and word_count <= 5:
            return "H1"
        elif features.get('title_case', False) and word_count <= 8:
            return "H2"
        else:
            return "H3"
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence
        
        # Strong indicators increase confidence
        if features.get('section_pattern'):
            confidence += 0.3
        if features.get('max_vocabulary_score', 0) > 0.4:
            confidence += 0.2
        if features.get('has_chapter_marker', False):
            confidence += 0.3
        
        # Consistent formatting increases confidence
        if (features.get('title_case', False) or 
            features.get('all_caps', False) or 
            features.get('first_word_capitalized', False)):
            confidence += 0.1
        
        # Appropriate length increases confidence
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 12:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def analyze_document_structure(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall document structure and heading relationships"""
        if not blocks:
            return {}
        
        # Analyze each block
        analyzed_blocks = []
        for block in blocks:
            text = block.get('text', '')
            semantic_analysis = self.analyze_text_semantics(text)
            
            # Combine with existing block data
            enhanced_block = block.copy()
            enhanced_block['semantic_analysis'] = semantic_analysis
            analyzed_blocks.append(enhanced_block)
        
        # Document-level analysis
        document_analysis = {
            'total_blocks': len(analyzed_blocks),
            'potential_headings': sum(1 for b in analyzed_blocks 
                                    if b['semantic_analysis']['heading_probability'] > 0.3),
            'language_distribution': self._analyze_language_distribution(analyzed_blocks),
            'heading_hierarchy': self._analyze_heading_hierarchy(analyzed_blocks),
            'content_domains': self._analyze_content_domains(analyzed_blocks)
        }
        
        return {
            'blocks': analyzed_blocks,
            'document_analysis': document_analysis
        }
    
    def _analyze_language_distribution(self, blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze language distribution in document"""
        languages = [block['semantic_analysis']['language'] for block in blocks]
        return dict(Counter(languages))
    
    def _analyze_heading_hierarchy(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential heading hierarchy"""
        heading_blocks = [b for b in blocks 
                         if b['semantic_analysis']['heading_probability'] > 0.3]
        
        levels = [b['semantic_analysis']['suggested_level'] for b in heading_blocks 
                 if b['semantic_analysis']['suggested_level']]
        
        return {
            'level_distribution': dict(Counter(levels)),
            'total_potential_headings': len(heading_blocks),
            'hierarchy_depth': len(set(levels))
        }

    def _analyze_content_domains(self, blocks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze dominant content domains in document and return normalized scores."""
        domain_scores = defaultdict(float)

        for block in blocks:
            vocab_scores = block.get('semantic_analysis', {}).get('features', {}).get('vocabulary_scores', {})
            for domain, score in vocab_scores.items():
                domain_scores[domain] += score

        total = sum(domain_scores.values())
        if total > 0:
            return {domain: score / total for domain, score in domain_scores.items()}
        return {}

    
    def analyze_text_semantics(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive semantic analysis of text
        
        Args:
            text: Text to analyze
            context: Additional context information
            
        Returns:
            Semantic analysis results
        """
        if not text.strip():
            return {'is_heading_likely': False, 'confidence': 0.0}
        
        analysis = {
            'text': text.strip(),
            'length': len(text),
            'word_count': len(text.split()),
            'language': self._detect_language(text),
            'heading_probability': 0.0,
            'suggested_level': None,
            'confidence': 0.0,
            'features': {}
        }
        
        # Extract linguistic features
        linguistic_features = self._extract_linguistic_features(text)
        analysis['features'].update(linguistic_features)
        
        # Pattern-based analysis
        pattern_analysis = self._analyze_patterns(text)
        analysis['features'].update(pattern_analysis)
        
        # Semantic analysis
        semantic_features = self._analyze_semantics(text)
        analysis['features'].update(semantic_features)
        
        # Multilingual analysis
        if analysis['language'] != 'english':
            multilingual_features = self._analyze_multilingual_patterns(text, analysis['language'])
            analysis['features'].update(multilingual_features)
        
        # Calculate overall heading probability
        analysis['heading_probability'] = self._calculate_heading_probability(analysis['features'])
        analysis['suggested_level'] = self._suggest_heading_level(analysis['features'])
        analysis['confidence'] = self._calculate_confidence(analysis['features'])
        
        return analysis
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        text = text.lower()
        
        # Japanese detection (Hiragana, Katakana, Kanji)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'japanese'
        
        # Chinese detection
        if re.search(r'[\u4E00-\u9FFF]', text):
            return 'chinese'
        
        # French detection
        french_chars = len(re.findall(r'[àâäçéèêëîïôöùûüÿ]', text))
        if french_chars > 0 or any(word in text for word in ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou']):
            return 'french'
        
        # German detection
        german_chars = len(re.findall(r'[äöüß]', text))
        if german_chars > 0 or any(word in text for word in ['der', 'die', 'das', 'und', 'oder', 'mit']):
            return 'german'
        
        # Spanish detection
        spanish_chars = len(re.findall(r'[áéíóúüñ]', text))
        if spanish_chars > 0 or any(word in text for word in ['el', 'la', 'los', 'las', 'y', 'o', 'con']):
            return 'spanish'
        
        return 'english'
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract basic linguistic features"""
        features = {}
        
        words = text.split()
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['sentence_count'] = len(sent_tokenize(text)) if text else 0
        
        # Character type ratios
        features['alpha_ratio'] = sum(c.isalpha() for c in text) / len(text) if text else 0
        features['digit_ratio'] = sum(c.isdigit() for c in text) / len(text) if text else 0
        features['punct_ratio'] = sum(c in '.,;:!?()[]{}"\'-' for c in text) / len(text) if text else 0
        features['space_ratio'] = sum(c.isspace() for c in text) / len(text) if text else 0
        features['upper_ratio'] = sum(c.isupper() for c in text) / len(text) if text else 0
        
        # Word-level features
        if words:
            features['first_word_capitalized'] = words[0][0].isupper() if words[0] else False
            features['all_words_capitalized'] = all(w[0].isupper() for w in words if w)
            features['title_case'] = text.istitle()
            features['all_caps'] = text.isupper()
            
            # Stop word analysis
            if self.stop_words:
                stop_word_count = sum(1 for w in words if w.lower() in self.stop_words)
                features['stop_word_ratio'] = stop_word_count / len(words)
            else:
                features['stop_word_ratio'] = 0
        
        return features
    
    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text patterns for heading indicators"""
        features = {}
        text_lower = text.lower().strip()
        
        # Numbering patterns
        features['has_number_prefix'] = bool(re.match(r'^\d+', text_lower))
        features['has_roman_numeral'] = bool(re.match(r'^[ivx]+\b', text_lower))
        features['has_letter_prefix'] = bool(re.match(r'^[a-z]\)', text_lower))
        
        # Section numbering patterns
        section_patterns = [
            (r'^\d+\.\s+', 'decimal_1'),
            (r'^\d+\.\d+\.\s+', 'decimal_2'),
            (r'^\d+\.\d+\.\d+\.\s+', 'decimal_3'),
            (r'^\d+\)\s+', 'paren_number'),
            (r'^[A-Z]\.\s+', 'letter_dot'),
            (r'^\([a-z]\)\s+', 'paren_letter'),
            (r'^[IVX]+\.\s+', 'roman_dot'),
        ]
        
        features['section_pattern'] = None
        features['section_level'] = 0
        
        for pattern, pattern_name in section_patterns:
            if re.match(pattern, text):
                features['section_pattern'] = pattern_name
                if 'decimal' in pattern_name:
                    features['section_level'] = int(pattern_name[-1])
                elif pattern_name in ['decimal_1', 'paren_number', 'letter_dot', 'roman_dot']:
                    features['section_level'] = 1
                elif pattern_name in ['paren_letter']:
                    features['section_level'] = 2
                break
        
        # Punctuation patterns
        features['ends_with_colon'] = text.endswith(':')
        features['ends_with_period'] = text.endswith('.')
        features['has_question_mark'] = '?' in text
        features['has_exclamation'] = '!' in text
        
        # Special formatting
        features['has_parentheses'] = '(' in text and ')' in text
        features['has_brackets'] = '[' in text and ']' in text
        features['has_quotes'] = '"' in text or "'" in text
        
        return features
    
    def _analyze_semantics(self, text: str) -> Dict[str, Any]:
        """Analyze semantic content for heading likelihood"""
        features = {}
        text_lower = text.lower()
        
        # Check against heading vocabularies
        vocabulary_scores = {}
        for domain, vocab in self.heading_vocabularies.items():
            words = set(text_lower.split())
            matches = words.intersection(vocab)
            vocabulary_scores[domain] = len(matches) / len(words) if words else 0
        
        features['vocabulary_scores'] = vocabulary_scores
        features['max_vocabulary_score'] = max(vocabulary_scores.values()) if vocabulary_scores else 0
        features['dominant_domain'] = max(vocabulary_scores.keys(), key=vocabulary_scores.get) if vocabulary_scores else None
        
        # Content type analysis
        content_indicators = {
            'question': ['what', 'how', 'why', 'when', 'where', 'who', 'which'],
            'instruction': ['step', 'procedure', 'process', 'method', 'approach'],
            'list': ['list', 'items', 'elements', 'components', 'parts'],
            'definition': ['definition', 'meaning', 'concept', 'term', 'explanation'],
            'example': ['example', 'instance', 'case', 'illustration', 'sample'],
            'comparison': ['versus', 'vs', 'compared', 'difference', 'similarity']
        }
        
        for content_type, indicators in content_indicators.items():
            features[f'is_{content_type}'] = any(indicator in text_lower for indicator in indicators)
        
        # Named entity-like analysis (simple version)
        features['has_proper_nouns'] = self._detect_proper_nouns(text)
        features['has_technical_terms'] = self._detect_technical_terms(text_lower)
        features['has_acronyms'] = bool(re.search(r'\b[A-Z]{2,}\b', text))
        
        return features
    
    def _analyze_multilingual_patterns(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze patterns specific to detected language"""
        features = {}
        
        if language not in self.multilingual_patterns:
            return features
        
        patterns = self.multilingual_patterns[language]
        
        # Check for language-specific chapter markers
        if 'chapter_markers' in patterns:
            features['has_chapter_marker'] = any(marker in text for marker in patterns['chapter_markers'])
        
        # Check for language-specific section patterns
        if 'section_patterns' in patterns:
            features['matches_section_pattern'] = any(
                re.search(pattern, text, re.IGNORECASE) 
                for pattern in patterns['section_patterns']
            )
        
        # Check for language-specific numbering
        if 'numbering' in patterns:
            features['has_native_numbering'] = any(num in text for num in patterns['numbering'])
        
        return features
    
    def _detect_proper_nouns(self, text: str) -> bool:
        """Simple proper noun detection"""
        words = text.split()
        if not words:
            return False
        
        # Count capitalized words (excluding first word of sentence)
        capitalized_count = sum(1 for i, word in enumerate(words) 
                              if i > 0 and word and word[0].isupper() and word.lower() not in self.stop_words)
        
        return capitalized_count > 0
    
    def _detect_technical_terms(self, text_lower: str) -> bool:
        """Detect technical terminology"""
        technical_indicators = [
            'algorithm', 'system', 'method', 'framework', 'model', 'architecture',
            'protocol', 'interface', 'implementation', 'specification', 'api',
            'database', 'server', 'client', 'network', 'security', 'encryption',
            'authentication', 'authorization', 'configuration', 'deployment'
        ]
        
        return any(term in text_lower for term in technical_indicators)
    
    def _calculate_heading_probability(self, features: Dict[str, Any]) -> float:
        """Calculate overall probability that text is a heading"""
        score = 0.0
        
        # Length factors
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 10:
            score += 0.3
        elif word_count <= 15:
            score += 0.1
        elif word_count > 20:
            score -= 0.2
        
        # Formatting factors
        if features.get('first_word_capitalized', False):
            score += 0.1
        if features.get('title_case', False):
            score += 0.2
        if features.get('all_caps', False) and word_count <= 8:
            score += 0.3
        
        # Pattern factors
        if features.get('section_pattern'):
            score += 0.4
        if features.get('has_number_prefix', False):
            score += 0.2
        
        # Content factors
        vocab_score = features.get('max_vocabulary_score', 0)
        score += vocab_score * 0.3
        
        # Multilingual factors
        if features.get('has_chapter_marker', False):
            score += 0.3
        if features.get('matches_section_pattern', False):
            score += 0.4
        
        # Punctuation factors
        if features.get('ends_with_colon', False):
            score += 0.1
        if not features.get('ends_with_period', False) and word_count > 3:
            score += 0.1  # Headings often don't end with periods
        
        # Stop word factor (headings tend to have fewer stop words)
        stop_word_ratio = features.get('stop_word_ratio', 0.5)
        if stop_word_ratio < 0.3:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _suggest_heading_level(self, features: Dict[str, Any]) -> Optional[str]:
        """Suggest heading level based on features"""
        section_level = features.get('section_level', 0)
        
        if section_level > 0:
            return f"H{min(section_level, 3)}"
        
        # Use vocabulary and pattern analysis
        vocab_scores = features.get('vocabulary_scores', {})
        max_score = features.get('max_vocabulary_score', 0)
        
        if max_score > 0.3:
            # High-level concepts suggest H1
            if any(features.get(f'is_{t}', False) for t in ['question', 'definition']):
                return "H1"
            # Implementation details suggest H3
            elif any(features.get(f'is_{t}', False) for t in ['instruction', 'example']):
                return "H3"
            else:
                return "H2"
        
        # Default based on capitalization and length
        word_count = features.get('word_count', 0)
        if features.get('all_caps', False) and word_count <= 5:
            return "H1"
        elif features.get('title_case', False) and word_count <= 8:
            return "H2"
        else:
            return "H3"
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence
        
        # Strong indicators increase confidence
        if features.get('section_pattern'):
            confidence += 0.3
        if features.get('max_vocabulary_score', 0) > 0.4:
            confidence += 0.2
        if features.get('has_chapter_marker', False):
            confidence += 0.3
        
        # Consistent formatting increases confidence
        if (features.get('title_case', False) or 
            features.get('all_caps', False) or 
            features.get('first_word_capitalized', False)):
            confidence += 0.1
        
        # Appropriate length increases confidence
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 12:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def analyze_document_structure(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall document structure and heading relationships"""
        if not blocks:
            return {}
        
        # Analyze each block
        analyzed_blocks = []
        for block in blocks:
            text = block.get('text', '')
            semantic_analysis = self.analyze_text_semantics(text)
            
            # Combine with existing block data
            enhanced_block = block.copy()
            enhanced_block['semantic_analysis'] = semantic_analysis
            analyzed_blocks.append(enhanced_block)
        
        # Document-level analysis
        document_analysis = {
            'total_blocks': len(analyzed_blocks),
            'potential_headings': sum(1 for b in analyzed_blocks 
                                    if b['semantic_analysis']['heading_probability'] > 0.3),
            'language_distribution': self._analyze_language_distribution(analyzed_blocks),
            'heading_hierarchy': self._analyze_heading_hierarchy(analyzed_blocks),
            'content_domains': self._analyze_content_domains(analyzed_blocks)
        }
        
        return {
            'blocks': analyzed_blocks,
            'document_analysis': document_analysis
        }
    
    def _analyze_language_distribution(self, blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze language distribution in document"""
        languages = [block['semantic_analysis']['language'] for block in blocks]
        return dict(Counter(languages))
    
    def _analyze_heading_hierarchy(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential heading hierarchy"""
        heading_blocks = [b for b in blocks if b['semantic_analysis']['heading_probability'] > 0.3]
        
        levels = [b['semantic_analysis']['suggested_level'] for b in heading_blocks if b['semantic_analysis']['suggested_level']]
        
        return {
            'level_distribution': dict(Counter(levels)),
            'total_potential_headings': len(heading_blocks),
            'hierarchy_depth': len(set(levels))
        }
    
    def _analyze_content_domains(self, blocks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze dominant content domains in document"""
        domain_scores = defaultdict(float)
        
        for block in blocks:
            vocab_scores = block['semantic_analysis']['features'].get('vocabulary_scores', {})
            for domain, score in vocab_scores.items():
                domain_scores[domain] += score
        
        total = sum(domain_scores.values())
        if total > 0:
            return {domain: score/total for domain, score in domain_scores.items()}
        
        return {}
    
    def analyze_text_semantics(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive semantic analysis of text
        
        Args:
            text: Text to analyze
            context: Additional context information
            
        Returns:
            Semantic analysis results
        """
        if not text.strip():
            return {'is_heading_likely': False, 'confidence': 0.0}
        
        analysis = {
            'text': text.strip(),
            'length': len(text),
            'word_count': len(text.split()),
            'language': self._detect_language(text),
            'heading_probability': 0.0,
            'suggested_level': None,
            'confidence': 0.0,
            'features': {}
        }
        
        # Extract linguistic features
        linguistic_features = self._extract_linguistic_features(text)
        analysis['features'].update(linguistic_features)
        
        # Pattern-based analysis
        pattern_analysis = self._analyze_patterns(text)
        analysis['features'].update(pattern_analysis)
        
        # Semantic analysis
        semantic_features = self._analyze_semantics(text)
        analysis['features'].update(semantic_features)
        
        # Multilingual analysis
        if analysis['language'] != 'english':
            multilingual_features = self._analyze_multilingual_patterns(text, analysis['language'])
            analysis['features'].update(multilingual_features)
        
        # Calculate overall heading probability
        analysis['heading_probability'] = self._calculate_heading_probability(analysis['features'])
        analysis['suggested_level'] = self._suggest_heading_level(analysis['features'])
        analysis['confidence'] = self._calculate_confidence(analysis['features'])
        
        return analysis
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        text = text.lower()
        
        # Japanese detection (Hiragana, Katakana, Kanji)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'japanese'
        
        # Chinese detection
        if re.search(r'[\u4E00-\u9FFF]', text):
            return 'chinese'
        
        # French detection
        french_chars = len(re.findall(r'[àâäçéèêëîïôöùûüÿ]', text))
        if french_chars > 0 or any(word in text for word in ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou']):
            return 'french'
        
        # German detection
        german_chars = len(re.findall(r'[äöüß]', text))
        if german_chars > 0 or any(word in text for word in ['der', 'die', 'das', 'und', 'oder', 'mit']):
            return 'german'
        
        # Spanish detection
        spanish_chars = len(re.findall(r'[áéíóúüñ]', text))
        if spanish_chars > 0 or any(word in text for word in ['el', 'la', 'los', 'las', 'y', 'o', 'con']):
            return 'spanish'
        
        return 'english'
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract basic linguistic features"""
        features = {}
        
        words = text.split()
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['sentence_count'] = len(sent_tokenize(text)) if text else 0
        
        # Character type ratios
        features['alpha_ratio'] = sum(c.isalpha() for c in text) / len(text) if text else 0
        features['digit_ratio'] = sum(c.isdigit() for c in text) / len(text) if text else 0
        features['punct_ratio'] = sum(c in '.,;:!?()[]{}"\'-' for c in text) / len(text) if text else 0
        features['space_ratio'] = sum(c.isspace() for c in text) / len(text) if text else 0
        features['upper_ratio'] = sum(c.isupper() for c in text) / len(text) if text else 0
        
        # Word-level features
        if words:
            features['first_word_capitalized'] = words[0][0].isupper() if words[0] else False
            features['all_words_capitalized'] = all(w[0].isupper() for w in words if w)
            features['title_case'] = text.istitle()
            features['all_caps'] = text.isupper()
            
            # Stop word analysis
            if self.stop_words:
                stop_word_count = sum(1 for w in words if w.lower() in self.stop_words)
                features['stop_word_ratio'] = stop_word_count / len(words)
            else:
                features['stop_word_ratio'] = 0
        
        return features
    
    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text patterns for heading indicators"""
        features = {}
        text_lower = text.lower().strip()
        
        # Numbering patterns
        features['has_number_prefix'] = bool(re.match(r'^\d+', text_lower))
        features['has_roman_numeral'] = bool(re.match(r'^[ivx]+\b', text_lower))
        features['has_letter_prefix'] = bool(re.match(r'^[a-z]\)', text_lower))
        
        # Section numbering patterns
        section_patterns = [
            (r'^\d+\.\s+', 'decimal_1'),
            (r'^\d+\.\d+\.\s+', 'decimal_2'),
            (r'^\d+\.\d+\.\d+\.\s+', 'decimal_3'),
            (r'^\d+\)\s+', 'paren_number'),
            (r'^[A-Z]\.\s+', 'letter_dot'),
            (r'^\([a-z]\)\s+', 'paren_letter'),
            (r'^[IVX]+\.\s+', 'roman_dot'),
        ]
        
        features['section_pattern'] = None
        features['section_level'] = 0
        
        for pattern, pattern_name in section_patterns:
            if re.match(pattern, text):
                features['section_pattern'] = pattern_name
                if 'decimal' in pattern_name:
                    features['section_level'] = int(pattern_name[-1])
                elif pattern_name in ['decimal_1', 'paren_number', 'letter_dot', 'roman_dot']:
                    features['section_level'] = 1
                elif pattern_name in ['paren_letter']:
                    features['section_level'] = 2
                break
        
        # Punctuation patterns
        features['ends_with_colon'] = text.endswith(':')
        features['ends_with_period'] = text.endswith('.')
        features['has_question_mark'] = '?' in text
        features['has_exclamation'] = '!' in text
        
        # Special formatting
        features['has_parentheses'] = '(' in text and ')' in text
        features['has_brackets'] = '[' in text and ']' in text
        features['has_quotes'] = '"' in text or "'" in text
        
        return features
    
    def _analyze_semantics(self, text: str) -> Dict[str, Any]:
        """Analyze semantic content for heading likelihood"""
        features = {}
        text_lower = text.lower()
        
        # Check against heading vocabularies
        vocabulary_scores = {}
        for domain, vocab in self.heading_vocabularies.items():
            words = set(text_lower.split())
            matches = words.intersection(vocab)
            vocabulary_scores[domain] = len(matches) / len(words) if words else 0
        
        features['vocabulary_scores'] = vocabulary_scores
        features['max_vocabulary_score'] = max(vocabulary_scores.values()) if vocabulary_scores else 0
        features['dominant_domain'] = max(vocabulary_scores.keys(), key=vocabulary_scores.get) if vocabulary_scores else None
        
        # Content type analysis
        content_indicators = {
            'question': ['what', 'how', 'why', 'when', 'where', 'who', 'which'],
            'instruction': ['step', 'procedure', 'process', 'method', 'approach'],
            'list': ['list', 'items', 'elements', 'components', 'parts'],
            'definition': ['definition', 'meaning', 'concept', 'term', 'explanation'],
            'example': ['example', 'instance', 'case', 'illustration', 'sample'],
            'comparison': ['versus', 'vs', 'compared', 'difference', 'similarity']
        }
        
        for content_type, indicators in content_indicators.items():
            features[f'is_{content_type}'] = any(indicator in text_lower for indicator in indicators)
        
        # Named entity-like analysis (simple version)
        features['has_proper_nouns'] = self._detect_proper_nouns(text)
        features['has_technical_terms'] = self._detect_technical_terms(text_lower)
        features['has_acronyms'] = bool(re.search(r'\b[A-Z]{2,}\b', text))
        
        return features
    
    def _analyze_multilingual_patterns(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze patterns specific to detected language"""
        features = {}
        
        if language not in self.multilingual_patterns:
            return features
        
        patterns = self.multilingual_patterns[language]
        
        # Check for language-specific chapter markers
        if 'chapter_markers' in patterns:
            features['has_chapter_marker'] = any(marker in text for marker in patterns['chapter_markers'])
        
        # Check for language-specific section patterns
        if 'section_patterns' in patterns:
            features['matches_section_pattern'] = any(
                re.search(pattern, text, re.IGNORECASE) 
                for pattern in patterns['section_patterns']
            )
        
        # Check for language-specific numbering
        if 'numbering' in patterns:
            features['has_native_numbering'] = any(num in text for num in patterns['numbering'])
        
        return features
    
    def _detect_proper_nouns(self, text: str) -> bool:
        """Simple proper noun detection"""
        words = text.split()
        if not words:
            return False
        
        # Count capitalized words (excluding first word of sentence)
        capitalized_count = sum(1 for i, word in enumerate(words) 
                              if i > 0 and word and word[0].isupper() and word.lower() not in self.stop_words)
        
        return capitalized_count > 0
    
    def _detect_technical_terms(self, text_lower: str) -> bool:
        """Detect technical terminology"""
        technical_indicators = [
            'algorithm', 'system', 'method', 'framework', 'model', 'architecture',
            'protocol', 'interface', 'implementation', 'specification', 'api',
            'database', 'server', 'client', 'network', 'security', 'encryption',
            'authentication', 'authorization', 'configuration', 'deployment'
        ]
        
        return any(term in text_lower for term in technical_indicators)
    
    def _calculate_heading_probability(self, features: Dict[str, Any]) -> float:
        """Calculate overall probability that text is a heading"""
        score = 0.0
        
        # Length factors
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 10:
            score += 0.3
        elif word_count <= 15:
            score += 0.1
        elif word_count > 20:
            score -= 0.2
        
        # Formatting factors
        if features.get('first_word_capitalized', False):
            score += 0.1
        if features.get('title_case', False):
            score += 0.2
        if features.get('all_caps', False) and word_count <= 8:
            score += 0.3
        
        # Pattern factors
        if features.get('section_pattern'):
            score += 0.4
        if features.get('has_number_prefix', False):
            score += 0.2
        
        # Content factors
        vocab_score = features.get('max_vocabulary_score', 0)
        score += vocab_score * 0.3
        
        # Multilingual factors
        if features.get('has_chapter_marker', False):
            score += 0.3
        if features.get('matches_section_pattern', False):
            score += 0.4
        
        # Punctuation factors
        if features.get('ends_with_colon', False):
            score += 0.1
        if not features.get('ends_with_period', False) and word_count > 3:
            score += 0.1  # Headings often don't end with periods
        
        # Stop word factor (headings tend to have fewer stop words)
        stop_word_ratio = features.get('stop_word_ratio', 0.5)
        if stop_word_ratio < 0.3:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _suggest_heading_level(self, features: Dict[str, Any]) -> Optional[str]:
        """Suggest heading level based on features"""
        section_level = features.get('section_level', 0)
        
        if section_level > 0:
            return f"H{min(section_level, 3)}"
        
        # Use vocabulary and pattern analysis
        vocab_scores = features.get('vocabulary_scores', {})
        max_score = features.get('max_vocabulary_score', 0)
        
        if max_score > 0.3:
            # High-level concepts suggest H1
            if any(features.get(f'is_{t}', False) for t in ['question', 'definition']):
                return "H1"
            # Implementation details suggest H3
            elif any(features.get(f'is_{t}', False) for t in ['instruction', 'example']):
                return "H3"
            else:
                return "H2"
        
        # Default based on capitalization and length
        word_count = features.get('word_count', 0)
        if features.get('all_caps', False) and word_count <= 5:
            return "H1"
        elif features.get('title_case', False) and word_count <= 8:
            return "H2"
        else:
            return "H3"
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence
        
        # Strong indicators increase confidence
        if features.get('section_pattern'):
            confidence += 0.3
        if features.get('max_vocabulary_score', 0) > 0.4:
            confidence += 0.2
        if features.get('has_chapter_marker', False):
            confidence += 0.3
        
        # Consistent formatting increases confidence
        if (features.get('title_case', False) or 
            features.get('all_caps', False) or 
            features.get('first_word_capitalized', False)):
            confidence += 0.1
        
        # Appropriate length increases confidence
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 12:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def analyze_document_structure(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall document structure and heading relationships"""
        if not blocks:
            return {}
        
        # Analyze each block
        analyzed_blocks = []
        for block in blocks:
            text = block.get('text', '')
            semantic_analysis = self.analyze_text_semantics(text)
            
            # Combine with existing block data
            enhanced_block = block.copy()
            enhanced_block['semantic_analysis'] = semantic_analysis
            analyzed_blocks.append(enhanced_block)
        
        # Document-level analysis
        document_analysis = {
            'total_blocks': len(analyzed_blocks),
            'potential_headings': sum(1 for b in analyzed_blocks 
                                    if b['semantic_analysis']['heading_probability'] > 0.3),
            'language_distribution': self._analyze_language_distribution(analyzed_blocks),
            'heading_hierarchy': self._analyze_heading_hierarchy(analyzed_blocks),
            'content_domains': self._analyze_content_domains(analyzed_blocks)
        }
        
        return {
            'blocks': analyzed_blocks,
            'document_analysis': document_analysis
        }
    
    def _analyze_language_distribution(self, blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze language distribution in document"""
        languages = [block['semantic_analysis']['language'] for block in blocks]
        return dict(Counter(languages))
    
    def _analyze_heading_hierarchy(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential heading hierarchy"""
        heading_blocks = [b for b in blocks if b['semantic_analysis']['heading_probability'] > 0.3]
        
        levels = [b['semantic_analysis']['suggested_level'] for b in heading_blocks if b['semantic_analysis']['suggested_level']]
        
        return {
            'level_distribution': dict(Counter(levels)),
            'total_potential_headings': len(heading_blocks),
            'hierarchy_depth': len(set(levels))
        }
    
    def _analyze_content_domains(self, blocks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze dominant content domains in document"""
        domain_scores = defaultdict(float)
        
        for block in blocks:
            vocab_scores = block['semantic_analysis']['features'].get('vocabulary_scores', {})
            for domain, score in vocab_scores.items():
                domain_scores[domain] += score
        
        total = sum(domain_scores.values())
        if total > 0:
            return {domain: score/total for domain, score in domain_scores.items()}
        
        return {}
    
    def analyze_text_semantics(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive semantic analysis of text
        
        Args:
            text: Text to analyze
            context: Additional context information
            
        Returns:
            Semantic analysis results
        """
        if not text.strip():
            return {'is_heading_likely': False, 'confidence': 0.0}
        
        analysis = {
            'text': text.strip(),
            'length': len(text),
            'word_count': len(text.split()),
            'language': self._detect_language(text),
            'heading_probability': 0.0,
            'suggested_level': None,
            'confidence': 0.0,
            'features': {}
        }
        
        # Extract linguistic features
        linguistic_features = self._extract_linguistic_features(text)
        analysis['features'].update(linguistic_features)
        
        # Pattern-based analysis
        pattern_analysis = self._analyze_patterns(text)
        analysis['features'].update(pattern_analysis)
        
        # Semantic analysis
        semantic_features = self._analyze_semantics(text)
        analysis['features'].update(semantic_features)
        
        # Multilingual analysis
        if analysis['language'] != 'english':
            multilingual_features = self._analyze_multilingual_patterns(text, analysis['language'])
            analysis['features'].update(multilingual_features)
        
        # Calculate overall heading probability
        analysis['heading_probability'] = self._calculate_heading_probability(analysis['features'])
        analysis['suggested_level'] = self._suggest_heading_level(analysis['features'])
        analysis['confidence'] = self._calculate_confidence(analysis['features'])
        
        return analysis
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        text = text.lower()
        
        # Japanese detection (Hiragana, Katakana, Kanji)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'japanese'
        
        # Chinese detection
        if re.search(r'[\u4E00-\u9FFF]', text):
            return 'chinese'
        
        # French detection
        french_chars = len(re.findall(r'[àâäçéèêëîïôöùûüÿ]', text))
        if french_chars > 0 or any(word in text for word in ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou']):
            return 'french'
        
        # German detection
        german_chars = len(re.findall(r'[äöüß]', text))
        if german_chars > 0 or any(word in text for word in ['der', 'die', 'das', 'und', 'oder', 'mit']):
            return 'german'
        
        # Spanish detection
        spanish_chars = len(re.findall(r'[áéíóúüñ]', text))
        if spanish_chars > 0 or any(word in text for word in ['el', 'la', 'los', 'las', 'y', 'o', 'con']):
            return 'spanish'
        
        return 'english'
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract basic linguistic features"""
        features = {}
        
        words = text.split()
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['sentence_count'] = len(sent_tokenize(text)) if text else 0
        
        # Character type ratios
        features['alpha_ratio'] = sum(c.isalpha() for c in text) / len(text) if text else 0
        features['digit_ratio'] = sum(c.isdigit() for c in text) / len(text) if text else 0
        features['punct_ratio'] = sum(c in '.,;:!?()[]{}"\'-' for c in text) / len(text) if text else 0
        features['space_ratio'] = sum(c.isspace() for c in text) / len(text) if text else 0
        features['upper_ratio'] = sum(c.isupper() for c in text) / len(text) if text else 0
        
        # Word-level features
        if words:
            features['first_word_capitalized'] = words[0][0].isupper() if words[0] else False
            features['all_words_capitalized'] = all(w[0].isupper() for w in words if w)
            features['title_case'] = text.istitle()
            features['all_caps'] = text.isupper()
            
            # Stop word analysis
            if self.stop_words:
                stop_word_count = sum(1 for w in words if w.lower() in self.stop_words)
                features['stop_word_ratio'] = stop_word_count / len(words)
            else:
                features['stop_word_ratio'] = 0
        
        return features
    
    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text patterns for heading indicators"""
        features = {}
        text_lower = text.lower().strip()
        
        # Numbering patterns
        features['has_number_prefix'] = bool(re.match(r'^\d+', text_lower))
        features['has_roman_numeral'] = bool(re.match(r'^[ivx]+\b', text_lower))
        features['has_letter_prefix'] = bool(re.match(r'^[a-z]\)', text_lower))
        
        # Section numbering patterns
        section_patterns = [
            (r'^\d+\.\s+', 'decimal_1'),
            (r'^\d+\.\d+\.\s+', 'decimal_2'),
            (r'^\d+\.\d+\.\d+\.\s+', 'decimal_3'),
            (r'^\d+\)\s+', 'paren_number'),
            (r'^[A-Z]\.\s+', 'letter_dot'),
            (r'^\([a-z]\)\s+', 'paren_letter'),
            (r'^[IVX]+\.\s+', 'roman_dot'),
        ]
        
        features['section_pattern'] = None
        features['section_level'] = 0
        
        for pattern, pattern_name in section_patterns:
            if re.match(pattern, text):
                features['section_pattern'] = pattern_name
                if 'decimal' in pattern_name:
                    features['section_level'] = int(pattern_name[-1])
                elif pattern_name in ['decimal_1', 'paren_number', 'letter_dot', 'roman_dot']:
                    features['section_level'] = 1
                elif pattern_name in ['paren_letter']:
                    features['section_level'] = 2
                break
        
        # Punctuation patterns
        features['ends_with_colon'] = text.endswith(':')
        features['ends_with_period'] = text.endswith('.')
        features['has_question_mark'] = '?' in text
        features['has_exclamation'] = '!' in text
        
        # Special formatting
        features['has_parentheses'] = '(' in text and ')' in text
        features['has_brackets'] = '[' in text and ']' in text
        features['has_quotes'] = '"' in text or "'" in text
        
        return features
    
    def _analyze_semantics(self, text: str) -> Dict[str, Any]:
        """Analyze semantic content for heading likelihood"""
        features = {}
        text_lower = text.lower()
        
        # Check against heading vocabularies
        vocabulary_scores = {}
        for domain, vocab in self.heading_vocabularies.items():
            words = set(text_lower.split())
            matches = words.intersection(vocab)
            vocabulary_scores[domain] = len(matches) / len(words) if words else 0
        
        features['vocabulary_scores'] = vocabulary_scores
        features['max_vocabulary_score'] = max(vocabulary_scores.values()) if vocabulary_scores else 0
        features['dominant_domain'] = max(vocabulary_scores.keys(), key=vocabulary_scores.get) if vocabulary_scores else None
        
        # Content type analysis
        content_indicators = {
            'question': ['what', 'how', 'why', 'when', 'where', 'who', 'which'],
            'instruction': ['step', 'procedure', 'process', 'method', 'approach'],
            'list': ['list', 'items', 'elements', 'components', 'parts'],
            'definition': ['definition', 'meaning', 'concept', 'term', 'explanation'],
            'example': ['example', 'instance', 'case', 'illustration', 'sample'],
            'comparison': ['versus', 'vs', 'compared', 'difference', 'similarity']
        }
        
        for content_type, indicators in content_indicators.items():
            features[f'is_{content_type}'] = any(indicator in text_lower for indicator in indicators)
        
        # Named entity-like analysis (simple version)
        features['has_proper_nouns'] = self._detect_proper_nouns(text)
        features['has_technical_terms'] = self._detect_technical_terms(text_lower)
        features['has_acronyms'] = bool(re.search(r'\b[A-Z]{2,}\b', text))
        
        return features
    
    def _analyze_multilingual_patterns(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze patterns specific to detected language"""
        features = {}
        
        if language not in self.multilingual_patterns:
            return features
        
        patterns = self.multilingual_patterns[language]
        
        # Check for language-specific chapter markers
        if 'chapter_markers' in patterns:
            features['has_chapter_marker'] = any(marker in text for marker in patterns['chapter_markers'])
        
        # Check for language-specific section patterns
        if 'section_patterns' in patterns:
            features['matches_section_pattern'] = any(
                re.search(pattern, text, re.IGNORECASE) 
                for pattern in patterns['section_patterns']
            )
        
        # Check for language-specific numbering
        if 'numbering' in patterns:
            features['has_native_numbering'] = any(num in text for num in patterns['numbering'])
        
        return features
    
    def _detect_proper_nouns(self, text: str) -> bool:
        """Simple proper noun detection"""
        words = text.split()
        if not words:
            return False
        
        # Count capitalized words (excluding first word of sentence)
        capitalized_count = sum(1 for i, word in enumerate(words) 
                              if i > 0 and word and word[0].isupper() and word.lower() not in self.stop_words)
        
        return capitalized_count > 0
    
    def _detect_technical_terms(self, text_lower: str) -> bool:
        """Detect technical terminology"""
        technical_indicators = [
            'algorithm', 'system', 'method', 'framework', 'model', 'architecture',
            'protocol', 'interface', 'implementation', 'specification', 'api',
            'database', 'server', 'client', 'network', 'security', 'encryption',
            'authentication', 'authorization', 'configuration', 'deployment'
        ]
        
        return any(term in text_lower for term in technical_indicators)
    
    def _calculate_heading_probability(self, features: Dict[str, Any]) -> float:
        """Calculate overall probability that text is a heading"""
        score = 0.0
        
        # Length factors
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 10:
            score += 0.3
        elif word_count <= 15:
            score += 0.1
        elif word_count > 20:
            score -= 0.2
        
        # Formatting factors
        if features.get('first_word_capitalized', False):
            score += 0.1
        if features.get('title_case', False):
            score += 0.2
        if features.get('all_caps', False) and word_count <= 8:
            score += 0.3
        
        # Pattern factors
        if features.get('section_pattern'):
            score += 0.4
        if features.get('has_number_prefix', False):
            score += 0.2
        
        # Content factors
        vocab_score = features.get('max_vocabulary_score', 0)
        score += vocab_score * 0.3
        
        # Multilingual factors
        if features.get('has_chapter_marker', False):
            score += 0.3
        if features.get('matches_section_pattern', False):
            score += 0.4
        
        # Punctuation factors
        if features.get('ends_with_colon', False):
            score += 0.1
        if not features.get('ends_with_period', False) and word_count > 3:
            score += 0.1  # Headings often don't end with periods
        
        # Stop word factor (headings tend to have fewer stop words)
        stop_word_ratio = features.get('stop_word_ratio', 0.5)
        if stop_word_ratio < 0.3:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _suggest_heading_level(self, features: Dict[str, Any]) -> Optional[str]:
        """Suggest heading level based on features"""
        section_level = features.get('section_level', 0)
        
        if section_level > 0:
            return f"H{min(section_level, 3)}"
        
        # Use vocabulary and pattern analysis
        vocab_scores = features.get('vocabulary_scores', {})
        max_score = features.get('max_vocabulary_score', 0)
        
        if max_score > 0.3:
            # High-level concepts suggest H1
            if any(features.get(f'is_{t}', False) for t in ['question', 'definition']):
                return "H1"
            # Implementation details suggest H3
            elif any(features.get(f'is_{t}', False) for t in ['instruction', 'example']):
                return "H3"
            else:
                return "H2"
        
        # Default based on capitalization and length
        word_count = features.get('word_count', 0)
        if features.get('all_caps', False) and word_count <= 5:
            return "H1"
        elif features.get('title_case', False) and word_count <= 8:
            return "H2"
        else:
            return "H3"
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence
        
        # Strong indicators increase confidence
        if features.get('section_pattern'):
            confidence += 0.3
        if features.get('max_vocabulary_score', 0) > 0.4:
            confidence += 0.2
        if features.get('has_chapter_marker', False):
            confidence += 0.3
        
        # Consistent formatting increases confidence
        if (features.get('title_case', False) or 
            features.get('all_caps', False) or 
            features.get('first_word_capitalized', False)):
            confidence += 0.1
        
        # Appropriate length increases confidence
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 12:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def analyze_document_structure(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall document structure and heading relationships"""
        if not blocks:
            return {}
        
        # Analyze each block
        analyzed_blocks = []
        for block in blocks:
            text = block.get('text', '')
            semantic_analysis = self.analyze_text_semantics(text)
            
            # Combine with existing block data
            enhanced_block = block.copy()
            enhanced_block['semantic_analysis'] = semantic_analysis
            analyzed_blocks.append(enhanced_block)
        
        # Document-level analysis
        document_analysis = {
            'total_blocks': len(analyzed_blocks),
            'potential_headings': sum(1 for b in analyzed_blocks 
                                    if b['semantic_analysis']['heading_probability'] > 0.3),
            'language_distribution': self._analyze_language_distribution(analyzed_blocks),
            'heading_hierarchy': self._analyze_heading_hierarchy(analyzed_blocks),
            'content_domains': self._analyze_content_domains(analyzed_blocks)
        }
        
        return {
            'blocks': analyzed_blocks,
            'document_analysis': document_analysis
        }
    
    def _analyze_language_distribution(self, blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze language distribution in document"""
        languages = [block['semantic_analysis']['language'] for block in blocks]
        return dict(Counter(languages))
    
    def _analyze_heading_hierarchy(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential heading hierarchy"""
        heading_blocks = [b for b in blocks 
                         if b['semantic_analysis']['heading_probability'] > 0.3]
        
        levels = [b['semantic_analysis']['suggested_level'] for b in heading_blocks 
                 if b['semantic_analysis']['suggested_level']]
        
        return {
            'level_distribution': dict(Counter(levels)),
            'total_potential_headings': len(heading_blocks),
            'hierarchy_depth': len(set(levels))
        }
    def _analyze_content_domains(self, blocks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze dominant content domains in document."""
        domain_scores = defaultdict(float)

        for block in blocks:
            vocab_scores = block.get('semantic_analysis', {}).get('features', {}).get('vocabulary_scores', {})
            for domain, score in vocab_scores.items():
                domain_scores[domain] += score

        total = sum(domain_scores.values())
        if total > 0:
            return {domain: score / total for domain, score in domain_scores.items()}

        return {}
    
    def _enhanced_pattern_analysis(self, text: str, language: str) -> Dict[str, Any]:
        """Enhanced pattern analysis with multilingual support."""
        features = {}
        text_clean = text.strip()
        text_lower = text_clean.lower()

        # Language-specific section patterns
        if language in self.multilingual_patterns:
            patterns = self.multilingual_patterns[language]

            features['matches_section_pattern'] = any(
                re.search(pattern, text_clean, re.IGNORECASE)
                for pattern in patterns.get('section_patterns', [])
            )

            features['has_chapter_marker'] = any(
                marker in text_lower
                for marker in patterns.get('chapter_markers', [])
            )

            features['is_common_heading'] = any(
                heading in text_lower
                for heading in patterns.get('common_headings', [])
            )
        else:
            # Default English patterns
            features['matches_section_pattern'] = bool(
                re.search(r'^\s*\d+\.(?:\d+\.)*\s+', text_clean) or
                re.search(r'^\s*[A-Z]\.(?:\d+\.)*\s+', text_clean) or
                re.search(r'^\s*[IVX]+\.(?:\d+\.)*\s+', text_clean, re.IGNORECASE)
            )

            features['has_chapter_marker'] = bool(
                re.search(r'\b(chapter|section|part|appendix)\s+\d+\b', text_lower, re.IGNORECASE)
            )

            features['is_common_heading'] = any(
                word in text_lower
                for word in ['introduction', 'conclusion', 'abstract', 'summary', 'references']
            )

        # Universal patterns
        features['starts_with_number'] = bool(re.match(r'^\s*\d+', text_clean))
        features['has_colon_ending'] = text_clean.endswith(':')
        features['has_question_mark'] = '?' in text_clean
        features['parentheses_pattern'] = bool(re.search(r'^\s*\([^)]+\)', text_clean))

        return features

    def _enhanced_semantic_analysis(self, text: str, language: str) -> Dict[str, Any]:
        """Enhanced semantic analysis with domain and language awareness"""
        features = {}
        text_lower = text.lower()
        
        # Domain-specific vocabulary analysis
        domain_scores = {}
        max_confidence_score = 0
        
        for domain, vocab_levels in self.heading_vocabularies.items():
            domain_score = 0
            
            # High confidence terms
            high_conf_matches = sum(1 for term in vocab_levels.get('high_confidence', set()) if term in text_lower)
            domain_score += high_conf_matches * 1.0
            
            # Medium confidence terms
            med_conf_matches = sum(1 for term in vocab_levels.get('medium_confidence', set()) if term in text_lower)
            domain_score += med_conf_matches * 0.6
            
            # Normalize by text length
            word_count = len(text.split())
            normalized_score = domain_score / max(word_count, 1)
            domain_scores[domain] = normalized_score
            
            max_confidence_score = max(max_confidence_score, normalized_score)
        
        features['domain_scores'] = domain_scores
        features['max_domain_score'] = max_confidence_score
        features['dominant_domain'] = max(domain_scores.keys(), key=domain_scores.get) if domain_scores else None
        
        # Content type classification
        content_types = self._classify_content_type(text_lower)
        features.update(content_types)
        
        return features
    
    def _classify_content_type(self, text_lower: str) -> Dict[str, bool]:
        """Classify content type for better heading detection"""
        
        content_indicators = {
            'is_question': any(indicator in text_lower for indicator in [
                'what', 'how', 'why', 'when', 'where', 'who', 'which', '?'
            ]),
            
            'is_instruction': any(indicator in text_lower for indicator in [
                'step', 'procedure', 'process', 'method', 'approach', 'install', 'configure'
            ]),
            
            'is_list_header': any(indicator in text_lower for indicator in [
                'list', 'items', 'elements', 'components', 'parts', 'contents'
            ]),
            
            'is_definition': any(indicator in text_lower for indicator in [
                'definition', 'meaning', 'concept', 'term', 'explanation', 'overview'
            ]),
            
            'is_example_section': any(indicator in text_lower for indicator in [
                'example', 'instance', 'case', 'illustration', 'sample', 'demo'
            ]),
            
            'is_comparison': any(indicator in text_lower for indicator in [
                'versus', 'vs', 'compared', 'difference', 'similarity', 'contrast'
            ])
        }
        
        return content_indicators
    
    def _calculate_enhanced_heading_probability(self, features: Dict[str, Any], language: str) -> float:
        """Calculate enhanced heading probability with language awareness"""
        score = 0.0
        
        # Length-based scoring (language-aware)
        word_count = features.get('word_count', 0)
        if language in ['japanese', 'chinese']:
            # Shorter headings are more common in CJK languages
            if 1 <= word_count <= 8:
                score += 0.4
            elif word_count <= 12:
                score += 0.2
        else:
            # Standard length scoring for Latin scripts
            if 2 <= word_count <= 10:
                score += 0.3
            elif word_count <= 15:
                score += 0.1
            elif word_count > 25:
                score -= 0.3  # Penalty for very long text
        
        # Formatting bonuses
        if features.get('first_word_capitalized', False):
            score += 0.1
        if features.get('title_case', False):
            score += 0.2
        if features.get('all_caps', False) and word_count <= 8:
            score += 0.3
        
        # Pattern bonuses (language-aware)
        if features.get('matches_section_pattern', False):
            score += 0.4
        if features.get('has_chapter_marker', False):
            score += 0.3
        if features.get('is_common_heading', False):
            score += 0.3
        
        # Domain vocabulary bonus
        max_domain_score = features.get('max_domain_score', 0)
        score += min(max_domain_score * 0.4, 0.3)
        
        # Content type bonuses
        if features.get('is_question', False) and word_count <= 12:
            score += 0.2
        if features.get('is_definition', False):
            score += 0.2
        
        # Punctuation patterns
        if features.get('has_colon_ending', False):
            score += 0.1
        if not features.get('has_verbs', False) and word_count <= 6:
            score += 0.1  # Noun phrases are good for headings
        
        # Penalties
        if features.get('stop_word_ratio', 0) > 0.6:
            score -= 0.2  # Too many stop words
        
        return min(1.0, max(0.0, score))
    
    def _suggest_heading_level_enhanced(self, features: Dict[str, Any], language: str) -> Optional[str]:
        """Enhanced heading level suggestion"""
        word_count = features.get('word_count', 0)
        
        # Pattern-based level detection first
        if features.get('matches_section_pattern', False):
            # Try to extract numbering depth
            text = features.get('text', '')
            dots = text.count('.')
            if dots >= 2:
                return 'H3'
            elif dots >= 1:
                return 'H2'
            else:
                return 'H1'
        
        # Content-based level suggestion
        domain_scores = features.get('domain_scores', {})
        max_score = features.get('max_domain_score', 0)
        
        if features.get('is_common_heading', False):
            return 'H1'  # Common headings are usually top-level
        
        if features.get('all_caps', False) and word_count <= 5:
            return 'H1'
        
        if features.get('title_case', False):
            if max_score > 0.3:
                return 'H1'
            else:
                return 'H2'
        
        # Default based on characteristics
        if word_count <= 4 and features.get('first_word_capitalized', False):
            return 'H2'
        
        return 'H3'
    
    def _calculate_enhanced_confidence(self, features: Dict[str, Any], lang_confidence: float) -> float:
        """Calculate enhanced confidence score"""
        confidence = 0.5  # Base confidence
        
        # Strong indicators
        if features.get('matches_section_pattern', False):
            confidence += 0.3
        if features.get('max_domain_score', 0) > 0.4:
            confidence += 0.2
        if features.get('has_chapter_marker', False):
            confidence += 0.2
        
        # Formatting consistency
        if features.get('title_case', False) or features.get('all_caps', False):
            confidence += 0.1
        
        # Length appropriateness
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 12:
            confidence += 0.1
        elif word_count > 20:
            confidence -= 0.2
        
        # Language detection confidence
        confidence += lang_confidence * 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def analyze_text_semantics(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive semantic analysis of text
        
        Args:
            text: Text to analyze
            context: Additional context information
            
        Returns:
            Semantic analysis results
        """
        if not text.strip():
            return {'is_heading_likely': False, 'confidence': 0.0}
        
        analysis = {
            'text': text.strip(),
            'length': len(text),
            'word_count': len(text.split()),
            'language': self._detect_language(text),
            'heading_probability': 0.0,
            'suggested_level': None,
            'confidence': 0.0,
            'features': {}
        }
        
        # Extract linguistic features
        linguistic_features = self._extract_linguistic_features(text)
        analysis['features'].update(linguistic_features)
        
        # Pattern-based analysis
        pattern_analysis = self._analyze_patterns(text)
        analysis['features'].update(pattern_analysis)
        
        # Semantic analysis
        semantic_features = self._analyze_semantics(text)
        analysis['features'].update(semantic_features)
        
        # Multilingual analysis
        if analysis['language'] != 'english':
            multilingual_features = self._analyze_multilingual_patterns(text, analysis['language'])
            analysis['features'].update(multilingual_features)
        
        # Calculate overall heading probability
        analysis['heading_probability'] = self._calculate_heading_probability(analysis['features'])
        analysis['suggested_level'] = self._suggest_heading_level(analysis['features'])
        analysis['confidence'] = self._calculate_confidence(analysis['features'])
        
        return analysis
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        text = text.lower()
        
        # Japanese detection (Hiragana, Katakana, Kanji)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'japanese'
        
        # Chinese detection
        if re.search(r'[\u4E00-\u9FFF]', text):
            return 'chinese'
        
        # French detection
        french_chars = len(re.findall(r'[àâäçéèêëîïôöùûüÿ]', text))
        if french_chars > 0 or any(word in text for word in ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou']):
            return 'french'
        
        # German detection
        german_chars = len(re.findall(r'[äöüß]', text))
        if german_chars > 0 or any(word in text for word in ['der', 'die', 'das', 'und', 'oder', 'mit']):
            return 'german'
        
        # Spanish detection
        spanish_chars = len(re.findall(r'[áéíóúüñ]', text))
        if spanish_chars > 0 or any(word in text for word in ['el', 'la', 'los', 'las', 'y', 'o', 'con']):
            return 'spanish'
        
        return 'english'
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract basic linguistic features"""
        features = {}
        
        words = text.split()
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['sentence_count'] = len(sent_tokenize(text)) if text else 0
        
        # Character type ratios
        features['alpha_ratio'] = sum(c.isalpha() for c in text) / len(text) if text else 0
        features['digit_ratio'] = sum(c.isdigit() for c in text) / len(text) if text else 0
        features['punct_ratio'] = sum(c in '.,;:!?()[]{}"\'-' for c in text) / len(text) if text else 0
        features['space_ratio'] = sum(c.isspace() for c in text) / len(text) if text else 0
        features['upper_ratio'] = sum(c.isupper() for c in text) / len(text) if text else 0
        
        # Word-level features
        if words:
            features['first_word_capitalized'] = words[0][0].isupper() if words[0] else False
            features['all_words_capitalized'] = all(w[0].isupper() for w in words if w)
            features['title_case'] = text.istitle()
            features['all_caps'] = text.isupper()
            
            # Stop word analysis
            if self.stop_words:
                stop_word_count = sum(1 for w in words if w.lower() in self.stop_words)
                features['stop_word_ratio'] = stop_word_count / len(words)
            else:
                features['stop_word_ratio'] = 0
        
        return features
    
    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text patterns for heading indicators"""
        features = {}
        text_lower = text.lower().strip()
        
        # Numbering patterns
        features['has_number_prefix'] = bool(re.match(r'^\d+', text_lower))
        features['has_roman_numeral'] = bool(re.match(r'^[ivx]+\b', text_lower))
        features['has_letter_prefix'] = bool(re.match(r'^[a-z]\)', text_lower))
        
        # Section numbering patterns
        section_patterns = [
            (r'^\d+\.\s+', 'decimal_1'),
            (r'^\d+\.\d+\.\s+', 'decimal_2'),
            (r'^\d+\.\d+\.\d+\.\s+', 'decimal_3'),
            (r'^\d+\)\s+', 'paren_number'),
            (r'^[A-Z]\.\s+', 'letter_dot'),
            (r'^\([a-z]\)\s+', 'paren_letter'),
            (r'^[IVX]+\.\s+', 'roman_dot'),
        ]
        
        features['section_pattern'] = None
        features['section_level'] = 0
        
        for pattern, pattern_name in section_patterns:
            if re.match(pattern, text):
                features['section_pattern'] = pattern_name
                if 'decimal' in pattern_name:
                    features['section_level'] = int(pattern_name[-1])
                elif pattern_name in ['decimal_1', 'paren_number', 'letter_dot', 'roman_dot']:
                    features['section_level'] = 1
                elif pattern_name in ['paren_letter']:
                    features['section_level'] = 2
                break
        
        # Punctuation patterns
        features['ends_with_colon'] = text.endswith(':')
        features['ends_with_period'] = text.endswith('.')
        features['has_question_mark'] = '?' in text
        features['has_exclamation'] = '!' in text
        
        # Special formatting
        features['has_parentheses'] = '(' in text and ')' in text
        features['has_brackets'] = '[' in text and ']' in text
        features['has_quotes'] = '"' in text or "'" in text
        
        return features
    
    def _analyze_semantics(self, text: str) -> Dict[str, Any]:
        """Analyze semantic content for heading likelihood"""
        features = {}
        text_lower = text.lower()
        
        # Check against heading vocabularies
        vocabulary_scores = {}
        for domain, vocab in self.heading_vocabularies.items():
            words = set(text_lower.split())
            matches = words.intersection(vocab)
            vocabulary_scores[domain] = len(matches) / len(words) if words else 0
        
        features['vocabulary_scores'] = vocabulary_scores
        features['max_vocabulary_score'] = max(vocabulary_scores.values()) if vocabulary_scores else 0
        features['dominant_domain'] = max(vocabulary_scores.keys(), key=vocabulary_scores.get) if vocabulary_scores else None
        
        # Content type analysis
        content_indicators = {
            'question': ['what', 'how', 'why', 'when', 'where', 'who', 'which'],
            'instruction': ['step', 'procedure', 'process', 'method', 'approach'],
            'list': ['list', 'items', 'elements', 'components', 'parts'],
            'definition': ['definition', 'meaning', 'concept', 'term', 'explanation'],
            'example': ['example', 'instance', 'case', 'illustration', 'sample'],
            'comparison': ['versus', 'vs', 'compared', 'difference', 'similarity']
        }
        
        for content_type, indicators in content_indicators.items():
            features[f'is_{content_type}'] = any(indicator in text_lower for indicator in indicators)
        
        # Named entity-like analysis (simple version)
        features['has_proper_nouns'] = self._detect_proper_nouns(text)
        features['has_technical_terms'] = self._detect_technical_terms(text_lower)
        features['has_acronyms'] = bool(re.search(r'\b[A-Z]{2,}\b', text))
        
        return features
    
    def _analyze_multilingual_patterns(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze patterns specific to detected language"""
        features = {}
        
        if language not in self.multilingual_patterns:
            return features
        
        patterns = self.multilingual_patterns[language]
        
        # Check for language-specific chapter markers
        if 'chapter_markers' in patterns:
            features['has_chapter_marker'] = any(marker in text for marker in patterns['chapter_markers'])
        
        # Check for language-specific section patterns
        if 'section_patterns' in patterns:
            features['matches_section_pattern'] = any(
                re.search(pattern, text, re.IGNORECASE) 
                for pattern in patterns['section_patterns']
            )
        
        # Check for language-specific numbering
        if 'numbering' in patterns:
            features['has_native_numbering'] = any(num in text for num in patterns['numbering'])
        
        return features
    
    def _detect_proper_nouns(self, text: str) -> bool:
        """Simple proper noun detection"""
        words = text.split()
        if not words:
            return False
        
        # Count capitalized words (excluding first word of sentence)
        capitalized_count = sum(1 for i, word in enumerate(words) 
                              if i > 0 and word and word[0].isupper() and word.lower() not in self.stop_words)
        
        return capitalized_count > 0
    
    def _detect_technical_terms(self, text_lower: str) -> bool:
        """Detect technical terminology"""
        technical_indicators = [
            'algorithm', 'system', 'method', 'framework', 'model', 'architecture',
            'protocol', 'interface', 'implementation', 'specification', 'api',
            'database', 'server', 'client', 'network', 'security', 'encryption',
            'authentication', 'authorization', 'configuration', 'deployment'
        ]
        
        return any(term in text_lower for term in technical_indicators)
    
    def _calculate_heading_probability(self, features: Dict[str, Any]) -> float:
        """Calculate overall probability that text is a heading"""
        score = 0.0
        
        # Length factors
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 10:
            score += 0.3
        elif word_count <= 15:
            score += 0.1
        elif word_count > 20:
            score -= 0.2
        
        # Formatting factors
        if features.get('first_word_capitalized', False):
            score += 0.1
        if features.get('title_case', False):
            score += 0.2
        if features.get('all_caps', False) and word_count <= 8:
            score += 0.3
        
        # Pattern factors
        if features.get('section_pattern'):
            score += 0.4
        if features.get('has_number_prefix', False):
            score += 0.2
        
        # Content factors
        vocab_score = features.get('max_vocabulary_score', 0)
        score += vocab_score * 0.3
        
        # Multilingual factors
        if features.get('has_chapter_marker', False):
            score += 0.3
        if features.get('matches_section_pattern', False):
            score += 0.4
        
        # Punctuation factors
        if features.get('ends_with_colon', False):
            score += 0.1
        if not features.get('ends_with_period', False) and word_count > 3:
            score += 0.1  # Headings often don't end with periods
        
        # Stop word factor (headings tend to have fewer stop words)
        stop_word_ratio = features.get('stop_word_ratio', 0.5)
        if stop_word_ratio < 0.3:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _suggest_heading_level(self, features: Dict[str, Any]) -> Optional[str]:
        """Suggest heading level based on features"""
        section_level = features.get('section_level', 0)
        
        if section_level > 0:
            return f"H{min(section_level, 3)}"
        
        # Use vocabulary and pattern analysis
        vocab_scores = features.get('vocabulary_scores', {})
        max_score = features.get('max_vocabulary_score', 0)
        
        if max_score > 0.3:
            # High-level concepts suggest H1
            if any(features.get(f'is_{t}', False) for t in ['question', 'definition']):
                return "H1"
            # Implementation details suggest H3
            elif any(features.get(f'is_{t}', False) for t in ['instruction', 'example']):
                return "H3"
            else:
                return "H2"
        
        # Default based on capitalization and length
        word_count = features.get('word_count', 0)
        if features.get('all_caps', False) and word_count <= 5:
            return "H1"
        elif features.get('title_case', False) and word_count <= 8:
            return "H2"
        else:
            return "H3"
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence
        
        # Strong indicators increase confidence
        if features.get('section_pattern'):
            confidence += 0.3
        if features.get('max_vocabulary_score', 0) > 0.4:
            confidence += 0.2
        if features.get('has_chapter_marker', False):
            confidence += 0.3
        
        # Consistent formatting increases confidence
        if (features.get('title_case', False) or 
            features.get('all_caps', False) or 
            features.get('first_word_capitalized', False)):
            confidence += 0.1
        
        # Appropriate length increases confidence
        word_count = features.get('word_count', 0)
        if 2 <= word_count <= 12:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def analyze_document_structure(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall document structure and heading relationships"""
        if not blocks:
            return {}
        
        # Analyze each block
        analyzed_blocks = []
        for block in blocks:
            text = block.get('text', '')
            semantic_analysis = self.analyze_text_semantics(text)
            
            # Combine with existing block data
            enhanced_block = block.copy()
            enhanced_block['semantic_analysis'] = semantic_analysis
            analyzed_blocks.append(enhanced_block)
        
        # Document-level analysis
        document_analysis = {
            'total_blocks': len(analyzed_blocks),
            'potential_headings': sum(1 for b in analyzed_blocks 
                                    if b['semantic_analysis']['heading_probability'] > 0.3),
            'language_distribution': self._analyze_language_distribution(analyzed_blocks),
            'heading_hierarchy': self._analyze_heading_hierarchy(analyzed_blocks),
            'content_domains': self._analyze_content_domains(analyzed_blocks)
        }
        
        return {
            'blocks': analyzed_blocks,
            'document_analysis': document_analysis
        }
    
    def _analyze_language_distribution(self, blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze language distribution in document"""
        languages = [block['semantic_analysis']['language'] for block in blocks]
        return dict(Counter(languages))
    
    def _analyze_heading_hierarchy(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential heading hierarchy"""
        heading_blocks = [b for b in blocks 
                         if b['semantic_analysis']['heading_probability'] > 0.3]
        
        levels = [b['semantic_analysis']['suggested_level'] for b in heading_blocks 
                 if b['semantic_analysis']['suggested_level']]
        
        return {
            'level_distribution': dict(Counter(levels)),
            'total_potential_headings': len(heading_blocks),
            'hierarchy_depth': len(set(levels))
        }
    
    def _analyze_content_domains(self, blocks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze dominant content domains in document"""
        domain_scores = defaultdict(float)
        
        for block in blocks:
            vocab_scores = block['semantic_analysis']['features'].get('vocabulary_scores', {})
            for domain, score in vocab_scores.items():
                domain_scores[domain] += score
        
        total = sum(domain_scores.values())
        if total > 0:
            return {domain: score/total for domain, score in domain_scores.items()}
        
        return {}