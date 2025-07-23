"""
Training Data Generator - Generate synthetic and real training data for heading classifier
"""

import json
import random
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np

class TrainingDataGenerator:
    """Generate training data for heading classification"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Heading templates by level
        self.heading_templates = {
            'H1': [
                'Introduction', 'Overview', 'Abstract', 'Summary', 'Executive Summary',
                'Background', 'Conclusion', 'Results', 'Discussion', 'References',
                'Chapter {num}', 'Part {num}', 'Section {num}', 'Appendix {letter}',
                'Methodology', 'Literature Review', 'Analysis', 'Implementation',
                'Future Work', 'Acknowledgments', 'Table of Contents', 'Index'
            ],
            'H2': [
                'Problem Statement', 'Objectives', 'Scope', 'Requirements',
                'Design Principles', 'Architecture', 'System Overview',
                'Data Collection', 'Experimental Setup', 'Case Study',
                'Performance Evaluation', 'Limitations', 'Related Work',
                '{num}.{num} {concept}', 'Technical Specifications',
                'Market Analysis', 'Risk Assessment', 'Quality Assurance'
            ],
            'H3': [
                'Data Preprocessing', 'Feature Selection', 'Model Training',
                'Hyperparameter Tuning', 'Cross Validation', 'Error Analysis',
                'Baseline Comparison', 'Statistical Significance', 
                'Implementation Details', 'Configuration Settings',
                '{num}.{num}.{num} {detail}', 'Code Examples',
                'Performance Metrics', 'User Interface', 'Database Schema'
            ]
        }
        
        # Non-heading templates
        self.non_heading_templates = [
            'This is a paragraph of regular text that explains concepts in detail.',
            'The methodology used in this research follows standard practices.',
            'Table {num} shows the results of the experiment conducted.',
            'Figure {num} illustrates the relationship between variables.',
            'According to the literature, this approach has been proven effective.',
            'The data was collected over a period of {num} months.',
            'Statistical analysis was performed using standard software.',
            'Participants were randomly assigned to treatment groups.',
            'The survey consisted of {num} questions about user preferences.',
            'Copyright © 2024 by the authors. All rights reserved.',
            'Page {num}', 'www.example.com', 'Version 1.2.3',
            'For more information, please contact support@company.com'
        ]
        
        # Font characteristics for different heading levels
        self.font_characteristics = {
            'H1': {
                'size_range': (16, 24),
                'bold_probability': 0.9,
                'italic_probability': 0.1,
                'all_caps_probability': 0.3
            },
            'H2': {
                'size_range': (14, 18),
                'bold_probability': 0.8,
                'italic_probability': 0.2,
                'all_caps_probability': 0.1
            },
            'H3': {
                'size_range': (12, 16),
                'bold_probability': 0.6,
                'italic_probability': 0.3,
                'all_caps_probability': 0.05
            },
            'text': {
                'size_range': (10, 12),
                'bold_probability': 0.1,
                'italic_probability': 0.1,
                'all_caps_probability': 0.02
            }
        }
    
    def generate_synthetic_training_data(self, num_samples: int = 1000) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Generate synthetic training data
        
        Args:
            num_samples: Number of training samples to generate
            
        Returns:
            Tuple of (training_blocks, labels)
        """
        self.logger.info(f"Generating {num_samples} synthetic training samples")
        
        training_blocks = []
        labels = []
        
        # Generate heading samples (40% of total)
        heading_samples = int(num_samples * 0.4)
        for level in ['H1', 'H2', 'H3']:
            level_samples = heading_samples // 3
            for _ in range(level_samples):
                block = self._generate_heading_sample(level)
                training_blocks.append(block)
                labels.append('heading')
        
        # Generate non-heading samples (60% of total)
        text_samples = num_samples - len(training_blocks)
        for _ in range(text_samples):
            block = self._generate_text_sample()
            training_blocks.append(block)
            labels.append('text')
        
        # Shuffle the data
        combined = list(zip(training_blocks, labels))
        random.shuffle(combined)
        training_blocks, labels = zip(*combined)
        
        self.logger.info(f"Generated {len(training_blocks)} total samples")
        self.logger.info(f"Heading samples: {labels.count('heading')}")
        self.logger.info(f"Text samples: {labels.count('text')}")
        
        return list(training_blocks), list(labels)
    
    def _generate_heading_sample(self, level: str) -> Dict[str, Any]:
        """Generate a synthetic heading sample"""
        templates = self.heading_templates[level]
        characteristics = self.font_characteristics[level]
        
        # Choose template and fill in placeholders
        template = random.choice(templates)
        text = self._fill_template_placeholders(template)
        
        # Apply random formatting
        if random.random() < characteristics['all_caps_probability']:
            text = text.upper()
        elif random.random() < 0.3:
            text = text.title()
        
        # Generate font characteristics
        font_size = random.uniform(*characteristics['size_range'])
        is_bold = random.random() < characteristics['bold_probability']
        is_italic = random.random() < characteristics['italic_probability']
        
        # Generate position characteristics
        page = random.randint(1, 50)
        x_position = random.uniform(50, 150)  # Headings often start near left margin
        y_position = random.uniform(100, 700)
        width = random.uniform(200, 400)
        height = random.uniform(15, 30)
        
        # Create block structure
        block = {
            'text': text,
            'page': page,
            'bbox': [x_position, y_position, x_position + width, y_position + height],
            'font_info': {
                'size': font_size,
                'is_bold': is_bold,
                'is_italic': is_italic,
                'name': random.choice(['arial', 'times', 'helvetica', 'calibri']),
                'flags': (16 if is_bold else 0) + (1 if is_italic else 0)
            },
            'width': width,
            'height': height,
            'suggested_level': level  # Ground truth for training
        }
        
        # Add some variability in positioning
        if level == 'H1':
            # H1 headings might be centered
            if random.random() < 0.3:
                block['bbox'][0] = random.uniform(150, 250)  # More centered
        
        return block
    
    def _generate_text_sample(self) -> Dict[str, Any]:
        """Generate a synthetic non-heading text sample"""
        templates = self.non_heading_templates
        characteristics = self.font_characteristics['text']
        
        # Choose template and fill in placeholders
        template = random.choice(templates)
        text = self._fill_template_placeholders(template)
        
        # Text samples are usually longer
        if random.random() < 0.5:
            # Add more content to make it longer
            additional = random.choice(templates)
            text += ' ' + self._fill_template_placeholders(additional)
        
        # Generate font characteristics (more uniform for body text)
        font_size = random.uniform(*characteristics['size_range'])
        is_bold = random.random() < characteristics['bold_probability']
        is_italic = random.random() < characteristics['italic_probability']
        
        # Generate position characteristics (more varied for body text)
        page = random.randint(1, 50)
        x_position = random.uniform(72, 200)  # Body text might be indented
        y_position = random.uniform(100, 700)
        width = random.uniform(300, 500)  # Body text usually wider
        height = random.uniform(10, 20)
        
        # Create block structure
        block = {
            'text': text,
            'page': page,
            'bbox': [x_position, y_position, x_position + width, y_position + height],
            'font_info': {
                'size': font_size,
                'is_bold': is_bold,
                'is_italic': is_italic,
                'name': random.choice(['arial', 'times', 'helvetica', 'calibri']),
                'flags': (16 if is_bold else 0) + (1 if is_italic else 0)
            },
            'width': width,
            'height': height,
            'suggested_level': None  # Not a heading
        }
        
        return block
    
    def _fill_template_placeholders(self, template: str) -> str:
        """Fill in placeholder values in templates"""
        # Replace {num} with random numbers
        while '{num}' in template:
            template = template.replace('{num}', str(random.randint(1, 20)), 1)
        
        # Replace {letter} with random letters
        while '{letter}' in template:
            template = template.replace('{letter}', random.choice('ABCDEFGHIJK'), 1)
        
        # Replace {concept} with random concept words
        concepts = [
            'Overview', 'Analysis', 'Implementation', 'Evaluation', 'Framework',
            'Architecture', 'Design', 'Testing', 'Validation', 'Optimization'
        ]
        while '{concept}' in template:
            template = template.replace('{concept}', random.choice(concepts), 1)
        
        # Replace {detail} with random detail words
        details = [
            'Configuration', 'Parameters', 'Settings', 'Specifications', 'Properties',
            'Attributes', 'Features', 'Components', 'Elements', 'Aspects'
        ]
        while '{detail}' in template:
            template = template.replace('{detail}', random.choice(details), 1)
        
        return template
    
    def generate_multilingual_samples(self, num_samples: int = 200) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Generate multilingual training samples for bonus points"""
        
        multilingual_templates = {
            'japanese': {
                'H1': ['第{num}章', '概要', '序論', '結論', '参考文献'],
                'H2': ['第{num}節', '手法', '実験', '評価', '考察'],
                'H3': ['第{num}項', '詳細', '実装', '結果', '分析']
            },
            'chinese': {
                'H1': ['第{num}章', '概述', '引言', '结论', '参考文献'],
                'H2': ['第{num}节', '方法', '实验', '评价', '讨论'],
                'H3': ['第{num}条', '详情', '实现', '结果', '分析']
            },
            'french': {
                'H1': ['Chapitre {num}', 'Introduction', 'Conclusion', 'Résumé', 'Références'],
                'H2': ['Section {num}', 'Méthodologie', 'Résultats', 'Discussion', 'Analyse'],
                'H3': ['Sous-section {num}', 'Détails', 'Implémentation', 'Évaluation', 'Tests']
            },
            'german': {
                'H1': ['Kapitel {num}', 'Einführung', 'Zusammenfassung', 'Schlussfolgerung', 'Literatur'],
                'H2': ['Abschnitt {num}', 'Methodik', 'Ergebnisse', 'Diskussion', 'Analyse'],
                'H3': ['Unterabschnitt {num}', 'Details', 'Implementierung', 'Bewertung', 'Tests']
            },
            'spanish': {
                'H1': ['Capítulo {num}', 'Introducción', 'Resumen', 'Conclusión', 'Referencias'],
                'H2': ['Sección {num}', 'Metodología', 'Resultados', 'Discusión', 'Análisis'],
                'H3': ['Subsección {num}', 'Detalles', 'Implementación', 'Evaluación', 'Pruebas']
            }
        }
        
        training_blocks = []
        labels = []
        
        languages = list(multilingual_templates.keys())
        samples_per_language = num_samples // len(languages)
        
        for language in languages:
            templates = multilingual_templates[language]
            
            for level in ['H1', 'H2', 'H3']:
                level_templates = templates[level]
                level_samples = samples_per_language // 3
                
                for _ in range(level_samples):
                    template = random.choice(level_templates)
                    text = self._fill_template_placeholders(template)
                    
                    # Create block with language-specific characteristics
                    block = self._generate_heading_sample(level)
                    block['text'] = text
                    block['language'] = language
                    
                    training_blocks.append(block)
                    labels.append('heading')
        
        self.logger.info(f"Generated {len(training_blocks)} multilingual samples")
        return training_blocks, labels
    
    def save_training_data(self, blocks: List[Dict[str, Any]], labels: List[str], 
                          output_path: str = 'training_data.json'):
        """Save training data to JSON file"""
        
        training_data = {
            'blocks': blocks,
            'labels': labels,
            'metadata': {
                'total_samples': len(blocks),
                'heading_samples': labels.count('heading'),
                'text_samples': labels.count('text'),
                'features': [
                    'text', 'page', 'bbox', 'font_info', 'width', 'height'
                ]
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Training data saved to {output_path}")
    
    def load_training_data(self, input_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Load training data from JSON file"""
        
        with open(input_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        blocks = training_data['blocks']
        labels = training_data['labels']
        
        self.logger.info(f"Loaded {len(blocks)} training samples from {input_path}")
        return blocks, labels
    
    def augment_real_data(self, real_blocks: List[Dict[str, Any]], 
                         real_labels: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Augment real data with variations for better training"""
        
        augmented_blocks = real_blocks.copy()
        augmented_labels = real_labels.copy()
        
        # Create variations of existing samples
        for i, (block, label) in enumerate(zip(real_blocks, real_labels)):
            if label == 'heading':
                # Create variations of heading samples
                variations = self._create_heading_variations(block)
                augmented_blocks.extend(variations)
                augmented_labels.extend(['heading'] * len(variations))
        
        self.logger.info(f"Augmented data from {len(real_blocks)} to {len(augmented_blocks)} samples")
        return augmented_blocks, augmented_labels
    
    def _create_heading_variations(self, heading_block: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create variations of a heading block for data augmentation"""
        variations = []
        original_text = heading_block['text']
        
        # Variation 1: Change capitalization
        if not original_text.isupper():
            upper_variation = heading_block.copy()
            upper_variation['text'] = original_text.upper()
            variations.append(upper_variation)
        
        # Variation 2: Title case
        if not original_text.istitle():
            title_variation = heading_block.copy()
            title_variation['text'] = original_text.title()
            variations.append(title_variation)
        
        # Variation 3: Slightly different font size
        font_variation = heading_block.copy()
        original_size = font_variation['font_info']['size']
        font_variation['font_info'] = font_variation['font_info'].copy()
        font_variation['font_info']['size'] = original_size + random.uniform(-1, 1)
        variations.append(font_variation)
        
        # Variation 4: Different position (if heading)
        if len(variations) < 3:
            position_variation = heading_block.copy()
            position_variation['bbox'] = position_variation['bbox'].copy()
            position_variation['bbox'][0] += random.uniform(-20, 20)  # Slight x position change
            variations.append(position_variation)
        
        return variations[:2]  # Return at most 2 variations to avoid overwhelming the data


def main():
    """Generate training data for heading classifier"""
    logging.basicConfig(level=logging.INFO)
    
    generator = TrainingDataGenerator()
    
    # Generate synthetic data
    blocks, labels = generator.generate_synthetic_training_data(1000)
    
    # Add multilingual samples
    multilingual_blocks, multilingual_labels = generator.generate_multilingual_samples(200)
    blocks.extend(multilingual_blocks)
    labels.extend(multilingual_labels)
    
    # Save training data
    Path('training').mkdir(exist_ok=True)
    generator.save_training_data(blocks, labels, 'training/synthetic_training_data.json')
    
    print(f"Generated {len(blocks)} training samples")
    print(f"Headings: {labels.count('heading')}")
    print(f"Text: {labels.count('text')}")


if __name__ == '__main__':
    main()