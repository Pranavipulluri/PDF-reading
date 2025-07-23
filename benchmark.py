#!/usr/bin/env python3
"""
Performance Benchmark Script
Tests processing speed, accuracy, and resource usage for different document types
"""

import sys
import time
import json
import tempfile
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

class PerformanceBenchmark:
    """Comprehensive performance benchmarking system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = []
        
    def create_test_document(self, page_count: int, complexity: str = "medium") -> List[Dict[str, Any]]:
        """Create synthetic test document with specified characteristics"""
        
        from training.generate_training_data import TrainingDataGenerator
        generator = TrainingDataGenerator()
        
        blocks = []
        blocks_per_page = {
            "simple": 10,     # Text-heavy documents
            "medium": 25,     # Mixed content
            "complex": 50     # Dense, complex layouts
        }
        
        target_blocks = blocks_per_page.get(complexity, 25)
        
        for page in range(1, page_count + 1):
            # Add some headings per page
            headings_per_page = 2 if complexity == "simple" else 3 if complexity == "medium" else 5
            
            for h in range(headings_per_page):
                # Generate heading
                if h == 0:
                    level = "H1"
                elif h == 1:
                    level = "H2"  
                else:
                    level = "H3"
                
                heading = generator._generate_heading_sample(level)
                heading['page'] = page
                blocks.append(heading)
            
            # Add body text blocks
            text_blocks = target_blocks - headings_per_page
            for t in range(text_blocks):
                text_block = generator._generate_text_sample()
                text_block['page'] = page
                blocks.append(text_block)
        
        return blocks
    
    def benchmark_processing_speed(self) -> Dict[str, Any]:
        """Benchmark processing speed for different document sizes"""
        
        print("🚀 Benchmarking Processing Speed...")
        
        from heading_detector import AdvancedHeadingDetector
        from title_extractor import TitleExtractor
        
        detector = AdvancedHeadingDetector()
        title_extractor = TitleExtractor()
        
        test_cases = [
            (10, "simple", "10-page simple document"),
            (25, "medium", "25-page medium complexity"),
            (50, "medium", "50-page medium complexity"),
            (50, "complex", "50-page complex layout")
        ]
        
        results = []
        
        for pages, complexity, description in test_cases:
            print(f"  Testing: {description}")
            
            # Create test document
            document_data = self.create_test_document(pages, complexity)
            
            # Benchmark title extraction
            start_time = time.time()
            title = title_extractor.extract_title(document_data, "benchmark_doc")
            title_time = time.time() - start_time
            
            # Benchmark heading detection
            start_time = time.time()
            headings = detector.detect_headings(document_data)
            detection_time = time.time() - start_time
            
            total_time = title_time + detection_time
            
            result = {
                'description': description,
                'pages': pages,
                'complexity': complexity,
                'blocks_processed': len(document_data),
                'title_extraction_time': title_time,
                'heading_detection_time': detection_time,
                'total_processing_time': total_time,
                'blocks_per_second': len(document_data) / total_time if total_time > 0 else 0,
                'pages_per_second': pages / total_time if total_time > 0 else 0,
                'headings_detected': len(headings),
                'meets_10s_requirement': total_time <= 10.0
            }
            
            results.append(result)
            
            print(f"    Time: {total_time:.3f}s | Headings: {len(headings)} | "
                  f"Speed: {result['pages_per_second']:.2f} pages/s")
        
        return {
            'test_type': 'processing_speed',
            'results': results,
            'average_time': statistics.mean([r['total_processing_time'] for r in results]),
            'max_time': max([r['total_processing_time'] for r in results]),
            'all_meet_requirement': all(r['meets_10s_requirement'] for r in results)
        }
    
    def benchmark_accuracy(self) -> Dict[str, Any]:
        """Benchmark accuracy using synthetic ground truth data"""
        
        print("🎯 Benchmarking Accuracy...")
        
        from heading_detector import AdvancedHeadingDetector
        from training.generate_training_data import TrainingDataGenerator
        
        generator = TrainingDataGenerator()
        detector = AdvancedHeadingDetector()
        
        # Generate test data with known ground truth
        test_blocks, test_labels = generator.generate_synthetic_training_data(200)
        
        # Add more challenging cases
        multilingual_blocks, multilingual_labels = generator.generate_multilingual_samples(50)
        test_blocks.extend(multilingual_blocks)
        test_labels.extend(multilingual_labels)
        
        print(f"  Testing on {len(test_blocks)} blocks with known labels")
        
        # Get predictions
        detected_headings = detector.detect_headings(test_blocks)
        
        # Convert to binary classification for evaluation
        predicted_labels = ['text'] * len(test_blocks)
        
        # Mark blocks that were detected as headings
        detected_texts = [h['text'].lower().strip() for h in detected_headings]
        
        for i, block in enumerate(test_blocks):
            block_text = block['text'].lower().strip()
            if block_text in detected_texts:
                predicted_labels[i] = 'heading'
        
        # Calculate metrics
        true_positives = sum(1 for t, p in zip(test_labels, predicted_labels) 
                           if t == 'heading' and p == 'heading')
        false_positives = sum(1 for t, p in zip(test_labels, predicted_labels) 
                            if t == 'text' and p == 'heading')
        false_negatives = sum(1 for t, p in zip(test_labels, predicted_labels) 
                            if t == 'heading' and p == 'text')
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'test_type': 'accuracy',
            'total_samples': len(test_blocks),
            'heading_samples': test_labels.count('heading'),
            'text_samples': test_labels.count('text'),
            'detected_headings': len(detected_headings),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage during processing"""
        
        print("💾 Benchmarking Memory Usage...")
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / (1024 * 1024)
            
            # Test with increasing document sizes
            memory_results = []
            
            test_sizes = [(10, "simple"), (25, "medium"), (50, "complex")]
            
            for pages, complexity in test_sizes:
                # Create test document
                document_data = self.create_test_document(pages, complexity)
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Measure memory before processing
                memory_before = process.memory_info().rss / (1024 * 1024)
                
                # Process document
                from heading_detector import AdvancedHeadingDetector
                detector = AdvancedHeadingDetector()
                headings = detector.detect_headings(document_data)
                
                # Measure memory after processing
                memory_after = process.memory_info().rss / (1024 * 1024)
                
                memory_used = memory_after - memory_before
                
                result = {
                    'pages': pages,
                    'complexity': complexity,
                    'blocks': len(document_data),
                    'memory_before_mb': memory_before,
                    'memory_after_mb': memory_after,
                    'memory_used_mb': memory_used,
                    'memory_per_page_mb': memory_used / pages if pages > 0 else 0
                }
                
                memory_results.append(result)
                
                print(f"  {pages} pages ({complexity}): {memory_used:.2f}MB used")
                
                # Cleanup
                del document_data, detector, headings
                gc.collect()
            
            return {
                'test_type': 'memory_usage',
                'baseline_memory_mb': baseline_memory,
                'results': memory_results,
                'peak_memory_mb': max([r['memory_after_mb'] for r in memory_results]),
                'max_memory_per_page_mb': max([r['memory_per_page_mb'] for r in memory_results])
            }
            
        except ImportError:
            print("  ⚠ psutil not available, skipping memory benchmark")
            return {'test_type': 'memory_usage', 'error': 'psutil not available'}
    
    def benchmark_feature_extraction(self) -> Dict[str, Any]:
        """Benchmark feature extraction performance"""
        
        print("⚙️ Benchmarking Feature Extraction...")
        
        from feature_extractor import ComprehensiveFeatureExtractor
        from training.generate_training_data import TrainingDataGenerator
        
        generator = TrainingDataGenerator()
        extractor = ComprehensiveFeatureExtractor()
        
        test_cases = [
            (50, "Feature extraction - 50 blocks"),
            (100, "Feature extraction - 100 blocks"),
            (200, "Feature extraction - 200 blocks")
        ]
        
        results = []
        
        for block_count, description in test_cases:
            # Generate test blocks
            blocks, _ = generator.generate_synthetic_training_data(block_count)
            
            # Benchmark feature extraction
            start_time = time.time()
            features_df = extractor.extract_comprehensive_features(blocks)
            extraction_time = time.time() - start_time
            
            result = {
                'description': description,
                'block_count': block_count,
                'feature_count': len(features_df.columns) if not features_df.empty else 0,
                'extraction_time': extraction_time,
                'blocks_per_second': block_count / extraction_time if extraction_time > 0 else 0,
                'features_per_block': len(features_df.columns) if not features_df.empty else 0
            }
            
            results.append(result)
            
            print(f"  {description}: {extraction_time:.3f}s | "
                  f"{result['features_per_block']} features | "
                  f"{result['blocks_per_second']:.1f} blocks/s")
        
        return {
            'test_type': 'feature_extraction',
            'results': results,
            'average_extraction_time': statistics.mean([r['extraction_time'] for r in results])
        }
    
    def benchmark_ml_training(self) -> Dict[str, Any]:
        """Benchmark ML model training performance"""
        
        print("🤖 Benchmarking ML Training...")
        
        from training.train_classifier import ClassifierTrainer
        
        trainer = ClassifierTrainer()
        
        # Test different training set sizes
        training_sizes = [100, 500, 1000]
        results = []
        
        for size in training_sizes:
            print(f"  Training model with {size} samples...")
            
            # Generate training data
            blocks, labels = trainer.data_generator.generate_synthetic_training_data(size)
            
            # Benchmark training
            start_time = time.time()
            metrics = trainer.classifier.train(blocks, labels, model_type='logistic_regression')
            training_time = time.time() - start_time
            
            result = {
                'training_samples': size,
                'training_time': training_time,
                'train_accuracy': metrics.get('train_accuracy', 0),
                'cv_accuracy': metrics.get('cv_mean', 0),
                'samples_per_second': size / training_time if training_time > 0 else 0
            }
            
            results.append(result)
            
            print(f"    Time: {training_time:.3f}s | "
                  f"Accuracy: {result['train_accuracy']:.3f} | "
                  f"CV: {result['cv_accuracy']:.3f}")
        
        return {
            'test_type': 'ml_training',
            'results': results,
            'average_training_time': statistics.mean([r['training_time'] for r in results])
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmark tests"""
        
        print("=" * 60)
        print("PDF OUTLINE EXTRACTOR v2.0 - PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        benchmark_results = {}
        
        # Run all benchmarks
        benchmarks = [
            ("Processing Speed", self.benchmark_processing_speed),
            ("Accuracy", self.benchmark_accuracy),
            ("Memory Usage", self.benchmark_memory_usage),
            ("Feature Extraction", self.benchmark_feature_extraction),
            ("ML Training", self.benchmark_ml_training)
        ]
        
        for benchmark_name, benchmark_func in benchmarks:
            print(f"\n{benchmark_name}")
            print("-" * 40)
            
            try:
                result = benchmark_func()
                benchmark_results[benchmark_name.lower().replace(" ", "_")] = result
            except Exception as e:
                print(f"❌ {benchmark_name} failed: {e}")
                benchmark_results[benchmark_name.lower().replace(" ", "_")] = {
                    'error': str(e)
                }
        
        # Generate summary report
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Speed summary
        speed_results = benchmark_results.get('processing_speed', {})
        if 'results' in speed_results:
            max_time = speed_results.get('max_time', 0)
            avg_time = speed_results.get('average_time', 0)
            meets_req = speed_results.get('all_meet_requirement', False)
            
            print(f"Processing Speed:")
            print(f"  Average time: {avg_time:.3f}s")
            print(f"  Maximum time: {max_time:.3f}s")
            print(f"  Meets 10s requirement: {'✅' if meets_req else '❌'}")
        
        # Accuracy summary
        accuracy_results = benchmark_results.get('accuracy', {})
        if 'precision' in accuracy_results:
            precision = accuracy_results['precision']
            recall = accuracy_results['recall']
            f1 = accuracy_results['f1_score']
            
            print(f"Accuracy:")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
        
        # Memory summary
        memory_results = benchmark_results.get('memory_usage', {})
        if 'peak_memory_mb' in memory_results:
            peak_memory = memory_results['peak_memory_mb']
            print(f"Memory Usage:")
            print(f"  Peak memory: {peak_memory:.2f}MB")
        
        print("=" * 60)
        
        return benchmark_results

def main():
    """Main benchmark function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance benchmark for PDF outline extractor')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    parser.add_argument('--test', choices=['speed', 'accuracy', 'memory', 'features', 'training'], 
                       help='Run specific test only')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during benchmarking
    
    # Run benchmarks
    benchmark = PerformanceBenchmark()
    
    if args.test:
        # Run specific test
        test_methods = {
            'speed': benchmark.benchmark_processing_speed,
            'accuracy': benchmark.benchmark_accuracy,
            'memory': benchmark.benchmark_memory_usage,
            'features': benchmark.benchmark_feature_extraction,
            'training': benchmark.benchmark_ml_training
        }
        
        results = test_methods[args.test]()
    else:
        # Run all tests
        results = benchmark.run_comprehensive_benchmark()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()