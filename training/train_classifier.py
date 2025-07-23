"""
Train Classifier - Train the ML heading classifier with synthetic and real data
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ml_classifier import HeadingMLClassifier
from feature_extractor import ComprehensiveFeatureExtractor
from generate_training_data import TrainingDataGenerator

class ClassifierTrainer:
    """Train heading classification models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.classifier = HeadingMLClassifier()
        self.feature_extractor = ComprehensiveFeatureExtractor()
        self.data_generator = TrainingDataGenerator()
        
    def train_from_synthetic_data(self, model_output_path: str = 'models/heading_classifier.pkl') -> Dict[str, float]:
        """Train classifier using synthetic training data"""
        
        self.logger.info("Starting training from synthetic data")
        
        # Step 1: Generate training data
        blocks, labels = self.data_generator.generate_synthetic_training_data(2000)
        
        # Add multilingual samples for bonus scoring
        multilingual_blocks, multilingual_labels = self.data_generator.generate_multilingual_samples(400)
        blocks.extend(multilingual_blocks)
        labels.extend(multilingual_labels)
        
        self.logger.info(f"Generated {len(blocks)} total training samples")
        
        # Step 2: Train classifier
        metrics = self.classifier.train(blocks, labels, model_type='random_forest')
        
        # Step 3: Save trained model
        Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(model_output_path)
        
        # Step 4: Evaluate and report
        self._log_training_results(metrics)
        
        return metrics
    
    def train_from_real_data(self, training_data_path: str, 
                           model_output_path: str = 'models/heading_classifier.pkl') -> Dict[str, float]:
        """Train classifier using real PDF data"""
        
        if not Path(training_data_path).exists():
            raise FileNotFoundError(f"Training data file not found: {training_data_path}")
        
        self.logger.info(f"Loading real training data from {training_data_path}")
        
        # Load real training data
        blocks, labels = self.data_generator.load_training_data(training_data_path)
        
        # Augment with variations
        augmented_blocks, augmented_labels = self.data_generator.augment_real_data(blocks, labels)
        
        # Add synthetic data to improve generalization
        synthetic_blocks, synthetic_labels = self.data_generator.generate_synthetic_training_data(1000)
        augmented_blocks.extend(synthetic_blocks)
        augmented_labels.extend(synthetic_labels)
        
        self.logger.info(f"Training on {len(augmented_blocks)} samples (real + synthetic)")
        
        # Train classifier
        metrics = self.classifier.train(augmented_blocks, augmented_labels, model_type='gradient_boosting')
        
        # Save model
        Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(model_output_path)
        
        self._log_training_results(metrics)
        return metrics
    
    def cross_validate_model(self, blocks: List[Dict[str, Any]], labels: List[str], 
                           cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation to evaluate model performance"""
        
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        self.logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        # Extract features
        features_df = self.feature_extractor.extract_comprehensive_features(blocks)
        y = np.array([1 if label == 'heading' else 0 for label in labels])
        
        # Perform cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(features_df, y)):
            self.logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            # Split data
            X_train = features_df.iloc[train_idx]
            X_test = features_df.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            train_blocks = [blocks[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            
            # Create and train model for this fold
            fold_classifier = HeadingMLClassifier()
            fold_classifier.train(train_blocks, train_labels)
            
            # Make predictions
            test_blocks = [blocks[i] for i in test_idx]
            predictions = fold_classifier.predict_headings(test_blocks, threshold=0.5)
            y_pred = np.array([1 if pred else 0 for pred in predictions])
            
            # Calculate metrics
            cv_scores['accuracy'].append(accuracy_score(y_test, y_pred))
            cv_scores['precision'].append(precision_score(y_test, y_pred))
            cv_scores['recall'].append(recall_score(y_test, y_pred))
            cv_scores['f1'].append(f1_score(y_test, y_pred))
        
        # Calculate mean and std for each metric
        results = {}
        for metric, scores in cv_scores.items():
            results[f'{metric}_mean'] = np.mean(scores)
            results[f'{metric}_std'] = np.std(scores)
        
        self._log_cv_results(results)
        return results
    
    def evaluate_feature_importance(self, blocks: List[Dict[str, Any]], 
                                  labels: List[str]) -> Dict[str, float]:
        """Analyze feature importance for the trained model"""
        
        self.logger.info("Analyzing feature importance")
        
        # Train model
        self.classifier.train(blocks, labels)
        
        # Get feature importance
        importance_scores = self.classifier.get_feature_importance()
        
        # Log top features
        self.logger.info("Top 10 most important features:")
        for i, (feature, score) in enumerate(list(importance_scores.items())[:10]):
            self.logger.info(f"{i+1}. {feature}: {score:.4f}")
        
        return importance_scores
    
    def optimize_hyperparameters(self, blocks: List[Dict[str, Any]], labels: List[str],
                                model_type: str = 'random_forest') -> Dict[str, Any]:
        """Optimize hyperparameters using grid search"""
        
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        
        self.logger.info(f"Optimizing hyperparameters for {model_type}")
        
        # Extract features
        features_df = self.feature_extractor.extract_comprehensive_features(blocks)
        y = np.array([1 if label == 'heading' else 0 for label in labels])
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        # Initialize model
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(random_state=42)
        elif model_type == 'gradient_boosting':
            base_model = GradientBoostingClassifier(random_state=42)
        else:
            base_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grids[model_type],
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(features_df, y)
        
        # Log results
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def _log_training_results(self, metrics: Dict[str, float]):
        """Log training results"""
        self.logger.info("Training completed successfully!")
        self.logger.info(f"Training accuracy: {metrics.get('train_accuracy', 0):.4f}")
        self.logger.info(f"Cross-validation accuracy: {metrics.get('cv_mean', 0):.4f} ± {metrics.get('cv_std', 0):.4f}")
        self.logger.info(f"Number of features: {metrics.get('n_features', 0)}")
        self.logger.info(f"Number of samples: {metrics.get('n_samples', 0)}")
        self.logger.info(f"Positive sample ratio: {metrics.get('positive_ratio', 0):.4f}")
    
    def _log_cv_results(self, results: Dict[str, float]):
        """Log cross-validation results"""
        self.logger.info("Cross-validation results:")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            self.logger.info(f"{metric.capitalize()}: {results[mean_key]:.4f} ± {results[std_key]:.4f}")
    
    def create_lightweight_model(self, blocks: List[Dict[str, Any]], labels: List[str],
                                max_model_size_mb: int = 50) -> str:
        """Create a lightweight model under size constraints"""
        
        self.logger.info(f"Creating lightweight model (max {max_model_size_mb}MB)")
        
        # Use fewer features for smaller model
        features_df = self.feature_extractor.extract_comprehensive_features(blocks, include_text_features=False)
        y = np.array([1 if label == 'heading' else 0 for label in labels])
        
        # Select top features to reduce model size
        if not features_df.empty:
            features_df, selected_features = self.feature_extractor.select_top_features(features_df, y, n_features=30)
            self.logger.info(f"Selected {len(selected_features)} features for lightweight model")
        
        # Train with smaller, faster model
        metrics = self.classifier.train(blocks, labels, model_type='logistic_regression')
        
        # Save lightweight model
        model_path = 'models/heading_classifier_lite.pkl'
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(model_path)
        
        # Check model size
        model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        self.logger.info(f"Lightweight model size: {model_size_mb:.2f}MB")
        
        if model_size_mb > max_model_size_mb:
            self.logger.warning(f"Model size ({model_size_mb:.2f}MB) exceeds limit ({max_model_size_mb}MB)")
        
        return model_path


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train heading classifier')
    parser.add_argument('--data-type', choices=['synthetic', 'real'], default='synthetic',
                       help='Type of training data to use')
    parser.add_argument('--data-path', type=str, help='Path to real training data (if using real)')
    parser.add_argument('--model-type', choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
                       default='random_forest', help='Type of model to train')
    parser.add_argument('--output', type=str, default='models/heading_classifier.pkl',
                       help='Output path for trained model')
    parser.add_argument('--cross-validate', action='store_true',
                       help='Perform cross-validation')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize hyperparameters')
    parser.add_argument('--lightweight', action='store_true',
                       help='Create lightweight model for deployment')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    trainer = ClassifierTrainer()
    
    try:
        if args.data_type == 'synthetic':
            metrics = trainer.train_from_synthetic_data(args.output)
        else:
            if not args.data_path:
                raise ValueError("--data-path required when using real data")
            metrics = trainer.train_from_real_data(args.data_path, args.output)
        
        # Generate training data for additional analysis
        blocks, labels = trainer.data_generator.generate_synthetic_training_data(1000)
        
        if args.cross_validate:
            cv_results = trainer.cross_validate_model(blocks, labels)
            print("\nCross-validation results:", cv_results)
        
        if args.optimize:
            optimization_results = trainer.optimize_hyperparameters(blocks, labels, args.model_type)
            print("\nHyperparameter optimization results:", optimization_results['best_params'])
        
        if args.lightweight:
            lite_model_path = trainer.create_lightweight_model(blocks, labels)
            print(f"\nLightweight model saved to: {lite_model_path}")
        
        # Feature importance analysis
        importance_scores = trainer.evaluate_feature_importance(blocks, labels)
        print(f"\nFeature analysis complete. Found {len(importance_scores)} features.")
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {args.output}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()