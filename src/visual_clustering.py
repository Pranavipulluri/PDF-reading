"""
Visual Font Clustering - Advanced unsupervised clustering for adaptive heading detection
Uses KMeans clustering on font properties to automatically discover heading hierarchy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import logging
from typing import List, Dict, Any, Tuple, Optional
import json
from pathlib import Path

class VisualFontClusterer:
    """Advanced font clustering for adaptive heading detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.clusterer = None
        self.cluster_labels = {}
        self.cluster_hierarchy = {}
        
        # Color scheme for visualization
        self.colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
    def extract_font_features(self, blocks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract clustering features from font properties
        
        Args:
            blocks: List of text blocks with font metadata
            
        Returns:
            Feature matrix for clustering
        """
        features = []
        
        for block in blocks:
            font_info = block.get('font_info', {})
            bbox = block.get('bbox', [0, 0, 0, 0])
            
            # Extract key visual features
            feature_vector = [
                font_info.get('size', 12),              # Font size (primary)
                int(font_info.get('is_bold', False)),   # Bold weight
                int(font_info.get('is_italic', False)), # Italic style  
                bbox[1],                                 # Y-position on page
                bbox[3] - bbox[1],                      # Text height
                len(block.get('text', '')),             # Text length
                len(block.get('text', '').split()),     # Word count
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def determine_optimal_clusters(self, features: np.ndarray, max_clusters: int = 6) -> int:
        """
        Determine optimal number of clusters using silhouette analysis
        
        Args:
            features: Feature matrix
            max_clusters: Maximum clusters to test
            
        Returns:
            Optimal number of clusters
        """
        if len(features) < 4:  # Need minimum samples
            return min(len(features), 3)
        
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters + 1, len(features)))
        
        for n_clusters in cluster_range:
            try:
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(features)
                score = silhouette_score(features, cluster_labels)
                silhouette_scores.append((n_clusters, score))
            except:
                continue
        
        if not silhouette_scores:
            return 3  # Default fallback
        
        # Choose cluster count with highest silhouette score
        optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
        
        self.logger.info(f"Optimal clusters: {optimal_clusters} (silhouette score: {max(silhouette_scores, key=lambda x: x[1])[1]:.3f})")
        
        return optimal_clusters
    
    def cluster_fonts(self, blocks: List[Dict[str, Any]]) -> Tuple[Dict[int, str], Dict[str, Any]]:
        """
        Perform font clustering and assign heading levels
        
        Args:
            blocks: List of text blocks
            
        Returns:
            Tuple of (cluster_labels, cluster_info)
        """
        if not blocks:
            return {}, {}
        
        self.logger.info(f"Clustering fonts from {len(blocks)} text blocks")
        
        # Extract features
        features = self.extract_font_features(blocks)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Determine optimal number of clusters
        n_clusters = self.determine_optimal_clusters(features_scaled)
        
        # Perform clustering
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_assignments = self.clusterer.fit_predict(features_scaled)
        
        # Analyze clusters and assign hierarchy
        cluster_info = self._analyze_clusters(blocks, features, cluster_assignments)
        cluster_labels = self._assign_hierarchy_labels(cluster_info)
        
        # Store results
        self.cluster_labels = cluster_labels
        self.cluster_hierarchy = cluster_info
        
        self.logger.info(f"Created {len(cluster_labels)} font clusters with automatic hierarchy")
        
        return cluster_labels, cluster_info
    
    def _analyze_clusters(self, blocks: List[Dict[str, Any]], 
                         features: np.ndarray, 
                         cluster_assignments: np.ndarray) -> Dict[str, Any]:
        """Analyze clusters and extract characteristics"""
        
        cluster_info = {}
        
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_blocks = [blocks[i] for i in range(len(blocks)) if cluster_mask[i]]
            cluster_features = features[cluster_mask]
            
            # Calculate cluster statistics
            avg_font_size = np.mean(cluster_features[:, 0])  # Font size
            avg_y_position = np.mean(cluster_features[:, 3])  # Y position
            bold_ratio = np.mean(cluster_features[:, 1])     # Bold ratio
            avg_word_count = np.mean(cluster_features[:, 6]) # Word count
            
            # Text analysis
            cluster_texts = [block['text'] for block in cluster_blocks]
            avg_text_length = np.mean([len(text) for text in cluster_texts])
            
            cluster_info[cluster_id] = {
                'size': len(cluster_blocks),
                'avg_font_size': avg_font_size,
                'avg_y_position': avg_y_position,
                'bold_ratio': bold_ratio,
                'avg_word_count': avg_word_count,
                'avg_text_length': avg_text_length,
                'sample_texts': cluster_texts[:3],  # First 3 examples
                'blocks': cluster_blocks
            }
        
        return cluster_info
    
    def _assign_hierarchy_labels(self, cluster_info: Dict[str, Any]) -> Dict[int, str]:
        """Assign H1, H2, H3 labels based on cluster characteristics"""
        
        if not cluster_info:
            return {}
        
        # Score each cluster for "heading-ness"
        cluster_scores = []
        
        for cluster_id, info in cluster_info.items():
            # Scoring factors
            font_size_score = info['avg_font_size'] / 12.0  # Relative to typical body text
            bold_score = info['bold_ratio']
            length_score = 1.0 / (1.0 + info['avg_word_count'] / 10.0)  # Shorter is better for headings
            
            # Combined score
            total_score = (font_size_score * 0.5) + (bold_score * 0.3) + (length_score * 0.2)
            
            cluster_scores.append((cluster_id, total_score, info))
        
        # Sort by score (highest first)
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Assign labels
        labels = {}
        hierarchy_levels = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
        
        for i, (cluster_id, score, info) in enumerate(cluster_scores):
            if i < len(hierarchy_levels) and score > 1.0:  # Threshold for heading-like clusters
                labels[cluster_id] = hierarchy_levels[i]
                self.logger.info(f"Cluster {cluster_id} → {hierarchy_levels[i]} "
                               f"(score: {score:.2f}, font_size: {info['avg_font_size']:.1f}, "
                               f"bold: {info['bold_ratio']:.2f})")
            else:
                labels[cluster_id] = 'TEXT'  # Non-heading cluster
        
        return labels
    
    def apply_cluster_labels(self, blocks: List[Dict[str, Any]], 
                           cluster_labels: Dict[int, str], 
                           cluster_assignments: np.ndarray) -> List[Dict[str, Any]]:
        """Apply cluster-based labels to blocks"""
        
        enhanced_blocks = []
        
        for i, block in enumerate(blocks):
            enhanced_block = block.copy()
            
            if i < len(cluster_assignments):
                cluster_id = cluster_assignments[i]
                assigned_level = cluster_labels.get(cluster_id, 'TEXT')
                
                enhanced_block['cluster_info'] = {
                    'cluster_id': int(cluster_id),
                    'assigned_level': assigned_level,
                    'is_heading_cluster': assigned_level.startswith('H')
                }
            
            enhanced_blocks.append(enhanced_block)
        
        return enhanced_blocks
    
    def create_cluster_visualization(self, blocks: List[Dict[str, Any]], 
                                   features: np.ndarray,
                                   cluster_assignments: np.ndarray,
                                   cluster_labels: Dict[int, str],
                                   output_path: str = "/app/output/font_clusters.png"):
        """
        Create visualization of font clusters
        
        Args:
            blocks: Text blocks
            features: Feature matrix
            cluster_assignments: Cluster assignments
            cluster_labels: Cluster hierarchy labels
            output_path: Path to save visualization
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('PDF Font Clustering Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Font Size vs Y Position
            ax1 = axes[0, 0]
            unique_clusters = np.unique(cluster_assignments)
            
            for i, cluster_id in enumerate(unique_clusters):
                cluster_mask = cluster_assignments == cluster_id
                cluster_features = features[cluster_mask]
                color = self.colors[i % len(self.colors)]
                label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
                
                ax1.scatter(cluster_features[:, 0], cluster_features[:, 3], 
                          c=color, label=label, alpha=0.7, s=60)
            
            ax1.set_xlabel('Font Size')
            ax1.set_ylabel('Y Position')
            ax1.set_title('Font Size vs Position Clusters')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Font Size Distribution by Cluster
            ax2 = axes[0, 1]
            font_sizes_by_cluster = []
            cluster_names = []
            
            for cluster_id in unique_clusters:
                cluster_mask = cluster_assignments == cluster_id
                cluster_font_sizes = features[cluster_mask, 0]
                font_sizes_by_cluster.append(cluster_font_sizes)
                cluster_names.append(cluster_labels.get(cluster_id, f'C{cluster_id}'))
            
            ax2.boxplot(font_sizes_by_cluster, labels=cluster_names)
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Font Size')
            ax2.set_title('Font Size Distribution by Cluster')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Word Count vs Font Size
            ax3 = axes[1, 0]
            
            for i, cluster_id in enumerate(unique_clusters):
                cluster_mask = cluster_assignments == cluster_id
                cluster_features = features[cluster_mask]
                color = self.colors[i % len(self.colors)]
                label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
                
                ax3.scatter(cluster_features[:, 6], cluster_features[:, 0], 
                          c=color, label=label, alpha=0.7, s=60)
            
            ax3.set_xlabel('Word Count')
            ax3.set_ylabel('Font Size')
            ax3.set_title('Word Count vs Font Size')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Cluster Summary Statistics
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Create summary text
            summary_text = "Cluster Summary:\n\n"
            for cluster_id in unique_clusters:
                info = self.cluster_hierarchy.get(cluster_id, {})
                label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
                
                summary_text += f"{label} (Cluster {cluster_id}):\n"
                summary_text += f"  Size: {info.get('size', 0)} blocks\n"
                summary_text += f"  Avg Font: {info.get('avg_font_size', 0):.1f}pt\n"
                summary_text += f"  Bold Ratio: {info.get('bold_ratio', 0):.2f}\n"
                summary_text += f"  Avg Words: {info.get('avg_word_count', 0):.1f}\n\n"
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            # Save the plot
            plt.tight_layout()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Cluster visualization saved to {output_path}")
            
            # Also save cluster data as JSON for debugging
            cluster_data_path = output_path.replace('.png', '_data.json')
            self._save_cluster_data(cluster_data_path, cluster_labels, self.cluster_hierarchy)
            
        except Exception as e:
            self.logger.error(f"Failed to create cluster visualization: {e}")
    
    def _save_cluster_data(self, output_path: str, cluster_labels: Dict[int, str], 
                          cluster_info: Dict[str, Any]):
        """Save cluster analysis data as JSON"""
        
        try:
            # Convert numpy types to Python types for JSON serialization
            serializable_data = {
                'cluster_labels': {str(k): v for k, v in cluster_labels.items()},
                'cluster_analysis': {}
            }
            
            for cluster_id, info in cluster_info.items():
                serializable_info = {}
                for key, value in info.items():
                    if key == 'blocks':
                        continue  # Skip blocks to reduce file size
                    elif isinstance(value, (np.integer, np.floating)):
                        serializable_info[key] = float(value)
                    elif isinstance(value, list):
                        serializable_info[key] = value[:3]  # Limit sample size
                    else:
                        serializable_info[key] = value
                
                serializable_data['cluster_analysis'][str(cluster_id)] = serializable_info
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Cluster data saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save cluster data: {e}")
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics"""
        
        if not self.cluster_hierarchy:
            return {}
        
        stats = {
            'total_clusters': len(self.cluster_hierarchy),
            'heading_clusters': len([k for k, v in self.cluster_labels.items() if v.startswith('H')]),
            'text_clusters': len([k for k, v in self.cluster_labels.items() if v == 'TEXT']),
            'hierarchy_levels': list(set(self.cluster_labels.values())),
            'cluster_details': self.cluster_hierarchy
        }
        
        return stats