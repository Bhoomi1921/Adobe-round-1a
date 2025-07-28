"""
Hierarchy Builder Module - Build document hierarchy from classified elements
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import AgglomerativeClustering, KMeans
from collections import defaultdict, Counter
import logging

from document_parser import DocumentElement

logger = logging.getLogger(__name__)

class HierarchyLevel:
    """Represents a level in the document hierarchy"""
    
    def __init__(self, level_id: str, elements: List[int], 
                 avg_font_size: float, confidence: float):
        self.level_id = level_id
        self.element_indices = elements
        self.avg_font_size = avg_font_size
        self.confidence = confidence
        self.characteristics = {}

class HierarchyBuilder:
    """Advanced document hierarchy builder using multiple signals"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize hierarchy builder with configuration"""
        self.config = config or {
            'max_hierarchy_levels': 6,
            'min_cluster_size': 2,
            'font_size_weight': 0.4,
            'position_weight': 0.2,
            'semantic_weight': 0.3,
            'visual_weight': 0.1,
            'confidence_threshold': 0.6
        }
        
    async def build_hierarchy(self, elements: List[DocumentElement], 
                            embeddings: List[np.ndarray]) -> List[Dict]:
        """
        Build document hierarchy using multi-modal clustering
        """
        if not elements or not embeddings:
            return []
        
        try:
            # Debug: Check page numbers at start
            input_page_nums = [elem.page_num for elem in elements]
            logger.info(f"Hierarchy builder input - Page range: {min(input_page_nums)} to {max(input_page_nums)}")
            
            # Step 1: Create feature vectors for clustering
            feature_vectors = self._create_feature_vectors(elements, embeddings)
            
            # Step 2: Perform hierarchical clustering
            cluster_labels = await self._perform_clustering(feature_vectors, elements)
            
            # Step 3: Analyze clusters and assign hierarchy levels
            hierarchy_levels = self._analyze_clusters(elements, cluster_labels)
            
            # Step 4: Create structured hierarchy
            structured_hierarchy = self._create_structured_output(
                elements, embeddings, hierarchy_levels, cluster_labels
            )
            
            # Debug: Check page numbers in output
            if structured_hierarchy:
                output_page_nums = [item.get('page', 0) for item in structured_hierarchy]
                logger.info(f"Hierarchy builder output - Page range: {min(output_page_nums)} to {max(output_page_nums)}")
            
            logger.info(f"Built hierarchy with {len(hierarchy_levels)} levels")
            return structured_hierarchy
            
        except Exception as e:
            logger.error(f"Error building hierarchy: {e}")
            return []
    
    def _create_feature_vectors(self, elements: List[DocumentElement], 
                              embeddings: List[np.ndarray]) -> np.ndarray:
        """Create multi-modal feature vectors for clustering"""
        features = []
        
        # Calculate global statistics for normalization
        font_sizes = [e.font_size for e in elements]
        max_font_size = max(font_sizes) if font_sizes else 1
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 1
        
        x_positions = [e.bbox[0] for e in elements]
        max_x = max(x_positions) if x_positions else 1
        
        y_positions = [e.bbox[1] for e in elements]
        max_y = max(y_positions) if y_positions else 1
        
        for i, (element, embedding) in enumerate(zip(elements, embeddings)):
            # Normalize semantic embedding
            embedding_norm = embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) > 0 else embedding
            
            # Visual features
            visual_features = np.array([
                element.font_size / max_font_size,  # Normalized font size
                (element.font_size - avg_font_size) / avg_font_size,  # Font size deviation
                1.0 if element.is_bold else 0.0,  # Bold indicator
                1.0 if element.is_italic else 0.0,  # Italic indicator
                element.bbox[0] / max_x,  # Normalized x position
                element.bbox[1] / max_y,  # Normalized y position
                len(element.content.split()) / 20.0,  # Normalized word count
                1.0 / max(element.page_num, 1),  # Page position (earlier = higher)
                element.capitalization_ratio,  # Capitalization ratio
                1.0 if element.has_numbers else 0.0  # Contains numbers
            ])
            
            # Combine features with weights
            semantic_part = embedding_norm * self.config['semantic_weight']
            visual_part = visual_features * self.config['visual_weight']
            
            # Create combined feature vector
            combined_features = np.concatenate([semantic_part, visual_part])
            features.append(combined_features)
        
        return np.array(features)
    
    async def _perform_clustering(self, feature_vectors: np.ndarray, 
                                elements: List[DocumentElement]) -> np.ndarray:
        """Perform adaptive clustering to identify hierarchy levels"""
        
        # Determine optimal number of clusters
        n_elements = len(elements)
        unique_font_sizes = len(set(e.font_size for e in elements))
        
        # Adaptive cluster count based on document characteristics
        max_clusters = min(
            self.config['max_hierarchy_levels'],
            max(2, unique_font_sizes),
            max(2, n_elements // 3)
        )
        
        best_labels = None
        best_score = -1
        
        # Try different clustering approaches
        for n_clusters in range(2, max_clusters + 1):
            try:
                # Agglomerative clustering (better for hierarchical structures)
                agg_clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward'
                )
                labels = agg_clustering.fit_predict(feature_vectors)
                
                # Evaluate clustering quality
                score = self._evaluate_clustering(elements, labels)
                
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    
            except Exception as e:
                logger.warning(f"Clustering with {n_clusters} clusters failed: {e}")
                continue
        
        if best_labels is None:
            # Fallback: simple font-size based clustering
            logger.warning("Using fallback font-size clustering")
            best_labels = self._fallback_font_clustering(elements)
        
        return best_labels
    
    def _evaluate_clustering(self, elements: List[DocumentElement], 
                           labels: np.ndarray) -> float:
        """Evaluate clustering quality using multiple metrics"""
        if len(set(labels)) < 2:
            return 0.0
        
        score = 0.0
        
        # Font size consistency within clusters
        cluster_font_consistency = 0.0
        for cluster_id in set(labels):
            cluster_elements = [elements[i] for i, label in enumerate(labels) if label == cluster_id]
            if len(cluster_elements) > 1:
                font_sizes = [e.font_size for e in cluster_elements]
                font_std = np.std(font_sizes)
                cluster_font_consistency += 1.0 / (1.0 + font_std)
        
        score += cluster_font_consistency / len(set(labels)) * 0.4
        
        # Visual consistency (bold, position)
        visual_consistency = 0.0
        for cluster_id in set(labels):
            cluster_elements = [elements[i] for i, label in enumerate(labels) if label == cluster_id]
            if len(cluster_elements) > 1:
                bold_ratio = sum(e.is_bold for e in cluster_elements) / len(cluster_elements)
                # Prefer clusters that are consistently bold or not bold
                visual_consistency += max(bold_ratio, 1 - bold_ratio)
        
        score += visual_consistency / len(set(labels)) * 0.3
        
        # Cluster size balance (prefer reasonable cluster sizes)
        cluster_sizes = Counter(labels)
        size_balance = 1.0 - np.std(list(cluster_sizes.values())) / np.mean(list(cluster_sizes.values()))
        score += max(0, size_balance) * 0.3
        
        return score
    
    def _fallback_font_clustering(self, elements: List[DocumentElement]) -> np.ndarray:
        """Fallback clustering based on font sizes"""
        font_sizes = [e.font_size for e in elements]
        unique_sizes = sorted(set(font_sizes), reverse=True)
        
        # Map font sizes to cluster labels
        size_to_cluster = {size: i for i, size in enumerate(unique_sizes)}
        labels = [size_to_cluster[e.font_size] for e in elements]
        
        return np.array(labels)
    
    def _analyze_clusters(self, elements: List[DocumentElement], 
                        labels: np.ndarray) -> Dict[int, HierarchyLevel]:
        """Analyze clusters and create hierarchy levels"""
        hierarchy_levels = {}
        
        for cluster_id in set(labels):
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            cluster_elements = [elements[i] for i in cluster_indices]
            
            # Calculate cluster characteristics
            avg_font_size = np.mean([e.font_size for e in cluster_elements])
            bold_ratio = sum(e.is_bold for e in cluster_elements) / len(cluster_elements)
            avg_page = np.mean([e.page_num for e in cluster_elements])
            avg_position = np.mean([e.bbox[1] for e in cluster_elements])
            
            # Calculate confidence based on cluster consistency
            confidence = self._calculate_cluster_confidence(cluster_elements)
            
            # Create hierarchy level
            hierarchy_levels[cluster_id] = HierarchyLevel(
                level_id=f"L{cluster_id}",
                elements=cluster_indices,
                avg_font_size=avg_font_size,
                confidence=confidence
            )
            
            hierarchy_levels[cluster_id].characteristics = {
                'bold_ratio': bold_ratio,
                'avg_page': avg_page,
                'avg_y_position': avg_position,
                'element_count': len(cluster_elements),
                'avg_word_count': np.mean([len(e.content.split()) for e in cluster_elements])
            }
        
        return hierarchy_levels
    
    def _calculate_cluster_confidence(self, cluster_elements: List[DocumentElement]) -> float:
        """Calculate confidence score for a cluster"""
        if len(cluster_elements) < 2:
            return 0.5
        
        confidence = 0.0
        
        # Font size consistency
        font_sizes = [e.font_size for e in cluster_elements]
        font_cv = np.std(font_sizes) / np.mean(font_sizes) if np.mean(font_sizes) > 0 else 1
        confidence += (1.0 - min(font_cv, 1.0)) * 0.4
        
        # Visual consistency (bold)
        bold_ratio = sum(e.is_bold for e in cluster_elements) / len(cluster_elements)
        bold_consistency = max(bold_ratio, 1 - bold_ratio)
        confidence += bold_consistency * 0.3
        
        # Position consistency
        y_positions = [e.bbox[1] for e in cluster_elements]
        y_cv = np.std(y_positions) / np.mean(y_positions) if np.mean(y_positions) > 0 else 1
        confidence += (1.0 - min(y_cv / 2, 1.0)) * 0.3
        
        return confidence
    
    def _create_structured_output(self, elements: List[DocumentElement], 
                                embeddings: List[np.ndarray],
                                hierarchy_levels: Dict[int, HierarchyLevel],
                                labels: np.ndarray) -> List[Dict]:
        """Create structured output with hierarchy information"""
        
        # Sort hierarchy levels by average font size (larger = higher level)
        sorted_levels = sorted(
            hierarchy_levels.items(),
            key=lambda x: x[1].avg_font_size,
            reverse=True
        )
        
        # Assign H1, H2, H3, etc. based on font size ranking
        level_mapping = {}
        for rank, (cluster_id, level_obj) in enumerate(sorted_levels):
            if level_obj.confidence >= self.config['confidence_threshold']:
                level_mapping[cluster_id] = f"H{rank + 1}"
            else:
                level_mapping[cluster_id] = f"H{min(rank + 2, 6)}"  # Lower confidence gets lower priority
        
        # Create output structure
        structured_output = []
        
        for i, (element, embedding, label) in enumerate(zip(elements, embeddings, labels)):
            # Calculate semantic relevance score
            semantic_score = self._calculate_semantic_relevance(
                element, embedding, elements, embeddings
            )
            
            # Get hierarchy level
            hierarchy_level = level_mapping.get(label, "H6")
            cluster_confidence = hierarchy_levels[label].confidence
            
            # CRITICAL FIX: Ensure page number is correctly preserved
            structured_output.append({
                'text': element.content,
                'level': hierarchy_level,
                'confidence': cluster_confidence,
                'page': element.page_num,  # FIXED: Use element.page_num directly
                'position': {
                    'x': element.bbox[0],
                    'y': element.bbox[1]
                },
                'font_properties': {
                    'size': element.font_size,
                    'size_ratio': element.font_size / hierarchy_levels[label].avg_font_size,
                    'name': element.font_name,
                    'bold': element.is_bold,
                    'italic': element.is_italic
                },
                'semantic_score': semantic_score,
                'section_type': self._classify_section_type(element),
                'word_count': len(element.content.split())
            })
        
        # Debug: Verify page numbers in output
        output_pages = [item['page'] for item in structured_output]
        if output_pages:
            logger.info(f"Structured output page range: {min(output_pages)} to {max(output_pages)}")
        
        return structured_output
    
    def _calculate_semantic_relevance(self, element: DocumentElement, 
                                    embedding: np.ndarray,
                                    all_elements: List[DocumentElement],
                                    all_embeddings: List[np.ndarray]) -> float:
        """Calculate semantic relevance score for an element"""
        
        # Base score from text characteristics
        base_score = 0.0
        
        # Length appropriateness for headings
        text_length = len(element.content)
        if 5 <= text_length <= 80:
            base_score += 0.3
        elif text_length <= 120:
            base_score += 0.1
        
        # Bold formatting bonus
        if element.is_bold:
            base_score += 0.2
        
        # Position bonus (earlier pages, higher positions)
        if element.page_num <= 2:
            base_score += 0.2
        
        if element.bbox[1] < 200:  # Top of page
            base_score += 0.1
        
        # Semantic uniqueness (how different from other content)
        if len(all_embeddings) > 1:
            similarities = []
            for other_emb in all_embeddings:
                if not np.array_equal(embedding, other_emb):
                    similarity = np.dot(embedding, other_emb) / (
                        np.linalg.norm(embedding) * np.linalg.norm(other_emb)
                    )
                    similarities.append(similarity)
            
            if similarities:
                uniqueness = 1.0 - max(similarities)
                base_score += uniqueness * 0.2
        
        return min(base_score, 1.0)
    
    def _classify_section_type(self, element: DocumentElement) -> str:
        """Classify the type of section based on content"""
        content_lower = element.content.lower()
        
        # Common heading patterns
        if any(word in content_lower for word in ['introduction', 'abstract', 'summary']):
            return 'introduction'
        elif any(word in content_lower for word in ['conclusion', 'summary', 'results']):
            return 'conclusion'
        elif any(word in content_lower for word in ['method', 'approach', 'procedure']):
            return 'methodology'
        elif any(word in content_lower for word in ['reference', 'bibliography', 'citation']):
            return 'references'
        elif element.content.count('.') > 2:  # Multiple sentences
            return 'content'
        else:
            return 'heading'