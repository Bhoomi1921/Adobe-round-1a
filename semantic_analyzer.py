"""
Semantic Analyzer Module - Generate embeddings and analyze semantic relationships
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import os

from document_parser import DocumentElement

logger = logging.getLogger(__name__)

class SemanticAnalyzer:
    """Advanced semantic analysis using transformer embeddings"""
    
    def __init__(self, model_path: str = "model"):
        """Initialize the semantic analyzer with ONNX model"""
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.onnx_session = None
        self.embedding_cache = {}
        
        # Disable tokenizer parallelism warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Initialize model components
        self._initialize_model()
    
    def _initialize_model(self):
        """Load tokenizer and ONNX model"""
        try:
            tokenizer_path = self.model_path / "tokenizer"
            model_file = self.model_path / "model.onnx"
            
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                local_files_only=True
            )
            
            # Load ONNX model
            self.onnx_session = InferenceSession(
                str(model_file),
                providers=["CPUExecutionProvider"]
            )
            
            logger.info("Semantic analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic analyzer: {e}")
            raise
    
    async def generate_embeddings(self, elements: List[DocumentElement]) -> List[np.ndarray]:
        """
        Generate semantic embeddings for document elements
        """
        if not elements:
            return []
        
        embeddings = []
        
        for element in elements:
            try:
                # Check cache first
                cache_key = self._get_cache_key(element.content)
                if cache_key in self.embedding_cache:
                    embeddings.append(self.embedding_cache[cache_key])
                    continue
                
                # Generate new embedding
                embedding = await self._compute_embedding(element.content)
                
                # Cache the result
                self.embedding_cache[cache_key] = embedding
                embeddings.append(embedding)
                
            except Exception as e:
                logger.warning(f"Failed to generate embedding for '{element.content[:50]}...': {e}")
                # Use zero vector as fallback
                embeddings.append(self._get_zero_embedding())
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    async def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a single text"""
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Tokenize
            tokens = self.tokenizer(
                processed_text,
                return_tensors="np",
                truncation=True,
                max_length=128,  # Increased from 32 for better context
                padding="max_length"
            )
            
            # Prepare ONNX inputs
            onnx_inputs = {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"]
            }
            
            # Add token_type_ids if available
            if "token_type_ids" in tokens:
                onnx_inputs["token_type_ids"] = tokens["token_type_ids"]
            
            # Run inference
            onnx_outputs = self.onnx_session.run(None, onnx_inputs)
            
            # Extract embedding (mean pooling over sequence length)
            embedding = np.mean(onnx_outputs[0], axis=1)[0]
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return self._get_zero_embedding()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better embedding quality"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Handle special characters and encoding issues
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Limit length for efficiency
        if len(text) > 500:
            text = text[:500]
        
        return text
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hash(text.lower().strip())
    
    def _get_zero_embedding(self) -> np.ndarray:
        """Return zero embedding as fallback"""
        return np.zeros(384, dtype=np.float32)  # MiniLM embedding dimension
    
    def calculate_semantic_similarity(self, embedding1: np.ndarray, 
                                    embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Ensure embeddings are normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            embedding1_norm = embedding1 / norm1
            embedding2_norm = embedding2 / norm2
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)
            
            # Clamp to [-1, 1] range
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_semantic_clusters(self, embeddings: List[np.ndarray], 
                             threshold: float = 0.7) -> List[List[int]]:
        """
        Find clusters of semantically similar elements
        """
        if not embeddings:
            return []
        
        clusters = []
        used_indices = set()
        
        for i, embedding in enumerate(embeddings):
            if i in used_indices:
                continue
            
            # Start new cluster
            cluster = [i]
            used_indices.add(i)
            
            # Find similar embeddings
            for j, other_embedding in enumerate(embeddings[i+1:], i+1):
                if j in used_indices:
                    continue
                
                similarity = self.calculate_semantic_similarity(embedding, other_embedding)
                if similarity >= threshold:
                    cluster.append(j)
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def analyze_semantic_patterns(self, elements: List[DocumentElement], 
                                embeddings: List[np.ndarray]) -> Dict:
        """
        Analyze semantic patterns in the document
        """
        if not elements or not embeddings:
            return {}
        
        patterns = {
            'heading_candidates': [],
            'content_blocks': [],
            'similar_sections': [],
            'outliers': []
        }
        
        # Find heading candidates based on semantic uniqueness
        for i, (element, embedding) in enumerate(zip(elements, embeddings)):
            semantic_uniqueness = self._calculate_semantic_uniqueness(
                embedding, embeddings[:i] + embeddings[i+1:]
            )
            
            if semantic_uniqueness > 0.3 and element.is_bold:
                patterns['heading_candidates'].append({
                    'index': i,
                    'text': element.content,
                    'uniqueness': semantic_uniqueness,
                    'font_size': element.font_size
                })
        
        # Find semantic clusters
        clusters = self.find_semantic_clusters(embeddings, threshold=0.6)
        patterns['semantic_clusters'] = len(clusters)
        patterns['cluster_sizes'] = [len(cluster) for cluster in clusters]
        
        return patterns
    
    def _calculate_semantic_uniqueness(self, target_embedding: np.ndarray, 
                                     other_embeddings: List[np.ndarray]) -> float:
        """
        Calculate how semantically unique an embedding is compared to others
        """
        if not other_embeddings:
            return 1.0
        
        similarities = [
            self.calculate_semantic_similarity(target_embedding, other_emb)
            for other_emb in other_embeddings
        ]
        
        # Calculate uniqueness as 1 - max_similarity
        max_similarity = max(similarities) if similarities else 0
        return 1.0 - max_similarity
    
    def get_embedding_statistics(self, embeddings: List[np.ndarray]) -> Dict:
        """Generate statistics about the embeddings"""
        if not embeddings:
            return {}
        
        # Convert to numpy array for easier computation
        embedding_matrix = np.array(embeddings)
        
        return {
            'total_embeddings': len(embeddings),
            'embedding_dimension': embedding_matrix.shape[1],
            'mean_norm': float(np.mean([np.linalg.norm(emb) for emb in embeddings])),
            'std_norm': float(np.std([np.linalg.norm(emb) for emb in embeddings])),
            'cache_hit_rate': len(self.embedding_cache) / max(len(embeddings), 1),
            'zero_embeddings': sum(1 for emb in embeddings if np.allclose(emb, 0))
        }