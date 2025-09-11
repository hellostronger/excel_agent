"""Embedding agent for semantic retrieval and caching (inspired by ST-Raptor)."""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import heapq

from ..models.base import AgentRequest, AgentResponse
from ..models.feature_tree import FeatureTree, TreeNode
from ..utils.cache_manager import get_cache_manager
from ..utils.config import get_config
from .base import BaseAgent


class EmbeddingAgent(BaseAgent):
    """Agent for embedding generation and semantic matching."""
    
    def __init__(self):
        super().__init__("embedding_agent")
        self.config = get_config()
        self.cache_manager = get_cache_manager()
        self._embedding_model = None
        
    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.embedding_model
                self._embedding_model = SentenceTransformer(model_name)
                self.logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                # Fallback to a simple embedding model
                try:
                    from sentence_transformers import SentenceTransformer
                    self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.logger.info("Using fallback embedding model: all-MiniLM-L6-v2")
                except Exception as e2:
                    self.logger.error(f"Failed to load fallback model: {e2}")
                    raise e2
        return self._embedding_model
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Get instruction format for embedding."""
        return f"Instruct: {task_description}\\nQuery: {query}"
    
    def generate_embeddings(self, texts: List[str], task_description: str = "Represent this text for semantic search") -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        try:
            model = self._get_embedding_model()
            
            # Format texts with instructions if needed
            if hasattr(model, 'encode') and 'instruct' in model.model_name.lower():
                formatted_texts = [self.get_detailed_instruct(task_description, text) for text in texts]
            else:
                formatted_texts = texts
            
            # Generate embeddings
            embeddings = model.encode(formatted_texts, convert_to_tensor=False, show_progress_bar=False)
            
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), 384))  # Default embedding dimension
    
    def compute_similarity(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and candidates."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, candidate_embeddings)
        return similarities.flatten()
    
    def find_topk_indices(self, similarities: np.ndarray, k: int) -> List[int]:
        """Find top-k indices with highest similarities."""
        topk_with_indices = heapq.nlargest(k, enumerate(similarities), key=lambda x: x[1])
        indices = [index for index, value in topk_with_indices]
        return indices
    
    def create_embedding_dict(self, feature_tree: FeatureTree) -> Dict[str, Any]:
        """Create embedding dictionary from feature tree (ST-Raptor style)."""
        all_values = feature_tree.all_value_list()
        
        if not all_values:
            return {}
        
        # Clean and prepare texts
        clean_values = []
        value_mapping = {}
        
        for i, value in enumerate(all_values):
            clean_value = str(value).strip()
            if clean_value and clean_value != "#":  # Filter empty and placeholder values
                clean_values.append(clean_value)
                value_mapping[len(clean_values) - 1] = i
        
        if not clean_values:
            return {}
        
        # Generate embeddings
        embeddings = self.generate_embeddings(clean_values, "Represent this table content for semantic search")
        
        # Create embedding dictionary
        embedding_dict = {
            "texts": clean_values,
            "embeddings": embeddings.tolist(),
            "original_indices": [value_mapping[i] for i in range(len(clean_values))],
            "model_name": self.config.embedding_model,
            "dimension": embeddings.shape[1] if embeddings.size > 0 else 0,
            "count": len(clean_values)
        }
        
        return embedding_dict
    
    def save_embedding_dict(self, embedding_dict: Dict[str, Any], file_path: str) -> bool:
        """Save embedding dictionary to file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(embedding_dict, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save embeddings to {file_path}: {e}")
            return False
    
    def load_embedding_dict(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load embedding dictionary from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                embedding_dict = json.load(f)
            
            # Convert embeddings back to numpy array
            if "embeddings" in embedding_dict:
                embedding_dict["embeddings"] = np.array(embedding_dict["embeddings"])
            
            return embedding_dict
        except Exception as e:
            self.logger.error(f"Failed to load embeddings from {file_path}: {e}")
            return None
    
    def match_query_to_content(self, query: str, embedding_dict: Dict[str, Any], top_k: int = 5) -> List[Tuple[str, float, int]]:
        """Match query to table content using embeddings."""
        if not embedding_dict or "texts" not in embedding_dict or "embeddings" not in embedding_dict:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query], "Represent this query for searching table content")
            
            if query_embedding.size == 0:
                return []
            
            # Compute similarities
            candidate_embeddings = embedding_dict["embeddings"]
            if isinstance(candidate_embeddings, list):
                candidate_embeddings = np.array(candidate_embeddings)
            
            similarities = self.compute_similarity(query_embedding, candidate_embeddings)
            
            # Find top matches
            top_indices = self.find_topk_indices(similarities, min(top_k, len(similarities)))
            
            results = []
            for idx in top_indices:
                text = embedding_dict["texts"][idx]
                similarity = float(similarities[idx])
                original_idx = embedding_dict.get("original_indices", [idx])[idx]
                results.append((text, similarity, original_idx))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to match query to content: {e}")
            return []
    
    def process_with_cache(self, request: AgentRequest) -> AgentResponse:
        """Process request with embedding caching."""
        try:
            file_id = request.data.get("file_id")
            feature_tree = request.data.get("feature_tree")
            query = request.data.get("query", "")
            use_cache = request.data.get("use_cache", True)
            
            if not file_id or not feature_tree:
                return AgentResponse(
                    request_id=request.request_id,
                    agent_name=self.name,
                    status="error",
                    error_message="Missing file_id or feature_tree in request"
                )
            
            embedding_dict = None
            
            # Try to load from cache first
            if use_cache and self.config.enable_embedding_cache:
                embedding_dict = self.cache_manager.load_embedding_cache(file_id)
                if embedding_dict:
                    self.logger.info(f"Loaded embeddings from cache for file {file_id}")
            
            # Generate embeddings if not in cache
            if embedding_dict is None:
                self.logger.info(f"Generating embeddings for file {file_id}")
                embedding_dict = self.create_embedding_dict(feature_tree)
                
                # Save to cache
                if use_cache and self.config.enable_embedding_cache:
                    self.cache_manager.save_embedding_cache(embedding_dict, file_id)
            
            # Perform query matching if query provided
            matches = []
            if query:
                matches = self.match_query_to_content(query, embedding_dict)
            
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status="completed",
                data={
                    "embedding_dict": {
                        "count": embedding_dict.get("count", 0),
                        "dimension": embedding_dict.get("dimension", 0),
                        "model_name": embedding_dict.get("model_name", "unknown")
                    },
                    "matches": matches,
                    "cached": embedding_dict is not None and use_cache
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in embedding processing: {e}")
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status="error",
                error_message=str(e)
            )
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process embedding request."""
        return self.process_with_cache(request)


def match_sub_table(query: str, feature_tree: FeatureTree, embedding_cache_file: Optional[str] = None, top_k: int = 5) -> List[Tuple[str, float, int]]:
    """Standalone function to match query to table content (ST-Raptor style)."""
    agent = EmbeddingAgent()
    
    # Load or create embedding dict
    embedding_dict = None
    if embedding_cache_file and agent.cache_manager._is_cache_valid(embedding_cache_file):
        embedding_dict = agent.load_embedding_dict(embedding_cache_file)
    
    if embedding_dict is None:
        embedding_dict = agent.create_embedding_dict(feature_tree)
        if embedding_cache_file:
            agent.save_embedding_dict(embedding_dict, embedding_cache_file)
    
    # Match query
    return agent.match_query_to_content(query, embedding_dict, top_k)


class EmbeddingModelMultilingualE5:
    """Compatibility class for ST-Raptor style embedding operations."""
    
    _instance = None
    
    def __init__(self):
        self.agent = EmbeddingAgent()
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_embedding_dict(self, texts: List[str]) -> Dict[str, Any]:
        """Get embedding dictionary from texts."""
        if not texts:
            return {}
        
        # Create a minimal feature tree for compatibility
        from ..models.feature_tree import FeatureTree, TreeNode, IndexNode
        
        tree = FeatureTree()
        root = IndexNode(value="root")
        
        for text in texts:
            node = TreeNode(value=str(text))
            root.add_child(node)
        
        tree.set_root(root)
        return self.agent.create_embedding_dict(tree)
    
    def save_embedding_dict(self, embedding_dict: Dict[str, Any], file_path: str):
        """Save embedding dictionary."""
        self.agent.save_embedding_dict(embedding_dict, file_path)
    
    def load_embedding_dict(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load embedding dictionary."""
        return self.agent.load_embedding_dict(file_path)