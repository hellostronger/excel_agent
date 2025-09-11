"""Cache management system inspired by ST-Raptor for performance optimization."""

import os
import pickle
import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta

from .config import get_config


class CacheManager:
    """Cache manager for storing and retrieving processed data."""
    
    def __init__(self):
        self.config = get_config()
        self.cache_dir = self.config.cache_dir
        
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_type: str, key: str, extension: str = 'pkl') -> str:
        """Get the cache file path."""
        cache_subdir = getattr(self.config, f"{cache_type}_cache_dir", self.cache_dir)
        return os.path.join(cache_subdir, f"{key}.{extension}")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache file is valid and not expired."""
        if not os.path.exists(cache_path):
            return False
            
        if not self.config.enable_cache:
            return False
            
        # Check if cache is expired
        cache_age = time.time() - os.path.getmtime(cache_path)
        max_age = self.config.cache_ttl_hours * 3600
        
        return cache_age < max_age
    
    def save_tree_cache(self, tree_obj: Any, file_id: str) -> bool:
        """Save feature tree to cache (inspired by ST-Raptor)."""
        try:
            cache_path = self._get_cache_path("tree", file_id, "pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(tree_obj, f)
            return True
        except Exception as e:
            print(f"WARNING: Failed to save tree cache: {e}")
            return False
    
    def load_tree_cache(self, file_id: str) -> Optional[Any]:
        """Load feature tree from cache."""
        try:
            cache_path = self._get_cache_path("tree", file_id, "pkl")
            if self._is_cache_valid(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"WARNING: Failed to load tree cache: {e}")
        return None
    
    def save_embedding_cache(self, embeddings: Dict[str, Any], file_id: str) -> bool:
        """Save embeddings to cache."""
        try:
            cache_path = self._get_cache_path("embedding", file_id, "json")
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"WARNING: Failed to save embedding cache: {e}")
            return False
    
    def load_embedding_cache(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Load embeddings from cache."""
        try:
            cache_path = self._get_cache_path("embedding", file_id, "json")
            if self._is_cache_valid(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"WARNING: Failed to load embedding cache: {e}")
        return None
    
    def save_metadata_cache(self, metadata: Dict[str, Any], file_id: str) -> bool:
        """Save file metadata to cache."""
        try:
            cache_path = self._get_cache_path("json", f"{file_id}_metadata", "json")
            metadata['cache_timestamp'] = datetime.now().isoformat()
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"WARNING: Failed to save metadata cache: {e}")
            return False
    
    def load_metadata_cache(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Load file metadata from cache."""
        try:
            cache_path = self._get_cache_path("json", f"{file_id}_metadata", "json")
            if self._is_cache_valid(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"WARNING: Failed to load metadata cache: {e}")
        return None
    
    def save_query_result_cache(self, result: Dict[str, Any], query_hash: str) -> bool:
        """Save query result to cache."""
        try:
            cache_path = self._get_cache_path("json", f"query_{query_hash}", "json")
            result['cache_timestamp'] = datetime.now().isoformat()
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"WARNING: Failed to save query result cache: {e}")
            return False
    
    def load_query_result_cache(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Load query result from cache."""
        try:
            cache_path = self._get_cache_path("json", f"query_{query_hash}", "json")
            if self._is_cache_valid(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"WARNING: Failed to load query result cache: {e}")
        return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'total_files': 0,
            'total_size_mb': 0.0,
            'cache_dirs': {},
            'oldest_file': None,
            'newest_file': None
        }
        
        cache_dirs = [
            ('tree', self.config.tree_cache_dir),
            ('embedding', self.config.embedding_cache_dir),
            ('json', self.config.json_cache_dir),
            ('excel', self.config.excel_cache_dir),
            ('schema', self.config.schema_cache_dir)
        ]
        
        oldest_time = float('inf')
        newest_time = 0
        
        for dir_name, dir_path in cache_dirs:
            if os.path.exists(dir_path):
                files = list(Path(dir_path).glob('*'))
                dir_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
                dir_count = len(files)
                
                stats['cache_dirs'][dir_name] = {
                    'files': dir_count,
                    'size_mb': round(dir_size, 2)
                }
                
                stats['total_files'] += dir_count
                stats['total_size_mb'] += dir_size
                
                for f in files:
                    mtime = f.stat().st_mtime
                    if mtime < oldest_time:
                        oldest_time = mtime
                        stats['oldest_file'] = str(f)
                    if mtime > newest_time:
                        newest_time = mtime
                        stats['newest_file'] = str(f)
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        
        if oldest_time != float('inf'):
            stats['oldest_file_age_hours'] = round((time.time() - oldest_time) / 3600, 2)
        if newest_time > 0:
            stats['newest_file_age_hours'] = round((time.time() - newest_time) / 3600, 2)
        
        return stats
    
    def cleanup_expired_cache(self) -> Dict[str, int]:
        """Clean up expired cache files."""
        cleaned = {'files': 0, 'size_mb': 0.0}
        
        if not self.config.enable_cache:
            return cleaned
        
        max_age = self.config.cache_ttl_hours * 3600
        current_time = time.time()
        
        cache_dirs = [
            self.config.tree_cache_dir,
            self.config.embedding_cache_dir,
            self.config.json_cache_dir,
            self.config.excel_cache_dir,
            self.config.schema_cache_dir
        ]
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                for cache_file in Path(cache_dir).glob('*'):
                    file_age = current_time - cache_file.stat().st_mtime
                    if file_age > max_age:
                        try:
                            file_size = cache_file.stat().st_size / (1024 * 1024)
                            cache_file.unlink()
                            cleaned['files'] += 1
                            cleaned['size_mb'] += file_size
                        except Exception as e:
                            print(f"WARNING: Failed to delete expired cache file {cache_file}: {e}")
        
        cleaned['size_mb'] = round(cleaned['size_mb'], 2)
        return cleaned
    
    def clear_all_cache(self) -> Dict[str, int]:
        """Clear all cache files."""
        cleared = {'files': 0, 'size_mb': 0.0}
        
        cache_dirs = [
            self.config.tree_cache_dir,
            self.config.embedding_cache_dir,
            self.config.json_cache_dir,
            self.config.excel_cache_dir,
            self.config.schema_cache_dir
        ]
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                for cache_file in Path(cache_dir).glob('*'):
                    try:
                        file_size = cache_file.stat().st_size / (1024 * 1024)
                        cache_file.unlink()
                        cleared['files'] += 1
                        cleared['size_mb'] += file_size
                    except Exception as e:
                        print(f"WARNING: Failed to delete cache file {cache_file}: {e}")
        
        cleared['size_mb'] = round(cleared['size_mb'], 2)
        return cleared


# Global cache manager instance
_cache_manager_instance = None

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager()
    return _cache_manager_instance