"""
File Manager for persistent file storage and retrieval.

This module provides persistent storage for uploaded files and their metadata
to prevent file not found errors when the server restarts.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class FileManager:
    """Manages uploaded files with persistent storage."""
    
    def __init__(self, storage_dir: str = None, metadata_file: str = None):
        """
        Initialize FileManager.
        
        Args:
            storage_dir: Directory to store uploaded files
            metadata_file: File to store metadata
        """
        self.storage_dir = Path(storage_dir or os.path.join(os.getcwd(), 'file_storage'))
        self.metadata_file = Path(metadata_file or os.path.join(self.storage_dir, 'file_metadata.json'))
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self._metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load file metadata from persistent storage."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(metadata)} files")
                return metadata
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save file metadata to persistent storage."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=2, default=str)
            logger.debug(f"Saved metadata for {len(self._metadata)} files")
        except IOError as e:
            logger.error(f"Error saving metadata: {e}")
    
    def store_file(self, file_id: str, source_path: str, file_info: Dict[str, Any]) -> str:
        """
        Store a file and its metadata.
        
        Args:
            file_id: Unique file identifier
            source_path: Path to the source file
            file_info: File metadata
            
        Returns:
            str: Path to the stored file
        """
        try:
            # Create file-specific directory
            file_dir = self.storage_dir / file_id
            file_dir.mkdir(exist_ok=True)
            
            # Determine file extension
            source_path = Path(source_path)
            file_extension = source_path.suffix
            stored_file_path = file_dir / f"data{file_extension}"
            
            # Copy file to storage
            shutil.copy2(source_path, stored_file_path)
            
            # Update metadata
            self._metadata[file_id] = {
                **file_info,
                'file_path': str(stored_file_path),
                'original_path': str(source_path),
                'stored_at': datetime.now().isoformat(),
                'file_size': stored_file_path.stat().st_size,
                'storage_dir': str(file_dir)
            }
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Stored file {file_id} at {stored_file_path}")
            return str(stored_file_path)
            
        except Exception as e:
            logger.error(f"Error storing file {file_id}: {e}")
            raise
    
    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get file information by ID.
        
        Args:
            file_id: File identifier
            
        Returns:
            File metadata or None if not found
        """
        if file_id in self._metadata:
            file_info = self._metadata[file_id].copy()
            
            # Verify file still exists
            file_path = file_info.get('file_path')
            if file_path and Path(file_path).exists():
                return file_info
            else:
                logger.warning(f"File {file_id} metadata exists but file not found at {file_path}")
                # Clean up metadata for missing file
                self.remove_file(file_id, cleanup_only=True)
                return None
        
        return None
    
    def file_exists(self, file_id: str) -> bool:
        """
        Check if file exists.
        
        Args:
            file_id: File identifier
            
        Returns:
            True if file exists, False otherwise
        """
        return self.get_file_info(file_id) is not None
    
    def list_files(self) -> Dict[str, Any]:
        """
        List all stored files.
        
        Returns:
            Dictionary of file metadata
        """
        # Clean up metadata for missing files
        valid_files = {}
        for file_id, file_info in self._metadata.items():
            file_path = file_info.get('file_path')
            if file_path and Path(file_path).exists():
                valid_files[file_id] = file_info
            else:
                logger.info(f"Removing metadata for missing file: {file_id}")
        
        # Update metadata if any files were removed
        if len(valid_files) != len(self._metadata):
            self._metadata = valid_files
            self._save_metadata()
        
        return valid_files
    
    def remove_file(self, file_id: str, cleanup_only: bool = False) -> bool:
        """
        Remove file and its metadata.
        
        Args:
            file_id: File identifier
            cleanup_only: If True, only remove metadata without deleting physical file
            
        Returns:
            True if removed successfully, False otherwise
        """
        try:
            if file_id in self._metadata:
                file_info = self._metadata[file_id]
                
                if not cleanup_only:
                    # Remove physical file and directory
                    storage_dir = file_info.get('storage_dir')
                    if storage_dir and Path(storage_dir).exists():
                        shutil.rmtree(storage_dir)
                        logger.info(f"Removed storage directory: {storage_dir}")
                
                # Remove metadata
                del self._metadata[file_id]
                self._save_metadata()
                
                logger.info(f"Removed file metadata: {file_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing file {file_id}: {e}")
            return False
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Remove files older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of files cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        removed_count = 0
        
        files_to_remove = []
        for file_id, file_info in self._metadata.items():
            stored_at = file_info.get('stored_at')
            if stored_at:
                try:
                    stored_time = datetime.fromisoformat(stored_at)
                    if stored_time < cutoff_time:
                        files_to_remove.append(file_id)
                except ValueError:
                    logger.warning(f"Invalid stored_at timestamp for file {file_id}: {stored_at}")
        
        # Remove old files
        for file_id in files_to_remove:
            if self.remove_file(file_id):
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old files")
        
        return removed_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Storage statistics
        """
        total_size = 0
        file_count = len(self._metadata)
        
        for file_info in self._metadata.values():
            total_size += file_info.get('file_size', 0)
        
        return {
            'file_count': file_count,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'storage_dir': str(self.storage_dir),
            'metadata_file': str(self.metadata_file)
        }


# Global file manager instance
file_manager = FileManager()