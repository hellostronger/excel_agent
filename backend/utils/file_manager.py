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

# Import text processor for text analysis
try:
    from .text_processor import text_processor
    TEXT_ANALYSIS_AVAILABLE = True
except ImportError:
    logger.warning("Text processor not available, text analysis will be disabled")
    TEXT_ANALYSIS_AVAILABLE = False

# Import markdown converter
try:
    from .markdown_converter import markdown_converter
    MARKDOWN_CONVERSION_AVAILABLE = True
except ImportError:
    logger.warning("Markdown converter not available, markdown conversion will be disabled")
    MARKDOWN_CONVERSION_AVAILABLE = False

# Import HTML converter
try:
    from .html_converter import html_converter
    HTML_CONVERSION_AVAILABLE = True
except ImportError:
    logger.warning("HTML converter not available, HTML conversion will be disabled")
    HTML_CONVERSION_AVAILABLE = False


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
            
            # Generate multiple formats for supported file types
            self._generate_multiple_formats(file_id)
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Stored file {file_id} at {stored_file_path}")
            return str(stored_file_path)
            
        except Exception as e:
            logger.error(f"Error storing file {file_id}: {e}")
            raise
    
    def analyze_file_text(self, file_id: str, max_rows: int = None) -> bool:
        """
        分析文件文本内容并更新元数据。
        
        Args:
            file_id: 文件ID
            max_rows: 最大分析行数
            
        Returns:
            bool: 分析是否成功
        """
        if not TEXT_ANALYSIS_AVAILABLE:
            logger.warning("Text analysis not available, skipping text analysis")
            return False
        
        if file_id not in self._metadata:
            logger.error(f"File {file_id} not found in metadata")
            return False
        
        file_info = self._metadata[file_id]
        file_path = file_info.get('file_path')
        
        if not file_path or not Path(file_path).exists():
            logger.error(f"File path not found or file doesn't exist: {file_path}")
            return False
        
        # 检查文件是否为Excel文件
        file_extension = Path(file_path).suffix.lower()
        if file_extension not in ['.xlsx', '.xls', '.xlsm']:
            logger.info(f"File {file_id} is not an Excel file, skipping text analysis")
            return False
        
        try:
            logger.info(f"Starting text analysis for file {file_id}")
            
            # 执行文本分析
            text_metadata = text_processor.get_text_metadata(file_path, max_rows)
            
            # 更新文件元数据
            self._metadata[file_id].update(text_metadata)
            
            # 添加分析时间戳
            self._metadata[file_id]['text_analyzed_at'] = datetime.now().isoformat()
            
            # 保存更新的元数据
            self._save_metadata()
            
            logger.info(f"Text analysis completed for file {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing text for file {file_id}: {e}")
            return False
    
    def get_text_analysis(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文件的文本分析结果。
        
        Args:
            file_id: 文件ID
            
        Returns:
            文本分析结果或None
        """
        if file_id not in self._metadata:
            return None
        
        file_info = self._metadata[file_id]
        return file_info.get('text_analysis')
    
    def search_files_by_keywords(self, keywords: List[str], match_any: bool = True) -> List[Dict[str, Any]]:
        """
        根据关键词搜索文件。
        
        Args:
            keywords: 关键词列表
            match_any: True表示匹配任意关键词，False表示匹配所有关键词
            
        Returns:
            匹配的文件信息列表
        """
        if not keywords:
            return []
        
        matching_files = []
        keywords_lower = [k.lower() for k in keywords]
        
        for file_id, file_info in self._metadata.items():
            text_analysis = file_info.get('text_analysis')
            if not text_analysis:
                continue
            
            # 检查文件中的关键词
            file_keywords = []
            
            # 从top_words中获取关键词
            top_words = text_analysis.get('top_words', {})
            file_keywords.extend([word.lower() for word in top_words.keys()])
            
            # 从各sheet的关键词中获取
            keywords_by_sheet = text_analysis.get('keywords_by_sheet', {})
            for sheet_keywords in keywords_by_sheet.values():
                file_keywords.extend([kw[0].lower() for kw in sheet_keywords if isinstance(kw, tuple)])
            
            # 检查匹配
            if match_any:
                # 匹配任意关键词
                if any(kw in file_keywords for kw in keywords_lower):
                    matching_files.append({
                        'file_id': file_id,
                        'file_info': file_info,
                        'matched_keywords': [kw for kw in keywords_lower if kw in file_keywords]
                    })
            else:
                # 匹配所有关键词
                if all(kw in file_keywords for kw in keywords_lower):
                    matching_files.append({
                        'file_id': file_id,
                        'file_info': file_info,
                        'matched_keywords': keywords_lower
                    })
        
        return matching_files
    
    def _generate_multiple_formats(self, file_id: str) -> Dict[str, bool]:
        """
        Generate multiple formats (markdown, HTML, etc.) for a file.
        
        Args:
            file_id: File identifier
            
        Returns:
            dict: Results of format generation attempts
        """
        results = {
            'markdown': False,
            'html': False
        }
        
        if file_id not in self._metadata:
            logger.error(f"File {file_id} not found in metadata")
            return results
        
        file_info = self._metadata[file_id]
        file_path = file_info.get('file_path')
        
        if not file_path or not Path(file_path).exists():
            logger.error(f"File path not found or file doesn't exist: {file_path}")
            return results
        
        # Check if file type is supported for conversion
        file_extension = Path(file_path).suffix.lower()
        if file_extension not in ['.xlsx', '.xls', '.xlsm', '.csv']:
            logger.info(f"File {file_id} is not supported for format conversion (extension: {file_extension})")
            return results
        
        storage_dir = Path(file_info['storage_dir'])
        
        # Generate markdown format
        results['markdown'] = self._convert_to_markdown(file_id, file_path, storage_dir)
        
        # Generate HTML format
        results['html'] = self._convert_to_html(file_id, file_path, storage_dir)
        
        # Update metadata with conversion results
        self._metadata[file_id]['format_conversions'] = {
            'markdown_success': results['markdown'],
            'html_success': results['html'],
            'converted_at': datetime.now().isoformat()
        }
        
        return results
    
    def _convert_to_markdown(self, file_id: str, file_path: str, storage_dir: Path) -> bool:
        """
        Convert file to markdown format and store it.
        
        Args:
            file_id: File identifier
            file_path: Path to the source file
            storage_dir: Directory to store the markdown file
            
        Returns:
            bool: True if conversion was successful, False otherwise
        """
        if not MARKDOWN_CONVERSION_AVAILABLE:
            logger.warning("Markdown conversion not available, skipping markdown conversion")
            return False
        
        try:
            logger.info(f"Converting file {file_id} to markdown")
            
            # Convert file to markdown
            markdown_data = markdown_converter.convert_file_to_markdown(file_path, file_id)
            
            # Save markdown file
            markdown_file_path = storage_dir / "data.md"
            saved_path = markdown_converter.save_markdown(markdown_data, str(markdown_file_path))
            
            # Update file metadata with markdown information
            self._metadata[file_id].update({
                'markdown_path': saved_path,
                'markdown_metadata_path': str(Path(saved_path).with_suffix('.metadata.json')),
                'markdown_converted_at': datetime.now().isoformat()
            })
            
            logger.info(f"Successfully converted file {file_id} to markdown")
            return True
            
        except Exception as e:
            logger.error(f"Error converting file {file_id} to markdown: {e}")
            self._metadata[file_id]['markdown_conversion_error'] = str(e)
            return False
    
    def _convert_to_html(self, file_id: str, file_path: str, storage_dir: Path) -> bool:
        """
        Convert file to HTML format and store it.
        
        Args:
            file_id: File identifier
            file_path: Path to the source file
            storage_dir: Directory to store the HTML file
            
        Returns:
            bool: True if conversion was successful, False otherwise
        """
        if not HTML_CONVERSION_AVAILABLE:
            logger.warning("HTML conversion not available, skipping HTML conversion")
            return False
        
        try:
            logger.info(f"Converting file {file_id} to HTML")
            
            # Convert file to HTML
            html_data = html_converter.convert_file_to_html(file_path, file_id)
            
            # Save HTML file
            html_file_path = storage_dir / "data.html"
            saved_path = html_converter.save_html(html_data, str(html_file_path))
            
            # Update file metadata with HTML information
            self._metadata[file_id].update({
                'html_path': saved_path,
                'html_metadata_path': str(Path(saved_path).with_suffix('.metadata.json')),
                'html_converted_at': datetime.now().isoformat()
            })
            
            logger.info(f"Successfully converted file {file_id} to HTML")
            return True
            
        except Exception as e:
            logger.error(f"Error converting file {file_id} to HTML: {e}")
            self._metadata[file_id]['html_conversion_error'] = str(e)
            return False
    
    def get_markdown_content(self, file_id: str) -> Optional[str]:
        """
        Get markdown content for a file.
        
        Args:
            file_id: File identifier
            
        Returns:
            Markdown content as string or None if not available
        """
        if file_id not in self._metadata:
            return None
        
        file_info = self._metadata[file_id]
        markdown_path = file_info.get('markdown_path')
        
        if not markdown_path or not Path(markdown_path).exists():
            return None
        
        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading markdown file for {file_id}: {e}")
            return None
    
    def get_markdown_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get markdown metadata for a file.
        
        Args:
            file_id: File identifier
            
        Returns:
            Markdown metadata or None if not available
        """
        if file_id not in self._metadata:
            return None
        
        file_info = self._metadata[file_id]
        metadata_path = file_info.get('markdown_metadata_path')
        
        if not metadata_path or not Path(metadata_path).exists():
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading markdown metadata for {file_id}: {e}")
            return None
    
    def get_html_content(self, file_id: str) -> Optional[str]:
        """
        Get HTML content for a file.
        
        Args:
            file_id: File identifier
            
        Returns:
            HTML content as string or None if not available
        """
        if file_id not in self._metadata:
            return None
        
        file_info = self._metadata[file_id]
        html_path = file_info.get('html_path')
        
        if not html_path or not Path(html_path).exists():
            return None
        
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading HTML file for {file_id}: {e}")
            return None
    
    def get_html_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get HTML metadata for a file.
        
        Args:
            file_id: File identifier
            
        Returns:
            HTML metadata or None if not available
        """
        if file_id not in self._metadata:
            return None
        
        file_info = self._metadata[file_id]
        metadata_path = file_info.get('html_metadata_path')
        
        if not metadata_path or not Path(metadata_path).exists():
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading HTML metadata for {file_id}: {e}")
            return None
    
    def has_markdown(self, file_id: str) -> bool:
        """
        Check if file has been converted to markdown.
        
        Args:
            file_id: File identifier
            
        Returns:
            True if markdown version exists, False otherwise
        """
        if file_id not in self._metadata:
            return False
        
        file_info = self._metadata[file_id]
        format_conversions = file_info.get('format_conversions', {})
        return (format_conversions.get('markdown_success', False) and
                file_info.get('markdown_path') and
                Path(file_info['markdown_path']).exists())
    
    def has_html(self, file_id: str) -> bool:
        """
        Check if file has been converted to HTML.
        
        Args:
            file_id: File identifier
            
        Returns:
            True if HTML version exists, False otherwise
        """
        if file_id not in self._metadata:
            return False
        
        file_info = self._metadata[file_id]
        format_conversions = file_info.get('format_conversions', {})
        return (format_conversions.get('html_success', False) and
                file_info.get('html_path') and
                Path(file_info['html_path']).exists())
    
    def get_available_formats(self, file_id: str) -> List[str]:
        """
        Get list of available formats for a file.
        
        Args:
            file_id: File identifier
            
        Returns:
            List of available format names
        """
        if file_id not in self._metadata:
            return []
        
        formats = ['original']  # Original file is always available
        
        if self.has_markdown(file_id):
            formats.append('markdown')
        
        if self.has_html(file_id):
            formats.append('html')
        
        return formats
    
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