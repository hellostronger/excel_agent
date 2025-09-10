"""Configuration storage and persistence for the Excel Intelligent Agent System."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .logging import get_logger

logger = get_logger(__name__)


class ConfigStorage:
    """Handle configuration persistence and storage."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "./config/user_config.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = {}
    
    def save_config(self, config_type: str, config_data: Dict[str, Any]) -> bool:
        """Save configuration data to storage."""
        try:
            # Load existing config
            all_config = self.load_all_config()
            
            # Update specific config type
            all_config[config_type] = {
                **config_data,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Save to file
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(all_config, f, indent=2, ensure_ascii=False)
            
            # Update cache
            self._cache = all_config
            
            logger.info(f"💾 [ConfigStorage] Saved {config_type} configuration")
            return True
            
        except Exception as e:
            logger.error(f"❌ [ConfigStorage] Failed to save {config_type} config: {e}")
            return False
    
    def load_config(self, config_type: str) -> Optional[Dict[str, Any]]:
        """Load specific configuration data from storage."""
        try:
            all_config = self.load_all_config()
            config_data = all_config.get(config_type)
            
            if config_data:
                logger.debug(f"📖 [ConfigStorage] Loaded {config_type} configuration")
                return config_data
            else:
                logger.debug(f"📖 [ConfigStorage] No {config_type} configuration found")
                return None
                
        except Exception as e:
            logger.error(f"❌ [ConfigStorage] Failed to load {config_type} config: {e}")
            return None
    
    def load_all_config(self) -> Dict[str, Any]:
        """Load all configuration data from storage."""
        try:
            if self._cache:
                return self._cache
            
            if not self.storage_path.exists():
                logger.debug("📖 [ConfigStorage] Config file doesn't exist, returning empty config")
                return {}
            
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self._cache = config_data
            logger.debug("📖 [ConfigStorage] Loaded all configuration from file")
            return config_data
            
        except Exception as e:
            logger.error(f"❌ [ConfigStorage] Failed to load config file: {e}")
            return {}
    
    def delete_config(self, config_type: str) -> bool:
        """Delete specific configuration data."""
        try:
            all_config = self.load_all_config()
            
            if config_type in all_config:
                del all_config[config_type]
                
                # Save updated config
                with open(self.storage_path, 'w', encoding='utf-8') as f:
                    json.dump(all_config, f, indent=2, ensure_ascii=False)
                
                # Update cache
                self._cache = all_config
                
                logger.info(f"🗑️ [ConfigStorage] Deleted {config_type} configuration")
                return True
            else:
                logger.warning(f"⚠️ [ConfigStorage] {config_type} configuration not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ [ConfigStorage] Failed to delete {config_type} config: {e}")
            return False
    
    def backup_config(self, backup_path: Optional[str] = None) -> bool:
        """Create a backup of current configuration."""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"./config/backup_config_{timestamp}.json"
            
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy current config to backup
            all_config = self.load_all_config()
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(all_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 [ConfigStorage] Created config backup at {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [ConfigStorage] Failed to create backup: {e}")
            return False
    
    def restore_config(self, backup_path: str) -> bool:
        """Restore configuration from backup."""
        try:
            backup_path = Path(backup_path)
            
            if not backup_path.exists():
                logger.error(f"❌ [ConfigStorage] Backup file not found: {backup_path}")
                return False
            
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_config = json.load(f)
            
            # Save as current config
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(backup_config, f, indent=2, ensure_ascii=False)
            
            # Clear cache
            self._cache = {}
            
            logger.info(f"🔄 [ConfigStorage] Restored config from backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [ConfigStorage] Failed to restore from backup: {e}")
            return False
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get information about stored configurations."""
        try:
            all_config = self.load_all_config()
            
            info = {
                'storage_path': str(self.storage_path),
                'file_exists': self.storage_path.exists(),
                'file_size': self.storage_path.stat().st_size if self.storage_path.exists() else 0,
                'last_modified': datetime.fromtimestamp(
                    self.storage_path.stat().st_mtime
                ).isoformat() if self.storage_path.exists() else None,
                'config_types': list(all_config.keys()),
                'total_configs': len(all_config)
            }
            
            # Add details for each config type
            config_details = {}
            for config_type, config_data in all_config.items():
                if isinstance(config_data, dict):
                    config_details[config_type] = {
                        'last_updated': config_data.get('last_updated'),
                        'version': config_data.get('version'),
                        'key_count': len(config_data)
                    }
                else:
                    config_details[config_type] = {
                        'type': type(config_data).__name__,
                        'size': len(str(config_data))
                    }
            
            info['config_details'] = config_details
            return info
            
        except Exception as e:
            logger.error(f"❌ [ConfigStorage] Failed to get config info: {e}")
            return {
                'storage_path': str(self.storage_path),
                'error': str(e)
            }


# Global storage instance
config_storage = ConfigStorage()


# Convenience functions for LLM configuration
def save_llm_config(config_params: Dict[str, Any]) -> bool:
    """Save LLM configuration parameters."""
    return config_storage.save_config('llm_params', config_params)


def load_llm_config() -> Optional[Dict[str, Any]]:
    """Load LLM configuration parameters."""
    return config_storage.load_config('llm_params')


def save_user_preferences(preferences: Dict[str, Any]) -> bool:
    """Save user preferences."""
    return config_storage.save_config('user_preferences', preferences)


def load_user_preferences() -> Optional[Dict[str, Any]]:
    """Load user preferences."""
    return config_storage.load_config('user_preferences')