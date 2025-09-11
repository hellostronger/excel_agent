"""Configuration management for the Excel Intelligent Agent System."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

# Constants for table analysis (inspired by ST-Raptor)
DELIMITER = "################"

# Table type constants
T_LIST = 1
T_ARRT = 2
T_SEMI = 3
T_MIX = 4
T_OTHER = -1

# Table size constants
SMALL_TABLE_ROWS = 3
SMALL_TABLE_COLUMNS = 3
BIG_TABLE_ROWS = 8
BIG_TABLE_COLUMNS = 8

# Default naming
DEFAULT_TABLE_NAME = "table"
DEFAULT_SUBTABLE_NAME = "subtable"
DEFAULT_SUBVALUE_NAME = "subvalue"
DEFAULT_SPLIT_SIG = "-"

# Schema detection
DIRECTION_KEY = "direction_key"
VLM_SCHEMA_KEY = "vlm_schema_key"
SCHEMA_TOP = True
SCHEMA_LEFT = False
SCHEMA_FAIL = -1

# Processing limits
MAX_ITER_META_INFORMATION_DETECTION = 5
MAX_ITER_PRIMITIVE = 5
MAX_RETRY_HOTREE = 3
MAX_RETRY_PRIMITIVE = 5

# Status constants
STATUS_END = 1
STATUS_RETRIEVE = 2
STATUS_AGG = 3
STATUS_SPLIT = 4

# Data type tags
TAG_DISCRETE = 1
TAG_CONTINUOUS = 2
TAG_TEXT = 3


class Config(BaseSettings):
    """Application configuration."""
    
    # SiliconFlow API Configuration
    siliconflow_api_key: str = Field(default="", env="SILICONFLOW_API_KEY")
    siliconflow_base_url: str = Field(
        default="https://api.siliconflow.cn/v1", 
        env="SILICONFLOW_BASE_URL"
    )
    
    # Model Configuration
    multimodal_model: str = Field(
        default="THUDM/GLM-4.1V-9B-Thinking", 
        env="MULTIMODAL_MODEL"
    )
    llm_model: str = Field(default="Qwen/Qwen3-8B", env="LLM_MODEL")
    embedding_model: str = Field(default="BAAI/bge-m3", env="EMBEDDING_MODEL")
    text_to_image_model: str = Field(
        default="Kwai-Kolors/Kolors", 
        env="TEXT_TO_IMAGE_MODEL"
    )
    
    # LLM Parameters Configuration (runtime configurable)
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: Optional[int] = Field(default=None, env="LLM_MAX_TOKENS")
    llm_top_p: float = Field(default=1.0, env="LLM_TOP_P")
    llm_frequency_penalty: float = Field(default=0.0, env="LLM_FREQUENCY_PENALTY")
    llm_presence_penalty: float = Field(default=0.0, env="LLM_PRESENCE_PENALTY")
    llm_stream: bool = Field(default=False, env="LLM_STREAM")
    
    # System Configuration
    log_level: str = Field(default="DEBUG", env="LOG_LEVEL")
    max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    temp_dir: str = Field(default="./tmp", env="TEMP_DIR")
    cache_dir: str = Field(default="./cache", env="CACHE_DIR")
    
    # Agent Configuration
    max_agents: int = Field(default=10, env="MAX_AGENTS")
    agent_timeout_seconds: int = Field(default=300, env="AGENT_TIMEOUT_SECONDS")
    memory_retention_days: int = Field(default=30, env="MEMORY_RETENTION_DAYS")
    
    # Cache Configuration (new)
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    cache_ttl_hours: int = Field(default=24, env="CACHE_TTL_HOURS")
    max_cache_size_mb: int = Field(default=500, env="MAX_CACHE_SIZE_MB")
    
    # Performance Configuration (new)
    enable_embedding_cache: bool = Field(default=True, env="ENABLE_EMBEDDING_CACHE")
    max_prompt_tokens: int = Field(default=4000, env="MAX_PROMPT_TOKENS")
    enable_query_decomposition: bool = Field(default=True, env="ENABLE_QUERY_DECOMPOSITION")
    
    # Cache subdirectories (computed fields)
    html_cache_dir: Optional[str] = Field(default=None, init=False)
    image_cache_dir: Optional[str] = Field(default=None, init=False)
    excel_cache_dir: Optional[str] = Field(default=None, init=False)
    schema_cache_dir: Optional[str] = Field(default=None, init=False)
    json_cache_dir: Optional[str] = Field(default=None, init=False)
    embedding_cache_dir: Optional[str] = Field(default=None, init=False)
    tree_cache_dir: Optional[str] = Field(default=None, init=False)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create cache subdirectories (inspired by ST-Raptor)
        self.html_cache_dir = os.path.join(self.cache_dir, "html")
        self.image_cache_dir = os.path.join(self.cache_dir, "image")
        self.excel_cache_dir = os.path.join(self.cache_dir, "excel")
        self.schema_cache_dir = os.path.join(self.cache_dir, "schema")
        self.json_cache_dir = os.path.join(self.cache_dir, "json")
        self.embedding_cache_dir = os.path.join(self.cache_dir, "embedding")
        self.tree_cache_dir = os.path.join(self.cache_dir, "tree")
        
        for cache_dir in [self.html_cache_dir, self.image_cache_dir, self.excel_cache_dir,
                         self.schema_cache_dir, self.json_cache_dir, self.embedding_cache_dir,
                         self.tree_cache_dir]:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Load saved configuration from storage
        self._load_saved_config()
    
    def update_llm_params(self, **params):
        """Update LLM parameters dynamically."""
        valid_params = {
            'temperature': 'llm_temperature',
            'max_tokens': 'llm_max_tokens', 
            'top_p': 'llm_top_p',
            'frequency_penalty': 'llm_frequency_penalty',
            'presence_penalty': 'llm_presence_penalty',
            'stream': 'llm_stream',
            'model': 'llm_model'
        }
        
        updated_params = {}
        
        for key, value in params.items():
            if key in valid_params:
                attr_name = valid_params[key]
                if hasattr(self, attr_name):
                    # Validate parameter ranges
                    if key == 'temperature' and not (0.0 <= value <= 2.0):
                        raise ValueError("Temperature must be between 0.0 and 2.0")
                    elif key == 'top_p' and not (0.0 <= value <= 1.0):
                        raise ValueError("Top_p must be between 0.0 and 1.0")
                    elif key in ['frequency_penalty', 'presence_penalty'] and not (-2.0 <= value <= 2.0):
                        raise ValueError("Penalties must be between -2.0 and 2.0")
                    elif key == 'max_tokens' and value is not None and value <= 0:
                        raise ValueError("Max tokens must be positive or None")
                    
                    setattr(self, attr_name, value)
                    updated_params[key] = value
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        # Save to persistent storage
        if updated_params:
            try:
                from .config_storage import save_llm_config
                current_params = self.get_llm_params()
                save_llm_config(current_params)
            except ImportError:
                pass  # config_storage not available
    
    def get_llm_params(self):
        """Get current LLM parameters."""
        return {
            'model': self.llm_model,
            'temperature': self.llm_temperature,
            'max_tokens': self.llm_max_tokens,
            'top_p': self.llm_top_p,
            'frequency_penalty': self.llm_frequency_penalty,
            'presence_penalty': self.llm_presence_penalty,
            'stream': self.llm_stream
        }
    
    def get_available_models(self):
        """Get list of available models."""
        return {
            'llm_models': [
                "Qwen/Qwen3-8B",
                "Qwen/Qwen2.5-7B-Instruct",
                "THUDM/glm-4-9b-chat",
                "01-ai/Yi-1.5-9B-Chat-16K",
                "microsoft/DialoGPT-medium",
                "meta-llama/Llama-2-7b-chat-hf"
            ],
            'multimodal_models': [
                "THUDM/GLM-4.1V-9B-Thinking",
                "THUDM/cogvlm2-llama3-chat-19B",
                "OpenGVLab/InternVL2-8B"
            ],
            'embedding_models': [
                "BAAI/bge-m3",
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-large-zh-v1.5"
            ]
        }
    
    def reset_llm_params_to_default(self):
        """Reset LLM parameters to default values."""
        self.llm_temperature = 0.7
        self.llm_max_tokens = None
        self.llm_top_p = 1.0
        self.llm_frequency_penalty = 0.0
        self.llm_presence_penalty = 0.0
        self.llm_stream = False
        
        # Save default values to storage
        try:
            from .config_storage import save_llm_config
            save_llm_config(self.get_llm_params())
        except ImportError:
            pass
    
    def _load_saved_config(self):
        """Load saved configuration from storage on startup."""
        try:
            from .config_storage import load_llm_config
            saved_config = load_llm_config()
            
            if saved_config:
                # Apply saved parameters (skip version and timestamp fields)
                saved_params = {k: v for k, v in saved_config.items() 
                              if k not in ['last_updated', 'version']}
                if saved_params:
                    self.update_llm_params(**saved_params)
                    print(f"SUCCESS: Loaded saved LLM configuration: {len(saved_params)} parameters")
        except Exception as e:
            print(f"WARNING: Failed to load saved configuration: {e}")
    
    def export_config(self, export_path: Optional[str] = None) -> bool:
        """Export current configuration to a file."""
        try:
            import json
            from datetime import datetime
            
            if not export_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"./config/exported_config_{timestamp}.json"
            
            config_data = {
                'export_timestamp': datetime.now().isoformat(),
                'llm_params': self.get_llm_params(),
                'available_models': self.get_available_models(),
                'system_config': {
                    'log_level': self.log_level,
                    'max_file_size_mb': self.max_file_size_mb,
                    'temp_dir': self.temp_dir,
                    'cache_dir': self.cache_dir,
                    'enable_cache': self.enable_cache,
                    'cache_ttl_hours': self.cache_ttl_hours,
                    'max_cache_size_mb': self.max_cache_size_mb,
                    'enable_embedding_cache': self.enable_embedding_cache,
                    'max_prompt_tokens': self.max_prompt_tokens,
                    'enable_query_decomposition': self.enable_query_decomposition
                }
            }
            
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Configuration exported to: {export_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to export configuration: {e}")
            return False


# Global configuration instance - use lazy initialization to avoid circular imports
_config_instance = None

def get_config() -> Config:
    """Get the global configuration instance with lazy initialization."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

# For backward compatibility
config = get_config()