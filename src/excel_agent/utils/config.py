"""Configuration management for the Excel Intelligent Agent System."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


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
    
    # System Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    temp_dir: str = Field(default="./tmp", env="TEMP_DIR")
    cache_dir: str = Field(default="./cache", env="CACHE_DIR")
    
    # Agent Configuration
    max_agents: int = Field(default=10, env="MAX_AGENTS")
    agent_timeout_seconds: int = Field(default=300, env="AGENT_TIMEOUT_SECONDS")
    memory_retention_days: int = Field(default=30, env="MEMORY_RETENTION_DAYS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)


# Global configuration instance
config = Config()