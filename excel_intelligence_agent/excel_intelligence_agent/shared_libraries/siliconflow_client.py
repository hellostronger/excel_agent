"""
SiliconFlow API Client for Excel Intelligence Agent

Custom model provider integration for using SiliconFlow models with ADK agents.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, AsyncIterator
import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SiliconFlowMessage(BaseModel):
    """Message format for SiliconFlow API"""
    role: str
    content: str


class SiliconFlowRequest(BaseModel):
    """Request format for SiliconFlow API"""
    model: str
    messages: List[SiliconFlowMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    top_p: float = 0.9
    top_k: int = 50


class SiliconFlowResponse(BaseModel):
    """Response format from SiliconFlow API"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


class SiliconFlowClient:
    """Client for SiliconFlow API integration"""
    
    # 硅基流动支持的模型列表
    SUPPORTED_MODELS = {
        # 通用对话模型
        "deepseek-chat": "deepseek-ai/deepseek-chat",
        "qwen-plus": "alibaba/Qwen2.5-7B-Instruct",
        "qwen-turbo": "alibaba/Qwen2.5-14B-Instruct", 
        "qwen-max": "alibaba/Qwen2.5-32B-Instruct",
        "yi-large": "01-ai/Yi-1.5-34B-Chat-16K",
        "glm-4-9b": "THUDM/glm-4-9b-chat",
        "internlm2_5-7b": "internlm/internlm2_5-7b-chat",
        
        # 代码专用模型
        "deepseek-coder": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "codegeex4": "THUDM/codegeex4-all-9b",
        
        # 数学推理模型
        "deepseek-math": "deepseek-ai/deepseek-math-7b-instruct",
        
        # 快速模型 (低成本)
        "qwen2-1.5b": "alibaba/Qwen2.5-1.5B-Instruct",
        "qwen2-7b": "alibaba/Qwen2.5-7B-Instruct",
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout: int = 300
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError("SiliconFlow API key is required. Set SILICONFLOW_API_KEY environment variable.")
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send chat completion request to SiliconFlow API
        """
        try:
            # 转换模型名称
            siliconflow_model = self.SUPPORTED_MODELS.get(model, model)
            
            # 构建请求
            request_data = SiliconFlowRequest(
                model=siliconflow_model,
                messages=[SiliconFlowMessage(role=msg["role"], content=msg["content"]) for msg in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 50)
            )
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=request_data.dict(),
                    headers=headers
                )
                
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"SiliconFlow API request failed: {str(e)}")
            raise
    
    async def generate_response(
        self,
        prompt: str,
        model: str = "qwen-plus",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response for a single prompt
        """
        messages = [{"role": "user", "content": prompt}]
        response = await self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response["choices"][0]["message"]["content"]
    
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available models"""
        return self.SUPPORTED_MODELS.copy()


# 创建全局客户端实例
def create_siliconflow_client(api_key: Optional[str] = None) -> Optional[SiliconFlowClient]:
    """创建硅基流动客户端实例"""
    import os
    
    api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        logger.warning("SILICONFLOW_API_KEY not found. SiliconFlow integration disabled.")
        return None
    
    try:
        return SiliconFlowClient(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to create SiliconFlow client: {e}")
        return None


# 与ADK集成的适配器函数
def convert_to_adk_response(siliconflow_response: Dict[str, Any]) -> Dict[str, Any]:
    """将硅基流动响应转换为ADK兼容格式"""
    try:
        content = siliconflow_response["choices"][0]["message"]["content"]
        
        # 构建ADK兼容的响应格式
        adk_response = {
            "content": {
                "parts": [{"text": content}]
            },
            "usage_metadata": siliconflow_response.get("usage", {}),
            "model": siliconflow_response.get("model"),
            "finish_reason": siliconflow_response["choices"][0].get("finish_reason", "stop")
        }
        
        return adk_response
        
    except (KeyError, IndexError) as e:
        logger.error(f"Failed to convert SiliconFlow response: {e}")
        raise ValueError("Invalid SiliconFlow response format")


def create_siliconflow_model_wrapper(model_name: str, client: SiliconFlowClient):
    """
    为ADK创建硅基流动模型包装器
    
    这个函数创建一个可以在ADK Agent中使用的模型包装器
    """
    class SiliconFlowModelWrapper:
        def __init__(self, model_name: str, client: SiliconFlowClient):
            self.model_name = model_name
            self.client = client
        
        async def generate_content_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
            """生成内容的异步方法，兼容ADK接口"""
            try:
                response = await self.client.chat_completion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens")
                )
                
                return convert_to_adk_response(response)
                
            except Exception as e:
                logger.error(f"SiliconFlow model generation failed: {e}")
                raise
    
    return SiliconFlowModelWrapper(model_name, client)