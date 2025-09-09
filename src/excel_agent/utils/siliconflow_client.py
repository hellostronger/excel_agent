"""SiliconFlow API client for external model integration."""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import httpx
from pydantic import BaseModel

from .config import config
from .logging import get_logger

logger = get_logger(__name__)


class SiliconFlowMessage(BaseModel):
    """Message format for SiliconFlow API."""
    role: str
    content: str


class SiliconFlowRequest(BaseModel):
    """Request format for SiliconFlow API."""
    model: str
    messages: List[SiliconFlowMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False


class SiliconFlowResponse(BaseModel):
    """Response format from SiliconFlow API."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


class SiliconFlowClient:
    """Client for interacting with SiliconFlow API."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 300
    ):
        self.api_key = api_key or config.siliconflow_api_key
        self.base_url = base_url or config.siliconflow_base_url
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError("SiliconFlow API key is required")
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout)
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Send chat completion request to SiliconFlow API."""
        model = model or config.llm_model
        
        request = SiliconFlowRequest(
            model=model,
            messages=[SiliconFlowMessage(**msg) for msg in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        logger.info(f"Sending chat completion request to model: {model}")
        
        try:
            if stream:
                return self._stream_completion(request)
            else:
                response = await self.client.post(
                    "/chat/completions",
                    json=request.model_dump()
                )
                response.raise_for_status()
                return response.json()
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise
    
    async def _stream_completion(
        self, 
        request: SiliconFlowRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming chat completion."""
        request.stream = True
        
        async with self.client.stream(
            "POST", 
            "/chat/completions",
            json=request.model_dump()
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        continue
    
    async def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Get text embeddings from SiliconFlow API."""
        model = model or config.embedding_model
        
        logger.info(f"Getting embeddings for {len(texts)} texts using model: {model}")
        
        try:
            response = await self.client.post(
                "/embeddings",
                json={
                    "model": model,
                    "input": texts
                }
            )
            response.raise_for_status()
            
            result = response.json()
            return [item["embedding"] for item in result["data"]]
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise
    
    async def multimodal_understanding(
        self,
        text: str,
        image_url: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform multimodal understanding using GLM-4.1V model."""
        model = model or config.multimodal_model
        
        messages = [
            {
                "role": "user",
                "content": text
            }
        ]
        
        if image_url:
            messages[0]["content"] = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        
        logger.info(f"Performing multimodal understanding with model: {model}")
        
        return await self.chat_completion(
            messages=messages,
            model=model,
            temperature=0.3
        )
    
    async def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: str = "1024x1024",
        quality: str = "standard"
    ) -> Dict[str, Any]:
        """Generate image using text-to-image model."""
        model = model or config.text_to_image_model
        
        logger.info(f"Generating image with model: {model}")
        
        try:
            response = await self.client.post(
                "/images/generations",
                json={
                    "model": model,
                    "prompt": prompt,
                    "size": size,
                    "quality": quality,
                    "n": 1
                }
            )
            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if the SiliconFlow API is accessible."""
        try:
            response = await self.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model=config.llm_model,
                max_tokens=10
            )
            return "choices" in response and len(response["choices"]) > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False