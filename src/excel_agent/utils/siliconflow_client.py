"""SiliconFlow API client for external model integration."""

import asyncio
import json
import uuid
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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stream: Optional[bool] = None,
        request_id: Optional[str] = None
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Send chat completion request to SiliconFlow API."""
        # Generate request ID for tracking
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]
        
        # Use config defaults if parameters not provided
        model = model or config.llm_model
        temperature = temperature if temperature is not None else config.llm_temperature
        max_tokens = max_tokens if max_tokens is not None else config.llm_max_tokens
        top_p = top_p if top_p is not None else config.llm_top_p
        frequency_penalty = frequency_penalty if frequency_penalty is not None else config.llm_frequency_penalty
        presence_penalty = presence_penalty if presence_penalty is not None else config.llm_presence_penalty
        stream = stream if stream is not None else config.llm_stream
        
        request = SiliconFlowRequest(
            model=model,
            messages=[SiliconFlowMessage(**msg) for msg in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        # Log detailed request information with request ID
        logger.info(f"🤖 [LLM Request {request_id}] ===== STARTING LLM REQUEST =====")
        logger.info(f"🤖 [LLM Request {request_id}] Model: {model}")
        logger.info(f"🤖 [LLM Request {request_id}] Temperature: {temperature}, Max Tokens: {max_tokens}")
        logger.info(f"🤖 [LLM Request {request_id}] Stream: {stream}, Top-P: {top_p}")
        logger.info(f"🤖 [LLM Request {request_id}] Frequency Penalty: {frequency_penalty}, Presence Penalty: {presence_penalty}")
        logger.info(f"🤖 [LLM Request {request_id}] Messages Count: {len(messages)}")
        
        # Log input messages (complete content, no truncation)
        for i, msg in enumerate(messages):
            content = msg.get('content', '')
            role = msg.get('role', 'unknown')
            
            logger.info(f"🤖 [LLM Request {request_id}] Input Message {i+1} [{role}] ({len(content)} chars):")
            logger.info(f"🤖 [LLM Request {request_id}] {content}")
            logger.info(f"🤖 [LLM Request {request_id}] --- End Message {i+1} ---")
        
        try:
            if stream:
                logger.info("🤖 [LLM] Using streaming response")
                return self._stream_completion(request)
            else:
                start_time = asyncio.get_event_loop().time()
                
                response = await self.client.post(
                    "/chat/completions",
                    json=request.model_dump()
                )
                response.raise_for_status()
                
                end_time = asyncio.get_event_loop().time()
                response_data = response.json()
                
                # Log response details
                response_time = end_time - start_time
                logger.info(f"🤖 [LLM Response] Time: {response_time:.2f}s, Status: {response.status_code}")
                
                if 'usage' in response_data:
                    usage = response_data['usage']
                    logger.info(f"🤖 [LLM Usage] Prompt: {usage.get('prompt_tokens', 'N/A')}, Completion: {usage.get('completion_tokens', 'N/A')}, Total: {usage.get('total_tokens', 'N/A')} tokens")
                
                # Log response content (complete content, no truncation)
                if 'choices' in response_data and response_data['choices']:
                    for i, choice in enumerate(response_data['choices']):
                        content = choice.get('message', {}).get('content', '')
                        finish_reason = choice.get('finish_reason', 'unknown')
                        
                        logger.info(f"🤖 [LLM Response {request_id}] Choice {i+1} (finish: {finish_reason}) ({len(content)} chars):")
                        logger.info(f"🤖 [LLM Response {request_id}] {content}")
                        logger.info(f"🤖 [LLM Response {request_id}] --- End Choice {i+1} ---")
                else:
                    logger.warning(f"🤖 [LLM Response {request_id}] No choices in response: {response_data}")
                
                logger.info(f"🤖 [LLM Response {request_id}] ===== COMPLETED LLM REQUEST =====")
                
                return response_data
        
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if hasattr(e.response, 'text') else 'No error detail'
            logger.error(f"❌ [LLM Error] HTTP {e.response.status_code}: {error_detail}")
            logger.error(f"❌ [LLM Error] Request URL: {e.request.url}")
            logger.error(f"❌ [LLM Error] Request model: {model}, messages: {len(messages)}")
            raise
        except Exception as e:
            logger.error(f"❌ [LLM Error] Request failed: {type(e).__name__}: {e}")
            logger.error(f"❌ [LLM Error] Request details - model: {model}, temp: {temperature}, max_tokens: {max_tokens}")
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
        
        logger.info(f"🔤 [Embeddings Request] Model: {model}, Texts: {len(texts)}")
        
        # Log all texts for debugging (complete content)
        for i, text in enumerate(texts):
            logger.debug(f"🔤 [Embeddings Input] Text {i+1} ({len(text)} chars): {text}")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            response = await self.client.post(
                "/embeddings",
                json={
                    "model": model,
                    "input": texts
                }
            )
            response.raise_for_status()
            
            end_time = asyncio.get_event_loop().time()
            result = response.json()
            
            # Log response details
            response_time = end_time - start_time
            logger.info(f"🔤 [Embeddings Response] Time: {response_time:.2f}s")
            
            embeddings = [item["embedding"] for item in result["data"]]
            logger.info(f"🔤 [Embeddings Response] Generated {len(embeddings)} embeddings, dimension: {len(embeddings[0]) if embeddings else 0}")
            
            if 'usage' in result:
                usage = result['usage']
                logger.info(f"🔤 [Embeddings Usage] Total tokens: {usage.get('total_tokens', 'N/A')}")
            
            return embeddings
        
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ [Embeddings Error] HTTP {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"❌ [Embeddings Error] Request failed: {e}")
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