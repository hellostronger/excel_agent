"""LLM client utilities for Excel Intelligent Agent System."""

from typing import Optional
from .siliconflow_client import SiliconFlowClient
from .config import config

# Global client instance for reuse
_llm_client: Optional[SiliconFlowClient] = None


async def get_llm_client() -> SiliconFlowClient:
    """Get or create a SiliconFlow LLM client instance."""
    global _llm_client
    
    if _llm_client is None:
        _llm_client = SiliconFlowClient(
            api_key=config.siliconflow_api_key,
            base_url=config.siliconflow_base_url
        )
    
    return _llm_client


class LLMClient:
    """Wrapper class for LLM operations with simplified interface."""
    
    def __init__(self, client: SiliconFlowClient):
        self.client = client
    
    async def complete(self, messages, max_tokens=None, temperature=None):
        """Simple completion interface that matches the expected API."""
        response = await self.client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Create a simple response object that matches expected format
        class SimpleResponse:
            def __init__(self, response_data):
                self.choices = []
                if 'choices' in response_data:
                    for choice in response_data['choices']:
                        choice_obj = type('Choice', (), {})()
                        choice_obj.message = type('Message', (), {})()
                        choice_obj.message.content = choice.get('message', {}).get('content', '')
                        self.choices.append(choice_obj)
        
        return SimpleResponse(response)


async def get_llm_client() -> LLMClient:
    """Get LLM client with simplified interface."""
    siliconflow_client = SiliconFlowClient(
        api_key=config.siliconflow_api_key,
        base_url=config.siliconflow_base_url
    )
    return LLMClient(siliconflow_client)