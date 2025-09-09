"""Base agent class for the Excel Intelligent Agent System."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from google.adk.agents import LlmAgent
from google.adk.models import Gemini

from ..models.base import AgentRequest, AgentResponse, AgentStatus
from ..utils.config import config
from ..utils.logging import get_logger
from ..utils.siliconflow_client import SiliconFlowClient


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(
        self, 
        name: str,
        description: str,
        model: Optional[str] = None,
        timeout: int = None
    ):
        self.name = name
        self.description = description
        self.model = model or config.llm_model
        self.timeout = timeout or config.agent_timeout_seconds
        self.logger = get_logger(f"{__name__}.{name}")
        
        # Initialize ADK agent
        self.adk_agent = LlmAgent(
            name=self.name,
            model=Gemini(model_name=self.model),
            description=self.description
        )
        
        self.siliconflow_client = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.siliconflow_client = SiliconFlowClient()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.siliconflow_client:
            await self.siliconflow_client.__aexit__(exc_type, exc_val, exc_tb)
    
    @abstractmethod
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process the agent request and return response."""
        pass
    
    async def execute_with_timeout(self, request: AgentRequest) -> AgentResponse:
        """Execute the agent with timeout handling."""
        start_time = time.time()
        
        try:
            # Execute with timeout
            response = await asyncio.wait_for(
                self.process(request),
                timeout=self.timeout
            )
            
            # Calculate execution time
            execution_time = int((time.time() - start_time) * 1000)
            response.execution_time_ms = execution_time
            
            self.logger.info(
                f"Agent {self.name} completed successfully in {execution_time}ms"
            )
            
            return response
            
        except asyncio.TimeoutError:
            self.logger.error(f"Agent {self.name} timed out after {self.timeout}s")
            return AgentResponse(
                agent_id=self.name,
                request_id=request.request_id,
                status=AgentStatus.TIMEOUT,
                error_log=f"Agent {self.name} timed out after {self.timeout}s",
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
            
        except Exception as e:
            self.logger.error(f"Agent {self.name} failed with error: {e}")
            return AgentResponse(
                agent_id=self.name,
                request_id=request.request_id,
                status=AgentStatus.FAILED,
                error_log=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def llm_completion(
        self, 
        messages: list,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Helper method for LLM completions."""
        if not self.siliconflow_client:
            raise RuntimeError("SiliconFlow client not initialized. Use async context manager.")
        
        return await self.siliconflow_client.chat_completion(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def get_embeddings(self, texts: list) -> list:
        """Helper method for getting embeddings."""
        if not self.siliconflow_client:
            raise RuntimeError("SiliconFlow client not initialized. Use async context manager.")
        
        return await self.siliconflow_client.get_embeddings(texts)
    
    def create_success_response(
        self, 
        request: AgentRequest, 
        result: Dict[str, Any]
    ) -> AgentResponse:
        """Create a successful response."""
        return AgentResponse(
            agent_id=self.name,
            request_id=request.request_id,
            status=AgentStatus.SUCCESS,
            result=result
        )
    
    def create_error_response(
        self, 
        request: AgentRequest, 
        error_message: str
    ) -> AgentResponse:
        """Create an error response."""
        return AgentResponse(
            agent_id=self.name,
            request_id=request.request_id,
            status=AgentStatus.FAILED,
            error_log=error_message
        )