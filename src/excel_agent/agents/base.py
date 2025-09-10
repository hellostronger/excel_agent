"""Base agent class for the Excel Intelligent Agent System."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from google.adk.agents import LlmAgent
from google.adk.models import Gemini

from ..models.base import AgentRequest, AgentResponse, AgentStatus
from ..utils.config import config
from ..utils.logging import get_logger
from ..utils.siliconflow_client import SiliconFlowClient
from ..mcp.base import MCPClient
from ..mcp.registry import mcp_registry


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(
        self, 
        name: str,
        description: str,
        model: Optional[str] = None,
        timeout: int = None,
        mcp_capabilities: Optional[List[str]] = None
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
        
        # MCP Integration
        self.mcp_client: Optional[MCPClient] = None
        self.mcp_capabilities = mcp_capabilities or []
        self._register_mcp_config()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.siliconflow_client = SiliconFlowClient()
        
        # Initialize MCP client if capabilities are defined
        if self.mcp_capabilities:
            await self._initialize_mcp()
        
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
    
    def _register_mcp_config(self):
        """Register MCP configuration for this agent."""
        if self.mcp_capabilities:
            from ..mcp.registry import AgentMCPConfig
            config = AgentMCPConfig(
                agent_name=self.name,
                capabilities=self.mcp_capabilities,
                auto_initialize=True
            )
            mcp_registry.register_agent_config(config)
    
    async def _initialize_mcp(self):
        """Initialize MCP client for this agent."""
        try:
            self.mcp_client = await mcp_registry.initialize_agent_mcp(self.name)
            if self.mcp_client:
                self.logger.info(f"MCP client initialized for agent: {self.name}")
            else:
                self.logger.warning(f"MCP client initialization failed for agent: {self.name}")
        except Exception as e:
            self.logger.error(f"MCP initialization error for agent {self.name}: {e}")
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """Call an MCP tool if available."""
        if not self.mcp_client:
            self.logger.warning(f"No MCP client available for agent {self.name}")
            return None
        
        try:
            result = await self.mcp_client.call_tool(tool_name, arguments)
            self.logger.debug(f"MCP tool call successful: {tool_name}")
            return result
        except Exception as e:
            self.logger.error(f"MCP tool call failed {tool_name}: {e}")
            return None
    
    async def read_mcp_resource(self, resource_uri: str) -> Optional[Any]:
        """Read an MCP resource if available."""
        if not self.mcp_client:
            self.logger.warning(f"No MCP client available for agent {self.name}")
            return None
        
        try:
            result = await self.mcp_client.read_resource(resource_uri)
            self.logger.debug(f"MCP resource read successful: {resource_uri}")
            return result
        except Exception as e:
            self.logger.error(f"MCP resource read failed {resource_uri}: {e}")
            return None
    
    async def get_mcp_prompt(self, prompt_name: str, arguments: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get an MCP prompt if available."""
        if not self.mcp_client:
            self.logger.warning(f"No MCP client available for agent {self.name}")
            return None
        
        try:
            result = await self.mcp_client.get_prompt(prompt_name, arguments)
            self.logger.debug(f"MCP prompt retrieved: {prompt_name}")
            return result
        except Exception as e:
            self.logger.error(f"MCP prompt retrieval failed {prompt_name}: {e}")
            return None
    
    async def list_mcp_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        if not self.mcp_client:
            return []
        
        try:
            return await self.mcp_client.list_tools()
        except Exception as e:
            self.logger.error(f"Failed to list MCP tools: {e}")
            return []
    
    async def list_mcp_resources(self) -> List[Dict[str, Any]]:
        """List available MCP resources."""
        if not self.mcp_client:
            return []
        
        try:
            return await self.mcp_client.list_resources()
        except Exception as e:
            self.logger.error(f"Failed to list MCP resources: {e}")
            return []
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        import json
        from datetime import datetime
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)