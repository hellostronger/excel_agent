"""Base MCP implementation for agent capabilities."""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from datetime import datetime

from ..utils.logging import get_logger


class MCPMessageType(str, Enum):
    """MCP message types."""
    INITIALIZE = "initialize"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"


@dataclass
class MCPTool:
    """MCP Tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MCPResource:
    """MCP Resource definition."""
    uri: str
    name: str
    description: str
    mime_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MCPPrompt:
    """MCP Prompt definition."""
    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MCPMessage:
    """MCP Protocol message."""
    id: str
    method: str
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class MCPCapability(ABC):
    """Base class for MCP capabilities."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = get_logger(f"mcp.{name}")
        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._prompts: Dict[str, MCPPrompt] = {}
        self._tool_handlers: Dict[str, Callable] = {}
        self._resource_handlers: Dict[str, Callable] = {}
        self._prompt_handlers: Dict[str, Callable] = {}
    
    def register_tool(self, tool: MCPTool, handler: Callable):
        """Register a tool with its handler."""
        self._tools[tool.name] = tool
        self._tool_handlers[tool.name] = handler
        self.logger.debug(f"Registered tool: {tool.name}")
    
    def register_resource(self, resource: MCPResource, handler: Callable):
        """Register a resource with its handler."""
        self._resources[resource.uri] = resource
        self._resource_handlers[resource.uri] = handler
        self.logger.debug(f"Registered resource: {resource.uri}")
    
    def register_prompt(self, prompt: MCPPrompt, handler: Callable):
        """Register a prompt with its handler."""
        self._prompts[prompt.name] = prompt
        self._prompt_handlers[prompt.name] = handler
        self.logger.debug(f"Registered prompt: {prompt.name}")
    
    def get_tools(self) -> List[MCPTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_resources(self) -> List[MCPResource]:
        """Get all registered resources."""
        return list(self._resources.values())
    
    def get_prompts(self) -> List[MCPPrompt]:
        """Get all registered prompts."""
        return list(self._prompts.values())
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by name."""
        if name not in self._tool_handlers:
            raise ValueError(f"Tool '{name}' not found")
        
        handler = self._tool_handlers[name]
        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler(**arguments)
            else:
                return handler(**arguments)
        except Exception as e:
            self.logger.error(f"Error calling tool '{name}': {e}")
            raise
    
    async def read_resource(self, uri: str) -> Any:
        """Read a resource by URI."""
        if uri not in self._resource_handlers:
            raise ValueError(f"Resource '{uri}' not found")
        
        handler = self._resource_handlers[uri]
        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler()
            else:
                return handler()
        except Exception as e:
            self.logger.error(f"Error reading resource '{uri}': {e}")
            raise
    
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Get a prompt by name."""
        if name not in self._prompt_handlers:
            raise ValueError(f"Prompt '{name}' not found")
        
        handler = self._prompt_handlers[name]
        try:
            args = arguments or {}
            if asyncio.iscoroutinefunction(handler):
                return await handler(**args)
            else:
                return handler(**args)
        except Exception as e:
            self.logger.error(f"Error getting prompt '{name}': {e}")
            raise
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the capability."""
        pass


class MCPServer:
    """MCP Server implementation."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.logger = get_logger(f"mcp.server.{name}")
        self.capabilities: Dict[str, MCPCapability] = {}
        self.initialized = False
    
    def register_capability(self, capability: MCPCapability):
        """Register a capability with the server."""
        self.capabilities[capability.name] = capability
        self.logger.info(f"Registered capability: {capability.name}")
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the MCP server."""
        if self.initialized:
            return {"status": "already_initialized"}
        
        # Initialize all capabilities
        for capability in self.capabilities.values():
            await capability.initialize()
        
        self.initialized = True
        self.logger.info(f"MCP Server '{self.name}' initialized")
        
        return {
            "protocolVersion": "2025-06-18",
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {}
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }
    
    async def handle_message(self, message: MCPMessage) -> MCPMessage:
        """Handle incoming MCP message."""
        try:
            if message.method == MCPMessageType.INITIALIZE:
                result = await self.initialize()
                return MCPMessage(id=message.id, method=message.method, result=result)
            
            elif message.method == MCPMessageType.TOOLS_LIST:
                tools = []
                for capability in self.capabilities.values():
                    tools.extend([tool.to_dict() for tool in capability.get_tools()])
                return MCPMessage(id=message.id, method=message.method, result={"tools": tools})
            
            elif message.method == MCPMessageType.TOOLS_CALL:
                tool_name = message.params.get("name")
                arguments = message.params.get("arguments", {})
                
                # Find capability that has this tool
                for capability in self.capabilities.values():
                    if tool_name in capability._tools:
                        result = await capability.call_tool(tool_name, arguments)
                        return MCPMessage(
                            id=message.id, 
                            method=message.method, 
                            result={"content": [{"type": "text", "text": str(result)}]}
                        )
                
                raise ValueError(f"Tool '{tool_name}' not found")
            
            elif message.method == MCPMessageType.RESOURCES_LIST:
                resources = []
                for capability in self.capabilities.values():
                    resources.extend([resource.to_dict() for resource in capability.get_resources()])
                return MCPMessage(id=message.id, method=message.method, result={"resources": resources})
            
            elif message.method == MCPMessageType.RESOURCES_READ:
                uri = message.params.get("uri")
                
                # Find capability that has this resource
                for capability in self.capabilities.values():
                    if uri in capability._resources:
                        content = await capability.read_resource(uri)
                        return MCPMessage(
                            id=message.id, 
                            method=message.method, 
                            result={"contents": [{"uri": uri, "text": str(content)}]}
                        )
                
                raise ValueError(f"Resource '{uri}' not found")
            
            elif message.method == MCPMessageType.PROMPTS_LIST:
                prompts = []
                for capability in self.capabilities.values():
                    prompts.extend([prompt.to_dict() for prompt in capability.get_prompts()])
                return MCPMessage(id=message.id, method=message.method, result={"prompts": prompts})
            
            elif message.method == MCPMessageType.PROMPTS_GET:
                prompt_name = message.params.get("name")
                arguments = message.params.get("arguments", {})
                
                # Find capability that has this prompt
                for capability in self.capabilities.values():
                    if prompt_name in capability._prompts:
                        content = await capability.get_prompt(prompt_name, arguments)
                        return MCPMessage(
                            id=message.id, 
                            method=message.method, 
                            result={"messages": [{"role": "user", "content": {"type": "text", "text": content}}]}
                        )
                
                raise ValueError(f"Prompt '{prompt_name}' not found")
            
            else:
                raise ValueError(f"Unknown method: {message.method}")
        
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            return MCPMessage(
                id=message.id,
                method=message.method,
                error={"code": -1, "message": str(e)}
            )


class MCPClient:
    """MCP Client implementation."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"mcp.client.{name}")
        self.server: Optional[MCPServer] = None
        self.message_id_counter = 0
    
    def connect_to_server(self, server: MCPServer):
        """Connect to an MCP server."""
        self.server = server
        self.logger.info(f"Connected to MCP server: {server.name}")
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID."""
        self.message_id_counter += 1
        return f"{self.name}_{self.message_id_counter}_{uuid.uuid4().hex[:8]}"
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize connection with server."""
        if not self.server:
            raise RuntimeError("No server connected")
        
        message = MCPMessage(
            id=self._generate_message_id(),
            method=MCPMessageType.INITIALIZE,
            params={"clientInfo": {"name": self.name, "version": "1.0.0"}}
        )
        
        response = await self.server.handle_message(message)
        if response.error:
            raise RuntimeError(f"Initialization failed: {response.error}")
        
        return response.result
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        if not self.server:
            raise RuntimeError("No server connected")
        
        message = MCPMessage(
            id=self._generate_message_id(),
            method=MCPMessageType.TOOLS_LIST
        )
        
        response = await self.server.handle_message(message)
        if response.error:
            raise RuntimeError(f"Failed to list tools: {response.error}")
        
        return response.result.get("tools", [])
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool."""
        if not self.server:
            raise RuntimeError("No server connected")
        
        message = MCPMessage(
            id=self._generate_message_id(),
            method=MCPMessageType.TOOLS_CALL,
            params={"name": name, "arguments": arguments}
        )
        
        response = await self.server.handle_message(message)
        if response.error:
            raise RuntimeError(f"Tool call failed: {response.error}")
        
        return response.result
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources."""
        if not self.server:
            raise RuntimeError("No server connected")
        
        message = MCPMessage(
            id=self._generate_message_id(),
            method=MCPMessageType.RESOURCES_LIST
        )
        
        response = await self.server.handle_message(message)
        if response.error:
            raise RuntimeError(f"Failed to list resources: {response.error}")
        
        return response.result.get("resources", [])
    
    async def read_resource(self, uri: str) -> Any:
        """Read a resource."""
        if not self.server:
            raise RuntimeError("No server connected")
        
        message = MCPMessage(
            id=self._generate_message_id(),
            method=MCPMessageType.RESOURCES_READ,
            params={"uri": uri}
        )
        
        response = await self.server.handle_message(message)
        if response.error:
            raise RuntimeError(f"Resource read failed: {response.error}")
        
        return response.result
    
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """List available prompts."""
        if not self.server:
            raise RuntimeError("No server connected")
        
        message = MCPMessage(
            id=self._generate_message_id(),
            method=MCPMessageType.PROMPTS_LIST
        )
        
        response = await self.server.handle_message(message)
        if response.error:
            raise RuntimeError(f"Failed to list prompts: {response.error}")
        
        return response.result.get("prompts", [])
    
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Get a prompt."""
        if not self.server:
            raise RuntimeError("No server connected")
        
        message = MCPMessage(
            id=self._generate_message_id(),
            method=MCPMessageType.PROMPTS_GET,
            params={"name": name, "arguments": arguments or {}}
        )
        
        response = await self.server.handle_message(message)
        if response.error:
            raise RuntimeError(f"Prompt get failed: {response.error}")
        
        return response.result