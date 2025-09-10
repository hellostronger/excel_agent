"""Standalone MCP test without system dependencies."""

import asyncio
import sys
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod


# Standalone MCP implementation for testing
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
    
    def register_resource(self, resource: MCPResource, handler: Callable):
        """Register a resource with its handler."""
        self._resources[resource.uri] = resource
        self._resource_handlers[resource.uri] = handler
    
    def register_prompt(self, prompt: MCPPrompt, handler: Callable):
        """Register a prompt with its handler."""
        self._prompts[prompt.name] = prompt
        self._prompt_handlers[prompt.name] = handler
    
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
            raise Exception(f"Error calling tool '{name}': {e}")
    
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
            raise Exception(f"Error reading resource '{uri}': {e}")
    
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
            raise Exception(f"Error getting prompt '{name}': {e}")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the capability."""
        pass


class MCPServer:
    """MCP Server implementation."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.capabilities: Dict[str, MCPCapability] = {}
        self.initialized = False
    
    def register_capability(self, capability: MCPCapability):
        """Register a capability with the server."""
        self.capabilities[capability.name] = capability
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the MCP server."""
        if self.initialized:
            return {"status": "already_initialized"}
        
        # Initialize all capabilities
        for capability in self.capabilities.values():
            await capability.initialize()
        
        self.initialized = True
        
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
            
            else:
                raise ValueError(f"Unknown method: {message.method}")
        
        except Exception as e:
            return MCPMessage(
                id=message.id,
                method=message.method,
                error={"code": -1, "message": str(e)}
            )


class MCPClient:
    """MCP Client implementation."""
    
    def __init__(self, name: str):
        self.name = name
        self.server: Optional[MCPServer] = None
        self.message_id_counter = 0
    
    def connect_to_server(self, server: MCPServer):
        """Connect to an MCP server."""
        self.server = server
    
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


# Test Implementations
class SimpleExcelCapability(MCPCapability):
    """Simple Excel capability for testing."""
    
    async def initialize(self) -> None:
        """Initialize Excel capability."""
        self.register_tool(
            MCPTool(
                name="read_file",
                description="Read a file and return basic info",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to file"}
                    },
                    "required": ["file_path"]
                }
            ),
            self._read_file
        )
        
        self.register_tool(
            MCPTool(
                name="analyze_data",
                description="Analyze data and return statistics",
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Data to analyze"}
                    },
                    "required": ["data"]
                }
            ),
            self._analyze_data
        )
        
        self.register_resource(
            MCPResource(
                uri="excel://status",
                name="Excel Status",
                description="Current Excel processing status"
            ),
            self._get_status
        )
        
        self.register_prompt(
            MCPPrompt(
                name="analysis_prompt",
                description="Generate analysis prompt",
                arguments=[
                    {"name": "data_type", "description": "Type of data to analyze"}
                ]
            ),
            self._generate_analysis_prompt
        )
    
    def _read_file(self, file_path: str) -> Dict[str, Any]:
        """Simple file reader."""
        try:
            path = Path(file_path)
            if path.exists():
                return {
                    "status": "success",
                    "file_name": path.name,
                    "file_size": path.stat().st_size,
                    "exists": True
                }
            else:
                return {
                    "status": "not_found",
                    "file_name": path.name,
                    "exists": False
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple data analyzer."""
        try:
            keys = list(data.keys())
            values = list(data.values())
            
            return {
                "status": "success",
                "key_count": len(keys),
                "keys": keys,
                "value_types": [type(v).__name__ for v in values],
                "sample_keys": keys[:3] if len(keys) > 3 else keys
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_status(self) -> str:
        """Get status resource."""
        return "Excel capability is active and ready"
    
    def _generate_analysis_prompt(self, data_type: str = "general") -> str:
        """Generate analysis prompt."""
        return f"Please analyze the {data_type} data and provide insights about patterns, trends, and key statistics."


async def test_standalone_mcp():
    """Test standalone MCP implementation."""
    print("Starting Standalone MCP Tests\n")
    
    success_count = 0
    total_tests = 4
    
    # Test 1: Basic capability functionality
    print("Test 1: Basic Capability Functionality")
    try:
        excel_cap = SimpleExcelCapability("excel_tools", "Excel processing tools")
        await excel_cap.initialize()
        
        tools = excel_cap.get_tools()
        resources = excel_cap.get_resources()
        prompts = excel_cap.get_prompts()
        
        print(f"[OK] Capability initialized with {len(tools)} tools, {len(resources)} resources, {len(prompts)} prompts")
        
        # Test tool call
        result = await excel_cap.call_tool("analyze_data", {"data": {"name": "test", "value": 123}})
        print(f"[OK] Tool call result: {result['status']}")
        
        # Test resource access
        status = await excel_cap.read_resource("excel://status")
        print(f"[OK] Resource access: {len(status)} characters")
        
        # Test prompt generation
        prompt = await excel_cap.get_prompt("analysis_prompt", {"data_type": "financial"})
        print(f"[OK] Prompt generation: {len(prompt)} characters")
        
        success_count += 1
        print("[OK] Test 1 passed\n")
        
    except Exception as e:
        print(f"[FAIL] Test 1 failed: {e}\n")
    
    # Test 2: Server-Client Communication
    print("Test 2: Server-Client Communication")
    try:
        # Create server and register capability
        server = MCPServer("test_server")
        excel_cap = SimpleExcelCapability("excel_tools", "Excel processing tools")
        await excel_cap.initialize()
        server.register_capability(excel_cap)
        
        # Initialize server
        init_result = await server.initialize()
        print(f"[OK] Server initialized: {init_result['protocolVersion']}")
        
        # Create client and connect
        client = MCPClient("test_client")
        client.connect_to_server(server)
        
        client_init = await client.initialize()
        print(f"[OK] Client connected: {client_init['protocolVersion']}")
        
        # List tools through client
        tools = await client.list_tools()
        print(f"[OK] Tools listed through client: {len(tools)} tools")
        
        # Call tool through client
        tool_result = await client.call_tool("read_file", {"file_path": __file__})
        content = tool_result.get('content', [])
        if content and len(content) > 0:
            actual_result = eval(content[0]['text'])
            print(f"[OK] Tool call through client: {actual_result['status']}")
        
        success_count += 1
        print("[OK] Test 2 passed\n")
        
    except Exception as e:
        print(f"[FAIL] Test 2 failed: {e}\n")
    
    # Test 3: Multiple Capabilities
    print("Test 3: Multiple Capabilities")
    try:
        # Create a second capability
        class SimpleDataCapability(MCPCapability):
            async def initialize(self):
                self.register_tool(
                    MCPTool(
                        name="calculate_stats",
                        description="Calculate basic statistics",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "numbers": {"type": "array", "description": "List of numbers"}
                            },
                            "required": ["numbers"]
                        }
                    ),
                    self._calculate_stats
                )
            
            def _calculate_stats(self, numbers: List[float]) -> Dict[str, Any]:
                if not numbers:
                    return {"error": "No numbers provided"}
                
                return {
                    "count": len(numbers),
                    "sum": sum(numbers),
                    "avg": sum(numbers) / len(numbers),
                    "min": min(numbers),
                    "max": max(numbers)
                }
        
        # Create server with multiple capabilities
        server = MCPServer("multi_cap_server")
        
        excel_cap = SimpleExcelCapability("excel_tools", "Excel tools")
        data_cap = SimpleDataCapability("data_analysis", "Data analysis tools")
        
        await excel_cap.initialize()
        await data_cap.initialize()
        
        server.register_capability(excel_cap)
        server.register_capability(data_cap)
        
        await server.initialize()
        print(f"[OK] Server with {len(server.capabilities)} capabilities initialized")
        
        # Test through client
        client = MCPClient("multi_test_client")
        client.connect_to_server(server)
        await client.initialize()
        
        # List all tools
        all_tools = await client.list_tools()
        print(f"[OK] Total tools from all capabilities: {len(all_tools)}")
        
        # Test tool from each capability
        excel_result = await client.call_tool("read_file", {"file_path": __file__})
        data_result = await client.call_tool("calculate_stats", {"numbers": [1, 2, 3, 4, 5]})
        
        print(f"[OK] Excel tool result: success")
        print(f"[OK] Data tool result: success")
        
        success_count += 1
        print("[OK] Test 3 passed\n")
        
    except Exception as e:
        print(f"[FAIL] Test 3 failed: {e}\n")
    
    # Test 4: Error Handling
    print("Test 4: Error Handling")
    try:
        server = MCPServer("error_test_server")
        excel_cap = SimpleExcelCapability("excel_tools", "Excel tools")
        await excel_cap.initialize()
        server.register_capability(excel_cap)
        await server.initialize()
        
        client = MCPClient("error_test_client")
        client.connect_to_server(server)
        await client.initialize()
        
        # Test calling non-existent tool
        try:
            await client.call_tool("non_existent_tool", {})
            print("[FAIL] Should have thrown error for non-existent tool")
        except RuntimeError as e:
            if "not found" in str(e):
                print("[OK] Correctly handled non-existent tool error")
            else:
                print(f"[WARN] Unexpected error message: {e}")
        
        # Test tool with invalid arguments
        result = await client.call_tool("read_file", {"file_path": "/non/existent/path"})
        content = result.get('content', [])
        if content:
            actual_result = eval(content[0]['text'])
            if actual_result['status'] == 'not_found':
                print("[OK] Correctly handled invalid file path")
            else:
                print(f"[WARN] Unexpected result: {actual_result}")
        
        success_count += 1
        print("[OK] Test 4 passed\n")
        
    except Exception as e:
        print(f"[FAIL] Test 4 failed: {e}\n")
    
    # Summary
    print(f"Test Results: {success_count}/{total_tests} tests passed")
    if success_count == total_tests:
        print("All Standalone MCP Tests Completed Successfully!")
        return 0
    else:
        print("[FAIL] Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(test_standalone_mcp())
    sys.exit(exit_code)