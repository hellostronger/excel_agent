"""MCP Registry for managing agent capabilities."""

import asyncio
from typing import Dict, List, Optional, Type
from dataclasses import dataclass
from datetime import datetime

from .base import MCPServer, MCPClient, MCPCapability
from ..utils.logging import get_logger


@dataclass
class AgentMCPConfig:
    """Configuration for agent MCP integration."""
    agent_name: str
    capabilities: List[str]
    server_name: Optional[str] = None
    auto_initialize: bool = True
    timeout_seconds: int = 30


class MCPRegistry:
    """Registry for managing MCP servers and capabilities across agents."""
    
    def __init__(self):
        self.logger = get_logger("mcp.registry")
        self.servers: Dict[str, MCPServer] = {}
        self.clients: Dict[str, MCPClient] = {}
        self.capabilities: Dict[str, MCPCapability] = {}
        self.agent_configs: Dict[str, AgentMCPConfig] = {}
        self._initialized = False
    
    def register_capability(self, capability: MCPCapability):
        """Register a new MCP capability."""
        self.capabilities[capability.name] = capability
        self.logger.info(f"Registered MCP capability: {capability.name}")
    
    def register_agent_config(self, config: AgentMCPConfig):
        """Register agent MCP configuration."""
        self.agent_configs[config.agent_name] = config
        self.logger.info(f"Registered agent config: {config.agent_name}")
    
    def create_server(self, name: str, capability_names: List[str]) -> MCPServer:
        """Create an MCP server with specified capabilities."""
        server = MCPServer(name)
        
        for cap_name in capability_names:
            if cap_name in self.capabilities:
                server.register_capability(self.capabilities[cap_name])
            else:
                self.logger.warning(f"Capability '{cap_name}' not found for server '{name}'")
        
        self.servers[name] = server
        self.logger.info(f"Created MCP server: {name} with {len(capability_names)} capabilities")
        return server
    
    def create_client(self, name: str) -> MCPClient:
        """Create an MCP client."""
        client = MCPClient(name)
        self.clients[name] = client
        self.logger.info(f"Created MCP client: {name}")
        return client
    
    async def initialize_agent_mcp(self, agent_name: str) -> Optional[MCPClient]:
        """Initialize MCP for a specific agent."""
        if agent_name not in self.agent_configs:
            self.logger.warning(f"No MCP config found for agent: {agent_name}")
            return None
        
        config = self.agent_configs[agent_name]
        
        # Create or get server
        server_name = config.server_name or f"{agent_name}_server"
        if server_name not in self.servers:
            server = self.create_server(server_name, config.capabilities)
        else:
            server = self.servers[server_name]
        
        # Create client
        client_name = f"{agent_name}_client"
        client = self.create_client(client_name)
        client.connect_to_server(server)
        
        if config.auto_initialize:
            try:
                await asyncio.wait_for(
                    client.initialize(),
                    timeout=config.timeout_seconds
                )
                self.logger.info(f"Initialized MCP for agent: {agent_name}")
            except asyncio.TimeoutError:
                self.logger.error(f"MCP initialization timeout for agent: {agent_name}")
                return None
            except Exception as e:
                self.logger.error(f"MCP initialization failed for agent {agent_name}: {e}")
                return None
        
        return client
    
    async def initialize_all(self):
        """Initialize MCP for all registered agents."""
        if self._initialized:
            return
        
        initialization_tasks = []
        for agent_name in self.agent_configs:
            task = self.initialize_agent_mcp(agent_name)
            initialization_tasks.append(task)
        
        if initialization_tasks:
            await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        self._initialized = True
        self.logger.info("MCP Registry fully initialized")
    
    def get_agent_client(self, agent_name: str) -> Optional[MCPClient]:
        """Get MCP client for an agent."""
        client_name = f"{agent_name}_client"
        return self.clients.get(client_name)
    
    def get_server(self, server_name: str) -> Optional[MCPServer]:
        """Get MCP server by name."""
        return self.servers.get(server_name)
    
    def get_capability(self, capability_name: str) -> Optional[MCPCapability]:
        """Get MCP capability by name."""
        return self.capabilities.get(capability_name)
    
    async def call_agent_tool(
        self, 
        agent_name: str, 
        tool_name: str, 
        arguments: Dict
    ) -> Optional[any]:
        """Call a tool through an agent's MCP client."""
        client = self.get_agent_client(agent_name)
        if not client:
            self.logger.error(f"No MCP client found for agent: {agent_name}")
            return None
        
        try:
            return await client.call_tool(tool_name, arguments)
        except Exception as e:
            self.logger.error(f"Tool call failed for {agent_name}.{tool_name}: {e}")
            return None
    
    async def read_agent_resource(
        self, 
        agent_name: str, 
        resource_uri: str
    ) -> Optional[any]:
        """Read a resource through an agent's MCP client."""
        client = self.get_agent_client(agent_name)
        if not client:
            self.logger.error(f"No MCP client found for agent: {agent_name}")
            return None
        
        try:
            return await client.read_resource(resource_uri)
        except Exception as e:
            self.logger.error(f"Resource read failed for {agent_name}: {e}")
            return None
    
    def list_available_tools(self, agent_name: Optional[str] = None) -> Dict[str, List[str]]:
        """List all available tools, optionally filtered by agent."""
        tools = {}
        
        if agent_name:
            # List tools for specific agent
            client = self.get_agent_client(agent_name)
            if client and client.server:
                agent_tools = []
                for capability in client.server.capabilities.values():
                    agent_tools.extend([tool.name for tool in capability.get_tools()])
                tools[agent_name] = agent_tools
        else:
            # List tools for all agents
            for client_name, client in self.clients.items():
                if client.server:
                    agent_tools = []
                    for capability in client.server.capabilities.values():
                        agent_tools.extend([tool.name for tool in capability.get_tools()])
                    # Extract agent name from client name
                    agent_name = client_name.replace("_client", "")
                    tools[agent_name] = agent_tools
        
        return tools
    
    def get_registry_status(self) -> Dict[str, any]:
        """Get current registry status."""
        return {
            "initialized": self._initialized,
            "servers_count": len(self.servers),
            "clients_count": len(self.clients),
            "capabilities_count": len(self.capabilities),
            "agent_configs_count": len(self.agent_configs),
            "servers": list(self.servers.keys()),
            "capabilities": list(self.capabilities.keys()),
            "agent_configs": list(self.agent_configs.keys()),
            "timestamp": datetime.now().isoformat()
        }


# Global registry instance
mcp_registry = MCPRegistry()