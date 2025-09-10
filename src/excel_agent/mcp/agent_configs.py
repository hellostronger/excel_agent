"""MCP Configuration for Excel Intelligent Agent System."""

from .registry import mcp_registry, AgentMCPConfig
from .capabilities import (
    ExcelToolsCapability,
    DataAnalysisCapability,
    FileManagementCapability,
    VisualizationCapability
)


def initialize_mcp_system():
    """Initialize the MCP system with all capabilities and agent configurations."""
    
    # Register MCP capabilities
    mcp_registry.register_capability(ExcelToolsCapability())
    mcp_registry.register_capability(DataAnalysisCapability())
    mcp_registry.register_capability(FileManagementCapability())
    mcp_registry.register_capability(VisualizationCapability())
    
    # Configure agents with their MCP capabilities
    agent_configs = [
        AgentMCPConfig(
            agent_name="FileIngestAgent",
            capabilities=["excel_tools", "file_management"],
            auto_initialize=True,
            timeout_seconds=30
        ),
        AgentMCPConfig(
            agent_name="StructureScanAgent",
            capabilities=["excel_tools"],
            auto_initialize=True,
            timeout_seconds=30
        ),
        AgentMCPConfig(
            agent_name="ColumnProfilingAgent",
            capabilities=["data_analysis", "excel_tools"],
            auto_initialize=True,
            timeout_seconds=60
        ),
        AgentMCPConfig(
            agent_name="MergeHandlingAgent",
            capabilities=["excel_tools"],
            auto_initialize=True,
            timeout_seconds=30
        ),
        AgentMCPConfig(
            agent_name="LabelingAgent",
            capabilities=["excel_tools"],
            auto_initialize=True,
            timeout_seconds=30
        ),
        AgentMCPConfig(
            agent_name="CodeGenerationAgent",
            capabilities=["excel_tools", "data_analysis", "visualization"],
            auto_initialize=True,
            timeout_seconds=120
        ),
        AgentMCPConfig(
            agent_name="ExecutionAgent",
            capabilities=["excel_tools", "file_management", "visualization"],
            auto_initialize=True,
            timeout_seconds=180
        ),
        AgentMCPConfig(
            agent_name="SummarizationAgent",
            capabilities=["data_analysis", "excel_tools"],
            auto_initialize=True,
            timeout_seconds=60
        ),
        AgentMCPConfig(
            agent_name="MemoryAgent",
            capabilities=["file_management"],
            auto_initialize=True,
            timeout_seconds=30
        ),
        AgentMCPConfig(
            agent_name="RelationDiscoveryAgent",
            capabilities=["data_analysis", "excel_tools"],
            auto_initialize=True,
            timeout_seconds=90
        ),
        AgentMCPConfig(
            agent_name="Orchestrator",
            capabilities=["excel_tools", "data_analysis", "file_management", "visualization"],
            auto_initialize=True,
            timeout_seconds=300
        )
    ]
    
    # Register all agent configurations
    for config in agent_configs:
        mcp_registry.register_agent_config(config)
    
    return mcp_registry


def get_mcp_registry():
    """Get the initialized MCP registry."""
    return mcp_registry


async def initialize_all_agent_mcp():
    """Initialize MCP for all configured agents."""
    await mcp_registry.initialize_all()
    return mcp_registry


def get_agent_mcp_status():
    """Get status of all MCP-enabled agents."""
    return {
        "registry_status": mcp_registry.get_registry_status(),
        "available_tools": mcp_registry.list_available_tools(),
        "initialized": mcp_registry._initialized
    }