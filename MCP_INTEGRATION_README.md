# MCP (Model Context Protocol) Integration for Excel Intelligent Agent System

This document describes the MCP integration that has been implemented for the Excel Intelligent Agent System, providing each agent with enhanced capabilities through the Model Context Protocol.

## Overview

The MCP integration enhances the Excel Intelligent Agent System by providing:

- **Standardized Tool Access**: Agents can access tools through the MCP protocol
- **Resource Management**: Structured access to data sources and files
- **Prompt Templates**: Reusable prompts for consistent interactions
- **Inter-Agent Communication**: Agents can leverage capabilities from other agents
- **Extensibility**: Easy addition of new capabilities and tools

## Architecture

### Core Components

1. **MCP Base Classes** (`src/excel_agent/mcp/base.py`)
   - `MCPCapability`: Base class for implementing MCP capabilities
   - `MCPServer`: Server implementation for hosting capabilities
   - `MCPClient`: Client implementation for accessing capabilities
   - `MCPTool`, `MCPResource`, `MCPPrompt`: Data structures for MCP entities

2. **MCP Registry** (`src/excel_agent/mcp/registry.py`)
   - `MCPRegistry`: Central registry for managing capabilities and agent configurations
   - `AgentMCPConfig`: Configuration for agent MCP integration

3. **MCP Capabilities** (`src/excel_agent/mcp/capabilities.py`)
   - `ExcelToolsCapability`: Excel file operations and manipulation
   - `DataAnalysisCapability`: Advanced data analysis and statistics
   - `FileManagementCapability`: File system operations
   - `VisualizationCapability`: Data visualization and chart generation

4. **Agent Configuration** (`src/excel_agent/mcp/agent_configs.py`)
   - Configuration and initialization functions for the MCP system

### Enhanced Agents

All agents in the system have been enhanced with MCP capabilities:

#### FileIngestAgent
- **Capabilities**: `excel_tools`, `file_management`
- **Enhancements**: 
  - Advanced Excel file analysis using MCP tools
  - Automatic backup creation
  - Enhanced metadata extraction

#### ColumnProfilingAgent
- **Capabilities**: `data_analysis`, `excel_tools`
- **Enhancements**:
  - Statistical analysis using MCP data analysis tools
  - Outlier detection
  - Enhanced column profiling with advanced metrics

#### Other Agents
Each agent has been configured with relevant MCP capabilities:
- **StructureScanAgent**: `excel_tools`
- **CodeGenerationAgent**: `excel_tools`, `data_analysis`, `visualization`
- **ExecutionAgent**: `excel_tools`, `file_management`, `visualization`
- **And more...**

## MCP Capabilities

### Excel Tools Capability

Provides tools for Excel file operations:

```python
# Tools available:
- read_excel_file(file_path, sheet_name=None)
- write_excel_file(data, file_path, sheet_name="Sheet1")
- get_sheet_names(file_path)

# Resources available:
- excel://workbook_structure

# Prompts available:
- excel_analysis_prompt(sheet_name, columns)
```

### Data Analysis Capability

Provides advanced data analysis tools:

```python
# Tools available:
- analyze_dataset(file_path, analysis_type, sheet_name=None)
- detect_outliers(data, method, columns=None)
- calculate_correlation(data, method="pearson")

# Resources available:
- analysis://latest_results
```

### File Management Capability

Provides file system operations:

```python
# Tools available:
- list_files(directory, pattern=None)
- create_backup(file_path, backup_dir=None)

# Resources available:
- files://recent_files
```

### Visualization Capability

Provides data visualization tools:

```python
# Tools available:
- create_chart(data, chart_type, x_column=None, y_column=None, title=None)
- create_correlation_heatmap(data, title=None)
```

## Usage Examples

### Basic MCP Tool Usage in an Agent

```python
class EnhancedAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="EnhancedAgent",
            description="Agent with MCP capabilities",
            mcp_capabilities=["excel_tools", "data_analysis"]
        )
    
    async def process_data(self, file_path: str):
        # Use MCP tool to read Excel file
        excel_data = await self.call_mcp_tool("read_excel_file", {
            "file_path": file_path
        })
        
        if excel_data and not excel_data.get("error"):
            # Use MCP tool for analysis
            analysis = await self.call_mcp_tool("analyze_dataset", {
                "file_path": file_path,
                "analysis_type": "statistical"
            })
            return analysis
        
        return None
```

### MCP Resource Access

```python
# Read a resource
workbook_info = await agent.read_mcp_resource("excel://workbook_structure")

# Get a prompt
analysis_prompt = await agent.get_mcp_prompt("excel_analysis_prompt", {
    "sheet_name": "Sales Data",
    "columns": ["Revenue", "Profit", "Date"]
})
```

### Direct MCP Registry Usage

```python
from excel_agent.mcp.registry import mcp_registry

# Get agent's MCP client
client = mcp_registry.get_agent_client("FileIngestAgent")

# Call tool through registry
result = await mcp_registry.call_agent_tool(
    "FileIngestAgent", 
    "read_excel_file", 
    {"file_path": "/path/to/file.xlsx"}
)

# List available tools for an agent
tools = mcp_registry.list_available_tools("ColumnProfilingAgent")
```

## Configuration

### Agent MCP Configuration

Agents are configured with MCP capabilities in `agent_configs.py`:

```python
AgentMCPConfig(
    agent_name="FileIngestAgent",
    capabilities=["excel_tools", "file_management"],
    auto_initialize=True,
    timeout_seconds=30
)
```

### System Initialization

The MCP system is initialized automatically when the orchestrator starts:

```python
from excel_agent.mcp.agent_configs import initialize_mcp_system, initialize_all_agent_mcp

# Initialize MCP system
registry = initialize_mcp_system()
await initialize_all_agent_mcp()
```

## Testing

### Running MCP Tests

```bash
# Run standalone MCP tests (no dependencies)
python test_mcp_standalone.py

# Run simple MCP tests (requires pandas)
python test_mcp_simple.py

# Run full integration tests (requires all dependencies)
python test_mcp_integration.py
```

### Test Results

The standalone test demonstrates:
- ✅ Basic MCP capability functionality
- ✅ Server-client communication  
- ✅ Multiple capabilities integration
- ✅ Error handling and validation

## Benefits of MCP Integration

1. **Modularity**: Each capability is self-contained and reusable
2. **Standardization**: Consistent interface across all tools and resources
3. **Extensibility**: Easy to add new capabilities and tools
4. **Inter-operability**: Agents can access capabilities from other agents
5. **Error Handling**: Robust error handling and validation
6. **Resource Management**: Structured access to data and resources
7. **Tool Discovery**: Agents can discover available tools dynamically

## Adding New Capabilities

To add a new MCP capability:

1. **Create Capability Class**:
```python
class NewCapability(MCPCapability):
    async def initialize(self):
        self.register_tool(
            MCPTool(
                name="new_tool",
                description="Description of new tool",
                input_schema={...}
            ),
            self._new_tool_handler
        )
    
    async def _new_tool_handler(self, **kwargs):
        # Implementation here
        pass
```

2. **Register in System**:
```python
# In agent_configs.py
mcp_registry.register_capability(NewCapability())
```

3. **Configure Agents**:
```python
AgentMCPConfig(
    agent_name="SomeAgent",
    capabilities=["new_capability"],
    auto_initialize=True
)
```

## Troubleshooting

### Common Issues

1. **Capability Not Found**: Ensure capability is registered in `agent_configs.py`
2. **Tool Not Available**: Check that agent is configured with the correct capabilities
3. **Initialization Failures**: Check logs for dependency issues or configuration errors
4. **Timeout Issues**: Increase timeout_seconds in agent configuration

### Debugging

```python
# Check MCP status
status = await orchestrator.get_mcp_status()
print(f"MCP Status: {status}")

# List available tools for an agent
tools = await agent.list_mcp_tools()
print(f"Available tools: {tools}")

# Check registry status
registry_status = mcp_registry.get_registry_status()
print(f"Registry: {registry_status}")
```

## Future Enhancements

1. **Remote MCP Servers**: Support for connecting to remote MCP servers
2. **Tool Versioning**: Version management for tools and capabilities
3. **Authentication**: Security and authentication for MCP communications
4. **Caching**: Intelligent caching of tool results and resources
5. **Monitoring**: Enhanced logging and monitoring of MCP operations
6. **Performance**: Optimization for high-throughput scenarios

## Conclusion

The MCP integration significantly enhances the Excel Intelligent Agent System by providing a standardized, extensible framework for agent capabilities. This enables more sophisticated analysis, better inter-agent communication, and easier system extension while maintaining clean separation of concerns.

The implementation follows MCP best practices and provides a solid foundation for future enhancements and extensions to the system.