"""Simple MCP test without full system dependencies."""

import asyncio
import sys
import tempfile
import pandas as pd
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_mcp_base_functionality():
    """Test basic MCP functionality."""
    print("Testing MCP Base Functionality...")
    
    try:
        # Test MCP base classes
        from excel_agent.mcp.base import MCPServer, MCPClient, MCPCapability, MCPTool
        
        # Create a simple test capability
        class TestCapability(MCPCapability):
            async def initialize(self):
                # Register a simple tool
                self.register_tool(
                    MCPTool(
                        name="test_tool",
                        description="A test tool",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "message": {"type": "string"}
                            }
                        }
                    ),
                    self._test_handler
                )
            
            async def _test_handler(self, message: str = "Hello MCP!"):
                return f"Test tool received: {message}"
        
        # Test capability
        test_cap = TestCapability("test_capability", "Test capability")
        await test_cap.initialize()
        
        tools = test_cap.get_tools()
        print(f"[OK] Test capability created with {len(tools)} tools")
        
        # Test tool execution
        result = await test_cap.call_tool("test_tool", {"message": "MCP is working!"})
        print(f"[OK] Tool execution result: {result}")
        
        # Test MCP Server
        server = MCPServer("test_server")
        server.register_capability(test_cap)
        
        init_result = await server.initialize()
        print(f"[OK] MCP Server initialized: {init_result.get('protocolVersion', 'unknown')}")
        
        # Test MCP Client
        client = MCPClient("test_client")
        client.connect_to_server(server)
        
        client_init = await client.initialize()
        print(f"[OK] MCP Client initialized: {client_init.get('protocolVersion', 'unknown')}")
        
        # List tools through client
        tools_list = await client.list_tools()
        print(f"[OK] Tools available through client: {len(tools_list)}")
        
        # Call tool through client
        tool_result = await client.call_tool("test_tool", {"message": "Via client!"})
        print(f"[OK] Tool call through client: {tool_result}")
        
        print("[OK] MCP Base Functionality tests completed\n")
        return True
        
    except Exception as e:
        print(f"[FAIL] MCP Base Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mcp_capabilities():
    """Test MCP capabilities without full system."""
    print("Testing MCP Capabilities...")
    
    try:
        from excel_agent.mcp.capabilities import ExcelToolsCapability, DataAnalysisCapability
        
        # Test Excel Tools
        excel_cap = ExcelToolsCapability()
        await excel_cap.initialize()
        
        excel_tools = excel_cap.get_tools()
        print(f"[OK] Excel Tools Capability: {len(excel_tools)} tools")
        for tool in excel_tools:
            print(f"   - {tool.name}: {tool.description}")
        
        # Test Data Analysis
        data_cap = DataAnalysisCapability()
        await data_cap.initialize()
        
        data_tools = data_cap.get_tools()
        print(f"[OK] Data Analysis Capability: {len(data_tools)} tools")
        for tool in data_tools:
            print(f"   - {tool.name}: {tool.description}")
        
        # Test Excel file operations with sample data
        test_data = {
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'Score': [85.5, 92.0, 78.5]
        }
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            test_file_path = tmp_file.name
            df = pd.DataFrame(test_data)
            df.to_excel(test_file_path, index=False)
        
        try:
            # Test reading Excel file
            read_result = await excel_cap.call_tool("read_excel_file", {
                "file_path": test_file_path
            })
            
            if read_result and not read_result.get("error"):
                print(f"[OK] Excel file read successfully")
                print(f"   - Sheets: {read_result.get('total_sheets', 0)}")
                print(f"   - Shape: {read_result.get('shape', 'unknown')}")
            else:
                print(f"[WARN] Excel file read had issues: {read_result}")
            
            # Test data analysis
            analysis_result = await data_cap.call_tool("analyze_dataset", {
                "file_path": test_file_path,
                "analysis_type": "basic"
            })
            
            if analysis_result and not analysis_result.get("error"):
                print(f"[OK] Data analysis completed")
                print(f"   - Analysis type: {analysis_result.get('analysis_type')}")
                print(f"   - Columns: {len(analysis_result.get('columns', []))}")
            else:
                print(f"[WARN] Data analysis had issues: {analysis_result}")
        
        finally:
            Path(test_file_path).unlink(missing_ok=True)
        
        print("[OK] MCP Capabilities tests completed\n")
        return True
        
    except Exception as e:
        print(f"[FAIL] MCP Capabilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mcp_registry():
    """Test MCP registry functionality."""
    print("Testing MCP Registry...")
    
    try:
        from excel_agent.mcp.registry import MCPRegistry, AgentMCPConfig
        from excel_agent.mcp.capabilities import ExcelToolsCapability, DataAnalysisCapability
        
        # Create registry
        registry = MCPRegistry()
        
        # Register capabilities
        excel_cap = ExcelToolsCapability()
        data_cap = DataAnalysisCapability()
        
        registry.register_capability(excel_cap)
        registry.register_capability(data_cap)
        
        # Register agent config
        agent_config = AgentMCPConfig(
            agent_name="TestAgent",
            capabilities=["excel_tools", "data_analysis"],
            auto_initialize=True
        )
        registry.register_agent_config(agent_config)
        
        # Test registry status
        status = registry.get_registry_status()
        print(f"[OK] Registry status: {status['capabilities_count']} capabilities, {status['agent_configs_count']} agent configs")
        
        # Create server and client for test agent
        server = registry.create_server("test_server", ["excel_tools", "data_analysis"])
        client = registry.create_client("test_client")
        
        print(f"[OK] Created server with {len(server.capabilities)} capabilities")
        
        # Initialize server
        await server.initialize()
        print(f"[OK] Server initialized: {server.initialized}")
        
        print("[OK] MCP Registry tests completed\n")
        return True
        
    except Exception as e:
        print(f"[FAIL] MCP Registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run simple MCP tests."""
    print("Starting Simple MCP Tests\n")
    
    success_count = 0
    total_tests = 3
    
    if await test_mcp_base_functionality():
        success_count += 1
    
    if await test_mcp_capabilities():
        success_count += 1
    
    if await test_mcp_registry():
        success_count += 1
    
    print(f"Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("All MCP Tests Completed Successfully!")
        return 0
    else:
        print("[FAIL] Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)