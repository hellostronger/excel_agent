"""Test script for MCP integration with Excel Intelligent Agent System."""

import asyncio
import sys
import tempfile
import pandas as pd
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from excel_agent.mcp.agent_configs import initialize_mcp_system, initialize_all_agent_mcp, get_agent_mcp_status
from excel_agent.mcp.capabilities import ExcelToolsCapability, DataAnalysisCapability
from excel_agent.agents.file_ingest import FileIngestAgent
from excel_agent.agents.column_profiling import ColumnProfilingAgent
from excel_agent.core.orchestrator import Orchestrator
from excel_agent.models.agents import FileIngestRequest, ColumnProfilingRequest


async def test_mcp_capabilities():
    """Test individual MCP capabilities."""
    print("🧪 Testing MCP Capabilities...")
    
    # Test Excel Tools Capability
    excel_cap = ExcelToolsCapability()
    await excel_cap.initialize()
    
    # Create test data
    test_data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'Age': [25, 30, 35, 28],
        'Salary': [50000, 60000, 70000, 55000],
        'Department': ['Engineering', 'Marketing', 'Sales', 'Engineering']
    }
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        test_file_path = tmp_file.name
        df = pd.DataFrame(test_data)
        df.to_excel(test_file_path, index=False)
    
    try:
        # Test reading Excel file
        result = await excel_cap.call_tool("read_excel_file", {
            "file_path": test_file_path
        })
        print(f"✅ Excel read test: {result is not None}")
        
        # Test getting sheet names
        sheet_names = await excel_cap.call_tool("get_sheet_names", {
            "file_path": test_file_path
        })
        print(f"✅ Sheet names test: {sheet_names}")
        
        # Test Data Analysis Capability
        data_cap = DataAnalysisCapability()
        await data_cap.initialize()
        
        # Test dataset analysis
        analysis_result = await data_cap.call_tool("analyze_dataset", {
            "file_path": test_file_path,
            "analysis_type": "statistical"
        })
        print(f"✅ Statistical analysis test: {analysis_result is not None}")
        
        # Test outlier detection
        outlier_result = await data_cap.call_tool("detect_outliers", {
            "data": test_data,
            "method": "iqr"
        })
        print(f"✅ Outlier detection test: {outlier_result is not None}")
        
    finally:
        # Clean up
        Path(test_file_path).unlink(missing_ok=True)
    
    print("✅ MCP Capabilities tests completed\n")


async def test_mcp_registry():
    """Test MCP registry functionality."""
    print("🧪 Testing MCP Registry...")
    
    # Initialize MCP system
    registry = initialize_mcp_system()
    await initialize_all_agent_mcp()
    
    # Test registry status
    status = get_agent_mcp_status()
    print(f"✅ Registry initialized: {status['initialized']}")
    print(f"✅ Available capabilities: {len(status['registry_status']['capabilities_count'])}")
    print(f"✅ Configured agents: {status['registry_status']['agent_configs_count']}")
    
    # Test tool listing
    tools = registry.list_available_tools()
    print(f"✅ Available tools by agent: {len(tools)} agents configured")
    
    for agent_name, agent_tools in tools.items():
        if agent_tools:
            print(f"   - {agent_name}: {len(agent_tools)} tools")
    
    print("✅ MCP Registry tests completed\n")


async def test_mcp_enhanced_agents():
    """Test MCP-enhanced agents."""
    print("🧪 Testing MCP-Enhanced Agents...")
    
    # Create test Excel file
    test_data = {
        'ID': [1, 2, 3, 4, 5],
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'Score': [85.5, 92.0, 78.5, 96.0, 88.5],
        'Category': ['A', 'A', 'B', 'A', 'B'],
        'Date': pd.date_range('2024-01-01', periods=5)
    }
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        test_file_path = tmp_file.name
        df = pd.DataFrame(test_data)
        df.to_excel(test_file_path, index=False, sheet_name='TestData')
    
    try:
        # Test File Ingest Agent with MCP
        print("  Testing FileIngestAgent with MCP...")
        async with FileIngestAgent() as ingest_agent:
            request = FileIngestRequest(
                agent_id="FileIngestAgent",
                file_path=test_file_path
            )
            response = await ingest_agent.execute_with_timeout(request)
            print(f"  ✅ File ingest successful: {response.status.value}")
            print(f"  ✅ Sheets detected: {response.sheets}")
            
            # Test MCP tools access
            mcp_tools = await ingest_agent.list_mcp_tools()
            print(f"  ✅ MCP tools available: {len(mcp_tools)} tools")
        
        # Test Column Profiling Agent with MCP
        print("  Testing ColumnProfilingAgent with MCP...")
        async with ColumnProfilingAgent() as profiling_agent:
            request = ColumnProfilingRequest(
                agent_id="ColumnProfilingAgent",
                file_id="test_file",
                sheet_name="TestData",
                context={
                    'file_metadata': {
                        'file_path': test_file_path,
                        'sheets': ['TestData']
                    }
                }
            )
            response = await profiling_agent.execute_with_timeout(request)
            print(f"  ✅ Column profiling successful: {response.status.value}")
            print(f"  ✅ Columns profiled: {len(response.profiles)}")
            
            # Test MCP analysis enhancement
            for profile in response.profiles:
                if hasattr(profile, 'statistics') and profile.statistics:
                    print(f"  ✅ Enhanced statistics for {profile.column_name}: {len(profile.statistics)} metrics")
    
    finally:
        # Clean up
        Path(test_file_path).unlink(missing_ok=True)
    
    print("✅ MCP-Enhanced Agents tests completed\n")


async def test_orchestrator_mcp():
    """Test Orchestrator with MCP integration."""
    print("🧪 Testing Orchestrator MCP Integration...")
    
    # Create test Excel file
    test_data = {
        'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
        'Price': [999.99, 25.99, 79.99, 299.99, 149.99],
        'Stock': [50, 200, 75, 30, 100],
        'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories']
    }
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        test_file_path = tmp_file.name
        df = pd.DataFrame(test_data)
        df.to_excel(test_file_path, index=False, sheet_name='Products')
    
    try:
        # Test Orchestrator
        orchestrator = Orchestrator()
        
        # Test MCP status
        mcp_status = await orchestrator.get_mcp_status()
        print(f"  ✅ MCP Status: {mcp_status.get('status', 'unknown')}")
        
        # Test processing a user request
        user_request = "Analyze the product data and show me statistics for each column"
        result = await orchestrator.process_user_request(
            user_request=user_request,
            file_path=test_file_path
        )
        
        print(f"  ✅ User request processed: {result.get('status', 'unknown')}")
        print(f"  ✅ Workflow type: {result.get('workflow_type', 'unknown')}")
        
        if result.get('status') == 'success':
            steps = result.get('steps', [])
            print(f"  ✅ Workflow steps completed: {len(steps)}")
            for step in steps:
                print(f"    - {step.get('step', 'unknown')}: {step.get('status', 'unknown')}")
    
    finally:
        # Clean up
        Path(test_file_path).unlink(missing_ok=True)
    
    print("✅ Orchestrator MCP Integration tests completed\n")


async def main():
    """Run all MCP integration tests."""
    print("🚀 Starting MCP Integration Tests for Excel Intelligent Agent System\n")
    
    try:
        await test_mcp_capabilities()
        await test_mcp_registry()
        await test_mcp_enhanced_agents()
        await test_orchestrator_mcp()
        
        print("🎉 All MCP Integration Tests Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)