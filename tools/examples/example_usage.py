"""
Excel Intelligent Agent System - Example Usage

This script demonstrates how to use the Excel Intelligent Agent System
for various types of Excel data processing tasks.
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to Python path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from excel_agent.core.orchestrator import Orchestrator
from excel_agent.utils.logging import get_logger

logger = get_logger(__name__)


async def demo_single_table_analysis():
    """Demonstrate single table analysis workflow."""
    print("\n" + "="*60)
    print("DEMO 1: Single Table Analysis")
    print("="*60)
    
    orchestrator = Orchestrator()
    
    # Test file path
    test_file = Path(__file__).parent / "data" / "synthetic" / "single_table_sales.xlsx"
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        print("Please run 'python data/synthetic/generate_test_data.py' first")
        return
    
    # Example queries
    queries = [
        "Show me the total sales by region",
        "What are the top 5 products by sales amount?",
        "Calculate the monthly sales trends",
        "Find the average unit price by category"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        try:
            result = await orchestrator.process_user_request(
                user_request=query,
                file_path=str(test_file),
                context={"demo": "single_table"}
            )
            
            print(f"Status: {result['status']}")
            if result['status'] == 'success':
                print(f"Workflow: {result['workflow_type']}")
                print(f"Steps completed: {len(result.get('steps', []))}")
                if 'generated_code' in result:
                    print("Generated Code Preview:")
                    code_lines = result['generated_code'].split('\n')[:5]
                    for line in code_lines:
                        print(f"  {line}")
                    if len(result['generated_code'].split('\n')) > 5:
                        print("  ...")
                print(f"Output: {result.get('output', 'No output')[:200]}...")
            else:
                print(f"Error: {result.get('error_message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            print(f"❌ Error: {e}")
        
        print()


async def demo_multi_table_analysis():
    """Demonstrate multi-table analysis workflow."""
    print("\n" + "="*60)
    print("DEMO 2: Multi-Table Analysis")
    print("="*60)
    
    orchestrator = Orchestrator()
    
    # Test file path
    test_file = Path(__file__).parent / "data" / "synthetic" / "multi_table_business.xlsx"
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    
    # Example multi-table queries
    queries = [
        "Join sales data with customer information and show sales by industry",
        "Find customers who haven't made any purchases",
        "Calculate total sales by customer region and company size",
        "Show inventory levels for products with high sales"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        try:
            result = await orchestrator.process_user_request(
                user_request=query,
                file_path=str(test_file),
                context={"demo": "multi_table"}
            )
            
            print(f"Status: {result['status']}")
            if result['status'] == 'success':
                print(f"Workflow: {result['workflow_type']}")
                print(f"Steps completed: {len(result.get('steps', []))}")
            else:
                print(f"Error: {result.get('error_message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            print(f"❌ Error: {e}")


async def demo_file_structure_analysis():
    """Demonstrate file structure analysis."""
    print("\n" + "="*60)
    print("DEMO 3: File Structure Analysis")
    print("="*60)
    
    from excel_agent.agents.file_ingest import FileIngestAgent
    from excel_agent.agents.structure_scan import StructureScanAgent
    from excel_agent.models.agents import FileIngestRequest, StructureScanRequest
    
    # Test complex structure file
    test_file = Path(__file__).parent / "data" / "synthetic" / "complex_structure.xlsx"
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    
    print(f"📁 Analyzing file: {test_file.name}")
    
    # File ingest
    file_agent = FileIngestAgent()
    ingest_request = FileIngestRequest(
        agent_id="FileIngestAgent",
        file_path=str(test_file)
    )
    
    async with file_agent:
        ingest_response = await file_agent.execute_with_timeout(ingest_request)
    
    if ingest_response.status.value == 'success':
        print("✅ File ingested successfully")
        print(f"Sheets found: {ingest_response.sheets}")
        print(f"File size: {ingest_response.result['metadata']['file_size']} bytes")
        print(f"Has merged cells: {ingest_response.result['metadata']['has_merged_cells']}")
        
        # Structure scan
        if ingest_response.sheets:
            structure_agent = StructureScanAgent()
            scan_request = StructureScanRequest(
                agent_id="StructureScanAgent",
                file_id=ingest_response.file_id,
                sheet_name=ingest_response.sheets[0],
                context={"file_metadata": ingest_response.result}
            )
            
            async with structure_agent:
                scan_response = await structure_agent.execute_with_timeout(scan_request)
            
            if scan_response.status.value == 'success':
                print(f"✅ Structure scan completed")
                print(f"Merged cells: {len(scan_response.merged_cells)}")
                print(f"Charts: {len(scan_response.charts)}")
                print(f"Images: {len(scan_response.images)}")
                print(f"Formulas: {len(scan_response.formulas)}")
            else:
                print(f"❌ Structure scan failed: {scan_response.error_log}")
    else:
        print(f"❌ File ingest failed: {ingest_response.error_log}")


async def demo_system_health():
    """Demonstrate system health check."""
    print("\n" + "="*60)
    print("DEMO 4: System Health Check")
    print("="*60)
    
    from excel_agent.utils.siliconflow_client import SiliconFlowClient
    from excel_agent.utils.config import config
    
    print("🔧 Checking system configuration...")
    print(f"SiliconFlow API Key: {'✅ Set' if config.siliconflow_api_key else '❌ Not set'}")
    print(f"LLM Model: {config.llm_model}")
    print(f"Embedding Model: {config.embedding_model}")
    print(f"Temp Directory: {config.temp_dir}")
    print(f"Max File Size: {config.max_file_size_mb} MB")
    
    print("\n🌐 Testing API connectivity...")
    try:
        async with SiliconFlowClient() as client:
            health_ok = await client.health_check()
            print(f"SiliconFlow API: {'✅ Connected' if health_ok else '❌ Connection failed'}")
    except Exception as e:
        print(f"❌ API connection error: {e}")
    
    print("\n📊 Workflow statistics...")
    orchestrator = Orchestrator()
    stats = await orchestrator.get_workflow_statistics()
    print(f"Workflow history: {stats}")


async def main():
    """Main demo function."""
    print("🚀 Excel Intelligent Agent System - Demo")
    print("=" * 60)
    print("This demo showcases the multi-agent Excel processing system")
    print("based on Google ADK framework with SiliconFlow AI integration.")
    
    # Check if API key is configured
    api_key = os.getenv('SILICONFLOW_API_KEY')
    if not api_key:
        print("\n⚠️  WARNING: SILICONFLOW_API_KEY not found in environment")
        print("Some features may not work correctly. Please check your .env file.")
    
    try:
        # Run demos
        await demo_file_structure_analysis()
        await demo_system_health()
        
        # Only run AI-powered demos if API key is available
        if api_key:
            await demo_single_table_analysis()
            await demo_multi_table_analysis()
        else:
            print("\n⚠️  Skipping AI-powered demos due to missing API key")
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\n❌ Demo failed with error: {e}")
    
    print("\n" + "="*60)
    print("✅ Demo completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())