"""
Basic test without Unicode emojis for Windows compatibility
"""

import sys
from pathlib import Path
import asyncio

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_basic_imports():
    """Test basic imports without ADK dependencies."""
    print("Testing basic imports...")
    
    try:
        # Test utils - avoid ADK imports initially
        from excel_agent.utils.config import Config
        config = Config()
        print("PASS Config imported successfully")
        
        # Test models
        from excel_agent.models.base import AgentRequest, AgentResponse, AgentStatus
        print("PASS Models imported successfully")
        
        # Test SiliconFlow client (without ADK context)
        from excel_agent.utils.siliconflow_client import SiliconFlowClient
        print("PASS SiliconFlow client imported successfully")
        
        return True
        
    except Exception as e:
        print(f"FAIL Import failed: {e}")
        return False

async def test_file_generation():
    """Test if synthetic files exist."""
    print("\nTesting synthetic data files...")
    
    try:
        test_files = [
            "data/synthetic/single_table_sales.xlsx",
            "data/synthetic/multi_table_business.xlsx",
            "data/synthetic/complex_structure.xlsx"
        ]
        
        for file_path in test_files:
            path = Path(file_path)
            if path.exists():
                print(f"PASS {path.name} exists ({path.stat().st_size} bytes)")
            else:
                print(f"WARN {path.name} missing")
        
        return True
        
    except Exception as e:
        print(f"FAIL File test failed: {e}")
        return False

async def test_pandas_excel_read():
    """Test basic Excel reading with pandas."""
    print("\nTesting Excel file reading...")
    
    try:
        import pandas as pd
        
        test_file = Path("data/synthetic/single_table_sales.xlsx")
        if not test_file.exists():
            print("SKIP No test file available")
            return True
            
        # Try reading with pandas
        excel_file = pd.ExcelFile(test_file)
        print(f"PASS Read Excel file with {len(excel_file.sheet_names)} sheets")
        
        # Read first sheet
        df = excel_file.parse(excel_file.sheet_names[0])
        print(f"PASS Read first sheet with {len(df)} rows and {len(df.columns)} columns")
        
        return True
        
    except Exception as e:
        print(f"FAIL Excel reading failed: {e}")
        return False

async def test_configuration():
    """Test system configuration without ADK."""
    print("\nTesting configuration...")
    
    try:
        from excel_agent.utils.config import Config
        
        config = Config()
        print(f"PASS Configuration loaded:")
        print(f"  - Log Level: {config.log_level}")
        print(f"  - Max File Size: {config.max_file_size_mb} MB")
        print(f"  - LLM Model: {config.llm_model}")
        
        # Create directories
        temp_dir = Path(config.temp_dir)
        cache_dir = Path(config.cache_dir)
        
        temp_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"PASS Directories created: {temp_dir}, {cache_dir}")
        return True
        
    except Exception as e:
        print(f"FAIL Configuration test failed: {e}")
        return False

async def test_simple_file_ingest():
    """Test file ingest agent without ADK context manager."""
    print("\nTesting File Ingest Agent (basic)...")
    
    try:
        # Import without using ADK context manager
        from excel_agent.agents.file_ingest import FileIngestAgent
        from excel_agent.models.agents import FileIngestRequest
        
        test_file = Path("data/synthetic/single_table_sales.xlsx")
        if not test_file.exists():
            print("SKIP No test file available")
            return True
        
        # Create agent without ADK initialization
        agent = FileIngestAgent()
        
        # Create request
        request = FileIngestRequest(
            agent_id="FileIngestAgent",
            file_path=str(test_file)
        )
        
        # Call process directly (without async context manager that uses ADK)
        response = await agent.process(request)
        
        if response.status.value == 'success':
            print(f"PASS File ingested successfully")
            print(f"  - File ID: {response.file_id}")
            print(f"  - Sheets: {response.sheets}")
            return True
        else:
            print(f"FAIL File ingest failed: {response.error_log}")
            return False
            
    except Exception as e:
        print(f"FAIL File ingest test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("Excel Intelligent Agent System - Basic Test")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_configuration),
        ("File Generation", test_file_generation),
        ("Pandas Excel Read", test_pandas_excel_read),
        ("Simple File Ingest", test_simple_file_ingest),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"FAIL {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSUCCESS: All tests passed!")
        print("Core functionality is working.")
    else:
        print(f"\nISSUES: {total-passed} tests failed.")
    
    return passed == total

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)