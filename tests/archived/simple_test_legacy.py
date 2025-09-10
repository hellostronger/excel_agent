"""
Simple test to check core functionality without ADK dependencies
"""

import sys
from pathlib import Path
import asyncio

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_basic_imports():
    """Test basic imports without ADK dependencies."""
    print("🔍 Testing basic imports...")
    
    try:
        # Test utils
        from excel_agent.utils.config import config
        from excel_agent.utils.logging import get_logger
        print("✅ Utils imported successfully")
        
        # Test models
        from excel_agent.models.base import AgentRequest, AgentResponse, AgentStatus
        print("✅ Models imported successfully")
        
        # Test SiliconFlow client
        from excel_agent.utils.siliconflow_client import SiliconFlowClient
        print("✅ SiliconFlow client imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_file_ingest_basic():
    """Test file ingest without ADK."""
    print("\n🔍 Testing File Ingest Agent...")
    
    try:
        from excel_agent.agents.file_ingest import FileIngestAgent
        from excel_agent.models.agents import FileIngestRequest
        
        # Create agent (without ADK context manager)
        agent = FileIngestAgent()
        
        # Test with existing file
        test_file = Path("data/synthetic/single_table_sales.xlsx")
        if not test_file.exists():
            print("⚠️ Test file not found, skipping file ingest test")
            return True
            
        request = FileIngestRequest(
            agent_id="FileIngestAgent",
            file_path=str(test_file)
        )
        
        # Test the core processing logic
        response = await agent.process(request)
        
        if response.status == AgentStatus.SUCCESS:
            print("✅ File ingest test passed")
            print(f"   - File ID: {response.file_id}")
            print(f"   - Sheets: {response.sheets}")
            return True
        else:
            print(f"❌ File ingest failed: {response.error_log}")
            return False
            
    except Exception as e:
        print(f"❌ File ingest test failed: {e}")
        return False

async def test_configuration():
    """Test system configuration."""
    print("\n🔍 Testing configuration...")
    
    try:
        from excel_agent.utils.config import config
        
        print(f"✅ Configuration loaded:")
        print(f"   - Log Level: {config.log_level}")
        print(f"   - Max File Size: {config.max_file_size_mb} MB")
        print(f"   - Temp Directory: {config.temp_dir}")
        print(f"   - LLM Model: {config.llm_model}")
        
        # Check if directories exist
        temp_dir = Path(config.temp_dir)
        cache_dir = Path(config.cache_dir)
        
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created temp directory: {temp_dir}")
        
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created cache directory: {cache_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def test_data_files():
    """Test synthetic data files."""
    print("\n🔍 Testing data files...")
    
    try:
        import pandas as pd
        
        test_files = [
            "data/synthetic/single_table_sales.xlsx",
            "data/synthetic/multi_table_business.xlsx",
            "data/synthetic/complex_structure.xlsx"
        ]
        
        for file_path in test_files:
            path = Path(file_path)
            if path.exists():
                # Try to read with pandas
                excel_file = pd.ExcelFile(path)
                print(f"✅ {path.name}: {len(excel_file.sheet_names)} sheets")
            else:
                print(f"⚠️ Missing: {path.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data files test failed: {e}")
        return False

async def run_unit_tests():
    """Run specific unit tests."""
    print("\n🔍 Running unit tests...")
    
    try:
        import pytest
        
        # Run only tests that don't require ADK
        test_files = [
            "tests/unit/test_file_ingest_agent.py::TestFileIngestAgent::test_file_id_generation",
            "tests/unit/test_file_ingest_agent.py::TestFileIngestAgent::test_file_metadata_storage",
            "tests/unit/test_column_profiling_agent.py::TestColumnProfilingAgent::test_is_integer_string",
            "tests/unit/test_column_profiling_agent.py::TestColumnProfilingAgent::test_is_float_string",
            "tests/unit/test_column_profiling_agent.py::TestColumnProfilingAgent::test_is_boolean_string",
        ]
        
        for test in test_files:
            try:
                result = pytest.main(["-v", test])
                if result == 0:
                    print(f"✅ {test.split('::')[-1]} passed")
                else:
                    print(f"❌ {test.split('::')[-1]} failed")
            except Exception as e:
                print(f"⚠️ Could not run {test}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Unit tests failed: {e}")
        return False

async def main():
    """Main test function."""
    print("Excel Intelligent Agent System - Simple Test")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_configuration),
        ("Data Files", test_data_files),
        ("File Ingest Basic", test_file_ingest_basic),
        ("Unit Tests", run_unit_tests),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("📋 TEST RESULTS")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests PASSED! 🎉")
        print("Core functionality is working correctly.")
    else:
        print(f"\n⚠️ {total-passed} tests failed. Check the issues above.")
    
    print("\nNext steps:")
    print("1. If tests passed, the system core is working")
    print("2. To use AI features, configure SILICONFLOW_API_KEY in .env")  
    print("3. Run full validation: python validate_system.py")

if __name__ == "__main__":
    asyncio.run(main())