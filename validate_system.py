"""
System Validation Script

This script performs comprehensive validation of the Excel Intelligent Agent System
to ensure all components are working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from excel_agent.utils.logging import get_logger
from excel_agent.utils.config import config

logger = get_logger(__name__)


def check_project_structure():
    """Check if all required files and directories exist."""
    print("🔍 Checking project structure...")
    
    required_paths = [
        "src/excel_agent/__init__.py",
        "src/excel_agent/agents/__init__.py", 
        "src/excel_agent/core/__init__.py",
        "src/excel_agent/models/__init__.py",
        "src/excel_agent/utils/__init__.py",
        "src/excel_agent/agents/file_ingest.py",
        "src/excel_agent/agents/structure_scan.py", 
        "src/excel_agent/agents/column_profiling.py",
        "src/excel_agent/agents/code_generation.py",
        "src/excel_agent/agents/execution.py",
        "src/excel_agent/core/orchestrator.py",
        "data/synthetic/single_table_sales.xlsx",
        "data/synthetic/multi_table_business.xlsx",
        "data/synthetic/complex_structure.xlsx",
        "requirements.txt",
        "pyproject.toml",
        ".env.example"
    ]
    
    missing_files = []
    for path in required_paths:
        if not Path(path).exists():
            missing_files.append(path)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ All required files present")
        return True


def check_imports():
    """Check if all modules can be imported successfully."""
    print("\n🔍 Checking imports...")
    
    try:
        # Core imports
        from excel_agent import Orchestrator
        from excel_agent.agents import FileIngestAgent, StructureScanAgent, ColumnProfilingAgent
        from excel_agent.agents import CodeGenerationAgent, ExecutionAgent
        from excel_agent.models.base import AgentRequest, AgentResponse
        from excel_agent.utils.config import Config
        from excel_agent.utils.siliconflow_client import SiliconFlowClient
        
        print("✅ Core modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'openpyxl', 'pydantic', 'httpx', 
        'loguru', 'python-dotenv', 'pytest'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages available")
        return True


def check_configuration():
    """Check system configuration."""
    print("\n🔍 Checking configuration...")
    
    issues = []
    
    # Check temp directory
    if not Path(config.temp_dir).exists():
        try:
            Path(config.temp_dir).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created temp directory: {config.temp_dir}")
        except Exception as e:
            issues.append(f"Cannot create temp directory: {e}")
    
    # Check cache directory
    if not Path(config.cache_dir).exists():
        try:
            Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created cache directory: {config.cache_dir}")
        except Exception as e:
            issues.append(f"Cannot create cache directory: {e}")
    
    # Check API key (warning only)
    if not config.siliconflow_api_key:
        print("⚠️  SiliconFlow API key not configured (some features will be limited)")
    else:
        print("✅ SiliconFlow API key configured")
    
    if issues:
        print("❌ Configuration issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Configuration looks good")
        return True


async def check_agent_initialization():
    """Check if agents can be initialized properly."""
    print("\n🔍 Checking agent initialization...")
    
    try:
        from excel_agent.agents import (
            FileIngestAgent, StructureScanAgent, ColumnProfilingAgent,
            CodeGenerationAgent, ExecutionAgent
        )
        from excel_agent.core.orchestrator import Orchestrator
        
        # Try to initialize each agent
        agents = [
            ("FileIngestAgent", FileIngestAgent),
            ("StructureScanAgent", StructureScanAgent), 
            ("ColumnProfilingAgent", ColumnProfilingAgent),
            ("CodeGenerationAgent", CodeGenerationAgent),
            ("ExecutionAgent", ExecutionAgent),
            ("Orchestrator", Orchestrator)
        ]
        
        for name, agent_class in agents:
            try:
                agent = agent_class()
                print(f"✅ {name} initialized successfully")
            except Exception as e:
                print(f"❌ {name} initialization failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Agent initialization check failed: {e}")
        return False


def check_test_data():
    """Check if test data files are valid."""
    print("\n🔍 Checking test data...")
    
    test_files = [
        "data/synthetic/single_table_sales.xlsx",
        "data/synthetic/multi_table_business.xlsx", 
        "data/synthetic/complex_structure.xlsx"
    ]
    
    try:
        import pandas as pd
        import openpyxl
        
        for file_path in test_files:
            path = Path(file_path)
            if not path.exists():
                print(f"❌ Test file missing: {file_path}")
                return False
            
            # Try to read with pandas
            try:
                excel_file = pd.ExcelFile(path)
                print(f"✅ {path.name}: {len(excel_file.sheet_names)} sheets")
            except Exception as e:
                print(f"❌ Cannot read {path.name}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test data validation failed: {e}")
        return False


async def run_basic_functionality_test():
    """Run basic functionality test."""
    print("\n🔍 Running basic functionality test...")
    
    try:
        from excel_agent.agents.file_ingest import FileIngestAgent
        from excel_agent.models.agents import FileIngestRequest
        
        # Test file ingestion
        test_file = Path("data/synthetic/single_table_sales.xlsx")
        if not test_file.exists():
            print("❌ Test file not found for functionality test")
            return False
        
        agent = FileIngestAgent()
        request = FileIngestRequest(
            agent_id="FileIngestAgent",
            file_path=str(test_file)
        )
        
        async with agent:
            response = await agent.execute_with_timeout(request)
        
        if response.status.value == 'success':
            print("✅ Basic file ingestion test passed")
            print(f"   - File ID: {response.file_id}")
            print(f"   - Sheets: {response.sheets}")
            return True
        else:
            print(f"❌ Basic file ingestion test failed: {response.error_log}")
            return False
            
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def print_system_summary():
    """Print system summary."""
    print("\n" + "="*60)
    print("📊 SYSTEM SUMMARY")
    print("="*60)
    
    print(f"🐍 Python Version: {sys.version}")
    print(f"📁 Project Directory: {Path.cwd()}")
    print(f"⚙️  Configuration:")
    print(f"   - Log Level: {config.log_level}")
    print(f"   - Max File Size: {config.max_file_size_mb} MB") 
    print(f"   - Temp Directory: {config.temp_dir}")
    print(f"   - Cache Directory: {config.cache_dir}")
    print(f"   - Max Agents: {config.max_agents}")
    print(f"   - Agent Timeout: {config.agent_timeout_seconds}s")
    
    print(f"\n🤖 AI Models:")
    print(f"   - LLM: {config.llm_model}")
    print(f"   - Embedding: {config.embedding_model}")
    print(f"   - Multimodal: {config.multimodal_model}")
    print(f"   - Text-to-Image: {config.text_to_image_model}")


async def main():
    """Main validation function."""
    print("🚀 Excel Intelligent Agent System - Validation")
    print("="*60)
    
    validation_steps = [
        ("Project Structure", check_project_structure),
        ("Dependencies", check_dependencies),
        ("Imports", check_imports),
        ("Configuration", check_configuration),
        ("Agent Initialization", check_agent_initialization),
        ("Test Data", check_test_data),
        ("Basic Functionality", run_basic_functionality_test)
    ]
    
    results = []
    
    for step_name, step_func in validation_steps:
        try:
            if asyncio.iscoroutinefunction(step_func):
                result = await step_func()
            else:
                result = step_func()
            results.append((step_name, result))
        except Exception as e:
            print(f"❌ {step_name} validation failed with exception: {e}")
            results.append((step_name, False))
    
    # Print results summary
    print("\n" + "="*60)
    print("📋 VALIDATION RESULTS")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for step_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {step_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 System validation SUCCESSFUL! 🎉")
        print("The Excel Intelligent Agent System is ready to use.")
        print("\nNext steps:")
        print("1. Configure your SiliconFlow API key in .env")
        print("2. Run: python example_usage.py")
        print("3. Check the README.md for detailed usage instructions")
    else:
        print(f"\n⚠️  System validation INCOMPLETE ({total-passed} issues found)")
        print("Please resolve the issues above before using the system.")
    
    print_system_summary()
    print("\n" + "="*60)


if __name__ == "__main__":
    asyncio.run(main())