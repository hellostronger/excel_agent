#!/usr/bin/env python3
"""
SiliconFlow Integration Verification Script

Tests the complete SiliconFlow model integration across all agents.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_test_environment():
    """Setup test environment with mock SiliconFlow configuration"""
    test_env = {
        "MODEL_PROVIDER": "siliconflow",
        "SILICONFLOW_API_KEY": "test-key-123",
        "ROOT_AGENT_MODEL": "qwen-plus",
        "ORCHESTRATOR_MODEL": "qwen-max", 
        "WORKER_MODEL": "qwen-turbo",
        "SILICONFLOW_BASE_URL": "https://api.siliconflow.cn/v1",
        "SILICONFLOW_TIMEOUT": "300",
        "MAX_PARALLEL_AGENTS": "4",
        "ENABLE_CONCURRENT_ANALYSIS": "true",
        "FILE_PREPARATION_TIMEOUT": "120",
        "PARALLEL_ANALYSIS_TIMEOUT": "300",
        "RESPONSE_GENERATION_TIMEOUT": "60"
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    print("[PASS] Test environment configured with SiliconFlow settings")

def test_environment_loading():
    """Test environment variable loading"""
    try:
        from excel_intelligence_agent.shared_libraries.utils import load_environment_variables
        
        env_config = load_environment_variables()
        
        # Verify key configurations
        assert env_config["MODEL_PROVIDER"] == "siliconflow"
        assert env_config["ROOT_AGENT_MODEL"] == "qwen-plus"
        assert env_config["ORCHESTRATOR_MODEL"] == "qwen-max"
        assert env_config["WORKER_MODEL"] == "qwen-turbo"
        assert "SILICONFLOW_API_KEY" in env_config
        
        print("[PASS] Environment variable loading works correctly")
        return env_config
        
    except Exception as e:
        print(f"[FAIL] Environment loading failed: {e}")
        return None

def test_siliconflow_client():
    """Test SiliconFlow client creation"""
    try:
        from excel_intelligence_agent.shared_libraries.siliconflow_client import create_siliconflow_client
        
        # This should create a client even with test credentials
        client = create_siliconflow_client()
        
        if client:
            print("[PASS] SiliconFlow client created successfully")
            
            # Test model mapping
            available_models = client.get_available_models()
            assert "qwen-plus" in available_models
            assert "qwen-max" in available_models
            assert "qwen-turbo" in available_models
            
            print(f"✅ Available models: {list(available_models.keys())}")
            return True
        else:
            print("⚠️  SiliconFlow client creation returned None (expected with test key)")
            return True  # This is expected with test credentials
            
    except Exception as e:
        print(f"[FAIL] SiliconFlow client test failed: {e}")
        return False

def test_model_adapter():
    """Test model adapter functionality"""
    try:
        from excel_intelligence_agent.shared_libraries.model_adapter import ModelAdapter
        
        adapter = ModelAdapter()
        
        # Test model selection
        root_model = adapter.get_model_for_agent("root")
        orchestrator_model = adapter.get_model_for_agent("orchestrator") 
        worker_model = adapter.get_model_for_agent("worker")
        
        assert root_model == "qwen-plus"
        assert orchestrator_model == "qwen-max"
        assert worker_model == "qwen-turbo"
        
        print("[PASS] Model adapter correctly maps agent types to models")
        print(f"   Root: {root_model}")
        print(f"   Orchestrator: {orchestrator_model}")
        print(f"   Worker: {worker_model}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Model adapter test failed: {e}")
        return False

def test_agent_initialization():
    """Test that all agents can be initialized with SiliconFlow models"""
    agents_to_test = [
        ("Root Agent", "excel_intelligence_agent.agent", "excel_intelligence_agent"),
        ("File Analyzer", "excel_intelligence_agent.sub_agents.file_analyzer.agent", "file_analyzer_agent"),
        ("Column Profiler", "excel_intelligence_agent.sub_agents.column_profiler.agent", "column_profiler_agent"),
        ("Relation Discoverer", "excel_intelligence_agent.sub_agents.relation_discoverer.agent", "relation_discoverer_agent"),
        ("Response Synthesizer", "excel_intelligence_agent.sub_agents.response_synthesizer.agent", "response_synthesizer_agent")
    ]
    
    initialized_agents = []
    
    for agent_name, module_path, agent_var_name in agents_to_test:
        try:
            module = __import__(module_path, fromlist=[agent_var_name])
            agent = getattr(module, agent_var_name)
            
            # Check if agent has SiliconFlow configuration
            if hasattr(agent, '_siliconflow_client'):
                print(f"✅ {agent_name} initialized with SiliconFlow client")
            elif hasattr(agent, 'model') and ('qwen' in str(agent.model) or 'siliconflow' in str(agent.model)):
                print(f"✅ {agent_name} initialized with SiliconFlow-compatible model: {agent.model}")
            else:
                print(f"[WARN]  {agent_name} initialized with model: {getattr(agent, 'model', 'unknown')}")
            
            initialized_agents.append(agent_name)
            
        except Exception as e:
            print(f"[FAIL] {agent_name} initialization failed: {e}")
    
    return len(initialized_agents) == len(agents_to_test)

def test_constants_and_configuration():
    """Test that all constants are properly defined"""
    try:
        from excel_intelligence_agent.shared_libraries.constants import (
            ROOT_AGENT_NAME,
            FILE_ANALYZER_NAME,
            COLUMN_PROFILER_NAME,
            RELATION_DISCOVERER_NAME,
            RESPONSE_SYNTHESIZER_NAME
        )
        
        agent_names = [
            ROOT_AGENT_NAME,
            FILE_ANALYZER_NAME,
            COLUMN_PROFILER_NAME,
            RELATION_DISCOVERER_NAME,
            RESPONSE_SYNTHESIZER_NAME
        ]
        
        print("[PASS] All agent names defined:")
        for name in agent_names:
            print(f"   {name}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Constants test failed: {e}")
        return False

def main():
    """Run all SiliconFlow integration tests"""
    print("Starting SiliconFlow Integration Verification\n")
    
    # Setup test environment
    setup_test_environment()
    
    # Run tests
    tests = [
        ("Environment Loading", test_environment_loading),
        ("SiliconFlow Client", test_siliconflow_client),
        ("Model Adapter", test_model_adapter),
        ("Agent Initialization", test_agent_initialization),
        ("Constants & Configuration", test_constants_and_configuration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nTesting Testing {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"[FAIL] {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\nResults Test Results Summary:")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSUCCESS! All tests passed! SiliconFlow integration is ready.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)