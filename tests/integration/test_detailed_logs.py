"""Test script for detailed logging functionality."""

import asyncio
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from excel_agent.utils.siliconflow_client import SiliconFlowClient
from excel_agent.utils.config import config

async def test_detailed_logging():
    """Test the detailed logging functionality."""
    print("Testing detailed logging functionality...")
    print("=" * 60)
    
    # Test a simple request
    test_messages = [
        {
            "role": "user", 
            "content": "请分析这个用户请求并判断工作流类型：'请统计销售数据的基本信息'"
        }
    ]
    
    try:
        async with SiliconFlowClient() as client:
            print("Sending test request to SiliconFlow API...")
            print(f"Current log level: {config.log_level}")
            
            response = await client.chat_completion(
                messages=test_messages,
                temperature=0.3,
                max_tokens=200,
                request_id="test_logs_123"
            )
            
            print("\nResponse received!")
            print(f"Response choices: {len(response.get('choices', []))}")
            if response.get('choices'):
                print(f"First choice content length: {len(response['choices'][0].get('message', {}).get('content', ''))}")
            
            return True
            
    except Exception as e:
        print(f"Test failed: {e}")
        return False

async def test_multiple_messages():
    """Test logging with multiple messages."""
    print("\nTesting multiple message logging...")
    print("=" * 60)
    
    test_messages = [
        {
            "role": "system",
            "content": "你是一个Excel数据分析专家，擅长理解用户意图并分析数据需求。"
        },
        {
            "role": "user", 
            "content": """请分析以下用户请求并分类：

用户请求："我想看看销售数据中哪些产品卖得最好，按照销量排序，并生成一个图表"

请判断这是：
1. 单表操作 (SINGLE_TABLE)
2. 单元格操作 (SINGLE_CELL) 
3. 多表操作 (MULTI_TABLE)

并解释原因。"""
        }
    ]
    
    try:
        async with SiliconFlowClient() as client:
            response = await client.chat_completion(
                messages=test_messages,
                temperature=0.5,
                max_tokens=300,
                request_id="multi_msg_test"
            )
            
            print("Multi-message test completed!")
            return True
            
    except Exception as e:
        print(f"Multi-message test failed: {e}")
        return False

if __name__ == "__main__":
    print("Detailed Logging Test Suite")
    print("=" * 60)
    print(f"Log level: {config.log_level}")
    print(f"Model: {config.llm_model}")
    print("=" * 60)
    
    # Run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Test 1: Simple request
        result1 = loop.run_until_complete(test_detailed_logging())
        
        # Test 2: Multiple messages
        result2 = loop.run_until_complete(test_multiple_messages())
        
        print("\n" + "=" * 60)
        print("Test Results:")
        print(f"Simple request test: {'✅ PASSED' if result1 else '❌ FAILED'}")
        print(f"Multiple messages test: {'✅ PASSED' if result2 else '❌ FAILED'}")
        print("=" * 60)
        
        if result1 and result2:
            print("🎉 All tests passed! Detailed logging is working correctly.")
        else:
            print("⚠️ Some tests failed. Check the logs above for details.")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest suite failed: {e}")
    finally:
        loop.close()