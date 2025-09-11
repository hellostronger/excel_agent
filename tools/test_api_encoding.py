"""
测试API响应的中文编码
"""

import requests
import json
import sys
import io

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_api_encoding():
    """测试API的中文响应编码"""
    print("🧪 测试API中文编码...")
    
    # API基础URL
    base_url = "http://localhost:5000/api"
    
    # 测试系统状态API
    print("\n1. 测试系统状态API")
    try:
        response = requests.get(f"{base_url}/status", timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 系统状态API响应正常")
            # 检查响应是否包含中文
            response_text = json.dumps(data, ensure_ascii=False)
            if any('\u4e00' <= char <= '\u9fff' for char in response_text):
                print("✅ 响应包含中文字符")
            else:
                print("ℹ️ 响应未包含中文字符（正常）")
        else:
            print(f"❌ 系统状态API请求失败: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保Flask应用正在运行")
        return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False
    
    # 测试文件搜索API（如果有文件的话）
    print("\n2. 测试关键词搜索API")
    try:
        search_data = {
            "keywords": ["销售", "数据"],
            "match_any": True
        }
        
        response = requests.post(
            f"{base_url}/search/keywords", 
            json=search_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"状态码: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        
        if response.status_code in [200, 400]:  # 400可能是没有文件
            try:
                data = response.json()
                print("✅ 关键词搜索API响应正常")
                
                # 检查响应内容
                response_text = json.dumps(data, ensure_ascii=False, indent=2)
                print("响应内容（前200字符）:")
                print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
                
                # 检查中文字符
                if any('\u4e00' <= char <= '\u9fff' for char in response_text):
                    print("✅ 响应包含中文字符且编码正确")
                else:
                    print("ℹ️ 响应未包含中文字符")
                    
            except json.JSONDecodeError:
                print("❌ 响应不是有效的JSON格式")
        else:
            print(f"❌ 关键词搜索API请求失败: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 关键词搜索请求失败: {e}")
    
    return True

def test_mock_query():
    """测试模拟查询的中文响应"""
    print("\n3. 测试模拟查询API")
    
    # 首先需要一个模拟的文件ID
    mock_file_id = "test_file_123"
    
    query_data = {
        "file_id": mock_file_id,
        "query": "请帮我分析这个表格的销售数据"
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/api/query",
            json=query_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"状态码: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        
        if response.status_code in [200, 404]:  # 404是因为文件不存在，但能测试编码
            try:
                data = response.json()
                print("✅ 查询API响应正常")
                
                # 检查响应内容
                response_text = json.dumps(data, ensure_ascii=False, indent=2)
                print("响应内容（前300字符）:")
                print(response_text[:300] + "..." if len(response_text) > 300 else response_text)
                
                # 检查中文字符
                if any('\u4e00' <= char <= '\u9fff' for char in response_text):
                    print("✅ 响应包含中文字符且编码正确")
                else:
                    print("⚠️ 响应未包含中文字符")
                    
            except json.JSONDecodeError as e:
                print(f"❌ 响应不是有效的JSON格式: {e}")
                print("原始响应内容:")
                print(response.text[:500])
        else:
            print(f"❌ 查询API请求失败: {response.status_code}")
            print("响应内容:")
            print(response.text[:300])
            
    except Exception as e:
        print(f"❌ 查询请求失败: {e}")

if __name__ == "__main__":
    print("🚀 开始API中文编码测试")
    print("=" * 50)
    print("⚠️ 请确保Flask应用正在运行 (python backend/app.py)")
    print("=" * 50)
    
    # 运行测试
    if test_api_encoding():
        test_mock_query()
    
    print("\n" + "=" * 50)
    print("🎉 API编码测试完成！")
    print("\n📝 验证结果:")
    print("- 检查控制台输出确认中文字符显示正确")
    print("- 检查Content-Type是否包含charset=utf-8")
    print("- 检查JSON响应是否使用UTF-8编码而非Unicode转义")
    print("\n🔧 如果仍有编码问题:")
    print("- 重启Flask应用")
    print("- 检查浏览器开发者工具Network标签")
    print("- 确认前端接收响应时的处理方式")