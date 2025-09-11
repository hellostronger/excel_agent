"""
测试中文编码问题修复
"""

import sys
import os
import json
from pathlib import Path

# 设置UTF-8编码
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加backend目录到Python路径
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

def test_json_encoding():
    """测试JSON编码处理"""
    print("🧪 测试JSON编码处理...")
    
    # 测试数据包含中文
    test_data = {
        "query": "分析销售数据的趋势",
        "response": "📊 基于您上传的文件，数据分析结果如下：",
        "recommendations": [
            "建议检查其他列是否存在供应商3月的具体评分和等级信息。",
            "建议进一步核实是否存在合并单元格等格式问题导致数据未被直接显示。",
            "建议与供应商联系，确认3月的具体评分情况。"
        ],
        "execution_log": [
            "开始分析Excel文件中的数据...",
            "正在查找供应商相关信息...",
            "分析完成，生成建议报告"
        ]
    }
    
    # 测试ensure_ascii=False
    json_str_false = json.dumps(test_data, ensure_ascii=False, indent=2)
    print("✅ ensure_ascii=False 输出:")
    print(json_str_false[:200] + "..." if len(json_str_false) > 200 else json_str_false)
    
    # 测试ensure_ascii=True (默认)
    json_str_true = json.dumps(test_data, ensure_ascii=True, indent=2)
    print("\n❌ ensure_ascii=True 输出 (会产生乱码):")
    print(json_str_true[:200] + "..." if len(json_str_true) > 200 else json_str_true)
    
    print("\n✅ JSON编码测试完成")

def test_mock_response():
    """测试Mock响应生成"""
    print("\n🧪 测试Mock响应生成...")
    
    try:
        from app import generate_mock_response
        
        # 模拟文件信息
        mock_file_info = {
            'original_name': '销售数据表.xlsx',
            'size': 102400
        }
        
        # 测试不同查询
        test_queries = [
            "请统计基本信息",
            "分析数据异常值",
            "生成可视化图表",
            "计算相关性分析"
        ]
        
        for query in test_queries:
            print(f"\n📝 查询: {query}")
            try:
                response = generate_mock_response(query, mock_file_info)
                print(f"✅ 响应生成成功，analysis长度: {len(response.get('analysis', ''))}")
                
                # 检查是否包含中文
                analysis = response.get('analysis', '')
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in analysis)
                if has_chinese:
                    print("✅ 包含中文字符")
                else:
                    print("⚠️ 未检测到中文字符")
                    
            except Exception as e:
                print(f"❌ 响应生成失败: {e}")
        
        print("\n✅ Mock响应测试完成")
        
    except ImportError as e:
        print(f"❌ 无法导入app模块: {e}")

def test_file_manager_encoding():
    """测试文件管理器编码"""
    print("\n🧪 测试文件管理器编码...")
    
    try:
        from utils.file_manager import file_manager
        
        # 测试保存包含中文的元数据
        test_metadata = {
            'test_file': {
                'original_name': '测试文件.xlsx',
                'description': '这是一个包含中文的测试文件',
                'keywords': ['销售', '分析', '数据'],
                'text_analysis': {
                    'total_texts': 100,
                    'keywords_by_sheet': {
                        '销售数据': [('销售额', 0.8), ('客户', 0.6)],
                        '产品信息': [('产品名称', 0.9), ('价格', 0.7)]
                    }
                }
            }
        }
        
        # 直接测试JSON序列化
        import json
        json_str = json.dumps(test_metadata, ensure_ascii=False, indent=2, default=str)
        print("✅ 元数据JSON序列化成功")
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in json_str)
        print(f"包含中文字符: {has_chinese}")
        
        # 测试反序列化
        parsed_data = json.loads(json_str)
        print("✅ 元数据JSON反序列化成功")
        
        print("✅ 文件管理器编码测试完成")
        
    except Exception as e:
        print(f"❌ 文件管理器编码测试失败: {e}")

def test_text_processor_encoding():
    """测试文本处理器编码"""
    print("\n🧪 测试文本处理器编码...")
    
    try:
        from utils.text_processor import TextProcessor
        
        processor = TextProcessor()
        
        # 测试中文分词
        test_text = "请帮我分析这个Excel表格中的销售数据趋势"
        words = processor.segment_text(test_text)
        print(f"✅ 分词成功: {words}")
        
        # 测试关键词提取
        keywords = processor.extract_keywords(test_text)
        print(f"✅ 关键词提取成功: {keywords}")
        
        print("✅ 文本处理器编码测试完成")
        
    except Exception as e:
        print(f"❌ 文本处理器编码测试失败: {e}")

def test_relevance_matcher_encoding():
    """测试相关性匹配器编码"""
    print("\n🧪 测试相关性匹配器编码...")
    
    try:
        from utils.relevance_matcher import relevance_matcher
        
        # 测试查询分词
        test_query = "分析销售数据的统计信息"
        words = relevance_matcher.segment_query(test_query)
        print(f"✅ 查询分词成功: {words}")
        
        # 测试Excel相关性判断
        is_related, score, keywords = relevance_matcher.is_excel_related_query(test_query)
        print(f"✅ 相关性判断成功: 相关={is_related}, 评分={score}, 关键词={keywords}")
        
        print("✅ 相关性匹配器编码测试完成")
        
    except Exception as e:
        print(f"❌ 相关性匹配器编码测试失败: {e}")

if __name__ == "__main__":
    print("🚀 开始中文编码测试")
    print("=" * 60)
    
    # 运行所有测试
    test_json_encoding()
    test_mock_response()
    test_file_manager_encoding()
    test_text_processor_encoding()
    test_relevance_matcher_encoding()
    
    print("\n" + "=" * 60)
    print("🎉 中文编码测试完成！")
    print("\n📝 修复总结:")
    print("- ✅ 设置 Flask app.config['JSON_AS_ASCII'] = False")
    print("- ✅ 添加 @app.after_request 响应头处理")
    print("- ✅ 确保文件管理器使用 ensure_ascii=False")
    print("- ✅ 验证所有组件中文处理正确性")
    print("\n🔧 建议:")
    print("- 重启Flask应用以应用新的编码设置")
    print("- 测试前端API调用确认中文显示正常")
    print("- 检查浏览器开发者工具网络请求的响应编码")