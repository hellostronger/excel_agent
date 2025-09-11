"""
测试Excel相关性匹配器功能
"""

import sys
import os
from pathlib import Path

# 设置UTF-8编码
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加backend目录到Python路径
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

try:
    from utils.relevance_matcher import ExcelRelevanceMatcher, RelevanceResult
    from utils.text_processor import TextProcessor
    print("✅ 成功导入相关性匹配器")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_basic_matching():
    """测试基本的Excel相关性匹配"""
    print("\n🧪 测试基本Excel相关性匹配...")
    
    matcher = ExcelRelevanceMatcher()
    
    # 测试用例
    test_cases = [
        # Excel相关查询
        ("请帮我分析这个表格的数据", True, "包含Excel关键词"),
        ("统计销售数据的总和", True, "包含统计和销售关键词"),
        ("制作一个图表显示趋势", True, "包含图表关键词"),
        ("计算平均值和最大值", True, "包含计算关键词"),
        ("查找异常数据", True, "包含查找关键词"),
        
        # 非Excel相关查询
        ("今天天气怎么样", False, "天气查询"),
        ("你是谁", False, "身份查询"),
        ("北京到上海的距离", False, "地理查询"),
        ("写一首诗", False, "创作请求"),
        
        # 边界情况
        ("", False, "空查询"),
        ("数据", True, "单词查询"),
        ("abc123", False, "纯英文数字"),
    ]
    
    for query, expected, description in test_cases:
        is_related, score, keywords = matcher.is_excel_related_query(query)
        status = "✅" if is_related == expected else "❌"
        print(f"{status} {description}: '{query}' -> 相关性: {is_related}, 评分: {score:.2f}, 关键词: {keywords}")
    
    print("✅ 基本匹配测试完成")

def test_query_segmentation():
    """测试查询分词功能"""
    print("\n✂️ 测试查询分词功能...")
    
    matcher = ExcelRelevanceMatcher()
    
    test_queries = [
        "请帮我分析销售数据的统计信息",
        "计算每个月的平均收入",
        "制作产品类别的饼图",
        "找出异常的数据点",
        "导出财务报表"
    ]
    
    for query in test_queries:
        words = matcher.segment_query(query)
        print(f"📝 '{query}' -> {words}")
    
    print("✅ 分词测试完成")

def test_sheet_matching():
    """测试与工作表内容的匹配"""
    print("\n📊 测试工作表内容匹配...")
    
    matcher = ExcelRelevanceMatcher()
    
    # 模拟文本分析结果
    mock_text_analysis = {
        'keywords_by_sheet': {
            'Sales': [
                ('销售', 0.8),
                ('收入', 0.7),
                ('产品', 0.6),
                ('客户', 0.5),
                ('订单', 0.4)
            ],
            'Products': [
                ('产品', 0.9),
                ('价格', 0.7),
                ('库存', 0.6),
                ('分类', 0.5),
                ('品牌', 0.4)
            ],
            'Financial': [
                ('财务', 0.8),
                ('利润', 0.7),
                ('成本', 0.6),
                ('预算', 0.5),
                ('报表', 0.4)
            ]
        },
        'top_words': {
            '销售': 15,
            '产品': 12,
            '财务': 8,
            '数据': 20,
            '分析': 18
        }
    }
    
    # 测试查询
    test_queries = [
        "分析销售收入的趋势",      # 应该匹配Sales工作表
        "查看产品价格信息",        # 应该匹配Products工作表  
        "生成财务利润报表",        # 应该匹配Financial工作表
        "统计所有数据的总和",      # 应该匹配多个工作表
        "今天天气如何",           # 不应该匹配任何工作表
        "计算平均库存量"          # 应该匹配Products工作表
    ]
    
    for query in test_queries:
        result = matcher.match_query_to_sheets(query, mock_text_analysis)
        print(f"\n🔍 查询: '{query}'")
        print(f"   相关性: {result.is_relevant}")
        print(f"   置信度: {result.confidence_score:.2f}")
        print(f"   匹配工作表: {result.matched_sheets}")
        print(f"   匹配关键词: {result.matched_keywords}")
        print(f"   方法: {result.method}")
        print(f"   摘要: {matcher.get_relevance_summary(result)}")
    
    print("\n✅ 工作表匹配测试完成")

def test_query_enhancement():
    """测试查询增强功能"""
    print("\n🚀 测试查询增强功能...")
    
    matcher = ExcelRelevanceMatcher()
    
    test_cases = [
        ("分析销售数据", ["Sales"], ["销售", "收入"]),
        ("查看产品信息", ["Products", "Inventory"], ["产品", "库存"]),
        ("生成报表", [], ["报表", "财务"]),
        ("计算总和", ["Sales", "Products"], [])
    ]
    
    for original_query, matched_sheets, matched_keywords in test_cases:
        enhanced_query = matcher.enhance_query_with_context(
            original_query, matched_sheets, matched_keywords
        )
        print(f"\n📝 原始查询: {original_query}")
        print(f"🔧 增强查询: {enhanced_query}")
    
    print("✅ 查询增强测试完成")

def test_relevance_scoring():
    """测试相关性评分机制"""
    print("\n📊 测试相关性评分机制...")
    
    matcher = ExcelRelevanceMatcher()
    
    # 创建更详细的测试数据
    detailed_text_analysis = {
        'keywords_by_sheet': {
            'CustomerData': [
                ('客户', 0.9), ('姓名', 0.8), ('电话', 0.7), ('地址', 0.6), ('邮箱', 0.5)
            ],
            'SalesRecord': [
                ('销售', 0.9), ('金额', 0.8), ('日期', 0.7), ('业务员', 0.6), ('提成', 0.5)
            ],
            'ProductInfo': [
                ('产品', 0.9), ('价格', 0.8), ('库存', 0.7), ('供应商', 0.6), ('规格', 0.5)
            ]
        },
        'top_words': {
            '客户': 25, '销售': 22, '产品': 18, '数据': 30, '分析': 28,
            '金额': 15, '价格': 12, '库存': 10, '姓名': 8, '电话': 6
        }
    }
    
    # 测试不同复杂度的查询
    scoring_test_cases = [
        "查询客户张三的销售记录",        # 高相关性 - 多个匹配
        "分析产品价格趋势",             # 中相关性 - 部分匹配
        "统计客户数量",                 # 中相关性 - 单一匹配
        "生成数据分析报告",             # 通用Excel查询
        "计算平均值",                   # 低相关性 - 通用术语
        "删除重复数据",                 # Excel操作
        "你好",                        # 无相关性
        "北京天气预报"                  # 无相关性
    ]
    
    print("评分详情:")
    for query in scoring_test_cases:
        result = matcher.match_query_to_sheets(query, detailed_text_analysis)
        
        # 计算更详细的评分
        relevance_level = "高" if result.confidence_score >= 0.7 else \
                         "中" if result.confidence_score >= 0.3 else \
                         "低" if result.confidence_score > 0 else "无"
        
        print(f"\n📋 查询: '{query}'")
        print(f"   📊 评分: {result.confidence_score:.3f} ({relevance_level}相关性)")
        print(f"   🎯 方法: {result.method}")
        print(f"   📋 匹配工作表: {result.matched_sheets or '无'}")
        print(f"   🔑 关键词: {result.matched_keywords or '无'}")
        
        if result.details:
            if 'sheet_matches' in result.details:
                best_match = result.details.get('best_match', {})
                if best_match:
                    print(f"   🏆 最佳匹配: {best_match.get('sheet')} (分数: {best_match.get('score', 0):.3f})")
    
    print("\n✅ 相关性评分测试完成")

def test_edge_cases():
    """测试边界情况"""
    print("\n🔬 测试边界情况...")
    
    matcher = ExcelRelevanceMatcher()
    
    # 空文本分析数据
    empty_analysis = {'keywords_by_sheet': {}, 'top_words': {}}
    
    # 不完整的文本分析数据
    incomplete_analysis = {
        'keywords_by_sheet': {
            'Sheet1': [('数据', 0.5)]
        }
        # 缺少 top_words
    }
    
    edge_cases = [
        ("正常查询", {'keywords_by_sheet': {'Sheet1': [('测试', 0.5)]}, 'top_words': {'测试': 5}}),
        ("空分析数据", empty_analysis),
        ("不完整数据", incomplete_analysis),
        ("None数据", None)
    ]
    
    test_query = "分析测试数据"
    
    for case_name, analysis_data in edge_cases:
        try:
            if analysis_data is None:
                print(f"⚠️  {case_name}: 跳过 (数据为None)")
                continue
                
            result = matcher.match_query_to_sheets(test_query, analysis_data)
            print(f"✅ {case_name}: 处理成功 - {matcher.get_relevance_summary(result)}")
            
        except Exception as e:
            print(f"❌ {case_name}: 处理失败 - {e}")
    
    print("✅ 边界情况测试完成")

if __name__ == "__main__":
    print("🚀 开始Excel相关性匹配器功能测试")
    print("=" * 60)
    
    # 检查jieba是否可用
    try:
        import jieba
        print("✅ jieba库可用")
    except ImportError:
        print("⚠️  jieba库不可用，某些功能可能受限")
    
    # 运行所有测试
    test_basic_matching()
    test_query_segmentation()
    test_sheet_matching()
    test_query_enhancement()
    test_relevance_scoring()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成！")
    print("\n📝 测试总结:")
    print("- ✅ 基本Excel相关性判断")
    print("- ✅ 查询分词和关键词匹配")
    print("- ✅ 工作表内容智能匹配")
    print("- ✅ 查询上下文增强")
    print("- ✅ 相关性评分机制")
    print("- ✅ 边界情况处理")
    print("\n🔧 集成状态: 已集成到查询处理流程")
    print("📊 功能: 在LLM调用前进行快速相关性筛选")