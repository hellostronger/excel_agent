"""
测试Excel文本分析功能
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
    from utils.text_processor import TextProcessor
    print("✅ 成功导入TextProcessor")
except ImportError as e:
    print(f"❌ 导入TextProcessor失败: {e}")
    sys.exit(1)

def test_text_processor():
    """测试文本处理器的基本功能"""
    print("\n🧪 开始测试文本处理器...")
    
    # 创建文本处理器实例
    processor = TextProcessor()
    print(f"✅ 创建TextProcessor实例，停用词数量: {len(processor.stopwords)}")
    
    # 测试文本清理
    test_text = "这是一个测试文本，包含一些<html>标签</html>和特殊字符@#$%^&*()！"
    cleaned_text = processor.clean_text(test_text)
    print(f"📝 原始文本: {test_text}")
    print(f"🧹 清理后文本: {cleaned_text}")
    
    # 测试分词
    words = processor.segment_text(test_text)
    print(f"✂️ 分词结果: {words}")
    
    # 测试关键词提取
    long_text = "Excel数据分析是一项重要的技能。通过分析销售数据，我们可以了解产品的市场表现。数据分析师经常使用Excel来处理和分析大量的业务数据。"
    keywords = processor.extract_keywords(long_text)
    print(f"🔑 关键词提取: {keywords}")
    
    print("✅ 文本处理器测试完成")

def test_excel_processing():
    """测试Excel文件处理"""
    print("\n📊 开始测试Excel文件处理...")
    
    # 查找测试Excel文件
    test_file = Path(__file__).parent.parent / "test_data.xlsx"
    
    if not test_file.exists():
        print(f"⚠️ 测试文件不存在: {test_file}")
        print("创建一个简单的测试用例...")
        return
    
    print(f"📂 找到测试文件: {test_file}")
    
    try:
        processor = TextProcessor()
        
        # 提取Excel文本
        sheet_texts = processor.extract_text_from_excel(str(test_file), max_rows=10)
        print(f"📄 提取到 {len(sheet_texts)} 个工作表的文本")
        
        for sheet_name, texts in sheet_texts.items():
            print(f"  📋 工作表 '{sheet_name}': {len(texts)} 个文本项")
            if texts:
                print(f"    示例文本: {texts[0][:50]}{'...' if len(texts[0]) > 50 else ''}")
        
        # 完整处理
        result = processor.process_excel_file(str(test_file), max_rows=10)
        print(f"🔢 分析结果:")
        print(f"  总文本数: {result['total_texts']}")
        print(f"  总词数: {result['total_words']}")
        print(f"  唯一词数: {result['unique_word_count']}")
        
        # 显示每个工作表的关键词
        for sheet_name, sheet_data in result['sheets'].items():
            keywords = sheet_data.get('keywords', [])[:5]  # 只显示前5个关键词
            print(f"  工作表 '{sheet_name}' 关键词: {[kw[0] for kw in keywords]}")
        
        print("✅ Excel文件处理测试完成")
        
    except Exception as e:
        print(f"❌ Excel文件处理测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_file_manager_integration():
    """测试与FileManager的集成"""
    print("\n🗃️ 开始测试FileManager集成...")
    
    try:
        from utils.file_manager import file_manager, TEXT_ANALYSIS_AVAILABLE
        
        if not TEXT_ANALYSIS_AVAILABLE:
            print("❌ 文本分析不可用")
            return
        
        print("✅ FileManager文本分析功能可用")
        
        # 获取当前存储的文件列表
        files = file_manager.list_files()
        print(f"📁 当前存储的文件数: {len(files)}")
        
        if files:
            file_id = list(files.keys())[0]
            file_info = files[file_id]
            print(f"🔍 测试文件: {file_info.get('original_name', 'Unknown')}")
            
            # 检查是否已有文本分析
            text_analysis = file_manager.get_text_analysis(file_id)
            if text_analysis:
                print("✅ 文件已有文本分析结果")
                print(f"  总文本数: {text_analysis.get('total_texts', 0)}")
                print(f"  总词数: {text_analysis.get('total_words', 0)}")
                print(f"  唯一词数: {text_analysis.get('unique_word_count', 0)}")
            else:
                print("ℹ️ 文件暂无文本分析结果")
                
                # 尝试分析文件
                print("🔄 执行文本分析...")
                success = file_manager.analyze_file_text(file_id, max_rows=100)
                if success:
                    print("✅ 文本分析完成")
                    text_analysis = file_manager.get_text_analysis(file_id)
                    if text_analysis:
                        print(f"  总文本数: {text_analysis.get('total_texts', 0)}")
                        print(f"  总词数: {text_analysis.get('total_words', 0)}")
                        print(f"  唯一词数: {text_analysis.get('unique_word_count', 0)}")
                else:
                    print("❌ 文本分析失败")
        else:
            print("ℹ️ 没有可测试的文件")
        
        print("✅ FileManager集成测试完成")
        
    except Exception as e:
        print(f"❌ FileManager集成测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 开始Excel文本分析功能测试")
    print("=" * 50)
    
    # 检查jieba是否可用
    try:
        import jieba
        print("✅ jieba库可用")
    except ImportError:
        print("❌ jieba库不可用，需要安装: pip install jieba")
        sys.exit(1)
    
    # 运行测试
    test_text_processor()
    test_excel_processing()
    test_file_manager_integration()
    
    print("\n" + "=" * 50)
    print("🎉 所有测试完成！")