# Excel文本分析功能说明

## 功能概述

已成功为Excel智能分析系统添加了文本提取和分词功能，可以自动提取Excel文件中的所有文本内容，使用jieba进行中文分词，并将分析结果作为文件元数据保存。

## 主要特性

### 1. 文本提取
- 从Excel文件的所有工作表中提取文本内容
- 支持多种Excel格式：`.xlsx`、`.xls`、`.xlsm`
- 自动跳过空值和无效数据
- 可配置最大读取行数

### 2. 中文分词
- 使用jieba库进行中文分词
- 内置88个常用停用词
- 支持自定义停用词
- 过滤单字符、纯数字和标点符号

### 3. 关键词提取
- 基于TF-IDF算法提取关键词
- 支持按sheet分别提取关键词
- 可配置提取关键词数量
- 包含词频统计

### 4. 元数据存储
- 自动将分析结果保存到文件元数据
- 支持按关键词搜索文件
- 持久化存储，服务器重启后数据不丢失

## 新增API端点

### 1. 获取文本分析结果
```http
GET /api/text-analysis/{file_id}
```

响应示例：
```json
{
  "success": true,
  "file_id": "abc123",
  "text_analysis": {
    "total_texts": 100,
    "total_words": 500,
    "unique_word_count": 200,
    "top_words": {
      "数据": 15,
      "分析": 12,
      "Excel": 8
    },
    "keywords_by_sheet": {
      "Sheet1": [["数据", 0.8], ["分析", 0.6]],
      "Sheet2": [["产品", 0.7], ["销售", 0.5]]
    }
  }
}
```

### 2. 关键词搜索文件
```http
POST /api/search/keywords
```

请求体：
```json
{
  "keywords": ["数据", "分析"],
  "match_any": true
}
```

响应示例：
```json
{
  "success": true,
  "keywords": ["数据", "分析"],
  "match_any": true,
  "matching_files": [
    {
      "file_id": "abc123",
      "file_info": {...},
      "matched_keywords": ["数据", "分析"]
    }
  ],
  "total_matches": 1
}
```

## 使用流程

### 自动处理
文本分析已集成到文件处理流程中：

1. **文件上传** → `/api/upload`
2. **文件处理** → `/api/process/{file_id}` 
   - 自动执行文本分析
   - 将结果保存到元数据
3. **查询分析结果** → `/api/text-analysis/{file_id}`
4. **关键词搜索** → `/api/search/keywords`

### 手动触发分析
```python
from backend.utils.file_manager import file_manager

# 为指定文件执行文本分析
success = file_manager.analyze_file_text(file_id, max_rows=1000)

# 获取分析结果
text_analysis = file_manager.get_text_analysis(file_id)
```

## 技术实现

### 核心组件

1. **TextProcessor类** (`backend/utils/text_processor.py`)
   - 文本清理和预处理
   - jieba分词集成
   - 关键词提取
   - Excel文件解析

2. **FileManager扩展** (`backend/utils/file_manager.py`)
   - 文本分析方法集成
   - 元数据持久化
   - 关键词搜索功能

3. **API集成** (`backend/app.py`)
   - 自动文本分析
   - REST API端点
   - 错误处理

### 依赖库
- `jieba>=0.42.1` - 中文分词
- `pandas>=2.0.0` - Excel文件处理
- `openpyxl>=3.1.0` - Excel文件读取

## 配置选项

### 停用词配置
```python
from backend.utils.text_processor import TextProcessor

# 使用自定义停用词
processor = TextProcessor(custom_stopwords={'的', '了', '在'})

# 添加停用词
processor.add_stopwords(['新词1', '新词2'])

# 移除停用词
processor.remove_stopwords(['不需要', '的词'])
```

### 分析参数
```python
# 限制分析行数
file_manager.analyze_file_text(file_id, max_rows=1000)

# 关键词提取数量
keywords = processor.extract_keywords(text, top_k=20)
```

## 测试验证

运行测试脚本验证功能：
```bash
python tools/test_text_analysis.py
```

测试内容包括：
- TextProcessor基本功能
- Excel文件处理
- FileManager集成
- API端点测试

## 性能考虑

- 大文件建议设置`max_rows`参数限制分析行数
- jieba首次加载会有1-2秒初始化时间
- 文本分析结果缓存在元数据中，避免重复计算
- 支持异步处理，不会阻塞其他操作

## 示例用例

### 1. 财务报表分析
上传财务Excel后，系统自动提取关键词：
- "收入"、"支出"、"利润"、"成本"等财务术语
- 按工作表分析不同财务项目
- 支持按财务关键词搜索相关文件

### 2. 产品数据分析
产品信息表格的文本分析：
- 提取产品名称、分类、特征关键词
- 分析产品描述中的高频词汇
- 基于产品特征进行文件分组

### 3. 客户数据挖掘
客户信息表的文本处理：
- 提取客户行业、地区、需求关键词
- 分析客户反馈和评价文本
- 支持按客户特征快速检索文件

## 更新记录

- ✅ 创建TextProcessor文本处理器
- ✅ 实现Excel文本提取功能
- ✅ 集成jieba分词和停用词过滤
- ✅ 将分词结果保存为sheet元数据
- ✅ 创建API端点和测试工具
- ✅ 集成到文件处理流程
- ✅ 添加关键词搜索功能

系统现在具备了完整的Excel文本分析能力，可以智能提取和分析Excel文件中的文本内容，为后续的数据分析和查询提供更丰富的语义信息支持。