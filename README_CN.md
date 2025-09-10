# Excel 智能代理系统

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Google ADK](https://img.shields.io/badge/powered%20by-Google%20ADK-red.svg)](https://google.github.io/adk-docs/)

一个基于 Google Agent Development Kit (ADK) 框架的精密多代理系统，专为 Excel 文件处理、分析和智能查询而设计。系统支持文件摄取、结构解析、智能查询、多表分析、数据摘要和用户记忆管理，具备自我测试和优化反馈循环功能。

## 🚀 功能特性

### 核心能力
- **智能文件处理**：自动Excel文件摄取和结构分析
- **多代理架构**：10个专业化代理协同工作
- **三种查询类型**：单表、单单元格和多表分析
- **代码生成**：从自然语言自动生成pandas/openpyxl代码
- **沙盒执行**：安全的代码执行环境，具备全面错误处理
- **AI驱动分析**：通过SiliconFlow API集成多种AI模型

### 支持的操作
- **数据分析**：列分析、统计分析、趋势检测
- **数据转换**：筛选、排序、聚合、透视表操作
- **多表操作**：连接、合并、跨表分析
- **结构分析**：合并单元格检测、公式分析、图表识别
- **导出操作**：多种输出格式，可自定义选项

## 🏗️ 系统架构

### 多代理系统设计
系统采用分层多代理架构，以**协调器代理**作为核心调度中心：

```
┌─────────────────┐
│    协调器       │  ← 主要协调器和意图解析器
│     代理        │
└─────────┬───────┘
          │
          ├─ 文件摄取代理        ← Excel文件加载和解析
          ├─ 结构扫描代理        ← 合并单元格、图表、公式
          ├─ 列分析代理          ← 数据类型分析和统计
          ├─ 合并处理代理        ← 合并单元格处理策略
          ├─ 标签代理            ← 智能列/单元格标签
          ├─ 代码生成代理        ← 自然语言 → Python代码
          ├─ 执行代理            ← 沙盒代码执行
          ├─ 摘要代理            ← 数据摘要和洞察
          ├─ 记忆代理            ← 用户偏好和历史记录
          └─ 关系发现代理        ← 多表关系检测
```

### 工作流类型

#### 1. 单表工作流
```
文件摄取 → 列分析 → 代码生成 → 执行
```

#### 2. 单单元格工作流  
```
文件摄取 → 分析 → 代码生成（筛选器） → 执行
```

#### 3. 多表工作流
```
文件摄取 → 多表分析 → 关系发现 → 用户确认 → 代码生成 → 执行
```

## 🛠️ 安装配置

### 系统要求
- Python 3.9 或更高版本
- SiliconFlow API密钥（用于AI功能）

### 安装步骤

1. **克隆仓库**
```bash
git clone <repository-url>
cd excel_agent
```

2. **安装依赖**
```bash
pip install -r requirements.txt
# 或使用pyproject.toml安装
pip install -e .
```

3. **配置环境**
```bash
cp .env.example .env
# 编辑.env文件，添加你的SiliconFlow API密钥
```

4. **生成测试数据**
```bash
python data/synthetic/generate_test_data.py
```

## 🔧 配置说明

### 环境变量
创建一个包含以下配置的`.env`文件：

```env
# SiliconFlow API 配置
SILICONFLOW_API_KEY=your-api-key-here
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# 模型配置  
MULTIMODAL_MODEL=THUDM/GLM-4.1V-9B-Thinking
LLM_MODEL=Qwen/Qwen3-8B
EMBEDDING_MODEL=BAAI/bge-m3
TEXT_TO_IMAGE_MODEL=Kwai-Kolors/Kolors

# 系统配置
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=100
TEMP_DIR=./tmp
CACHE_DIR=./cache

# 代理配置
MAX_AGENTS=10
AGENT_TIMEOUT_SECONDS=300
MEMORY_RETENTION_DAYS=30
```

## 📖 使用指南

### 快速开始

```python
import asyncio
from excel_agent.core.orchestrator import Orchestrator

async def main():
    orchestrator = Orchestrator()
    
    # 处理用户查询
    result = await orchestrator.process_user_request(
        user_request="显示各地区的总销售额",
        file_path="./data/sales_data.xlsx"
    )
    
    print(f"状态: {result['status']}")
    print(f"生成的代码: {result['generated_code']}")
    print(f"输出: {result['output']}")

asyncio.run(main())
```

### 示例查询

#### 单表分析
```python
queries = [
    "显示各地区的总销售额",
    "销售额排名前5的产品是什么？", 
    "计算月度销售趋势",
    "按类别计算平均单价"
]
```

#### 多表分析
```python  
queries = [
    "将销售数据与客户信息连接，显示各行业的销售情况",
    "找出没有任何购买记录的客户",
    "按客户地区和公司规模计算总销售额"
]
```

#### 单单元格操作
```python
queries = [
    "获取B5单元格的值",
    "求A1:A10范围的和", 
    "找到所有包含'总计'的单元格"
]
```

### 运行演示

```bash
python example_usage.py
```

这将演示：
- 文件结构分析
- 系统健康检查  
- 单表分析工作流
- 多表分析工作流

## 🧪 测试

### 运行单元测试
```bash
pytest tests/unit/
```

### 运行集成测试  
```bash
pytest tests/integration/
```

### 运行所有测试
```bash
pytest
```

### 测试覆盖率
```bash
pytest --cov=src/excel_agent tests/
```

## 📊 支持的文件格式

- **Excel文件**：`.xlsx`、`.xls`、`.xlsm`
- **输出格式**：Excel、CSV、JSON
- **复杂结构**：合并单元格、公式、图表、图片

## 🔒 安全功能

### 代码执行安全
- **沙盒环境**：隔离执行上下文
- **模块限制**：仅允许安全模块
- **路径限制**：文件系统访问控制  
- **超时保护**：防止无限循环
- **内存限制**：资源使用监控

### 数据隐私
- **本地处理**：除AI查询外，数据不发送到外部服务
- **临时文件**：自动清理生成的文件
- **访问日志**：全面的审计跟踪

## 🎯 性能优化

### 优化功能
- **智能缓存**：结果和元数据缓存
- **异步处理**：非阻塞操作执行
- **批量操作**：高效的多文件处理
- **内存管理**：自动垃圾回收
- **进度跟踪**：实时执行监控

### 可扩展性
- **水平扩展**：多代理实例
- **负载均衡**：请求分发
- **资源监控**：内存和CPU跟踪
- **错误恢复**：自动重试机制

## 🐛 故障排除

### 常见问题

#### 缺少API密钥
```
⚠️ 警告：环境中未找到SILICONFLOW_API_KEY
```
**解决方案**：在`.env`文件中添加你的SiliconFlow API密钥

#### 文件未找到
```
❌ 文件不存在：/path/to/file.xlsx  
```
**解决方案**：检查文件路径和权限

#### 内存问题
```
❌ 文件过大：150.0MB > 100MB
```
**解决方案**：在配置中增加`MAX_FILE_SIZE_MB`或分割文件

### 调试模式
启用详细日志：
```bash
export LOG_LEVEL=DEBUG
python example_usage.py
```

## 🤝 参与贡献

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`) 
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

### 开发环境设置
```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行代码格式化
black src/ tests/
isort src/ tests/

# 运行代码检查
flake8 src/ tests/
mypy src/
```

## 📚 API参考

### 核心类

#### `Orchestrator`
处理用户请求的主要协调器类。

**方法：**
- `process_user_request(user_request, file_path, context)` - 处理自然语言查询
- `get_workflow_statistics()` - 获取系统性能指标

#### `FileIngestAgent`
处理Excel文件加载和元数据提取。

**方法：**
- `process(request)` - 处理文件摄取请求
- `get_file_metadata(file_id)` - 检索文件元数据

#### `CodeGenerationAgent` 
将自然语言转换为可执行的Python代码。

**方法：**
- `process(request)` - 从用户请求生成代码
- `get_code_template(operation_type)` - 获取代码模板

### 数据模型

所有代理使用标准化的`AgentRequest`和`AgentResponse`模型，包含：
- 用于跟踪的请求/响应ID
- 执行状态和时间
- 全面的错误日志  
- 类型安全的数据结构

## 📄 许可证

本项目使用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 🙏 致谢

- **Google ADK团队** - 提供优秀的Agent Development Kit框架
- **SiliconFlow** - 提供AI模型API访问
- **pandas和openpyxl团队** - 提供强大的Excel处理库
- **开源社区** - 提供令人惊叹的Python工具生态系统

## 📞 支持

- **文档**：[项目Wiki](link-to-wiki)
- **问题**：[GitHub Issues](link-to-issues)  
- **讨论**：[GitHub Discussions](link-to-discussions)
- **邮箱**：[support@example.com](mailto:support@example.com)

---

**使用❤️、Google ADK和Python构建**