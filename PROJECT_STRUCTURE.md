# Excel 智能代理系统 - 项目结构

## 项目概览

Excel智能代理系统是一个基于AI的Excel数据处理和分析系统，支持自然语言查询、数据分析、代码生成和执行。

## 目录结构

```
excel_agent/
├── README.md                    # 项目主说明文档
├── README_CN.md                 # 中文说明文档
├── PROJECT_STRUCTURE.md         # 本文件 - 项目结构说明
├── requirements.txt             # Python依赖包列表
├── frontend_requirements.txt    # 前端依赖包列表
├── pyproject.toml              # Python项目配置
├── logging_config.py           # 日志配置文件
├── run_frontend.py             # 前端运行脚本
├── test_data.xlsx              # 测试数据文件
│
├── src/excel_agent/            # 核心代码目录
│   ├── __init__.py
│   ├── core/                   # 核心组件
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # 主协调器
│   │   └── workflow.py             # 工作流管理
│   ├── agents/                 # 智能代理
│   │   ├── __init__.py
│   │   ├── base.py                 # 基础代理类
│   │   ├── file_ingest.py          # 文件摄取代理
│   │   ├── column_profiling.py     # 列分析代理
│   │   ├── code_generation.py      # 代码生成代理
│   │   ├── execution.py            # 代码执行代理
│   │   ├── structure_scan.py       # 结构扫描代理
│   │   ├── relation_discovery.py   # 关系发现代理
│   │   ├── summarization.py        # 摘要代理
│   │   ├── labeling.py            # 标签代理
│   │   ├── merge_handling.py       # 合并处理代理
│   │   └── memory.py              # 记忆代理
│   ├── models/                 # 数据模型
│   │   ├── __init__.py
│   │   ├── base.py                # 基础模型定义
│   │   └── agents.py              # 代理模型定义
│   ├── utils/                  # 工具模块
│   │   ├── __init__.py
│   │   ├── config.py              # 配置管理
│   │   ├── config_storage.py      # 配置存储
│   │   ├── logging.py             # 日志工具
│   │   └── siliconflow_client.py  # SiliconFlow客户端
│   └── mcp/                    # MCP (Model Context Protocol) 集成
│       ├── __init__.py
│       ├── base.py               # MCP基础实现
│       ├── registry.py           # MCP注册表
│       ├── capabilities.py       # MCP功能定义
│       └── agent_configs.py      # 代理配置
│
├── backend/                    # 后端服务
│   ├── app.py                    # Flask应用主文件
│   ├── templates/               # HTML模板
│   │   ├── index.html            # 主页面
│   │   └── chat.html            # 聊天页面
│   └── utils/                   # 后端工具
│       └── excel_to_html.py      # Excel转HTML工具
│
├── tests/                      # 测试文件 (新整理)
│   ├── README.md                # 测试说明文档
│   ├── integration/            # 集成测试
│   │   ├── test_basic_integration.py
│   │   ├── test_basic_integration_fixed.py
│   │   ├── test_detailed_logs.py
│   │   └── test_preview.py
│   ├── unit/                   # 单元测试
│   │   ├── test_file_ingest_agent.py
│   │   └── test_column_profiling_agent.py
│   ├── mcp/                    # MCP测试
│   │   ├── test_mcp_integration.py
│   │   ├── test_mcp_simple.py
│   │   └── test_mcp_standalone.py
│   └── archived/               # 归档测试
│       └── simple_test_legacy.py
│
├── tools/                      # 开发工具 (新整理)
│   ├── README.md                # 工具说明文档
│   ├── debug/                  # 调试工具
│   │   └── debug_query.py
│   ├── examples/               # 使用示例
│   │   └── example_usage.py
│   ├── validation/             # 验证工具
│   │   └── validate_system.py
│   └── archived/               # 归档工具
│
├── docs/                       # 文档目录
│   ├── FRONTEND_README.md       # 前端文档
│   ├── MCP_INTEGRATION_README.md # MCP集成文档
│   ├── TODO.md                 # 待办事项
│   └── FINAL_SUMMARY.md        # 最终总结
│
└── excel_agent_env/            # Python虚拟环境 (gitignore)

```

## 核心组件说明

### 1. 核心系统 (src/excel_agent/core/)
- **orchestrator.py**: 主协调器，负责意图解析、工作流路由和代理协调
- **workflow.py**: 工作流管理，定义不同类型的处理流程

### 2. 智能代理 (src/excel_agent/agents/)
- **base.py**: 所有代理的基础类，提供通用功能
- **file_ingest.py**: 文件摄取代理，处理Excel文件读取和解析
- **column_profiling.py**: 列分析代理，分析数据列的特征和统计信息
- **code_generation.py**: 代码生成代理，根据用户需求生成Python代码
- **execution.py**: 代码执行代理，在沙箱环境中执行生成的代码

### 3. 数据模型 (src/excel_agent/models/)
- **base.py**: 基础数据模型定义
- **agents.py**: 代理特定的数据模型

### 4. 工具模块 (src/excel_agent/utils/)
- **config.py**: 系统配置管理
- **logging.py**: 日志记录工具
- **siliconflow_client.py**: AI模型调用客户端

### 5. MCP集成 (src/excel_agent/mcp/)
- **base.py**: MCP协议基础实现
- **registry.py**: MCP服务注册表
- **capabilities.py**: MCP功能定义

### 6. Web后端 (backend/)
- **app.py**: Flask Web应用，提供REST API
- **utils/excel_to_html.py**: Excel文件转HTML预览工具

## 工作流程

1. **用户请求**: 用户通过Web界面或API提交Excel文件和查询
2. **意图解析**: Orchestrator解析用户意图，确定处理类型
3. **工作流路由**: 根据意图选择合适的工作流（单表、多表、单元格）
4. **代理协调**: 按顺序调用相关代理处理数据
5. **结果返回**: 整合各代理结果，返回给用户

## 开发指南

### 新增代理
1. 继承 `BaseAgent` 类
2. 实现 `process` 方法
3. 在 `agents/__init__.py` 中注册
4. 更新 `orchestrator.py` 中的工作流

### 测试
- 单元测试：`pytest tests/unit/`
- 集成测试：`pytest tests/integration/`
- MCP测试：`pytest tests/mcp/`

### 调试
- 使用 `tools/debug/debug_query.py` 调试查询
- 使用 `tools/validation/validate_system.py` 验证系统状态

## 部署

### 开发环境
```bash
pip install -r requirements.txt
python backend/app.py
```

### 生产环境
参考 `README.md` 中的详细部署说明。

## 最近更新

- ✅ 优化了Orchestrator代码结构和文档
- ✅ 重新组织了测试文件架构
- ✅ 整理了开发工具目录结构  
- ✅ 修复了Excel转HTML的合并单元格处理
- ✅ 改进了前端数据预览功能

## 贡献

请参考各子目录的README文件了解具体的贡献指南。