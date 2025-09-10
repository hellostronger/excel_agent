# 测试文件目录

本目录包含了Excel智能代理系统的所有测试文件，按功能和类型进行了分类组织。

## 目录结构

```
tests/
├── README.md                    # 本文件
├── integration/                 # 集成测试
│   ├── test_basic_integration.py      # 基础集成测试
│   ├── test_basic_integration_fixed.py # 修复后的集成测试
│   ├── test_detailed_logs.py          # 详细日志测试
│   └── test_preview.py               # 预览功能测试
├── unit/                        # 单元测试
│   ├── test_file_ingest_agent.py     # 文件摄取代理单元测试
│   └── test_column_profiling_agent.py # 列分析代理单元测试
├── mcp/                         # MCP相关测试
│   ├── test_mcp_integration.py       # MCP集成测试
│   ├── test_mcp_simple.py           # MCP简单测试
│   └── test_mcp_standalone.py       # MCP独立测试
└── archived/                    # 归档的测试文件
    └── simple_test_legacy.py        # 旧版简单测试

```

## 测试运行指南

### 单元测试
```bash
# 运行所有单元测试
pytest tests/unit/ -v

# 运行特定单元测试
pytest tests/unit/test_file_ingest_agent.py -v
```

### 集成测试
```bash
# 运行所有集成测试
pytest tests/integration/ -v

# 运行特定集成测试
pytest tests/integration/test_basic_integration.py -v
```

### MCP测试
```bash
# 运行所有MCP测试
pytest tests/mcp/ -v

# 运行特定MCP测试
pytest tests/mcp/test_mcp_integration.py -v
```

### 运行所有测试
```bash
# 运行所有测试（排除归档）
pytest tests/unit tests/integration tests/mcp -v

# 运行所有测试包括归档
pytest tests/ -v
```

## 测试覆盖率

使用以下命令生成测试覆盖率报告：
```bash
pytest tests/ --cov=src/excel_agent --cov-report=html --cov-report=term
```

## 测试文件说明

### 集成测试
- **test_basic_integration.py**: 测试基本的端到端工作流
- **test_basic_integration_fixed.py**: 修复后的集成测试，包含错误处理
- **test_detailed_logs.py**: 测试详细日志记录功能
- **test_preview.py**: 测试数据预览功能

### 单元测试  
- **test_file_ingest_agent.py**: 文件摄取代理的单元测试
- **test_column_profiling_agent.py**: 列分析代理的单元测试

### MCP测试
- **test_mcp_integration.py**: MCP系统集成测试
- **test_mcp_simple.py**: MCP基本功能测试
- **test_mcp_standalone.py**: MCP独立运行测试

## 注意事项

1. 运行测试前确保已安装所有依赖：`pip install -r requirements.txt`
2. 某些测试可能需要实际的Excel文件，确保测试数据文件存在
3. MCP测试需要MCP系统正常初始化
4. 集成测试可能需要较长时间完成

## 贡献指南

添加新测试时请遵循以下规范：
1. 单元测试放在 `unit/` 目录
2. 集成测试放在 `integration/` 目录  
3. MCP相关测试放在 `mcp/` 目录
4. 使用清晰的测试文件命名：`test_<component>_<test_type>.py`
5. 包含适当的文档字符串和注释