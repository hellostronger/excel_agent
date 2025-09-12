# Excel Intelligence Agent

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ADK](https://img.shields.io/badge/Google%20ADK-1.5.0+-green.svg)](https://github.com/google/adk-python)

<img src="https://github.com/google/adk-docs/blob/main/docs/assets/agent-development-kit.png" alt="Agent Development Kit Logo" width="150">

A comprehensive **multi-agent Excel file analysis system** built with Google ADK (Agent Development Kit). This system provides intelligent analysis of Excel files through coordinated specialized agents, delivering accurate insights and answering user questions with data-driven intelligence.

## ✨ Features

### 🎯 **Four-Stage Progressive Analysis**
1. **File Preparation**: Comprehensive structure and metadata extraction
2. **Concurrent Analysis**: Specialized agents working in parallel
3. **Data Integration**: Results synthesis and relationship building
4. **Response Generation**: Intelligent, contextual answers

### 🤖 **Specialized Agent Architecture**
- **File Analyzer**: Deep Excel structure analysis and metadata extraction
- **Column Profiler**: Data quality, types, and statistical analysis  
- **Relation Discoverer**: Cross-table relationships and dependencies
- **Response Synthesizer**: Intelligent answer generation

### 🔧 **Advanced Capabilities**
- **Real ADK Integration**: Built with Google's Agent Development Kit
- **Concurrent Processing**: Multiple agents working simultaneously
- **Quality Assessment**: Comprehensive data quality analysis
- **Relationship Discovery**: Cross-sheet data connections
- **Business Intelligence**: Actionable insights and recommendations

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**: Ensure you have Python 3.9 or later installed
- **Google Cloud Project**: Required for ADK and Vertex AI services
- **Poetry**: For dependency management (recommended)

### Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd excel_intelligence_agent
   ```

2. **Install Dependencies**
   ```bash
   # Using Poetry (recommended)
   poetry install
   
   # Or using pip
   pip install -e .
   ```

3. **Set Up Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Configure Google Cloud Credentials**
   ```bash
   # Set up Google Cloud authentication
   gcloud auth application-default login
   
   # Or set the environment variable
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
   ```

### Environment Configuration

Create a `.env` file with the following variables:

```env
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# API Keys
GEMINI_API_KEY=your-gemini-api-key

# Agent Models Configuration  
ROOT_AGENT_MODEL=gemini-2.5-flash
ORCHESTRATOR_MODEL=gemini-2.5-pro
WORKER_MODEL=gemini-2.5-flash

# Processing Configuration
MAX_PARALLEL_AGENTS=4
FILE_ANALYSIS_DEPTH=comprehensive
ENABLE_CONCURRENT_ANALYSIS=true

# Timeouts (seconds)
FILE_PREPARATION_TIMEOUT=120
PARALLEL_ANALYSIS_TIMEOUT=300
RESPONSE_GENERATION_TIMEOUT=60
```

## 💻 Usage

### Basic Usage

```python
from excel_intelligence_agent.agent import excel_intelligence_agent
from google.adk.tools import ToolContext

# Create a tool context for the analysis session
tool_context = ToolContext()

# Run analysis using ADK
result = await excel_intelligence_agent.run_async(
    args={
        "user_query": "Analyze the data quality and relationships in this sales report",
        "file_path": "/path/to/your/excel-file.xlsx",
        "analysis_depth": "comprehensive"
    },
    tool_context=tool_context
)

print(result)
```

### Using ADK CLI

```bash
# Run the agent using ADK CLI
adk run . --query "What are the data quality issues in my Excel file?" --file_path "/path/to/file.xlsx"

# Start ADK web interface
adk web
# Then navigate to http://localhost:8080
```

### Advanced Usage with Focus Areas

```python
# Analyze with specific focus areas
result = await excel_intelligence_agent.run_async(
    args={
        "user_query": "Focus on data relationships and business insights",
        "file_path": "/path/to/your/excel-file.xlsx",
        "focus_areas": ["relationships", "business_value", "data_quality"]
    },
    tool_context=tool_context
)
```

## 🏗️ Architecture

### Multi-Agent System Design

```
Excel Intelligence Agent (Root)
├── 📁 File Analyzer Agent
│   ├── Structure analysis
│   ├── Metadata extraction
│   └── Complexity assessment
│
├── 📊 Column Profiler Agent
│   ├── Data type analysis
│   ├── Quality assessment
│   └── Statistical profiling
│
├── 🔗 Relation Discoverer Agent
│   ├── Cross-sheet relationships
│   ├── Data dependencies
│   └── Business connections
│
└── 💬 Response Synthesizer Agent
    ├── Result integration
    ├── Insight generation
    └── Intelligent responses
```

### Processing Flow

1. **User Query** → Root Agent receives request
2. **File Preparation** → File Analyzer extracts structure & metadata  
3. **Concurrent Analysis** → Column Profiler + Relation Discoverer work in parallel
4. **Data Integration** → Results synthesized and relationships mapped
5. **Response Generation** → Response Synthesizer creates intelligent answer

## 🛠️ Development

### Project Structure

```
excel_intelligence_agent/
├── excel_intelligence_agent/
│   ├── __init__.py              # Main module
│   ├── agent.py                 # Root orchestrator agent
│   ├── prompts.py               # All agent prompts
│   ├── tools.py                 # Orchestrator tools
│   ├── sub_agents/              # Specialized sub-agents
│   │   ├── file_analyzer/       # File structure analysis
│   │   ├── column_profiler/     # Data quality analysis
│   │   ├── relation_discoverer/ # Relationship discovery
│   │   └── response_synthesizer/# Response generation
│   └── shared_libraries/        # Common utilities
│       ├── types.py             # Data models
│       ├── constants.py         # System constants
│       └── utils.py             # Helper functions
├── tests/                       # Test suite
├── deployment/                  # Deployment configs
├── eval/                       # Evaluation framework
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

### Adding New Agents

1. Create new agent directory under `sub_agents/`
2. Implement `agent.py` with ADK Agent class
3. Add agent-specific tools in `tools.py`
4. Define prompts in `prompt.py`
5. Export agent in `__init__.py`
6. Update root agent to include new sub-agent

### Testing

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_agents.py

# Run with coverage
poetry run pytest --cov=excel_intelligence_agent
```

## 📊 Performance

### Benchmarks

| File Size | Sheets | Processing Time | Memory Usage |
|-----------|--------|----------------|--------------|
| < 1MB     | 1-3    | 30-60s        | < 100MB      |
| 1-5MB     | 3-10   | 1-3 min       | 100-300MB    |
| 5-20MB    | 10+    | 3-8 min       | 300-500MB    |

### Optimization Features

- **Concurrent Processing**: Multiple agents work simultaneously
- **Smart Sampling**: Efficient analysis of large datasets
- **Memory Management**: Optimized memory usage patterns
- **Caching**: State persistence across analysis stages

## 🔧 Configuration

### Agent Models

Configure different models for different agents based on your needs:

```python
# High-performance configuration
ROOT_AGENT_MODEL = "gemini-2.5-pro"      # Best quality
ORCHESTRATOR_MODEL = "gemini-2.5-pro"    # Complex reasoning
WORKER_MODEL = "gemini-2.5-flash"        # Fast processing

# Cost-optimized configuration  
ROOT_AGENT_MODEL = "gemini-2.5-flash"
ORCHESTRATOR_MODEL = "gemini-2.5-flash"
WORKER_MODEL = "gemini-2.5-flash"
```

### Processing Limits

```python
MAX_FILE_SIZE_MB = 100           # Maximum file size
MAX_SHEETS_COUNT = 50            # Maximum sheets to process
MAX_COLUMNS_PER_SHEET = 500      # Maximum columns per sheet
MAX_ROWS_SAMPLE = 1000           # Sample size for large datasets
```

## 🚀 Deployment

### Local Development

```bash
# Start development server
poetry run adk web

# Run specific analysis
poetry run python -m excel_intelligence_agent.agent analyze --file="data.xlsx"
```

### Google Cloud Deployment

```bash
# Deploy to Vertex AI Agent Engine
python deployment/deploy.py --project-id=your-project --region=us-central1
```

### Docker Deployment

```bash
# Build image
docker build -t excel-intelligence-agent .

# Run container
docker run -p 8080:8080 -e GOOGLE_CLOUD_PROJECT=your-project excel-intelligence-agent
```

## 📖 Examples

### Example 1: Data Quality Analysis

```python
# Analyze data quality issues
query = "What are the data quality problems in my sales data?"
result = await analyze_excel(query, "sales_data.xlsx")
```

**Output:**
```
## Analysis Results
Analyzed Excel file with 3 worksheets (2,456,789 bytes).

### Data Quality Assessment
Overall data quality: **Good** across 47 columns analyzed.
Key quality considerations: Missing values in customer_id column, Inconsistent date formats, Duplicate entries detected

### Recommendations
• Address missing data in 3 critical columns before analysis
• Standardize date formats across all sheets for consistency
• Remove 156 duplicate records found in customer data
```

### Example 2: Relationship Discovery

```python
# Find data relationships
query = "How are the different sheets connected? What relationships exist?"
result = await analyze_excel(query, "financial_data.xlsx", focus_areas=["relationships"])
```

**Output:**
```
### Data Relationships
Discovered 8 data relationships enabling integrated analysis.
• Customer_ID connects Orders and Customer_Details sheets
• Product_Code links Products and Sales_Transactions
• Date fields enable temporal analysis across all sheets
```

### Example 3: Business Insights

```python
# Get business insights
query = "What insights can you provide about this business data?"
result = await analyze_excel(query, "business_report.xlsx", focus_areas=["business_value"])
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure tests pass: `poetry run pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full documentation](https://docs.example.com/excel-intelligence-agent)
- **Issues**: [GitHub Issues](https://github.com/your-org/excel-intelligence-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/excel-intelligence-agent/discussions)

## 🙏 Acknowledgments

- Built with [Google Agent Development Kit (ADK)](https://github.com/google/adk-python)
- Powered by [Google Vertex AI](https://cloud.google.com/vertex-ai)
- Inspired by the [ADK Samples](https://github.com/google/adk-samples) repository

## ⚠️ Disclaimers

This project is intended for demonstration and educational purposes. It showcases how to build sophisticated multi-agent systems using Google ADK for Excel file analysis.

- Not intended for production use without additional security and performance considerations
- Requires Google Cloud services and may incur costs
- Analysis accuracy depends on data quality and file complexity

---

**Happy Analyzing!** 🎉

For more information about the Agent Development Kit, visit the [ADK Documentation](https://google.github.io/adk-docs/).