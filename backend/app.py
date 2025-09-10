"""Flask backend for Excel Intelligent Agent System with RAG capabilities."""

import os
import asyncio
import sys
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try to import the agent system with fallback
try:
    from excel_agent.core.orchestrator import Orchestrator
    from excel_agent.mcp.agent_configs import initialize_mcp_system, initialize_all_agent_mcp
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Agent system not available: {e}")
    AGENT_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'xlsm'}

# Global variables
orchestrator = None
uploaded_files = {}
mcp_initialized = False


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


async def initialize_agent_system():
    """Initialize the agent system with MCP capabilities."""
    global orchestrator, mcp_initialized
    
    if not AGENT_AVAILABLE:
        return False
    
    try:
        print("Initializing Agent System...")
        
        # Initialize MCP system
        initialize_mcp_system()
        await initialize_all_agent_mcp()
        mcp_initialized = True
        
        # Initialize orchestrator
        orchestrator = Orchestrator()
        
        print("Agent System initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Failed to initialize agent system: {e}")
        return False


def run_async(coro):
    """Run async function in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new task
            return loop.create_task(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, create a new one
        return asyncio.run(coro)


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get system status."""
    status = {
        'agent_available': AGENT_AVAILABLE,
        'mcp_initialized': mcp_initialized,
        'orchestrator_ready': orchestrator is not None,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'max_file_size': app.config['MAX_CONTENT_LENGTH'],
        'allowed_extensions': list(ALLOWED_EXTENSIONS)
    }
    
    if orchestrator and mcp_initialized:
        try:
            # Get MCP status if available
            mcp_status = asyncio.run(orchestrator.get_mcp_status())
            status['mcp_status'] = mcp_status
        except Exception as e:
            status['mcp_error'] = str(e)
    
    return jsonify(status)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload Excel files (.xlsx, .xls, .xlsm)'}), 400
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        
        # Save file
        file.save(file_path)
        
        # Store file info
        file_info = {
            'file_id': file_id,
            'original_name': filename,
            'file_path': file_path,
            'size': os.path.getsize(file_path),
            'upload_time': datetime.now().isoformat(),
            'processed': False
        }
        
        uploaded_files[file_id] = file_info
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'filename': filename,
            'size': file_info['size'],
            'message': 'File uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/process/<file_id>', methods=['POST'])
def process_file(file_id):
    """Process uploaded file using the agent system."""
    try:
        if file_id not in uploaded_files:
            return jsonify({'error': 'File not found'}), 404
        
        file_info = uploaded_files[file_id]
        
        if not AGENT_AVAILABLE or not orchestrator:
            # Return mock response if agent system not available
            return jsonify({
                'success': True,
                'file_id': file_id,
                'processed': True,
                'mock_mode': True,
                'steps': [
                    {'step': 'upload', 'status': 'completed', 'message': 'File uploaded'},
                    {'step': 'ingest', 'status': 'completed', 'message': 'File parsed (mock)'},
                    {'step': 'profile', 'status': 'completed', 'message': 'Data analyzed (mock)'},
                    {'step': 'ready', 'status': 'completed', 'message': 'Ready for queries (mock)'}
                ],
                'metadata': {
                    'sheets': ['Sheet1', 'Data', 'Summary'],
                    'total_rows': 1234,
                    'total_columns': 8,
                    'file_size': file_info['size']
                },
                'preview': {
                    'columns': ['Product', 'Sales', 'Quantity', 'Date', 'Category', 'Price', 'Profit', 'Region'],
                    'sample_data': [
                        ['Laptop', 15000, 5, '2024-01-15', 'Electronics', 3000, 2000, 'North'],
                        ['Phone', 8000, 10, '2024-01-16', 'Electronics', 800, 1200, 'South'],
                        ['Tablet', 6000, 8, '2024-01-17', 'Electronics', 750, 800, 'East']
                    ]
                }
            })
        
        # Process with real agent system
        async def process_with_agents():
            file_path = file_info['file_path']
            
            # Use orchestrator to process file
            result = await orchestrator.process_user_request(
                user_request="Please ingest and analyze this Excel file",
                file_path=file_path,
                context={'file_id': file_id}
            )
            
            return result
        
        # Run async processing
        result = run_async(process_with_agents())
        
        # Update file info
        file_info['processed'] = True
        file_info['process_result'] = result
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'processed': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/api/query', methods=['POST'])
def query_data():
    """Handle user queries about the data."""
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if file_id not in uploaded_files:
            return jsonify({'error': 'File not found'}), 404
        
        file_info = uploaded_files[file_id]
        
        if not AGENT_AVAILABLE or not orchestrator:
            # Return mock response if agent system not available
            mock_responses = generate_mock_response(query, file_info)
            return jsonify({
                'success': True,
                'query': query,
                'response': mock_responses['analysis'],
                'generated_code': mock_responses['code'],
                'mock_mode': True,
                'execution_time': 2.5,
                'insights': mock_responses.get('insights', []),
                'visualizations': mock_responses.get('visualizations', [])
            })
        
        # Process with real agent system
        async def query_with_agents():
            file_path = file_info['file_path']
            
            # Use orchestrator to process query
            result = await orchestrator.process_user_request(
                user_request=query,
                file_path=file_path,
                context={
                    'file_id': file_id,
                    'previous_processing': file_info.get('process_result')
                }
            )
            
            return result
        
        # Run async query processing
        result = run_async(query_with_agents())
        
        return jsonify({
            'success': True,
            'query': query,
            'response': result.get('final_result', 'Analysis completed'),
            'generated_code': result.get('generated_code', ''),
            'workflow_steps': result.get('steps', []),
            'execution_time': result.get('execution_time', 0),
            'status': result.get('status', 'completed')
        })
        
    except Exception as e:
        return jsonify({'error': f'Query failed: {str(e)}'}), 500


def generate_mock_response(query, file_info):
    """Generate mock responses for testing when agent system is not available."""
    query_lower = query.lower()
    
    if '统计' in query_lower or '基本信息' in query_lower:
        return {
            'analysis': f"""📊 数据统计分析结果

基于您上传的文件 "{file_info['original_name']}"：

📋 数据概览：
• 总行数: 1,234 行
• 总列数: 8 列
• 文件大小: {file_info['size']} bytes
• 数据类型: 4个数值列, 3个文本列, 1个日期列

📈 数值列统计：
• 销售额: 平均值 ¥8,547, 最大值 ¥25,000, 最小值 ¥1,200
• 数量: 平均值 15.6, 最大值 100, 最小值 1
• 利润率: 平均值 23.4%, 标准差 8.9%

🔍 数据质量：
• 完整性: 98.7% (16个缺失值)
• 重复记录: 3条
• 异常值: 在销售额列发现5个潜在异常值

💡 关键洞察：
1. 电子产品类别销售额占总销售额的68%
2. Q4销售额比Q3增长了15.3%
3. 周五和周六是销售高峰期""",
            
            'code': f'''import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel('{file_info['original_name']}')

# 基本统计信息
print("数据形状:", df.shape)
print("数据类型:")
print(df.dtypes)

# 数值列统计
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("数值列统计:")
print(df[numeric_cols].describe())

# 缺失值检查
print("缺失值统计:")
print(df.isnull().sum())'''
        }
        
    elif '异常' in query_lower or '离群点' in query_lower:
        return {
            'analysis': """🔍 异常值检测结果

使用IQR方法和Z-Score方法检测异常值：

🚨 发现的异常值：
• 销售额列: 5个异常值
  - 第67行: ¥87,500 (Z-score: 4.2)
  - 第156行: ¥0 (可能录入错误)
  - 第234行: ¥125,000 (Z-score: 6.8)

• 数量列: 2个异常值
  - 第123行: 999 (批发订单)
  - 第567行: 0 (零数量异常)

💡 处理建议：
1. 验证极值的真实性
2. 检查零值和负值
3. 考虑业务规则进行清洗""",
            
            'code': '''import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# 检测异常值
outliers = detect_outliers_iqr(df, 'sales_amount')
print(f"发现 {len(outliers)} 个异常值")'''
        }
        
    elif '相关' in query_lower or '关联' in query_lower:
        return {
            'analysis': """🔗 相关性分析结果

数据相关性分析显示：

📈 强正相关 (r > 0.7):
• 销售额 ↔ 利润: r = 0.89
• 价格 ↔ 利润率: r = 0.76
• 广告投入 ↔ 销售额: r = 0.72

📉 负相关 (r < -0.3):
• 折扣率 ↔ 利润率: r = -0.65
• 库存量 ↔ 销售速度: r = -0.45

💡 业务洞察：
1. 销售额与利润高度相关，定价策略稳定
2. 广告投入ROI良好
3. 折扣策略需要优化""",
            
            'code': '''import seaborn as sns
import matplotlib.pyplot as plt

# 计算相关性矩阵
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

# 创建热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('数据相关性热力图')
plt.show()'''
        }
        
    elif '图表' in query_lower or '可视化' in query_lower:
        return {
            'analysis': """📊 数据可视化分析

已为您生成多种可视化图表：

🎨 生成的图表：
1. 📈 销售趋势线图 - 显示月度变化
2. 📊 产品类别饼图 - 类别分布
3. 📋 销售额分布直方图 - 金额分布
4. 🔥 相关性热力图 - 变量关系

💡 可视化洞察：
• 明显的季节性销售模式
• 产品类别分布不均匀
• 大部分为中小额订单
• 变量间存在有趣关联

所有图表已生成，可用于报告展示。""",
            
            'code': '''import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 销售趋势
monthly_sales = df.groupby('month')['sales'].sum()
axes[0,0].plot(monthly_sales.index, monthly_sales.values, marker='o')
axes[0,0].set_title('月度销售趋势')

# 2. 类别分布
category_counts = df['category'].value_counts()
axes[0,1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
axes[0,1].set_title('产品类别分布')

# 3. 销售额分布
axes[1,0].hist(df['sales_amount'], bins=30, alpha=0.7)
axes[1,0].set_title('销售额分布')

# 4. 相关性热力图
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1,1])
axes[1,1].set_title('相关性热力图')

plt.tight_layout()
plt.show()'''
        }
    
    else:
        return {
            'analysis': f"""🤖 AI分析回复

针对您的问题："{query}"

基于上传的文件 "{file_info['original_name']}"，我提供以下分析：

📊 数据概况：
• 文件大小: {file_info['size']} bytes
• 上传时间: {file_info['upload_time']}
• 处理状态: {"已处理" if file_info.get('processed') else "待处理"}

🔍 建议的分析方向：
1. 基本统计信息分析
2. 数据质量检查
3. 异常值检测
4. 相关性分析
5. 趋势分析
6. 可视化展示

💡 您可以尝试以下具体问题：
• "分析基本统计信息"
• "检测异常值和离群点"
• "分析数据相关性"
• "生成可视化图表"

我随时准备为您提供更详细的分析！""",
            
            'code': f'''import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel('{file_info['original_name']}')

# 基本信息
print("文件基本信息:")
print(f"行数: {{df.shape[0]}}")
print(f"列数: {{df.shape[1]}}")

# 数据预览
print("数据预览:")
print(df.head())

# 数据类型
print("数据类型:")
print(df.dtypes)'''
        }


@app.route('/api/files')
def list_files():
    """List all uploaded files."""
    files = []
    for file_id, file_info in uploaded_files.items():
        files.append({
            'file_id': file_id,
            'filename': file_info['original_name'],
            'size': file_info['size'],
            'upload_time': file_info['upload_time'],
            'processed': file_info['processed']
        })
    
    return jsonify({'files': files})


@app.route('/api/files/<file_id>', methods=['DELETE'])
def delete_file(file_id):
    """Delete uploaded file."""
    try:
        if file_id not in uploaded_files:
            return jsonify({'error': 'File not found'}), 404
        
        file_info = uploaded_files[file_id]
        
        # Delete physical file
        if os.path.exists(file_info['file_path']):
            os.remove(file_info['file_path'])
        
        # Remove from memory
        del uploaded_files[file_id]
        
        return jsonify({'success': True, 'message': 'File deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Delete failed: {str(e)}'}), 500


if __name__ == '__main__':
    # Initialize agent system
    if AGENT_AVAILABLE:
        print("Attempting to initialize agent system...")
        try:
            success = asyncio.run(initialize_agent_system())
            if success:
                print("✅ Agent system initialized successfully!")
            else:
                print("⚠️  Agent system initialization failed, running in mock mode")
        except Exception as e:
            print(f"⚠️  Agent system initialization error: {e}")
            print("Running in mock mode...")
    else:
        print("⚠️  Agent system not available, running in mock mode")
    
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)