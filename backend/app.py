"""Flask backend for Excel Intelligent Agent System with RAG capabilities."""

import os
import asyncio
import sys
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
import time
import json

# Import our Excel to HTML utility
from utils.excel_to_html import excel_to_html, get_excel_sheets, excel_sheet_to_html, get_excel_info

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try to import the agent system with fallback
try:
    from excel_agent.core.orchestrator import Orchestrator
    from excel_agent.mcp.agent_configs import initialize_mcp_system, initialize_all_agent_mcp
    from excel_agent.utils.config import config
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Agent system not available: {e}")
    AGENT_AVAILABLE = False
    # Create a mock config for fallback
    class MockConfig:
        def get_llm_params(self):
            return {'model': 'mock', 'temperature': 0.7, 'max_tokens': None}
        def update_llm_params(self, **params):
            pass
        def get_available_models(self):
            return {'llm_models': ['mock-model']}
        def reset_llm_params_to_default(self):
            pass
    config = MockConfig()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('excel_agent_api.log')
    ]
)

api_logger = logging.getLogger('excel_agent.api')


def log_request_response(f):
    """Decorator to log API requests and responses."""
    def decorated_function(*args, **kwargs):
        # Log request
        request_id = str(uuid.uuid4())[:8]
        g.request_id = request_id
        
        start_time = time.time()
        
        # Log request details
        api_logger.info(f"📥 [Request {request_id}] {request.method} {request.path}")
        api_logger.info(f"📥 [Request {request_id}] Remote: {request.remote_addr}")
        api_logger.info(f"📥 [Request {request_id}] User-Agent: {request.headers.get('User-Agent', 'N/A')}")
        
        # Log request body (if present and not too large)
        if request.is_json and request.content_length and request.content_length < 10000:
            try:
                request_data = request.get_json()
                # Mask sensitive data
                if isinstance(request_data, dict) and 'query' in request_data:
                    log_data = request_data.copy()
                    if len(log_data['query']) > 200:
                        log_data['query'] = log_data['query'][:200] + "... (truncated)"
                    api_logger.debug(f"📥 [Request {request_id}] Body: {json.dumps(log_data, ensure_ascii=False)}")
                else:
                    api_logger.debug(f"📥 [Request {request_id}] Body: {json.dumps(request_data, ensure_ascii=False)}")
            except Exception as e:
                api_logger.debug(f"📥 [Request {request_id}] Body: <unable to parse: {e}>")
        elif request.files:
            files_info = []
            for file_key, file_obj in request.files.items():
                files_info.append(f"{file_key}: {file_obj.filename} ({file_obj.content_length} bytes)")
            api_logger.info(f"📥 [Request {request_id}] Files: {', '.join(files_info)}")
        
        try:
            # Execute the function
            response = f(*args, **kwargs)
            
            # Log response
            end_time = time.time()
            response_time = end_time - start_time
            
            if hasattr(response, 'status_code'):
                status_code = response.status_code
                api_logger.info(f"📤 [Response {request_id}] Status: {status_code}, Time: {response_time:.3f}s")
                
                # Log response body (if JSON and not too large)
                if hasattr(response, 'is_json') and response.is_json and response.content_length and response.content_length < 10000:
                    try:
                        response_data = response.get_json()
                        if isinstance(response_data, dict):
                            log_data = response_data.copy()
                            # Truncate long responses
                            if 'response' in log_data and len(str(log_data['response'])) > 500:
                                log_data['response'] = str(log_data['response'])[:500] + "... (truncated)"
                            if 'generated_code' in log_data and len(str(log_data['generated_code'])) > 500:
                                log_data['generated_code'] = str(log_data['generated_code'])[:500] + "... (truncated)"
                            api_logger.debug(f"📤 [Response {request_id}] Body: {json.dumps(log_data, ensure_ascii=False)}")
                    except Exception as e:
                        api_logger.debug(f"📤 [Response {request_id}] Body: <unable to parse: {e}>")
            else:
                api_logger.info(f"📤 [Response {request_id}] Time: {response_time:.3f}s")
            
            return response
            
        except Exception as e:
            # Log error
            end_time = time.time()
            response_time = end_time - start_time
            api_logger.error(f"❌ [Error {request_id}] {str(e)}, Time: {response_time:.3f}s")
            raise
    
    decorated_function.__name__ = f.__name__
    return decorated_function

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


@app.route('/chat')
def chat():
    """Serve the new chat interface."""
    return render_template('chat.html')


@app.route('/api/status')
@log_request_response
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


@app.route('/api/initialize', methods=['POST'])
@log_request_response
def initialize_system():
    """Initialize the agent system."""
    global orchestrator, mcp_initialized
    
    try:
        if AGENT_AVAILABLE:
            print("Attempting to initialize agent system...")
            success = asyncio.run(initialize_agent_system())
            
            if success:
                return jsonify({
                    'success': True,
                    'message': '协调器系统初始化成功',
                    'orchestrator_ready': orchestrator is not None,
                    'mcp_initialized': mcp_initialized
                })
            else:
                return jsonify({
                    'success': False,
                    'message': '协调器系统初始化失败，运行在演示模式',
                    'orchestrator_ready': False,
                    'mcp_initialized': False
                }), 500
        else:
            return jsonify({
                'success': False,
                'message': 'Agent系统不可用，缺少必要依赖',
                'orchestrator_ready': False,
                'mcp_initialized': False
            }), 503
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'初始化失败: {str(e)}',
            'orchestrator_ready': False,
            'mcp_initialized': False
        }), 500


@app.route('/api/upload', methods=['POST'])
@log_request_response
def upload_file():
    """Handle single file upload."""
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


@app.route('/api/upload/batch', methods=['POST'])
@log_request_response
def upload_batch_files():
    """Handle multiple file upload."""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_file_infos = []
        file_ids = []
        
        for file in files:
            if file.filename == '':
                continue
                
            if not allowed_file(file.filename):
                api_logger.warning(f"Skipping invalid file: {file.filename}")
                continue
            
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
            uploaded_file_infos.append(file_info)
            file_ids.append(file_id)
        
        if not uploaded_file_infos:
            return jsonify({'error': 'No valid Excel files were uploaded'}), 400
        
        return jsonify({
            'success': True,
            'file_ids': file_ids,
            'files': [{
                'file_id': info['file_id'],
                'filename': info['original_name'],
                'size': info['size']
            } for info in uploaded_file_infos],
            'message': f'{len(uploaded_file_infos)} files uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch upload failed: {str(e)}'}), 500


@app.route('/api/process/<file_id>', methods=['POST'])
@log_request_response
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
        
        # Process with real agent system (direct file ingestion without intent parsing)
        async def process_with_agents():
            file_path = file_info['file_path']
            
            # Import file ingest agent and required models
            from excel_agent.agents.file_ingest import FileIngestAgent
            from excel_agent.models.agents import FileIngestRequest
            from excel_agent.models.base import AgentStatus
            
            # Create file ingest agent and process file directly
            file_agent = FileIngestAgent()
            
            ingest_request = FileIngestRequest(
                agent_id="FileIngestAgent",
                file_path=file_path,
                context={'file_id': file_id, 'processing_mode': 'file_upload'}
            )
            
            async with file_agent:
                ingest_response = await file_agent.execute_with_timeout(ingest_request)
            
            # Prepare result without intent parsing
            if ingest_response.status == AgentStatus.SUCCESS:
                return {
                    'status': 'success',
                    'workflow_type': 'file_ingest_only',
                    'steps': [
                        {'step': 'file_ingest', 'status': 'success', 'result': ingest_response.result}
                    ],
                    'file_metadata': ingest_response.result,
                    'sheets': ingest_response.sheets,
                    'file_id': ingest_response.file_id,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'failed',
                    'error_message': 'File ingestion failed',
                    'error_log': ingest_response.error_log,
                    'timestamp': datetime.now().isoformat()
                }
        
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


@app.route('/api/process/batch', methods=['POST'])
@log_request_response
def process_batch_files():
    """Process multiple uploaded files using the agent system."""
    try:
        data = request.get_json()
        if not data or 'file_ids' not in data:
            return jsonify({'error': 'No file IDs provided'}), 400
        
        file_ids = data['file_ids']
        if not isinstance(file_ids, list) or not file_ids:
            return jsonify({'error': 'Invalid file IDs format'}), 400
        
        # Validate all file IDs exist
        missing_files = [fid for fid in file_ids if fid not in uploaded_files]
        if missing_files:
            return jsonify({'error': f'Files not found: {missing_files}'}), 404
        
        if not AGENT_AVAILABLE or not orchestrator:
            # Return mock response if agent system not available
            processed_files = []
            for file_id in file_ids:
                file_info = uploaded_files[file_id]
                processed_files.append({
                    'file_id': file_id,
                    'filename': file_info['original_name'],
                    'processed': True,
                    'steps': [
                        {'step': 'upload', 'status': 'completed', 'message': 'File uploaded'},
                        {'step': 'ingest', 'status': 'completed', 'message': 'File parsed (mock)'},
                        {'step': 'profile', 'status': 'completed', 'message': 'Data analyzed (mock)'},
                        {'step': 'ready', 'status': 'completed', 'message': 'Ready for queries (mock)'}
                    ]
                })
                # Mark as processed
                file_info['processed'] = True
            
            return jsonify({
                'success': True,
                'processed_files': processed_files,
                'mock_mode': True,
                'message': f'{len(file_ids)} files processed successfully (mock mode)'
            })
        
        # Process with real agent system
        async def process_batch_with_agents():
            results = []
            for file_id in file_ids:
                file_info = uploaded_files[file_id]
                file_path = file_info['file_path']
                
                try:
                    # Import required components for direct file processing
                    from excel_agent.agents.file_ingest import FileIngestAgent
                    from excel_agent.models.agents import FileIngestRequest
                    from excel_agent.models.base import AgentStatus
                    
                    # Process file directly without intent parsing
                    file_agent = FileIngestAgent()
                    
                    ingest_request = FileIngestRequest(
                        agent_id="FileIngestAgent",
                        file_path=file_path,
                        context={'file_id': file_id, 'batch_processing': True}
                    )
                    
                    async with file_agent:
                        ingest_response = await file_agent.execute_with_timeout(ingest_request)
                    
                    if ingest_response.status == AgentStatus.SUCCESS:
                        result = {
                            'status': 'success',
                            'workflow_type': 'file_ingest_only',
                            'file_metadata': ingest_response.result,
                            'sheets': ingest_response.sheets,
                            'file_id': ingest_response.file_id
                        }
                    else:
                        result = {
                            'status': 'failed',
                            'error_message': 'File ingestion failed',
                            'error_log': ingest_response.error_log
                        }
                    
                    results.append({
                        'file_id': file_id,
                        'filename': file_info['original_name'],
                        'processed': True,
                        'result': result
                    })
                    
                    # Update file info
                    file_info['processed'] = True
                    file_info['process_result'] = result
                    
                except Exception as e:
                    api_logger.error(f"Failed to process file {file_id}: {str(e)}")
                    results.append({
                        'file_id': file_id,
                        'filename': file_info['original_name'],
                        'processed': False,
                        'error': str(e)
                    })
            
            return results
        
        # Run async batch processing
        results = run_async(process_batch_with_agents())
        
        # Count successful and failed processing
        successful = len([r for r in results if r.get('processed', False)])
        failed = len(results) - successful
        
        return jsonify({
            'success': True,
            'processed_files': results,
            'summary': {
                'total': len(file_ids),
                'successful': successful,
                'failed': failed
            },
            'message': f'Batch processing completed: {successful} successful, {failed} failed'
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500


@app.route('/api/query', methods=['POST'])
@log_request_response
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
            
            # Create mock workflow steps
            mock_workflow_steps = [
                {'step': 'intent_parsing', 'status': 'success', 'agent': 'Orchestrator', 'description': '意图解析完成'},
                {'step': 'file_ingest', 'status': 'success', 'agent': 'FileIngestAgent', 'description': '文件数据加载成功'},
                {'step': 'column_profiling', 'status': 'success', 'agent': 'ColumnProfilingAgent', 'description': '数据列分析完成'},
                {'step': 'code_generation', 'status': 'success', 'agent': 'CodeGenerationAgent', 'description': '分析代码生成成功'},
                {'step': 'execution', 'status': 'success', 'agent': 'ExecutionAgent', 'description': '代码执行完成'}
            ]
            
            return jsonify({
                'success': True,
                'query': query,
                'response': mock_responses['analysis'],
                'generated_code': mock_responses['code'],
                'workflow_steps': mock_workflow_steps,
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
        
        # Check if processing was successful
        if result.get('status') == 'failed':
            api_logger.error(f"Query processing failed for file {file_id}: {result}")
            return jsonify({
                'success': False,
                'error': result.get('error', 'Query processing failed'),
                'error_message': result.get('error_message', result.get('error', 'Unknown error')),
                'error_type': result.get('error_type', 'ProcessingError'),
                'query': query,
                'file_id': file_id,
                'execution_time': result.get('execution_time', 0),
                'workflow_steps': result.get('workflow_steps', result.get('steps', [])),
                'timestamp': result.get('timestamp'),
                'debug_info': {
                    'file_path': file_info.get('file_path'),
                    'file_processed': file_info.get('processed', False),
                    'agent_available': AGENT_AVAILABLE,
                    'orchestrator_ready': orchestrator is not None
                }
            }), 500
        
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


@app.route('/api/query/batch', methods=['POST'])
@log_request_response
def query_batch_data():
    """Handle user queries about multiple files data."""
    try:
        data = request.get_json()
        file_ids = data.get('file_ids', [])
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if not isinstance(file_ids, list) or not file_ids:
            return jsonify({'error': 'No file IDs provided'}), 400
        
        # Validate all file IDs exist
        missing_files = [fid for fid in file_ids if fid not in uploaded_files]
        if missing_files:
            return jsonify({'error': f'Files not found: {missing_files}'}), 404
        
        # Get file info for all files
        file_infos = [uploaded_files[fid] for fid in file_ids]
        file_names = [info['original_name'] for info in file_infos]
        
        if not AGENT_AVAILABLE or not orchestrator:
            # Return mock response for batch query
            mock_responses = generate_mock_batch_response(query, file_infos)
            
            # Create mock workflow steps
            mock_workflow_steps = [
                {'step': 'intent_parsing', 'status': 'success', 'agent': 'Orchestrator', 'description': '意图解析完成'},
                {'step': 'batch_file_ingest', 'status': 'success', 'agent': 'FileIngestAgent', 'description': f'{len(file_ids)}个文件数据加载成功'},
                {'step': 'data_merging', 'status': 'success', 'agent': 'DataMergeAgent', 'description': '数据合并完成'},
                {'step': 'column_profiling', 'status': 'success', 'agent': 'ColumnProfilingAgent', 'description': '批量数据分析完成'},
                {'step': 'code_generation', 'status': 'success', 'agent': 'CodeGenerationAgent', 'description': '分析代码生成成功'},
                {'step': 'execution', 'status': 'success', 'agent': 'ExecutionAgent', 'description': '代码执行完成'}
            ]
            
            return jsonify({
                'success': True,
                'query': query,
                'file_ids': file_ids,
                'file_names': file_names,
                'response': mock_responses['analysis'],
                'generated_code': mock_responses['code'],
                'workflow_steps': mock_workflow_steps,
                'mock_mode': True,
                'execution_time': 4.2,
                'insights': mock_responses.get('insights', []),
                'visualizations': mock_responses.get('visualizations', [])
            })
        
        # Process with real agent system
        async def query_batch_with_agents():
            file_paths = [info['file_path'] for info in file_infos]
            
            # Use orchestrator to process batch query
            result = await orchestrator.process_user_request(
                user_request=f"{query} (分析这{len(file_ids)}个文件: {', '.join(file_names)})",
                file_path=file_paths[0],  # Primary file
                context={
                    'file_ids': file_ids,
                    'file_paths': file_paths,
                    'file_names': file_names,
                    'batch_query': True,
                    'previous_processing': [info.get('process_result') for info in file_infos]
                }
            )
            
            return result
        
        # Run async batch query processing
        result = run_async(query_batch_with_agents())
        
        return jsonify({
            'success': True,
            'query': query,
            'file_ids': file_ids,
            'file_names': file_names,
            'response': result.get('final_result', 'Batch analysis completed'),
            'generated_code': result.get('generated_code', ''),
            'workflow_steps': result.get('steps', []),
            'execution_time': result.get('execution_time', 0),
            'status': result.get('status', 'completed')
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch query failed: {str(e)}'}), 500


def generate_mock_batch_response(query, file_infos):
    """Generate mock responses for batch queries when agent system is not available."""
    query_lower = query.lower()
    file_count = len(file_infos)
    total_size = sum(info['size'] for info in file_infos)
    file_names = [info['original_name'] for info in file_infos]
    
    if '统计' in query_lower or '基本信息' in query_lower:
        return {
            'analysis': f"""📊 批量数据统计分析结果

基于您上传的 {file_count} 个文件：
{chr(10).join([f'• {name}' for name in file_names])}

📋 整体数据概览：
• 文件总数: {file_count} 个
• 总文件大小: {total_size} bytes
• 合并后总行数: {file_count * 1234:,} 行
• 合并后总列数: 8 列（标准化后）
• 数据类型: 4个数值列, 3个文本列, 1个日期列

📈 合并数据统计：
• 销售额: 平均值 ¥{8547 * file_count:,}, 最大值 ¥{25000 * file_count:,}, 最小值 ¥1,200
• 数量: 平均值 {15.6 * file_count:.1f}, 最大值 {100 * file_count}, 最小值 1
• 利润率: 平均值 23.4%, 标准差 8.9%

🔍 数据质量：
• 完整性: 98.7% ({16 * file_count}个缺失值)
• 重复记录: {3 * file_count}条
• 异常值: 在销售额列发现{5 * file_count}个潜在异常值

💡 跨文件洞察：
1. 所有文件的电子产品类别销售额占总销售额的68%
2. 文件间数据一致性良好，格式统一
3. 合并后数据量增加了{file_count}倍，分析更全面
4. 发现跨文件的时间序列模式""",
            
            'code': f'''import pandas as pd
import numpy as np

# 读取并合并多个Excel文件
file_list = {[info['original_name'] for info in file_infos]}
all_data = []

for file in file_list:
    df = pd.read_excel(file)
    df['source_file'] = file  # 添加源文件标识
    all_data.append(df)
    print(f"加载文件: {{file}}, 行数: {{df.shape[0]}}, 列数: {{df.shape[1]}}")

# 合并所有数据
combined_df = pd.concat(all_data, ignore_index=True)
print(f"\\n合并后数据形状: {{combined_df.shape}}")

# 基本统计信息
print("\\n数据类型:")
print(combined_df.dtypes)

# 数值列统计
numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
print("\\n数值列统计:")
print(combined_df[numeric_cols].describe())

# 按源文件分组统计
print("\\n各文件统计:")
print(combined_df.groupby('source_file').size())

# 缺失值检查
print("\\n缺失值统计:")
print(combined_df.isnull().sum())'''
        }
    
    else:
        return {
            'analysis': f"""🤖 批量AI分析回复

针对您的问题："{query}"

基于上传的 {file_count} 个文件：
{chr(10).join([f'• {name}' for name in file_names])}

📊 批量数据概况：
• 文件总数: {file_count} 个
• 总文件大小: {total_size} bytes
• 预计合并后数据量: {file_count * 1234:,} 行

🔍 建议的批量分析方向：
1. 跨文件数据一致性检查
2. 合并数据的统计分析
3. 文件间数据差异对比
4. 批量异常值检测
5. 跨文件趋势分析
6. 综合可视化报告

💡 您可以尝试以下批量分析问题：
• "对比分析这些文件的统计信息"
• "检测所有文件中的异常值"
• "分析文件间的数据差异"
• "生成综合数据报告"

我随时准备为您提供更详细的批量分析！""",
            
            'code': f'''import pandas as pd
import numpy as np

# 批量读取Excel文件
file_list = {[info['original_name'] for info in file_infos]}
file_data = {{}}

for file in file_list:
    df = pd.read_excel(file)
    file_data[file] = df
    print(f"文件 {{file}}: {{df.shape[0]}} 行 x {{df.shape[1]}} 列")

# 显示所有文件的基本信息
print(f"\\n共加载 {{len(file_list)}} 个文件")
print(f"总数据量预估: {{sum([df.shape[0] for df in file_data.values()]):,}} 行")

# 合并所有数据（如果结构一致）
try:
    combined_df = pd.concat([df.assign(source=name) for name, df in file_data.items()], ignore_index=True)
    print(f"\\n数据合并成功: {{combined_df.shape}}")
    print(combined_df.head())
except Exception as e:
    print(f"\\n数据合并失败: {{e}}")
    print("文件结构可能不一致，需要单独分析")'''
        }


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
@log_request_response
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


@app.route('/api/preview/<file_id>', methods=['GET'])
@log_request_response
def preview_file(file_id):
    """Preview uploaded Excel file as HTML."""
    try:
        if file_id not in uploaded_files:
            return jsonify({'error': 'File not found'}), 404
        
        file_info = uploaded_files[file_id]
        file_path = file_info['file_path']
        
        # Get query parameters
        sheet_name = request.args.get('sheet')
        max_rows = min(int(request.args.get('max_rows', 100)), 1000)  # Cap at 1000 rows
        max_cols = min(int(request.args.get('max_cols', 20)), 50)    # Cap at 50 columns
        
        # Convert Excel to HTML
        result = excel_to_html(
            file_path=file_path,
            sheet_name=sheet_name,
            max_rows=max_rows,
            max_cols=max_cols,
            include_styling=True
        )
        
        if result['success']:
            return jsonify({
                'success': True,
                'file_id': file_id,
                'html': result['html'],
                'metadata': result['metadata'],
                'preview_data': result['preview_data']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error'],
                'html': result['html']
            }), 500
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Preview failed: {str(e)}',
            'html': f'<div class="error">Preview error: {str(e)}</div>'
        }), 500


@app.route('/api/preview/<file_id>/sheets', methods=['GET'])
@log_request_response
def get_file_sheets(file_id):
    """Get list of sheets in the Excel file."""
    try:
        if file_id not in uploaded_files:
            return jsonify({'error': 'File not found'}), 404
        
        file_info = uploaded_files[file_id]
        file_path = file_info['file_path']
        
        # Get Excel file information
        result = get_excel_info(file_path)
        
        if result['success']:
            return jsonify({
                'success': True,
                'file_id': file_id,
                'info': result['info']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Failed to get sheet info: {str(e)}'
        }), 500


@app.route('/api/preview/<file_id>/sheet/<sheet_name>', methods=['GET'])
@log_request_response  
def preview_specific_sheet(file_id, sheet_name):
    """Preview specific sheet of Excel file."""
    try:
        if file_id not in uploaded_files:
            return jsonify({'error': 'File not found'}), 404
        
        file_info = uploaded_files[file_id]
        file_path = file_info['file_path']
        
        # Get range parameters
        start_row = int(request.args.get('start_row', 0))
        end_row = request.args.get('end_row')
        if end_row:
            end_row = int(end_row)
        start_col = int(request.args.get('start_col', 0))  
        end_col = request.args.get('end_col')
        if end_col:
            end_col = int(end_col)
        
        # Convert specific sheet range to HTML
        result = excel_sheet_to_html(
            file_path=file_path,
            sheet_name=sheet_name,
            start_row=start_row,
            end_row=end_row,
            start_col=start_col,
            end_col=end_col
        )
        
        if result['success']:
            return jsonify({
                'success': True,
                'file_id': file_id,
                'sheet_name': sheet_name,
                'html': result['html'],
                'metadata': result['metadata']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error'],
                'html': result['html']
            }), 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Sheet preview failed: {str(e)}',
            'html': f'<div class="error">Sheet preview error: {str(e)}</div>'
        }), 500


# ================================
# LLM Configuration API Endpoints
# ================================

@app.route('/api/llm/config', methods=['GET'])
@log_request_response
def get_llm_config():
    """Get current LLM configuration."""
    try:
        current_params = config.get_llm_params()
        available_models = config.get_available_models()
        
        return jsonify({
            'success': True,
            'current_params': current_params,
            'available_models': available_models,
            'parameter_limits': {
                'temperature': {'min': 0.0, 'max': 2.0, 'step': 0.1},
                'max_tokens': {'min': 1, 'max': 4000, 'step': 1},
                'top_p': {'min': 0.0, 'max': 1.0, 'step': 0.01},
                'frequency_penalty': {'min': -2.0, 'max': 2.0, 'step': 0.1},
                'presence_penalty': {'min': -2.0, 'max': 2.0, 'step': 0.1}
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get LLM config: {str(e)}'
        }), 500


@app.route('/api/llm/config', methods=['POST'])
@log_request_response
def update_llm_config():
    """Update LLM configuration."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No configuration data provided'}), 400
        
        # Extract valid parameters
        valid_params = {}
        for key in ['model', 'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', 'stream']:
            if key in data:
                valid_params[key] = data[key]
        
        if not valid_params:
            return jsonify({'error': 'No valid parameters provided'}), 400
        
        # Update configuration
        config.update_llm_params(**valid_params)
        
        # Log the configuration change
        api_logger.info(f"🔧 [Config] LLM parameters updated: {valid_params}")
        
        return jsonify({
            'success': True,
            'message': 'LLM configuration updated successfully',
            'updated_params': valid_params,
            'current_params': config.get_llm_params()
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid parameter value: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to update LLM config: {str(e)}'
        }), 500


@app.route('/api/llm/config/reset', methods=['POST'])
@log_request_response
def reset_llm_config():
    """Reset LLM configuration to default values."""
    try:
        config.reset_llm_params_to_default()
        
        api_logger.info("🔧 [Config] LLM parameters reset to default values")
        
        return jsonify({
            'success': True,
            'message': 'LLM configuration reset to default values',
            'current_params': config.get_llm_params()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to reset LLM config: {str(e)}'
        }), 500


@app.route('/api/llm/models', methods=['GET'])
@log_request_response
def get_available_models():
    """Get list of available models."""
    try:
        models = config.get_available_models()
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get models: {str(e)}'
        }), 500


@app.route('/api/llm/presets', methods=['GET'])
@log_request_response
def get_llm_presets():
    """Get LLM configuration presets."""
    presets = {
        'creative': {
            'name': '创意模式',
            'description': '适合创造性任务，更多变化和创新',
            'params': {
                'temperature': 1.0,
                'top_p': 0.9,
                'frequency_penalty': 0.5,
                'presence_penalty': 0.3,
                'max_tokens': 2000
            }
        },
        'balanced': {
            'name': '平衡模式',
            'description': '适合一般任务，平衡创造性和准确性',
            'params': {
                'temperature': 0.7,
                'top_p': 1.0,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0,
                'max_tokens': 1500
            }
        },
        'precise': {
            'name': '精确模式',
            'description': '适合需要准确答案的任务，更确定性',
            'params': {
                'temperature': 0.3,
                'top_p': 0.8,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0,
                'max_tokens': 1000
            }
        },
        'analytical': {
            'name': '分析模式',
            'description': '适合数据分析和逻辑推理',
            'params': {
                'temperature': 0.1,
                'top_p': 0.95,
                'frequency_penalty': 0.2,
                'presence_penalty': 0.1,
                'max_tokens': 3000
            }
        }
    }
    
    return jsonify({
        'success': True,
        'presets': presets
    })


@app.route('/api/llm/presets/<preset_name>', methods=['POST'])
@log_request_response
def apply_llm_preset(preset_name):
    """Apply a specific LLM configuration preset."""
    try:
        # Get presets
        presets_response = get_llm_presets()
        presets_data = presets_response.get_json()
        
        if not presets_data['success']:
            return jsonify({'error': 'Failed to get presets'}), 500
        
        presets = presets_data['presets']
        
        if preset_name not in presets:
            return jsonify({'error': f'Preset "{preset_name}" not found'}), 404
        
        # Apply preset parameters
        preset_params = presets[preset_name]['params']
        config.update_llm_params(**preset_params)
        
        api_logger.info(f"🔧 [Config] Applied LLM preset '{preset_name}': {preset_params}")
        
        return jsonify({
            'success': True,
            'message': f'Applied preset: {presets[preset_name]["name"]}',
            'preset': presets[preset_name],
            'current_params': config.get_llm_params()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to apply preset: {str(e)}'
        }), 500


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