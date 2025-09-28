"""Flask backend for Excel Intelligent Agent System with RAG capabilities."""

import os
import asyncio
import sys
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from flask import Flask, request, jsonify, render_template, g, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
import time
import json

# Import our Excel to HTML utility
from utils.excel_to_html import excel_to_html, get_excel_sheets, excel_sheet_to_html, get_excel_info
from utils.file_manager import file_manager
from utils.relevance_matcher import relevance_matcher
from utils.progress_tracker import progress_tracker, ProgressUpdate

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
app.config['JSON_AS_ASCII'] = False  # 确保JSON响应支持UTF-8编码

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
logger = logging.getLogger('excel_agent.lightweight_processing')

# 确保所有响应都使用UTF-8编码
@app.after_request
def after_request(response):
    """设置响应头确保UTF-8编码"""
    if response.content_type.startswith('application/json'):
        response.content_type = 'application/json; charset=utf-8'
    response.headers['Content-Type'] = response.content_type
    return response


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

# Progress tracking for Server-Sent Events
progress_connections = {}  # request_id -> list of connections

# Store query results for later retrieval
query_results = {}  # request_id -> query result data

# Initialize file recovery at startup
def initialize_file_recovery():
    """Recover uploaded files from persistent storage."""
    global uploaded_files
    try:
        stored_files = file_manager.list_files()
        uploaded_files.update(stored_files)
        print(f"Recovered {len(stored_files)} files from persistent storage")
        
        # Clean up old files (older than 24 hours)
        cleaned_count = file_manager.cleanup_old_files(max_age_hours=24)
        if cleaned_count > 0:
            print(f"Cleaned up {cleaned_count} old files")
            
    except Exception as e:
        print(f"Error during file recovery: {e}")

# Recover files on startup
initialize_file_recovery()


# Progress tracking helper functions
def send_progress_to_client(progress_update: ProgressUpdate):
    """Send progress update to connected clients via SSE."""
    request_id = progress_update.request_id
    if request_id in progress_connections:
        # Convert to JSON
        data = {
            'type': 'progress',
            'request_id': progress_update.request_id,
            'current_step': progress_update.current_step,
            'current_step_name': progress_update.current_step_name,
            'progress_percent': progress_update.progress_percent,
            'status': progress_update.status,
            'message': progress_update.message,
            'agent': progress_update.agent,
            'timestamp': progress_update.timestamp,
            'details': progress_update.details,
            'all_steps': [
                {
                    'step_id': step.step_id,
                    'step_name': step.step_name,
                    'status': step.status,
                    'agent': step.agent,
                    'description': step.description,
                    'started_at': step.started_at,
                    'completed_at': step.completed_at,
                    'progress_percent': step.progress_percent,
                    'details': step.details
                }
                for step in progress_update.all_steps
            ]
        }
        
        # Format as Server-Sent Event
        sse_data = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        
        # Send to all connections for this request
        queues_to_remove = []
        for message_queue in progress_connections[request_id]:
            try:
                message_queue.put(sse_data)
            except Exception as e:
                api_logger.error(f"Failed to send progress to client: {e}")
                queues_to_remove.append(message_queue)
        
        # Remove failed connections
        for queue_obj in queues_to_remove:
            progress_connections[request_id].remove(queue_obj)
        
        # Clean up empty connection lists
        if not progress_connections[request_id]:
            del progress_connections[request_id]


def cleanup_progress_connections():
    """Periodically clean up old connections."""
    # This would be called by a background task
    pass


# Register progress callback
progress_tracker.add_progress_callback(send_progress_to_client)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_info_safe(file_id: str):
    """
    Safely get file info, trying memory cache first, then persistent storage.
    
    Returns:
        file_info dict or None if not found
    """
    global uploaded_files
    
    # Check in-memory cache first
    if file_id in uploaded_files:
        return uploaded_files[file_id]
    
    # Try to recover from persistent storage
    file_info = file_manager.get_file_info(file_id)
    if file_info is not None:
        # Add to in-memory cache
        uploaded_files[file_id] = file_info
        return file_info
    
    return None


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
    except RuntimeError as e:
        # No event loop in current thread, create a new one
        if "There is no current event loop" in str(e):
            return asyncio.run(coro)
        # For other RuntimeErrors, try to create new event loop
        try:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
            asyncio.set_event_loop(None)


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


@app.route('/api/progress/<request_id>')
def progress_stream(request_id):
    """Server-Sent Events stream for progress updates."""
    import queue
    import threading
    
    def event_generator():
        # Create a message queue for this connection
        message_queue = queue.Queue()
        connection_active = {'active': True}
        
        # Add this queue to progress connections
        if request_id not in progress_connections:
            progress_connections[request_id] = []
        progress_connections[request_id].append(message_queue)
        
        try:
            # Send initial connection message
            initial_message = {
                'type': 'connected',
                'request_id': request_id,
                'timestamp': datetime.now().isoformat(),
                'message': '连接已建立，等待进度更新...'
            }
            
            yield f"data: {json.dumps(initial_message, ensure_ascii=False)}\n\n"
            
            # Check for existing progress data
            existing_progress = progress_tracker.get_request_progress(request_id)
            if existing_progress:
                # Send current progress state
                progress_data = {
                    'type': 'progress',
                    'request_id': request_id,
                    'current_step': existing_progress['steps'][existing_progress['current_step_index']].step_id if existing_progress['steps'] else '',
                    'current_step_name': existing_progress['steps'][existing_progress['current_step_index']].step_name if existing_progress['steps'] else '',
                    'progress_percent': existing_progress['progress_percent'],
                    'status': existing_progress['status'],
                    'message': f"当前进度: {existing_progress['progress_percent']}%",
                    'agent': existing_progress['steps'][existing_progress['current_step_index']].agent if existing_progress['steps'] else '',
                    'timestamp': datetime.now().isoformat(),
                    'all_steps': [
                        {
                            'step_id': step.step_id,
                            'step_name': step.step_name,
                            'status': step.status,
                            'agent': step.agent,
                            'description': step.description,
                            'progress_percent': step.progress_percent
                        }
                        for step in existing_progress['steps']
                    ]
                }
                yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"
            
            # Keep connection alive and check for messages
            last_heartbeat = time.time()
            
            while connection_active['active']:
                try:
                    # Check for new messages with timeout
                    try:
                        message_data = message_queue.get(timeout=5.0)
                        yield message_data
                        message_queue.task_done()
                    except queue.Empty:
                        # Send heartbeat if needed
                        current_time = time.time()
                        if current_time - last_heartbeat > 25:  # Every 25 seconds
                            heartbeat = {
                                'type': 'heartbeat',
                                'timestamp': datetime.now().isoformat()
                            }
                            yield f"data: {json.dumps(heartbeat, ensure_ascii=False)}\n\n"
                            last_heartbeat = current_time
                        
                        # Check if request is completed
                        current_progress = progress_tracker.get_request_progress(request_id)
                        if current_progress and current_progress['status'] in ['completed', 'failed']:
                            api_logger.info(f"Progress stream for {request_id} ending - status: {current_progress['status']}")
                            break
                            
                except Exception as e:
                    api_logger.error(f"Error in progress stream for {request_id}: {e}")
                    break
        
        except GeneratorExit:
            # Client disconnected
            api_logger.info(f"Progress stream for {request_id} closed by client")
        except Exception as e:
            api_logger.error(f"SSE connection error for {request_id}: {e}")
        finally:
            # Mark connection as inactive
            connection_active['active'] = False
            
            # Clean up connection
            if request_id in progress_connections:
                if message_queue in progress_connections[request_id]:
                    progress_connections[request_id].remove(message_queue)
                if not progress_connections[request_id]:
                    del progress_connections[request_id]
                    api_logger.info(f"All connections closed for request {request_id}")
    
    # Return Server-Sent Events response
    response = Response(
        event_generator(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Cache-Control'
        }
    )
    
    return response


@app.route('/api/progress/<request_id>/info', methods=['GET'])
@log_request_response
def get_progress_info(request_id):
    """Get current progress information for a request."""
    progress_data = progress_tracker.get_request_progress(request_id)
    
    if progress_data is None:
        return jsonify({'error': 'Request not found or not being tracked'}), 404
    
    # Convert progress data to JSON-serializable format
    response_data = {
        'request_id': request_id,
        'workflow_type': progress_data['workflow_type'],
        'started_at': progress_data['started_at'],
        'status': progress_data['status'],
        'progress_percent': progress_data['progress_percent'],
        'current_step_index': progress_data['current_step_index'],
        'steps': [
            {
                'step_id': step.step_id,
                'step_name': step.step_name,
                'status': step.status,
                'agent': step.agent,
                'description': step.description,
                'started_at': step.started_at,
                'completed_at': step.completed_at,
                'progress_percent': step.progress_percent,
                'details': step.details
            }
            for step in progress_data['steps']
        ]
    }
    
    if 'finished_at' in progress_data:
        response_data['finished_at'] = progress_data['finished_at']
    
    return jsonify({
        'success': True,
        'progress': response_data
    })


@app.route('/api/progress/<request_id>/result', methods=['GET'])
@log_request_response
def get_progress_result(request_id):
    """Get the final result for a completed request."""
    # Check if request exists in progress tracker
    progress_data = progress_tracker.get_request_progress(request_id)

    if progress_data is None:
        return jsonify({'error': 'Request not found'}), 404

    # Check if request is completed
    if progress_data['status'] not in ['completed', 'failed']:
        return jsonify({
            'error': 'Request not yet completed',
            'status': progress_data['status'],
            'progress_percent': progress_data['progress_percent']
        }), 202  # Accepted but not ready

    # Get stored result
    stored_result = query_results.get(request_id)

    if stored_result is None:
        return jsonify({
            'error': 'Result not found',
            'message': 'Request completed but result data is not available'
        }), 404

    return jsonify({
        'success': True,
        'request_id': request_id,
        'status': progress_data['status'],
        'result': stored_result,  # This is what frontend expects
        'finished_at': progress_data.get('finished_at'),
        'progress_percent': progress_data['progress_percent']
    })


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
        
        # Store file persistently
        stored_path = file_manager.store_file(file_id, file_path, file_info)
        file_info['file_path'] = stored_path
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
            
            # Store file persistently
            stored_path = file_manager.store_file(file_id, file_path, file_info)
            file_info['file_path'] = stored_path
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
        file_info = get_file_info_safe(file_id)
        if file_info is None:
            return jsonify({'error': 'File not found'}), 404
        
        if not AGENT_AVAILABLE or not orchestrator:
            # Perform text analysis even in mock mode
            api_logger.info(f"Performing text analysis for file {file_id}")
            text_analysis_success = file_manager.analyze_file_text(file_id, max_rows=1000)
            
            # Return mock response if agent system not available
            response_data = {
                'success': True,
                'file_id': file_id,
                'processed': True,
                'mock_mode': True,
                'steps': [
                    {'step': 'upload', 'status': 'completed', 'message': 'File uploaded'},
                    {'step': 'ingest', 'status': 'completed', 'message': 'File parsed (mock)'},
                    {'step': 'text_analysis', 'status': 'completed' if text_analysis_success else 'skipped', 
                     'message': 'Text analysis completed' if text_analysis_success else 'Text analysis skipped'},
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
            }
            
            # Add text analysis results if available
            if text_analysis_success:
                text_analysis = file_manager.get_text_analysis(file_id)
                if text_analysis:
                    response_data['text_analysis'] = {
                        'total_texts': text_analysis.get('total_texts', 0),
                        'total_words': text_analysis.get('total_words', 0),
                        'unique_word_count': text_analysis.get('unique_word_count', 0),
                        'top_keywords': list(text_analysis.get('top_words', {}).keys())[:10]
                    }
            
            return jsonify(response_data)
        
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
        
        # Perform text analysis for real agent processing
        api_logger.info(f"Performing text analysis for file {file_id}")
        text_analysis_success = file_manager.analyze_file_text(file_id, max_rows=1000)
        
        # Add text analysis results to the response
        if text_analysis_success:
            text_analysis = file_manager.get_text_analysis(file_id)
            if text_analysis:
                result['text_analysis'] = {
                    'total_texts': text_analysis.get('total_texts', 0),
                    'total_words': text_analysis.get('total_words', 0),
                    'unique_word_count': text_analysis.get('unique_word_count', 0),
                    'top_keywords': list(text_analysis.get('top_words', {}).keys())[:10]
                }
        
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


@app.route('/api/text-analysis/<file_id>', methods=['GET'])
@log_request_response
def get_text_analysis(file_id):
    """Get text analysis results for a file."""
    try:
        file_info = get_file_info_safe(file_id)
        if file_info is None:
            return jsonify({'error': 'File not found'}), 404
        
        text_analysis = file_manager.get_text_analysis(file_id)
        if text_analysis is None:
            return jsonify({'error': 'Text analysis not available for this file'}), 404
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'text_analysis': text_analysis
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get text analysis: {str(e)}'}), 500


@app.route('/api/search/keywords', methods=['POST'])
@log_request_response
def search_files_by_keywords():
    """Search files by keywords in their text content."""
    try:
        data = request.get_json()
        keywords = data.get('keywords', [])
        match_any = data.get('match_any', True)
        
        if not keywords:
            return jsonify({'error': 'Keywords are required'}), 400
        
        matching_files = file_manager.search_files_by_keywords(keywords, match_any)
        
        return jsonify({
            'success': True,
            'keywords': keywords,
            'match_any': match_any,
            'matching_files': matching_files,
            'total_matches': len(matching_files)
        })
        
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500


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
    """Handle user queries about single or multiple files."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # 统一处理文件ID - 支持单文件和多文件
        file_ids = []
        if 'file_ids' in data:
            # 多文件模式
            file_ids = data.get('file_ids', [])
            if not isinstance(file_ids, list) or not file_ids:
                return jsonify({'error': 'No file IDs provided'}), 400
        elif 'file_id' in data:
            # 单文件模式（向后兼容）
            file_id = data.get('file_id')
            if file_id:
                file_ids = [file_id]
        
        if not file_ids:
            return jsonify({'error': 'No file ID(s) provided'}), 400
        
        # 检查是否为批量查询
        is_batch_query = len(file_ids) > 1
        
        # Generate request ID for progress tracking
        query_request_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(query) % 10000}"
        
        # Start progress tracking
        progress_tracker.start_request(query_request_id, "single_table")
        
        # Get file info safely with automatic recovery
        file_info = get_file_info_safe(file_id)
        if file_info is None:
            progress_tracker.finish_request(query_request_id, "failed", "文件未找到")
            return jsonify({
                'error': 'File not found',
                'message': f'File ID {file_id} does not exist or has been cleaned up',
                'suggestion': 'Please re-upload your file and try again',
                'request_id': query_request_id
            }), 404
        
        # 智能相关性判断 - 先通过分词匹配
        progress_tracker.update_step(query_request_id, "relevance_analysis", "in_progress", "正在分析查询相关性...")
        api_logger.info(f"开始相关性分析: {query}")
        text_analysis = file_manager.get_text_analysis(file_id)
        
        relevance_result = None
        enhanced_query = query
        
        if text_analysis:
            try:
                # 使用相关性匹配器分析查询
                relevance_result = relevance_matcher.match_query_to_sheets(query, text_analysis)
                api_logger.info(f"相关性分析结果: {relevance_matcher.get_relevance_summary(relevance_result)}")
                
                # 如果找到关键词匹配，增强查询上下文
                if relevance_result.is_relevant and relevance_result.method == 'keyword_match':
                    enhanced_query = relevance_matcher.enhance_query_with_context(
                        query, 
                        relevance_result.matched_sheets, 
                        relevance_result.matched_keywords
                    )
                    api_logger.info(f"查询已增强: {enhanced_query}")
                
                progress_tracker.update_step(query_request_id, "relevance_analysis", "completed", 
                                           f"相关性分析完成 - {relevance_matcher.get_relevance_summary(relevance_result)}")
                
                # 检查是否可以使用轻量级处理模式（通过markdown内容直接回答）
                lightweight_result = check_lightweight_processing(file_id, relevance_result, query_request_id, query)
                if lightweight_result:
                    api_logger.info("使用轻量级处理模式，直接基于markdown内容回答")

                    # Fix response format for frontend compatibility
                    if 'response' in lightweight_result and isinstance(lightweight_result['response'], dict):
                        if 'analysis' in lightweight_result['response']:
                            # Extract the actual text from nested response structure
                            lightweight_result['response'] = lightweight_result['response']['analysis']

                    # Store result for later retrieval via /result endpoint
                    query_results[query_request_id] = lightweight_result
                    return jsonify(lightweight_result)
                
            except Exception as e:
                api_logger.error(f"相关性分析失败: {e}")
                relevance_result = None
                progress_tracker.update_step(query_request_id, "relevance_analysis", "warning", 
                                           f"相关性分析失败: {str(e)}")
        else:
            api_logger.warning(f"文件 {file_id} 没有文本分析数据，跳过相关性匹配")
            progress_tracker.update_step(query_request_id, "relevance_analysis", "completed", 
                                       "无文本分析数据，跳过相关性匹配")

        if not AGENT_AVAILABLE or not orchestrator:
            # Return mock response if agent system not available
            # Simulate workflow steps with progress updates
            progress_tracker.update_step(query_request_id, "intent_parsing", "in_progress", "解析查询意图...")
            time.sleep(0.5)  # Simulate processing time
            progress_tracker.update_step(query_request_id, "intent_parsing", "completed", "意图解析完成")
            
            progress_tracker.update_step(query_request_id, "file_ingest", "in_progress", "加载文件数据...")
            time.sleep(0.5)
            progress_tracker.update_step(query_request_id, "file_ingest", "completed", "文件数据加载成功")
            
            progress_tracker.update_step(query_request_id, "column_profiling", "in_progress", "分析数据列信息...")
            time.sleep(0.5)
            progress_tracker.update_step(query_request_id, "column_profiling", "completed", "数据列分析完成")
            
            progress_tracker.update_step(query_request_id, "code_generation", "in_progress", "生成分析代码...")
            mock_responses = generate_mock_response(enhanced_query, file_info)
            time.sleep(0.5)
            progress_tracker.update_step(query_request_id, "code_generation", "completed", "分析代码生成成功")
            
            progress_tracker.update_step(query_request_id, "execution", "in_progress", "执行分析代码...")
            time.sleep(0.5)
            progress_tracker.update_step(query_request_id, "execution", "completed", "代码执行完成")
            
            progress_tracker.update_step(query_request_id, "response_generation", "in_progress", "生成用户回答...")
            time.sleep(0.5)
            progress_tracker.update_step(query_request_id, "response_generation", "completed", "回答生成完成")
            
            # Create mock workflow steps with relevance info
            mock_workflow_steps = [
                {'step': 'relevance_analysis', 'status': 'success', 'agent': 'RelevanceMatcher', 
                 'description': f'智能相关性分析: {relevance_matcher.get_relevance_summary(relevance_result) if relevance_result else "无文本分析数据"}'},
                {'step': 'intent_parsing', 'status': 'success', 'agent': 'Orchestrator', 'description': '意图解析完成'},
                {'step': 'file_ingest', 'status': 'success', 'agent': 'FileIngestAgent', 'description': '文件数据加载成功'},
                {'step': 'column_profiling', 'status': 'success', 'agent': 'ColumnProfilingAgent', 'description': '数据列分析完成'},
                {'step': 'code_generation', 'status': 'success', 'agent': 'CodeGenerationAgent', 'description': '分析代码生成成功'},
                {'step': 'execution', 'status': 'success', 'agent': 'ExecutionAgent', 'description': '代码执行完成'}
            ]
            
            progress_tracker.finish_request(query_request_id, "completed", "分析完成！")
            
            response_data = {
                'success': True,
                'request_id': query_request_id,
                'query': query,
                'enhanced_query': enhanced_query if enhanced_query != query else None,
                'response': mock_responses['analysis'],
                'generated_code': mock_responses['code'],
                'workflow_steps': mock_workflow_steps,
                'mock_mode': True,
                'execution_time': 2.5,
                'insights': mock_responses.get('insights', []),
                'visualizations': mock_responses.get('visualizations', [])
            }
            
            # 添加相关性分析结果
            if relevance_result:
                response_data['relevance_analysis'] = {
                    'is_relevant': relevance_result.is_relevant,
                    'confidence_score': relevance_result.confidence_score,
                    'matched_sheets': relevance_result.matched_sheets,
                    'matched_keywords': relevance_result.matched_keywords,
                    'method': relevance_result.method,
                    'summary': relevance_matcher.get_relevance_summary(relevance_result)
                }
            
            return jsonify(response_data)
        
        # Process with real agent system
        async def query_with_agents():
            file_path = file_info['file_path']
            
            # 决定是否需要LLM进一步判断相关性
            actual_query = enhanced_query
            context = {
                'file_id': file_id,
                'previous_processing': file_info.get('process_result')
            }
            
            # 添加相关性分析信息到上下文
            if relevance_result:
                context['relevance_analysis'] = {
                    'is_relevant': relevance_result.is_relevant,
                    'confidence_score': relevance_result.confidence_score,
                    'matched_sheets': relevance_result.matched_sheets,
                    'matched_keywords': relevance_result.matched_keywords,
                    'method': relevance_result.method
                }
                
                # 如果关键词匹配失败，可能需要LLM进一步判断
                if relevance_result.method == 'needs_llm_fallback':
                    context['needs_relevance_check'] = True
                    api_logger.info("关键词匹配未找到相关性，将由LLM进行进一步判断")
                elif not relevance_result.is_relevant:
                    # 明确不相关，可以提前返回或让LLM做最终判断
                    context['low_relevance_warning'] = True
                    api_logger.warning("查询与文件内容相关性较低")
            
            # Use orchestrator to process query
            result = await orchestrator.process_user_request(
                user_request=actual_query,
                file_path=file_path,
                context=context,
                progress_tracker=progress_tracker,
                request_id=query_request_id
            )
            
            return result
        
        # Run async query processing
        result = run_async(query_with_agents())
        
        # Check if processing was successful
        if result.get('status') == 'failed':
            progress_tracker.finish_request(query_request_id, "failed", "处理失败")
            api_logger.error(f"Query processing failed for file {file_id}: {result}")
            return jsonify({
                'success': False,
                'request_id': query_request_id,
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
        
        # Mark as completed
        progress_tracker.finish_request(query_request_id, "completed", "分析完成！")
        
        # Handle different response types from orchestrator
        response_data = {
            'success': True,
            'request_id': query_request_id,
            'query': query,
            'enhanced_query': enhanced_query if enhanced_query != query else None,
            'execution_time': result.get('execution_time', 0),
            'status': result.get('status', 'completed')
        }
        
        # 添加相关性分析结果到真实响应
        if relevance_result:
            response_data['relevance_analysis'] = {
                'is_relevant': relevance_result.is_relevant,
                'confidence_score': relevance_result.confidence_score,
                'matched_sheets': relevance_result.matched_sheets,
                'matched_keywords': relevance_result.matched_keywords,
                'method': relevance_result.method,
                'summary': relevance_matcher.get_relevance_summary(relevance_result)
            }
        
        # Check if this is a non-Excel request response
        if result.get('response_type') == 'general_llm_response':
            # Non-Excel request: use 'answer' field and include additional metadata
            response_data.update({
                'response': result.get('answer', 'No response generated'),
                'generated_code': '',  # Non-Excel requests don't generate code
                'workflow_steps': [],  # Non-Excel requests don't have workflow steps
                'excel_data_used': result.get('excel_data_used', False),
                'file_processed': result.get('file_processed', False),
                'note': result.get('note', ''),
                'relevance_analysis': result.get('relevance_analysis', {}),
                'response_type': 'general_llm_response'
            })
        else:
            # Excel request: use existing field mapping
            response_data.update({
                'response': result.get('final_result', result.get('user_response', 'Analysis completed')),
                'generated_code': result.get('generated_code', ''),
                'workflow_steps': result.get('steps', []),
                'excel_data_used': result.get('excel_data_used', True),
                'file_processed': result.get('file_processed', True),
                'note': result.get('note', ''),
                'response_type': 'excel_data_processing'
            })
            
            # Include additional Excel workflow data if available
            if 'user_response' in result:
                response_data['user_response'] = result['user_response']
            if 'response_summary' in result:
                response_data['response_summary'] = result['response_summary']
            if 'recommendations' in result:
                response_data['recommendations'] = result['recommendations']
            if 'technical_details' in result:
                response_data['technical_details'] = result['technical_details']

        # Fix response format for frontend compatibility
        if 'response' in response_data and isinstance(response_data['response'], dict):
            if 'analysis' in response_data['response']:
                # Extract the actual text from nested response structure
                response_data['response'] = response_data['response']['analysis']

        # Store result for later retrieval via /result endpoint
        query_results[query_request_id] = response_data

        return jsonify(response_data)
        
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
            
            # 并发进行多文件相关性检测
            import asyncio
            from backend.utils.relevance_matcher import relevance_matcher
            from backend.utils.file_manager import file_manager
            
            batch_relevance_results = {}
            
            async def check_file_relevance(file_id, file_name):
                """检查单个文件的相关性"""
                try:
                    api_logger.info(f"开始检查文件 {file_name} ({file_id}) 的相关性")
                    text_analysis = file_manager.get_text_analysis(file_id)
                    
                    if text_analysis:
                        relevance_result = relevance_matcher.match_query_to_sheets(query, text_analysis)
                        api_logger.info(f"文件 {file_name} 相关性分析结果: {relevance_matcher.get_relevance_summary(relevance_result)}")
                        return file_id, relevance_result
                    else:
                        api_logger.warning(f"文件 {file_name} ({file_id}) 没有文本分析数据")
                        return file_id, None
                except Exception as e:
                    api_logger.error(f"文件 {file_name} ({file_id}) 相关性检测失败: {e}")
                    return file_id, None
            
            # 并发检查所有文件的相关性
            api_logger.info(f"开始并发检查 {len(file_ids)} 个文件的相关性")
            tasks = [check_file_relevance(file_id, file_infos[i]['original_name']) for i, file_id in enumerate(file_ids)]
            relevance_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理相关性检测结果
            relevant_files = []
            all_matched_sheets = []
            all_matched_keywords = []
            
            for result in relevance_results:
                if isinstance(result, Exception):
                    api_logger.error(f"相关性检测任务失败: {result}")
                    continue
                    
                file_id, relevance_result = result
                batch_relevance_results[file_id] = relevance_result
                
                if relevance_result and relevance_result.is_relevant:
                    relevant_files.append(file_id)
                    all_matched_sheets.extend(relevance_result.matched_sheets)
                    all_matched_keywords.extend(relevance_result.matched_keywords)
            
            # 去重和统计
            unique_matched_sheets = list(set(all_matched_sheets))
            unique_matched_keywords = list(set(all_matched_keywords))
            
            api_logger.info(f"批量相关性检测完成: {len(relevant_files)}/{len(file_ids)} 个文件相关")
            api_logger.info(f"匹配工作表: {unique_matched_sheets[:5]}...")  # 显示前5个
            api_logger.info(f"匹配关键词: {unique_matched_keywords[:10]}...")  # 显示前10个
            
            # 如果有相关文件，优化查询上下文
            enhanced_query = query
            if relevant_files and unique_matched_keywords:
                context_parts = []
                if unique_matched_sheets:
                    sheets_str = "、".join(unique_matched_sheets[:5])
                    context_parts.append(f"重点关注工作表：{sheets_str}")
                if unique_matched_keywords:
                    keywords_str = "、".join(unique_matched_keywords[:8])
                    context_parts.append(f"关键词：{keywords_str}")
                
                if context_parts:
                    enhanced_query = f"{query}\n\n上下文提示：{' | '.join(context_parts)}"
                    api_logger.info(f"批量查询已增强: {enhanced_query}")
            
            # Use orchestrator to process batch query with enhanced context
            result = await orchestrator.process_user_request(
                user_request=enhanced_query,
                file_path=file_paths[0],  # Primary file
                context={
                    'file_ids': file_ids,
                    'file_paths': file_paths,
                    'file_names': file_names,
                    'batch_query': True,
                    'previous_processing': [info.get('process_result') for info in file_infos],
                    'batch_relevance_analysis': {
                        'total_files': len(file_ids),
                        'relevant_files': relevant_files,
                        'relevance_results': batch_relevance_results,
                        'matched_sheets': unique_matched_sheets,
                        'matched_keywords': unique_matched_keywords,
                        'relevance_count': len(relevant_files)
                    }
                }
            )
            
            return result
        
        # Run async batch query processing
        result = run_async(query_batch_with_agents())
        
        # 提取批量相关性分析结果
        batch_relevance = result.get('context', {}).get('batch_relevance_analysis', {})
        
        response_data = {
            'success': True,
            'query': query,
            'file_ids': file_ids,
            'file_names': file_names,
            'response': result.get('final_result', 'Batch analysis completed'),
            'generated_code': result.get('generated_code', ''),
            'workflow_steps': result.get('steps', []),
            'execution_time': result.get('execution_time', 0),
            'status': result.get('status', 'completed')
        }
        
        # 添加批量相关性分析结果
        if batch_relevance:
            response_data['batch_relevance_analysis'] = {
                'total_files': batch_relevance.get('total_files', len(file_ids)),
                'relevant_files_count': batch_relevance.get('relevance_count', 0),
                'relevant_file_ids': batch_relevance.get('relevant_files', []),
                'matched_sheets': batch_relevance.get('matched_sheets', []),
                'matched_keywords': batch_relevance.get('matched_keywords', []),
                'summary': f"{batch_relevance.get('relevance_count', 0)}/{batch_relevance.get('total_files', len(file_ids))} 个文件与查询相关"
            }
            
            # 为每个文件添加相关性详情
            file_relevance_details = {}
            relevance_results = batch_relevance.get('relevance_results', {})
            for file_id in file_ids:
                relevance_result = relevance_results.get(file_id)
                if relevance_result:
                    file_relevance_details[file_id] = {
                        'is_relevant': relevance_result.is_relevant,
                        'confidence_score': relevance_result.confidence_score,
                        'matched_sheets': relevance_result.matched_sheets,
                        'matched_keywords': relevance_result.matched_keywords,
                        'method': relevance_result.method
                    }
                else:
                    file_relevance_details[file_id] = {
                        'is_relevant': False,
                        'confidence_score': 0.0,
                        'matched_sheets': [],
                        'matched_keywords': [],
                        'method': 'no_analysis'
                    }
            
            response_data['file_relevance_details'] = file_relevance_details
        
        return jsonify(response_data)
        
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


def extract_sheet_markdown_content(file_id: str, sheet_names: List[str], full_content: bool = False) -> Optional[str]:
    """
    提取指定工作表的markdown内容
    
    Args:
        file_id: 文件ID
        sheet_names: 工作表名称列表
        
    Returns:
        合并后的markdown内容，如果失败则返回None
    """
    try:
        if full_content:
            # 需要完整内容时，重新生成不截断的markdown
            from backend.utils.markdown_converter import MarkdownConverter
            file_info = file_manager.get_file_info(file_id)
            if not file_info or 'file_path' not in file_info:
                logger.warning(f"文件 {file_id} 信息不完整")
                return None

            # 创建支持完整预览的markdown转换器
            converter = MarkdownConverter(preview_rows=10000)  # 设置很大的preview_rows
            result = converter.convert_file_to_markdown(file_info['file_path'], file_id)
            full_markdown = result.get('markdown_content', '')
        else:
            # 使用已缓存的markdown内容（可能截断）
            full_markdown = file_manager.get_markdown_content(file_id)

        if not full_markdown:
            logger.warning(f"文件 {file_id} 没有markdown内容")
            return None
        
        # 如果没有指定特定工作表，返回完整内容
        if not sheet_names:
            return full_markdown
        
        # 按工作表分割markdown内容
        sheet_contents = []
        lines = full_markdown.split('\n')
        current_sheet = None
        current_content = []
        
        for line in lines:
            # 检查是否是工作表标题行 (## Sheet_Name)
            if line.startswith('## ') and len(line) > 3:
                # 保存之前的工作表内容
                if current_sheet and current_sheet in sheet_names:
                    sheet_contents.append('\n'.join(current_content))
                
                # 开始新的工作表
                current_sheet = line[3:].strip()
                current_content = [line]
            else:
                # 添加内容到当前工作表
                current_content.append(line)
        
        # 处理最后一个工作表
        if current_sheet and current_sheet in sheet_names:
            sheet_contents.append('\n'.join(current_content))
        
        if sheet_contents:
            return '\n\n---\n\n'.join(sheet_contents)
        else:
            # 如果没有找到匹配的工作表，返回完整内容的一部分
            logger.warning(f"未找到指定工作表 {sheet_names}，返回完整内容")
            return full_markdown[:20000]  # 限制在20k字符内
            
    except Exception as e:
        logger.error(f"提取工作表markdown内容失败: {e}")
        return None


def check_lightweight_processing(file_id: str, relevance_result, request_id: str, query: str = "") -> Optional[Dict[str, Any]]:
    """
    检查是否可以使用轻量级处理模式（基于markdown内容直接回答）
    
    Args:
        file_id: 文件ID
        relevance_result: 相关性匹配结果
        request_id: 请求ID
        
    Returns:
        如果可以使用轻量级模式，返回响应数据；否则返回None
    """
    # 只有在明确匹配到特定工作表时才使用轻量级模式
    if not (relevance_result and relevance_result.is_relevant and 
            relevance_result.matched_sheets and 
            relevance_result.method == 'keyword_match'):
        return None
    
    try:
        # 首先检查存储的完整markdown长度，判断是否应该使用轻量级模式
        from backend.utils.file_manager import file_manager
        text_analysis = file_manager.get_text_analysis(file_id)

        if text_analysis and 'full_markdown_lengths' in text_analysis:
            # 计算匹配工作表的总长度
            full_lengths = text_analysis['full_markdown_lengths']
            total_length = sum(full_lengths.get(sheet, 0) for sheet in relevance_result.matched_sheets)

            # 如果完整内容超过2万字符，不使用轻量级模式
            if total_length > 20000:
                logger.info(f"匹配工作表完整内容过长 ({total_length} 字符)，回退到完整处理模式")
                return None

            logger.info(f"匹配工作表完整内容长度: {total_length} 字符，可以使用轻量级模式")

        # 提取匹配工作表的完整markdown内容（不截断）
        progress_tracker.update_step(request_id, "lightweight_processing", "in_progress", "提取匹配工作表完整内容...")

        sheet_markdown = extract_sheet_markdown_content(file_id, relevance_result.matched_sheets, full_content=True)
        if not sheet_markdown:
            logger.warning("无法提取工作表markdown内容，回退到完整处理模式")
            return None
        
        logger.info(f"提取到 {len(relevance_result.matched_sheets)} 个工作表的内容，共 {len(sheet_markdown)} 字符")
        
        # 使用汇总agent处理
        progress_tracker.update_step(request_id, "lightweight_processing", "in_progress", "使用汇总代理处理查询...")
        
        if AGENT_AVAILABLE:
            # 使用真实的汇总agent
            async def process_with_summarization_agent():
                from excel_agent.agents.summarization import SummarizationAgent
                from excel_agent.models.agents import SummarizationRequest
                from excel_agent.models.base import AgentStatus
                
                try:
                    summarization_agent = SummarizationAgent()
                    
                    # 构造汇总请求
                    summarization_request = SummarizationRequest(
                        agent_id="SummarizationAgent",
                        text_content=sheet_markdown,
                        query=query,
                        context={
                            'file_id': file_id,
                            'matched_sheets': relevance_result.matched_sheets,
                            'matched_keywords': relevance_result.matched_keywords,
                            'processing_mode': 'lightweight_markdown'
                        }
                    )
                    
                    async with summarization_agent:
                        response = await summarization_agent.execute_with_timeout(summarization_request)
                    
                    if response.status == AgentStatus.SUCCESS:
                        return {
                            'success': True,
                            'request_id': request_id,
                            'query': query,
                            'response': response.result,
                            'processing_mode': 'lightweight_markdown',
                            'matched_sheets': relevance_result.matched_sheets,
                            'matched_keywords': relevance_result.matched_keywords,
                            'content_length': len(sheet_markdown),
                            'execution_time': 0.5,
                            'workflow_steps': [
                                {'step': 'relevance_analysis', 'status': 'success', 'agent': 'RelevanceMatcher'},
                                {'step': 'content_extraction', 'status': 'success', 'agent': 'MarkdownExtractor'},
                                {'step': 'summarization', 'status': 'success', 'agent': 'SummarizationAgent'}
                            ],
                            'relevance_analysis': {
                                'is_relevant': relevance_result.is_relevant,
                                'confidence_score': relevance_result.confidence_score,
                                'matched_sheets': relevance_result.matched_sheets,
                                'matched_keywords': relevance_result.matched_keywords,
                                'method': relevance_result.method
                            }
                        }
                    else:
                        logger.error(f"汇总代理执行失败: {response.error_log}")
                        return None
                        
                except Exception as e:
                    logger.error(f"汇总代理处理失败: {e}")
                    return None
            
            result = run_async(process_with_summarization_agent())
            if result:
                progress_tracker.update_step(request_id, "lightweight_processing", "completed", "轻量级处理完成")
                progress_tracker.finish_request(request_id, "completed", "查询处理完成！")
                return result
        
        # 如果agent不可用，返回mock响应
        progress_tracker.update_step(request_id, "lightweight_processing", "completed", "轻量级处理完成（模拟模式）")
        progress_tracker.finish_request(request_id, "completed", "查询处理完成！")
        
        return {
            'success': True,
            'request_id': request_id,
            'query': query,
            'response': f"""基于匹配工作表的分析结果：

匹配的工作表：{', '.join(relevance_result.matched_sheets)}
匹配的关键词：{', '.join(relevance_result.matched_keywords)}

根据提取的内容（{len(sheet_markdown)} 字符），这是一个关于 {', '.join(relevance_result.matched_keywords)} 的查询。

内容摘要：
{sheet_markdown[:500]}...

由于系统当前运行在演示模式下，这是一个模拟回答。实际部署时会使用AI汇总代理提供更详细的分析。""",
            'processing_mode': 'lightweight_markdown_mock',
            'matched_sheets': relevance_result.matched_sheets,
            'matched_keywords': relevance_result.matched_keywords,
            'content_length': len(sheet_markdown),
            'mock_mode': True,
            'execution_time': 0.3,
            'workflow_steps': [
                {'step': 'relevance_analysis', 'status': 'success', 'agent': 'RelevanceMatcher'},
                {'step': 'content_extraction', 'status': 'success', 'agent': 'MarkdownExtractor'},
                {'step': 'summarization', 'status': 'success', 'agent': 'SummarizationAgent (Mock)'}
            ],
            'relevance_analysis': {
                'is_relevant': relevance_result.is_relevant,
                'confidence_score': relevance_result.confidence_score,
                'matched_sheets': relevance_result.matched_sheets,
                'matched_keywords': relevance_result.matched_keywords,
                'method': relevance_result.method
            }
        }
        
    except Exception as e:
        logger.error(f"轻量级处理失败: {e}")
        progress_tracker.update_step(request_id, "lightweight_processing", "failed", f"轻量级处理失败: {str(e)}")
        return None


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


@app.route('/api/files/storage-stats')
@log_request_response
def get_storage_stats():
    """Get file storage statistics."""
    try:
        stats = file_manager.get_storage_stats()
        return jsonify({
            'success': True,
            'stats': stats,
            'memory_cache_count': len(uploaded_files),
            'persistent_storage_count': stats['file_count']
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get storage stats: {str(e)}'}), 500


@app.route('/api/files')
def list_files():
    """List all uploaded files."""
    # Ensure we have the latest files from persistent storage
    try:
        stored_files = file_manager.list_files()
        uploaded_files.update(stored_files)
    except Exception as e:
        print(f"Warning: Failed to sync files from storage: {e}")
    
    files = []
    for file_id, file_info in uploaded_files.items():
        files.append({
            'file_id': file_id,
            'filename': file_info['original_name'],
            'size': file_info['size'],
            'upload_time': file_info['upload_time'],
            'processed': file_info['processed'],
            'available_formats': file_manager.get_available_formats(file_id),
            'has_markdown': file_manager.has_markdown(file_id),
            'has_html': file_manager.has_html(file_id),
            'format_conversions': file_info.get('format_conversions', {}),
            'markdown_converted_at': file_info.get('markdown_converted_at'),
            'html_converted_at': file_info.get('html_converted_at'),
            'has_text_analysis': file_manager.get_text_analysis(file_id) is not None,
            'text_analyzed_at': file_info.get('text_analyzed_at')
        })
    
    return jsonify({'files': files})


@app.route('/api/files/<file_id>', methods=['DELETE'])
@log_request_response
def delete_file(file_id):
    """Delete uploaded file."""
    try:
        file_info = get_file_info_safe(file_id)
        if file_info is None:
            return jsonify({'error': 'File not found'}), 404
        
        # Remove from persistent storage (this also deletes physical files)
        file_manager.remove_file(file_id)
        
        # Remove from memory cache
        if file_id in uploaded_files:
            del uploaded_files[file_id]
        
        return jsonify({'success': True, 'message': 'File deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Delete failed: {str(e)}'}), 500


@app.route('/api/files/<file_id>/formats', methods=['GET'])
@log_request_response
def get_file_formats(file_id):
    """Get available formats for a file."""
    try:
        file_info = get_file_info_safe(file_id)
        if file_info is None:
            return jsonify({'error': 'File not found'}), 404
        
        formats = file_manager.get_available_formats(file_id)
        format_details = {}
        
        for format_type in formats:
            if format_type == 'original':
                format_details['original'] = {
                    'available': True,
                    'path': file_info.get('file_path'),
                    'size': file_info.get('file_size'),
                    'type': 'original'
                }
            elif format_type == 'markdown':
                format_details['markdown'] = {
                    'available': file_manager.has_markdown(file_id),
                    'path': file_info.get('markdown_path'),
                    'converted_at': file_info.get('markdown_converted_at'),
                    'type': 'markdown'
                }
            elif format_type == 'html':
                format_details['html'] = {
                    'available': file_manager.has_html(file_id),
                    'path': file_info.get('html_path'),
                    'converted_at': file_info.get('html_converted_at'),
                    'type': 'html'
                }
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'available_formats': formats,
            'format_details': format_details
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get file formats: {str(e)}'}), 500


@app.route('/api/files/<file_id>/content/<format_type>', methods=['GET'])
@log_request_response
def get_file_content_by_format(file_id, format_type):
    """Get file content in a specific format."""
    try:
        file_info = get_file_info_safe(file_id)
        if file_info is None:
            return jsonify({'error': 'File not found'}), 404
        
        available_formats = file_manager.get_available_formats(file_id)
        if format_type not in available_formats:
            return jsonify({'error': f'Format {format_type} not available for this file'}), 404
        
        content = None
        metadata = None
        content_type = 'application/json'
        
        if format_type == 'original':
            # For original files, return file info rather than content
            content = "Original file content (binary data)"
            content_type = 'text/plain'
        elif format_type == 'markdown':
            content = file_manager.get_markdown_content(file_id)
            metadata = file_manager.get_markdown_metadata(file_id)
            content_type = 'text/markdown'
        elif format_type == 'html':
            content = file_manager.get_html_content(file_id)
            metadata = file_manager.get_html_metadata(file_id)
            content_type = 'text/html'
        
        if content is None:
            return jsonify({'error': f'Could not retrieve content for format {format_type}'}), 500
        
        # Return as JSON response with content and metadata
        return jsonify({
            'success': True,
            'file_id': file_id,
            'format': format_type,
            'content': content,
            'metadata': metadata,
            'content_type': content_type
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get file content: {str(e)}'}), 500


@app.route('/api/files/<file_id>/keywords', methods=['GET'])
@log_request_response
def get_file_keywords(file_id):
    """获取文件的关键词和文本分析结果。"""
    try:
        file_info = get_file_info_safe(file_id)
        if file_info is None:
            return jsonify({'error': 'File not found'}), 404
        
        # 获取文本分析结果
        text_analysis = file_manager.get_text_analysis(file_id)
        if not text_analysis:
            return jsonify({'error': 'Text analysis not available for this file'}), 404
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'filename': file_info.get('original_name', 'Unknown'),
            'text_analysis': text_analysis,
            'has_text_analysis': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get keywords: {str(e)}'}), 500


@app.route('/api/files/<file_id>/keywords/detailed', methods=['GET'])
@log_request_response
def get_detailed_keywords(file_id):
    """获取文件的详细关键词分析结果（包含分词信息）。"""
    try:
        file_info = get_file_info_safe(file_id)
        if file_info is None:
            return jsonify({'error': 'File not found'}), 404
        
        file_path = file_info.get('file_path')
        if not file_path:
            return jsonify({'error': 'File path not available'}), 404
        
        # 检查文件是否为Excel文件
        if not file_path.lower().endswith(('.xlsx', '.xls', '.xlsm')):
            return jsonify({'error': 'File is not an Excel file'}), 400
        
        # 执行详细的文本分析
        from utils.text_processor import text_processor
        
        try:
            detailed_result = text_processor.process_excel_file(file_path, max_rows=1000)
            
            return jsonify({
                'success': True,
                'file_id': file_id,
                'filename': file_info.get('original_name', 'Unknown'),
                'detailed_analysis': detailed_result,
                'analysis_timestamp': datetime.now().isoformat()
            })
            
        except Exception as analysis_error:
            return jsonify({
                'error': f'Failed to analyze file: {str(analysis_error)}'
            }), 500
        
    except Exception as e:
        return jsonify({'error': f'Failed to get detailed keywords: {str(e)}'}), 500



@app.route('/api/preview/<file_id>', methods=['GET'])
@log_request_response
def preview_file(file_id):
    """Preview uploaded Excel file as HTML."""
    try:
        file_info = get_file_info_safe(file_id)
        if file_info is None:
            return jsonify({'error': 'File not found'}), 404
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
        file_info = get_file_info_safe(file_id)
        if file_info is None:
            return jsonify({'error': 'File not found'}), 404
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
        print(f"[DEBUG] Previewing sheet '{sheet_name}' for file_id: {file_id}")

        file_info = get_file_info_safe(file_id)
        if file_info is None:
            return jsonify({'error': 'File not found'}), 404
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

        print(f"[DEBUG] Range: rows {start_row}-{end_row}, cols {start_col}-{end_col}")

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
                print("Agent system initialized successfully!")
            else:
                print("Agent system initialization failed, running in mock mode")
        except Exception as e:
            print(f"Agent system initialization error: {e}")
            print("Running in mock mode...")
    else:
        print("Agent system not available, running in mock mode")
    
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)