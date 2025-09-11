"""
Orchestrator Agent - Core coordinator for all agents in the system.

This module provides the main orchestration logic for the Excel Intelligent Agent System,
handling intent parsing, workflow routing, and agent coordination.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.models import Gemini

from ..agents.base import BaseAgent
from ..agents import (
    FileIngestAgent, StructureScanAgent, ColumnProfilingAgent,
    CodeGenerationAgent, ExecutionAgent, ResponseGenerationAgent
)
from ..models.base import RequestType, AgentRequest, AgentResponse, AgentStatus
from ..models.agents import *
from ..utils.config import config
from ..utils.logging import get_logger
from ..utils.siliconflow_client import SiliconFlowClient
from ..mcp.agent_configs import initialize_mcp_system, initialize_all_agent_mcp


class WorkflowType(str, Enum):
    """Types of workflows supported by the orchestrator."""
    SINGLE_TABLE = "single_table"
    SINGLE_CELL = "single_cell"
    MULTI_TABLE = "multi_table"


class Orchestrator(BaseAgent):
    """Main orchestrator agent responsible for intent parsing, agent coordination, and result integration."""
    
    def __init__(self):
        super().__init__(
            name="Orchestrator",
            description="Main coordinator for all agents, handles intent parsing and workflow orchestration",
            mcp_capabilities=["excel_tools", "data_analysis", "file_management", "visualization"]
        )
        
        # Initialize MCP system
        self.mcp_registry = initialize_mcp_system()
        self.mcp_initialized = False
        
        # Initialize sub-agents
        self.file_ingest_agent = FileIngestAgent()
        self.structure_scan_agent = StructureScanAgent()
        self.column_profiling_agent = ColumnProfilingAgent()
        self.code_generation_agent = CodeGenerationAgent()
        self.execution_agent = ExecutionAgent()
        self.response_generation_agent = ResponseGenerationAgent()
        
        # Workflow history for optimization
        self.workflow_history = []
        
        # Error tracking for debugging
        self.error_log = []
    
    async def process_user_request(
        self, 
        user_request: str, 
        file_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        progress_tracker=None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a user request end-to-end."""
        context = context or {}
        
        if request_id is None:
            request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self) % 10000}"
        context['request_id'] = request_id
        
        try:
            self.logger.info(f"🚀 [Orchestrator {request_id}] Starting request processing")
            self.logger.info(f"🚀 [Orchestrator {request_id}] User request: {user_request[:200]}{'...' if len(user_request) > 200 else ''}")
            self.logger.info(f"🚀 [Orchestrator {request_id}] File path: {file_path}")
            self.logger.debug(f"🚀 [Orchestrator {request_id}] Context: {context}")
            
            start_time = datetime.now()
            
            # Initialize MCP system if not already done
            if not self.mcp_initialized:
                self.logger.info(f"🔄 [Orchestrator {request_id}] Initializing MCP system...")
                await self._initialize_mcp_system()
            
            # Step 0: Check if request is Excel-related
            self.logger.info(f"🔍 [Orchestrator {request_id}] Step 0: Checking Excel relevance...")
            if progress_tracker:
                progress_tracker.update_step(request_id, "excel_relevance_check", "in_progress", "检查请求与Excel的相关性...")
            excel_relevance = await self._check_excel_relevance(user_request, file_path, context)
            
            if not excel_relevance['is_excel_related']:
                if progress_tracker:
                    progress_tracker.update_step(request_id, "excel_relevance_check", "completed", "检测到非Excel相关请求")
                self.logger.info(f"💬 [Orchestrator {request_id}] Non-Excel request detected, using direct LLM response")
                return await self._handle_non_excel_request(user_request, excel_relevance, request_id, start_time, progress_tracker)
            
            if progress_tracker:
                progress_tracker.update_step(request_id, "excel_relevance_check", "completed", "确认为Excel相关请求")
            self.logger.info(f"📊 [Orchestrator {request_id}] Excel-related request confirmed, proceeding with workflow")
            
            # Parse intent and determine workflow type
            self.logger.info(f"🧠 [Orchestrator {request_id}] Step 1: Parsing user intent...")
            if progress_tracker:
                progress_tracker.update_step(request_id, "intent_parsing", "in_progress", "解析用户意图和工作流类型...")
            intent_result = await self._parse_intent(user_request, context)
            workflow_type = intent_result['workflow_type']
            
            if progress_tracker:
                progress_tracker.update_step(request_id, "intent_parsing", "completed", f"解析完成 - 工作流类型: {workflow_type}")
            self.logger.info(f"🧠 [Orchestrator {request_id}] Intent parsed - Workflow type: {workflow_type}")
            self.logger.debug(f"🧠 [Orchestrator {request_id}] Intent details: {intent_result}")
            
            # Execute appropriate workflow
            self.logger.info(f"⚙️  [Orchestrator {request_id}] Step 2: Executing {workflow_type} workflow...")
            
            if workflow_type == WorkflowType.SINGLE_TABLE:
                result = await self._execute_single_table_workflow(
                    user_request, file_path, intent_result, request_id, progress_tracker
                )
            elif workflow_type == WorkflowType.SINGLE_CELL:
                result = await self._execute_single_cell_workflow(
                    user_request, file_path, intent_result, request_id, progress_tracker
                )
            elif workflow_type == WorkflowType.MULTI_TABLE:
                result = await self._execute_multi_table_workflow(
                    user_request, file_path, intent_result, request_id, progress_tracker
                )
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")
            
            # Record workflow execution
            self._record_workflow_execution(workflow_type, result)
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            self.logger.info(f"✅ [Orchestrator {request_id}] Request completed successfully in {total_time:.2f}s")
            self.logger.info(f"✅ [Orchestrator {request_id}] Final status: {result.get('status', 'unknown')}")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds() if 'start_time' in locals() else 0
            
            self.logger.error(f"❌ [Orchestrator {request_id}] Request failed after {total_time:.2f}s: {e}")
            self.logger.error(f"❌ [Orchestrator {request_id}] Exception type: {type(e).__name__}")
            
            error_result = {
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now().isoformat(),
                'execution_time': total_time
            }
            self.error_log.append(error_result)
            return error_result
    
    async def _parse_intent(
        self, 
        user_request: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse user intent and determine workflow type."""
        
        # Use LLM to analyze the request
        # Prepare context for JSON serialization
        serializable_context = self._make_json_serializable(context)
        
        intent_prompt = f"""
Analyze this user request for Excel data processing and classify it:

User Request: "{user_request}"
Context: {json.dumps(serializable_context, indent=2)}

Classify the request as one of:
1. SINGLE_TABLE - Operations on one table/sheet (filter, sort, aggregate, analyze)
2. SINGLE_CELL - Operations targeting specific cells or ranges
3. MULTI_TABLE - Operations involving multiple tables/sheets (join, merge, cross-analysis)

Also extract:
- Primary intent (what the user wants to do)
- Target sheets/tables mentioned
- Specific columns or cells mentioned
- Operation type (read, analyze, transform, export, etc.)
- Any special requirements

Respond in JSON format:
{{
    "workflow_type": "SINGLE_TABLE|SINGLE_CELL|MULTI_TABLE",
    "primary_intent": "description of main intent",
    "target_sheets": ["sheet1", "sheet2"],
    "target_columns": ["col1", "col2"],
    "target_cells": ["A1", "B2:C5"],
    "operation_type": "read|analyze|transform|export|aggregate",
    "requirements": ["requirement1", "requirement2"],
    "confidence": 0.9
}}
"""
        
        from ..utils.logging import get_logger
        
        logger = get_logger(__name__)
        request_id = context.get('request_id', 'orchestrator_intent')
        
        # Log request start
        start_time = time.time()
        logger.info(f"🤖 [Orchestrator] Starting intent analysis LLM request {request_id}")
        
        async with SiliconFlowClient() as client:
            try:
                response = await client.chat_completion(
                    messages=[{"role": "user", "content": intent_prompt}],
                    temperature=0.3,
                    request_id=request_id
                )
                
                # Log successful completion
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"🤖 [Orchestrator] Intent analysis LLM request {request_id} completed in {duration:.2f}s")
                
            except Exception as e:
                # Log error
                end_time = time.time()
                duration = end_time - start_time
                logger.error(f"❌ [Orchestrator] Intent analysis LLM request {request_id} failed after {duration:.2f}s: {type(e).__name__}: {e}")
                raise
        
        # Parse the response
        try:
            intent_text = response['choices'][0]['message']['content']
            # Extract JSON from response
            if '```json' in intent_text:
                json_start = intent_text.find('```json') + 7
                json_end = intent_text.find('```', json_start)
                intent_text = intent_text[json_start:json_end]
            
            intent_data = json.loads(intent_text.strip())
            
            # Validate and set default workflow type
            workflow_type = intent_data.get('workflow_type', 'SINGLE_TABLE')
            if workflow_type not in ['SINGLE_TABLE', 'SINGLE_CELL', 'MULTI_TABLE']:
                workflow_type = 'SINGLE_TABLE'
            
            # Convert to lowercase with underscores for enum
            workflow_type_mapping = {
                'SINGLE_TABLE': 'single_table',
                'SINGLE_CELL': 'single_cell', 
                'MULTI_TABLE': 'multi_table'
            }
            enum_value = workflow_type_mapping.get(workflow_type, 'single_table')
            intent_data['workflow_type'] = WorkflowType(enum_value)
            
            return intent_data
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Failed to parse intent response: {e}")
            # Fallback to simple heuristics
            return self._fallback_intent_parsing(user_request)
    
    def _fallback_intent_parsing(self, user_request: str) -> Dict[str, Any]:
        """Fallback intent parsing using simple heuristics."""
        request_lower = user_request.lower()
        
        # Simple keyword-based classification
        if any(word in request_lower for word in ['join', 'merge', 'combine', 'multiple']):
            workflow_type = WorkflowType.MULTI_TABLE
        elif any(word in request_lower for word in ['cell', 'range', 'specific']):
            workflow_type = WorkflowType.SINGLE_CELL
        else:
            workflow_type = WorkflowType.SINGLE_TABLE
        
        return {
            'workflow_type': workflow_type,
            'primary_intent': user_request,
            'target_sheets': [],
            'target_columns': [],
            'target_cells': [],
            'operation_type': 'analyze',
            'requirements': [],
            'confidence': 0.5
        }
    
    async def _execute_single_table_workflow(
        self,
        user_request: str,
        file_path: str,
        intent_result: Dict[str, Any],
        request_id: str = "unknown",
        progress_tracker=None
    ) -> Dict[str, Any]:
        """Execute single-table workflow: File Ingest → Column Profiling → Code Generation → Execution."""
        
        workflow_steps = []
        
        try:
            # Step 1: File Ingest
            self.logger.info(f"📁 [FileIngest {request_id}] Step 1: File Ingest starting...")
            self.logger.info(f"📁 [FileIngest {request_id}] Processing file: {file_path}")
            if progress_tracker:
                progress_tracker.update_step(request_id, "file_ingest", "in_progress", f"正在读取和解析Excel文件...")
            ingest_request = FileIngestRequest(
                agent_id="FileIngestAgent",
                file_path=file_path,
                context=intent_result
            )
            
            async with self.file_ingest_agent:
                ingest_response = await self.file_ingest_agent.execute_with_timeout(ingest_request)
            
            if ingest_response.status != AgentStatus.SUCCESS:
                return self._create_error_result("File ingest failed", ingest_response.error_log)
            
            workflow_steps.append({"step": "file_ingest", "status": "success", "result": ingest_response.result})
            
            # Step 2: Column Profiling (for primary sheet)
            primary_sheet = ingest_response.sheets[0] if ingest_response.sheets else None
            if primary_sheet:
                self.logger.info(f"Step 2: Column Profiling for sheet '{primary_sheet}'")
                
                profiling_request = ColumnProfilingRequest(
                    agent_id="ColumnProfilingAgent",
                    file_id=ingest_response.file_id,
                    sheet_name=primary_sheet,
                    context={"file_metadata": ingest_response.result}
                )
                
                async with self.column_profiling_agent:
                    profiling_response = await self.column_profiling_agent.execute_with_timeout(profiling_request)
                
                if profiling_response.status == AgentStatus.SUCCESS:
                    workflow_steps.append({"step": "column_profiling", "status": "success", "result": profiling_response.result})
                else:
                    workflow_steps.append({"step": "column_profiling", "status": "warning", "error": profiling_response.error_log})
            
            # Step 3: Code Generation
            self.logger.info("Step 3: Code Generation")
            
            code_context = {
                "file_info": ingest_response.result,
                "profiling_info": profiling_response.result if 'profiling_response' in locals() else None,
                "intent": intent_result
            }
            
            code_request = CodeGenerationRequest(
                agent_id="CodeGenerationAgent",
                user_request=user_request,
                context=code_context,
                file_info=ingest_response.result
            )
            
            async with self.code_generation_agent:
                code_response = await self.code_generation_agent.execute_with_timeout(code_request)
            
            if code_response.status != AgentStatus.SUCCESS:
                return self._create_error_result("Code generation failed", code_response.error_log)
            
            workflow_steps.append({"step": "code_generation", "status": "success", "result": code_response.result})
            
            # Step 4: Execution
            self.logger.info("Step 4: Code Execution")
            
            execution_request = ExecutionRequest(
                agent_id="ExecutionAgent",
                file_id=ingest_response.file_id,
                code=code_response.code,
                dry_run=False,
                context=code_context
            )
            
            async with self.execution_agent:
                execution_response = await self.execution_agent.execute_with_timeout(execution_request)
            
            workflow_steps.append({
                "step": "execution", 
                "status": "success" if execution_response.status == AgentStatus.SUCCESS else "failed",
                "result": execution_response.result
            })
            
            # Step 5: Response Generation
            self.logger.info("Step 5: Response Generation")
            
            # Prepare workflow results for response generation
            workflow_results = {
                'status': 'success',
                'workflow_type': 'single_table',
                'steps': workflow_steps,
                'final_result': execution_response.result,
                'generated_code': code_response.code,
                'dry_run_plan': code_response.dry_run_plan,
                'output': execution_response.output,
                'result_file': execution_response.result_file,
                'execution_time': 0,  # Will be calculated later
                'excel_data_used': True,
                'file_processed': True,
                'processed_file': file_path
            }
            
            response_request = ResponseGenerationRequest(
                agent_id="ResponseGenerationAgent",
                user_query=user_request,
                workflow_results=workflow_results,
                file_info=ingest_response.result,
                context=code_context
            )
            
            async with self.response_generation_agent:
                response_generation_result = await self.response_generation_agent.execute_with_timeout(response_request)
            
            workflow_steps.append({
                "step": "response_generation",
                "status": "success" if response_generation_result.status == AgentStatus.SUCCESS else "warning",
                "result": response_generation_result.result if response_generation_result.status == AgentStatus.SUCCESS else None
            })
            
            # Prepare final result with generated response
            final_result = {
                'status': 'success',
                'workflow_type': 'single_table', 
                'steps': workflow_steps,
                'final_result': execution_response.result,
                'generated_code': code_response.code,
                'dry_run_plan': code_response.dry_run_plan,
                'output': execution_response.output,
                'result_file': execution_response.result_file,
                'excel_data_used': True,
                'file_processed': True,
                'processed_file': file_path,
                'timestamp': datetime.now().isoformat(),
                'note': '此回答基于Excel数据处理结果'
            }
            
            # Add response generation results if successful
            if response_generation_result.status == AgentStatus.SUCCESS:
                final_result.update({
                    'user_response': response_generation_result.response,
                    'response_summary': response_generation_result.summary,
                    'recommendations': response_generation_result.recommendations,
                    'technical_details': response_generation_result.technical_details
                })
            else:
                # Fallback response if response generation fails
                final_result.update({
                    'user_response': f"已完成数据处理任务：{user_request}\n\n执行结果：\n{execution_response.output}",
                    'response_summary': '数据处理完成，但响应生成出现问题',
                    'recommendations': ['请检查执行结果', '如有疑问请重新提问'],
                    'technical_details': None
                })
            
            return final_result
            
        except Exception as e:
            workflow_steps.append({"step": "error", "status": "failed", "error": str(e)})
            return self._create_error_result(f"Workflow failed: {e}", str(e), workflow_steps)
    
    async def _check_excel_relevance(
        self,
        user_request: str,
        file_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Check if the user request is related to Excel data processing."""
        
        # Quick heuristics check first
        request_lower = user_request.lower()
        excel_keywords = [
            'excel', 'xlsx', 'xls', 'csv', 'spreadsheet', 'table', 'data',
            'column', 'row', 'cell', 'sheet', 'workbook', 'pivot', 'chart',
            'filter', 'sort', 'aggregate', 'sum', 'count', 'average', 'merge'
        ]
        
        # Check for Excel keywords first (more decisive)
        keyword_matches = [keyword for keyword in excel_keywords if keyword in request_lower]
        if keyword_matches:
            confidence = min(0.8, len(keyword_matches) * 0.2)
            return {
                'is_excel_related': True,
                'confidence': confidence,
                'reason': f'Excel keywords found: {keyword_matches}',
                'method': 'keyword_matching'
            }
        
        # Check for obvious non-Excel questions
        non_excel_patterns = [
            '你是谁', 'who are you', '你好', 'hello', '天气', 'weather', 
            '时间', 'time', '今天', 'today', '新闻', 'news',
            '什么是', 'what is', 'how to', '如何', '为什么', 'why'
        ]
        
        for pattern in non_excel_patterns:
            if pattern in request_lower:
                return {
                    'is_excel_related': False,
                    'confidence': 0.8,
                    'reason': f'General question pattern detected: {pattern}',
                    'method': 'pattern_matching'
                }
        
        # If file_path is provided and is an Excel file, but no Excel keywords and no general patterns,
        # use LLM to make the decision
        file_is_excel = file_path and (file_path.endswith('.xlsx') or file_path.endswith('.xls') or file_path.endswith('.csv'))
        
        # Use LLM for more sophisticated analysis
        serializable_context = self._make_json_serializable(context or {})
        
        relevance_prompt = f"""
Analyze this user request to determine if it's related to Excel, spreadsheet, or data processing tasks:

User Request: "{user_request}"
File Path: {file_path or 'None'}
Excel File Detected: {file_is_excel}
Context: {json.dumps(serializable_context, indent=2)}

Important: Even if an Excel file is provided, the user may ask general questions that are NOT about data processing.

Determine if this request involves:
1. Excel file operations (reading, writing, analyzing Excel files)
2. Spreadsheet data processing (filtering, sorting, calculations)
3. Data analysis tasks that would use the Excel data
4. Table/tabular data operations

Do NOT classify as Excel-related if the request is:
- General conversation ("你是谁", "hello", "how are you")
- Non-data questions ("what is AI", "weather", "time")
- Questions about topics unrelated to the Excel file

Respond in JSON format:
{{
    "is_excel_related": true/false,
    "confidence": 0.0-1.0,
    "reason": "explanation of decision",
    "suggested_category": "excel_data_processing|general_question|other_domain"
}}

Examples:
- "你是谁?" -> false (general conversation, not about data)
- "What's the weather today?" -> false (general question)
- "分析这个表格的数据" -> true (Excel data analysis)
- "How do I calculate sum in column A?" -> true (Excel operation)
- "What is machine learning?" -> false (general knowledge question)
"""
        
        try:
            # Log request start
            logger = get_logger(__name__)
            request_id = context.get('request_id', 'relevance_check') if context else 'relevance_check'
            start_time = time.time()
            logger.info(f"🤖 [Orchestrator] Starting relevance check LLM request {request_id}")
            
            async with SiliconFlowClient() as client:
                try:
                    response = await client.chat_completion(
                        messages=[{"role": "user", "content": relevance_prompt}],
                        temperature=0.2,
                        request_id=request_id
                    )
                    
                    # Log successful completion
                    end_time = time.time()
                    duration = end_time - start_time
                    logger.info(f"🤖 [Orchestrator] Relevance check LLM request {request_id} completed in {duration:.2f}s")
                    
                except Exception as e:
                    # Log error
                    end_time = time.time()
                    duration = end_time - start_time
                    logger.error(f"❌ [Orchestrator] Relevance check LLM request {request_id} failed after {duration:.2f}s: {type(e).__name__}: {e}")
                    raise
            
            # Parse the response
            relevance_text = response['choices'][0]['message']['content']
            if '```json' in relevance_text:
                json_start = relevance_text.find('```json') + 7
                json_end = relevance_text.find('```', json_start)
                relevance_text = relevance_text[json_start:json_end]
            
            relevance_data = json.loads(relevance_text.strip())
            relevance_data['method'] = 'llm_analysis'
            
            return relevance_data
            
        except Exception as e:
            self.logger.warning(f"LLM relevance check failed: {e}, falling back to heuristic analysis")
            # Fallback: if we detected non-Excel patterns, use that; otherwise be conservative
            for pattern in non_excel_patterns:
                if pattern in request_lower:
                    return {
                        'is_excel_related': False,
                        'confidence': 0.6,
                        'reason': f'Fallback: non-Excel pattern detected: {pattern}',
                        'method': 'fallback_pattern'
                    }
            
            # If Excel file is provided and no clear non-Excel patterns, assume Excel-related
            if file_is_excel:
                return {
                    'is_excel_related': True,
                    'confidence': 0.5,
                    'reason': 'Fallback: Excel file provided, assuming data-related',
                    'method': 'fallback_file_based'
                }
            
            # No file, no clear patterns - default to non-Excel
            return {
                'is_excel_related': False,
                'confidence': 0.3,
                'reason': 'Fallback: unclear request, defaulting to general',
                'method': 'fallback_default'
            }
    
    async def _handle_non_excel_request(
        self,
        user_request: str,
        excel_relevance: Dict[str, Any],
        request_id: str,
        start_time: datetime,
        progress_tracker=None
    ) -> Dict[str, Any]:
        """Handle non-Excel related requests with direct LLM response."""
        
        from ..utils.logging import get_logger
        logger = get_logger(__name__)
        
        try:
            # Prepare a general conversation prompt
            general_prompt = f"""
You are a helpful AI assistant. The user has asked a question that doesn't appear to be related to Excel or data processing.

User Request: "{user_request}"

Please provide a helpful, accurate, and informative response to their question. Be concise but thorough.

Important: This response will be marked as not using any Excel data or file processing capabilities.
"""
            
            # Log request start
            llm_start_time = time.time()
            logger.info(f"🤖 [Orchestrator] Starting general response LLM request {request_id}")
            
            async with SiliconFlowClient() as client:
                try:
                    response = await client.chat_completion(
                        messages=[{"role": "user", "content": general_prompt}],
                        temperature=0.7,
                        request_id=request_id
                    )
                    
                    # Log successful completion
                    llm_end_time = time.time()
                    duration = llm_end_time - llm_start_time
                    logger.info(f"🤖 [Orchestrator] General response LLM request {request_id} completed in {duration:.2f}s")
                    
                except Exception as e:
                    # Log error
                    llm_end_time = time.time()
                    duration = llm_end_time - llm_start_time
                    logger.error(f"❌ [Orchestrator] General response LLM request {request_id} failed after {duration:.2f}s: {type(e).__name__}: {e}")
                    raise
            
            llm_response = response['choices'][0]['message']['content']
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            self.logger.info(f"💬 [Orchestrator {request_id}] Non-Excel request completed in {total_time:.2f}s")
            
            return {
                'status': 'success',
                'response_type': 'general_llm_response',
                'answer': llm_response,
                'excel_data_used': False,
                'file_processed': False,
                'relevance_analysis': excel_relevance,
                'execution_time': total_time,
                'timestamp': datetime.now().isoformat(),
                'note': '此回答未引用任何Excel数据或文件处理功能'
            }
            
        except Exception as e:
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            self.logger.error(f"❌ [Orchestrator {request_id}] Non-Excel request failed: {e}")
            
            return {
                'status': 'failed',
                'response_type': 'general_llm_response',
                'error': str(e),
                'excel_data_used': False,
                'file_processed': False,
                'relevance_analysis': excel_relevance,
                'execution_time': total_time,
                'timestamp': datetime.now().isoformat(),
                'note': '此回答未引用任何Excel数据或文件处理功能'
            }
    
    async def _execute_single_cell_workflow(
        self,
        user_request: str,
        file_path: str,
        intent_result: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """Execute single-cell workflow: File Ingest → Profiling → Code Generation (filters) → Execution."""
        
        # Similar to single table but with cell-specific processing
        # For now, delegate to single table workflow with cell-specific context
        intent_result['operation_focus'] = 'cell_specific'
        return await self._execute_single_table_workflow(user_request, file_path, intent_result, request_id)
    
    async def _execute_multi_table_workflow(
        self,
        user_request: str,
        file_path: str,
        intent_result: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """Execute multi-table workflow: File Ingest → Multi-table Profiling → Relation Discovery → Code Generation → Execution."""
        
        # For now, implement basic multi-table workflow
        # This would need relation discovery agent implementation
        workflow_steps = []
        
        try:
            # File ingest (same as single table)
            ingest_request = FileIngestRequest(
                agent_id="FileIngestAgent",
                file_path=file_path,
                context=intent_result
            )
            
            async with self.file_ingest_agent:
                ingest_response = await self.file_ingest_agent.execute_with_timeout(ingest_request)
            
            if ingest_response.status != AgentStatus.SUCCESS:
                return self._create_error_result("File ingest failed", ingest_response.error_log)
            
            workflow_steps.append({"step": "file_ingest", "status": "success", "result": ingest_response.result})
            
            # Profile multiple sheets
            profiling_results = {}
            for sheet_name in ingest_response.sheets:
                profiling_request = ColumnProfilingRequest(
                    agent_id="ColumnProfilingAgent",
                    file_id=ingest_response.file_id,
                    sheet_name=sheet_name,
                    context={"file_metadata": ingest_response.result}
                )
                
                async with self.column_profiling_agent:
                    profiling_response = await self.column_profiling_agent.execute_with_timeout(profiling_request)
                
                if profiling_response.status == AgentStatus.SUCCESS:
                    profiling_results[sheet_name] = profiling_response.result
            
            workflow_steps.append({"step": "multi_sheet_profiling", "status": "success", "result": profiling_results})
            
            # Code generation with multi-table context
            code_context = {
                "file_info": ingest_response.result,
                "multi_sheet_profiling": profiling_results,
                "intent": intent_result,
                "workflow_type": "multi_table"
            }
            
            code_request = CodeGenerationRequest(
                agent_id="CodeGenerationAgent",
                user_request=user_request,
                context=code_context,
                file_info=ingest_response.result
            )
            
            async with self.code_generation_agent:
                code_response = await self.code_generation_agent.execute_with_timeout(code_request)
            
            if code_response.status != AgentStatus.SUCCESS:
                return self._create_error_result("Code generation failed", code_response.error_log)
            
            workflow_steps.append({"step": "code_generation", "status": "success", "result": code_response.result})
            
            # Execute generated code
            execution_request = ExecutionRequest(
                agent_id="ExecutionAgent",
                file_id=ingest_response.file_id,
                code=code_response.code,
                dry_run=False,
                context=code_context
            )
            
            async with self.execution_agent:
                execution_response = await self.execution_agent.execute_with_timeout(execution_request)
            
            workflow_steps.append({
                "step": "execution", 
                "status": "success" if execution_response.status == AgentStatus.SUCCESS else "failed",
                "result": execution_response.result
            })
            
            # Add Response Generation step
            workflow_results = {
                'status': 'success',
                'workflow_type': 'multi_table',
                'steps': workflow_steps,
                'final_result': execution_response.result,
                'generated_code': code_response.code,
                'output': execution_response.output,
                'result_file': execution_response.result_file,
                'execution_time': 0,
                'excel_data_used': True,
                'file_processed': True,
                'processed_file': file_path
            }
            
            response_request = ResponseGenerationRequest(
                agent_id="ResponseGenerationAgent",
                user_query=user_request,
                workflow_results=workflow_results,
                file_info=ingest_response.result,
                context=code_context
            )
            
            async with self.response_generation_agent:
                response_generation_result = await self.response_generation_agent.execute_with_timeout(response_request)
            
            workflow_steps.append({
                "step": "response_generation",
                "status": "success" if response_generation_result.status == AgentStatus.SUCCESS else "warning",
                "result": response_generation_result.result if response_generation_result.status == AgentStatus.SUCCESS else None
            })
            
            final_result = {
                'status': 'success',
                'workflow_type': 'multi_table',
                'steps': workflow_steps,
                'final_result': execution_response.result,
                'generated_code': code_response.code,
                'output': execution_response.output,
                'result_file': execution_response.result_file,
                'excel_data_used': True,
                'file_processed': True,
                'processed_file': file_path,
                'timestamp': datetime.now().isoformat(),
                'note': '此回答基于多表Excel数据处理结果'
            }
            
            # Add response generation results
            if response_generation_result.status == AgentStatus.SUCCESS:
                final_result.update({
                    'user_response': response_generation_result.response,
                    'response_summary': response_generation_result.summary,
                    'recommendations': response_generation_result.recommendations,
                    'technical_details': response_generation_result.technical_details
                })
            else:
                final_result.update({
                    'user_response': f"已完成多表数据处理任务：{user_request}\n\n执行结果：\n{execution_response.output}",
                    'response_summary': '多表数据处理完成，但响应生成出现问题',
                    'recommendations': ['请检查执行结果', '如有疑问请重新提问'],
                    'technical_details': None
                })
            
            return final_result
            
        except Exception as e:
            workflow_steps.append({"step": "error", "status": "failed", "error": str(e)})
            return self._create_error_result(f"Multi-table workflow failed: {e}", str(e), workflow_steps)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    def _create_error_result(
        self, 
        message: str, 
        error_log: str, 
        workflow_steps: List[Dict] = None,
        excel_data_used: bool = True,
        file_processed: bool = False
    ) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'status': 'failed',
            'error_message': message,
            'error_log': error_log,
            'workflow_steps': workflow_steps or [],
            'excel_data_used': excel_data_used,
            'file_processed': file_processed,
            'timestamp': datetime.now().isoformat(),
            'note': '处理Excel数据时出现错误' if excel_data_used else '处理请求时出现错误'
        }
    
    def _record_workflow_execution(
        self, 
        workflow_type: WorkflowType, 
        result: Dict[str, Any]
    ):
        """Record workflow execution for optimization and learning."""
        execution_record = {
            'workflow_type': workflow_type,
            'timestamp': datetime.now().isoformat(),
            'status': result.get('status'),
            'execution_time': result.get('execution_time'),
            'steps_completed': len(result.get('workflow_steps', [])),
            'success': result.get('status') == 'success'
        }
        
        self.workflow_history.append(execution_record)
        
        # Keep only last 100 executions
        if len(self.workflow_history) > 100:
            self.workflow_history = self.workflow_history[-100:]
    
    async def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow execution statistics for optimization."""
        if not self.workflow_history:
            return {"message": "No workflow history available"}
        
        total_executions = len(self.workflow_history)
        successful_executions = sum(1 for record in self.workflow_history if record['success'])
        success_rate = successful_executions / total_executions if total_executions > 0 else 0
        
        workflow_type_stats = {}
        for record in self.workflow_history:
            workflow_type = record['workflow_type']
            if workflow_type not in workflow_type_stats:
                workflow_type_stats[workflow_type] = {'count': 0, 'success': 0}
            workflow_type_stats[workflow_type]['count'] += 1
            if record['success']:
                workflow_type_stats[workflow_type]['success'] += 1
        
        return {
            'total_executions': total_executions,
            'success_rate': success_rate,
            'workflow_type_stats': workflow_type_stats,
            'recent_errors': [
                record for record in self.workflow_history[-10:]
                if not record['success']
            ]
        }
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Base agent interface - not used directly by orchestrator."""
        return self.create_error_response(
            request,
            "Orchestrator should be called via process_user_request method"
        )
    
    async def _initialize_mcp_system(self):
        """Initialize the MCP system for all agents."""
        try:
            self.logger.info("Initializing MCP system...")
            await initialize_all_agent_mcp()
            self.mcp_initialized = True
            self.logger.info("MCP system initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP system: {e}")
            # Continue without MCP capabilities
            self.mcp_initialized = False
    
    async def get_mcp_status(self) -> Dict[str, Any]:
        """Get MCP system status."""
        if not self.mcp_initialized:
            return {"status": "not_initialized", "error": "MCP system not initialized"}
        
        return {
            "status": "initialized",
            "registry_status": self.mcp_registry.get_registry_status(),
            "available_tools": self.mcp_registry.list_available_tools(),
            "agent_configs": len(self.mcp_registry.agent_configs)
        }