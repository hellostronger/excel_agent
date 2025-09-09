"""Orchestrator Agent - Core coordinator for all agents in the system."""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.models import Gemini

from ..agents.base import BaseAgent
from ..agents import (
    FileIngestAgent, StructureScanAgent, ColumnProfilingAgent,
    CodeGenerationAgent, ExecutionAgent
)
from ..models.base import RequestType, AgentRequest, AgentResponse, AgentStatus
from ..models.agents import *
from ..utils.config import config
from ..utils.logging import get_logger
from ..utils.siliconflow_client import SiliconFlowClient


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
            description="Main coordinator for all agents, handles intent parsing and workflow orchestration"
        )
        
        # Initialize sub-agents
        self.file_ingest_agent = FileIngestAgent()
        self.structure_scan_agent = StructureScanAgent()
        self.column_profiling_agent = ColumnProfilingAgent()
        self.code_generation_agent = CodeGenerationAgent()
        self.execution_agent = ExecutionAgent()
        
        # Workflow history for optimization
        self.workflow_history = []
        
        # Error tracking for debugging
        self.error_log = []
    
    async def process_user_request(
        self, 
        user_request: str, 
        file_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a user request end-to-end."""
        context = context or {}
        
        try:
            self.logger.info(f"Processing user request: {user_request[:100]}...")
            
            # Parse intent and determine workflow type
            intent_result = await self._parse_intent(user_request, context)
            workflow_type = intent_result['workflow_type']
            
            self.logger.info(f"Determined workflow type: {workflow_type}")
            
            # Execute appropriate workflow
            if workflow_type == WorkflowType.SINGLE_TABLE:
                result = await self._execute_single_table_workflow(
                    user_request, file_path, intent_result
                )
            elif workflow_type == WorkflowType.SINGLE_CELL:
                result = await self._execute_single_cell_workflow(
                    user_request, file_path, intent_result
                )
            elif workflow_type == WorkflowType.MULTI_TABLE:
                result = await self._execute_multi_table_workflow(
                    user_request, file_path, intent_result
                )
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")
            
            # Record workflow execution
            self._record_workflow_execution(workflow_type, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing user request: {e}")
            error_result = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
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
        intent_prompt = f"""
Analyze this user request for Excel data processing and classify it:

User Request: "{user_request}"
Context: {json.dumps(context, indent=2)}

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
        
        async with SiliconFlowClient() as client:
            response = await client.chat_completion(
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=0.3,
                max_tokens=500
            )
        
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
            
            intent_data['workflow_type'] = WorkflowType(workflow_type)
            
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
        intent_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute single-table workflow: File Ingest → Column Profiling → Code Generation → Execution."""
        
        workflow_steps = []
        
        try:
            # Step 1: File Ingest
            self.logger.info("Step 1: File Ingest")
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
            
            # Prepare final result
            return {
                'status': 'success',
                'workflow_type': 'single_table',
                'steps': workflow_steps,
                'final_result': execution_response.result,
                'generated_code': code_response.code,
                'dry_run_plan': code_response.dry_run_plan,
                'output': execution_response.output,
                'result_file': execution_response.result_file,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            workflow_steps.append({"step": "error", "status": "failed", "error": str(e)})
            return self._create_error_result(f"Workflow failed: {e}", str(e), workflow_steps)
    
    async def _execute_single_cell_workflow(
        self,
        user_request: str,
        file_path: str,
        intent_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute single-cell workflow: File Ingest → Profiling → Code Generation (filters) → Execution."""
        
        # Similar to single table but with cell-specific processing
        # For now, delegate to single table workflow with cell-specific context
        intent_result['operation_focus'] = 'cell_specific'
        return await self._execute_single_table_workflow(user_request, file_path, intent_result)
    
    async def _execute_multi_table_workflow(
        self,
        user_request: str,
        file_path: str,
        intent_result: Dict[str, Any]
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
            
            return {
                'status': 'success',
                'workflow_type': 'multi_table',
                'steps': workflow_steps,
                'final_result': execution_response.result,
                'generated_code': code_response.code,
                'output': execution_response.output,
                'result_file': execution_response.result_file,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            workflow_steps.append({"step": "error", "status": "failed", "error": str(e)})
            return self._create_error_result(f"Multi-table workflow failed: {e}", str(e), workflow_steps)
    
    def _create_error_result(
        self, 
        message: str, 
        error_log: str, 
        workflow_steps: List[Dict] = None
    ) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'status': 'failed',
            'error_message': message,
            'error_log': error_log,
            'workflow_steps': workflow_steps or [],
            'timestamp': datetime.now().isoformat()
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