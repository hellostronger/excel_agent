"""
Real-time progress tracking utility for Excel Agent System.

This module provides functionality to track and broadcast progress updates
during user request processing, allowing the frontend to display real-time
progress information to users.
"""

import time
import json
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import logging

# Get logger
logger = logging.getLogger(__name__)


@dataclass
class ProgressStep:
    """Represents a single progress step."""
    step_id: str
    step_name: str
    status: str  # 'pending', 'in_progress', 'completed', 'failed', 'warning'
    agent: str = ""
    description: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress_percent: int = 0
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class ProgressUpdate:
    """Represents a progress update message."""
    request_id: str
    current_step: str
    current_step_name: str
    progress_percent: int
    status: str
    message: str
    agent: str = ""
    timestamp: str = ""
    details: Dict[str, Any] = None
    all_steps: List[ProgressStep] = None
    
    def __post_init__(self):
        if self.timestamp == "":
            self.timestamp = datetime.now().isoformat()
        if self.details is None:
            self.details = {}
        if self.all_steps is None:
            self.all_steps = []


class ProgressTracker:
    """
    Tracks and manages progress for user requests.
    Supports real-time progress updates via callbacks.
    """
    
    def __init__(self):
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.progress_callbacks: List[Callable[[ProgressUpdate], None]] = []
        self.step_definitions = self._get_default_step_definitions()
        
    def _get_default_step_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Define default workflow steps and their progress weights."""
        return {
            # Excel relevance check workflow
            "excel_relevance_check": {
                "name": "Excel相关性检查",
                "agent": "Orchestrator",
                "description": "检查用户请求是否与Excel数据处理相关",
                "weight": 5
            },
            
            # Intent parsing
            "intent_parsing": {
                "name": "意图解析",
                "agent": "Orchestrator", 
                "description": "分析用户请求，确定处理工作流类型",
                "weight": 10
            },
            
            # File processing workflow
            "file_ingest": {
                "name": "文件数据载入",
                "agent": "FileIngestAgent",
                "description": "读取并解析Excel文件数据",
                "weight": 15
            },
            
            "structure_scan": {
                "name": "数据结构扫描",
                "agent": "StructureScanAgent",
                "description": "分析数据表结构和工作表信息",
                "weight": 10
            },
            
            "column_profiling": {
                "name": "数据列分析",
                "agent": "ColumnProfilingAgent",
                "description": "分析各列的数据类型、统计信息和质量",
                "weight": 15
            },
            
            "relation_discovery": {
                "name": "关系发现",
                "agent": "RelationDiscoveryAgent",
                "description": "发现数据表间的关联关系",
                "weight": 10
            },
            
            # Analysis and generation
            "code_generation": {
                "name": "分析代码生成",
                "agent": "CodeGenerationAgent",
                "description": "基于用户请求生成数据分析代码",
                "weight": 15
            },
            
            "execution": {
                "name": "代码执行",
                "agent": "ExecutionAgent", 
                "description": "执行生成的分析代码并获取结果",
                "weight": 15
            },
            
            "response_generation": {
                "name": "回答生成",
                "agent": "ResponseGenerationAgent",
                "description": "基于分析结果生成用户友好的回答",
                "weight": 15
            },
            
            # Specialized steps
            "relevance_analysis": {
                "name": "相关性分析",
                "agent": "RelevanceMatcher",
                "description": "分析查询与文件内容的相关性",
                "weight": 8
            },
            
            "text_analysis": {
                "name": "文本内容分析", 
                "agent": "TextProcessor",
                "description": "分析文件中的文本内容和关键词",
                "weight": 12
            },
            
            # Error handling
            "error_handling": {
                "name": "错误处理",
                "agent": "System",
                "description": "处理执行过程中的错误",
                "weight": 5
            }
        }
    
    def start_request(self, request_id: str, workflow_type: str = "single_table") -> None:
        """Start tracking a new request."""
        workflow_steps = self._get_workflow_steps(workflow_type)
        
        self.active_requests[request_id] = {
            'request_id': request_id,
            'workflow_type': workflow_type,
            'started_at': datetime.now().isoformat(),
            'current_step_index': 0,
            'steps': workflow_steps,
            'status': 'in_progress',
            'progress_percent': 0
        }
        
        # Send initial progress update
        self._send_progress_update(request_id, "Initializing...", "pending")
        
        logger.info(f"📊 [Progress {request_id}] Started tracking {workflow_type} workflow")
    
    def _get_workflow_steps(self, workflow_type: str) -> List[ProgressStep]:
        """Get the appropriate workflow steps based on workflow type."""
        if workflow_type == "single_table":
            step_ids = [
                "excel_relevance_check",
                "intent_parsing", 
                "file_ingest",
                "column_profiling",
                "code_generation",
                "execution",
                "response_generation"
            ]
        elif workflow_type == "multi_table":
            step_ids = [
                "excel_relevance_check",
                "intent_parsing",
                "file_ingest", 
                "structure_scan",
                "column_profiling",
                "relation_discovery",
                "code_generation",
                "execution",
                "response_generation"
            ]
        elif workflow_type == "single_cell":
            step_ids = [
                "excel_relevance_check",
                "intent_parsing",
                "file_ingest",
                "column_profiling", 
                "code_generation",
                "execution",
                "response_generation"
            ]
        elif workflow_type == "general_llm_response":
            step_ids = [
                "excel_relevance_check",
                "response_generation"
            ]
        else:
            # Default workflow
            step_ids = [
                "intent_parsing",
                "file_ingest", 
                "code_generation",
                "execution",
                "response_generation"
            ]
        
        steps = []
        for step_id in step_ids:
            step_def = self.step_definitions.get(step_id, {})
            steps.append(ProgressStep(
                step_id=step_id,
                step_name=step_def.get('name', step_id),
                status='pending',
                agent=step_def.get('agent', ''),
                description=step_def.get('description', ''),
                progress_percent=0
            ))
        
        return steps
    
    def update_step(
        self, 
        request_id: str, 
        step_id: str, 
        status: str,
        message: str = "",
        progress_percent: Optional[int] = None,
        details: Dict[str, Any] = None
    ) -> None:
        """Update progress for a specific step."""
        if request_id not in self.active_requests:
            logger.warning(f"Progress update for unknown request: {request_id}")
            return
        
        request_data = self.active_requests[request_id]
        steps = request_data['steps']
        
        # Find the step to update
        step_to_update = None
        step_index = -1
        for i, step in enumerate(steps):
            if step.step_id == step_id:
                step_to_update = step
                step_index = i
                break
        
        if step_to_update is None:
            logger.warning(f"Unknown step {step_id} for request {request_id}")
            return
        
        # Update step
        step_to_update.status = status
        if status == "in_progress" and step_to_update.started_at is None:
            step_to_update.started_at = datetime.now().isoformat()
        elif status in ["completed", "failed", "warning"]:
            step_to_update.completed_at = datetime.now().isoformat()
            if progress_percent is None:
                step_to_update.progress_percent = 100
        
        if progress_percent is not None:
            step_to_update.progress_percent = progress_percent
        
        if details:
            step_to_update.details.update(details)
        
        # Update current step index
        if status == "in_progress" and step_index > request_data['current_step_index']:
            request_data['current_step_index'] = step_index
        
        # Calculate overall progress
        overall_progress = self._calculate_overall_progress(request_id)
        request_data['progress_percent'] = overall_progress
        
        # Send progress update
        effective_message = message or f"{step_to_update.step_name}: {self._get_status_message(status)}"
        self._send_progress_update(
            request_id, 
            effective_message, 
            status,
            step_to_update.agent,
            step_to_update.step_name,
            overall_progress,
            details
        )
        
        logger.info(f"📊 [Progress {request_id}] Step '{step_id}' -> {status} ({overall_progress}%)")
    
    def _calculate_overall_progress(self, request_id: str) -> int:
        """Calculate overall progress percentage for a request."""
        if request_id not in self.active_requests:
            return 0
        
        steps = self.active_requests[request_id]['steps']
        total_weight = sum(
            self.step_definitions.get(step.step_id, {}).get('weight', 10)
            for step in steps
        )
        
        completed_weight = 0
        for step in steps:
            step_weight = self.step_definitions.get(step.step_id, {}).get('weight', 10)
            if step.status == "completed":
                completed_weight += step_weight
            elif step.status == "in_progress":
                completed_weight += step_weight * (step.progress_percent / 100)
            elif step.status in ["warning"]:
                # Count warnings as partially completed
                completed_weight += step_weight * 0.8
        
        return min(int((completed_weight / total_weight) * 100), 100) if total_weight > 0 else 0
    
    def _get_status_message(self, status: str) -> str:
        """Get user-friendly status message."""
        status_messages = {
            'pending': '等待中...',
            'in_progress': '进行中...',
            'completed': '完成',
            'failed': '失败',
            'warning': '完成（有警告）'
        }
        return status_messages.get(status, status)
    
    def finish_request(self, request_id: str, status: str = "completed", message: str = "") -> None:
        """Mark request as finished."""
        if request_id not in self.active_requests:
            return
        
        request_data = self.active_requests[request_id]
        request_data['status'] = status
        request_data['finished_at'] = datetime.now().isoformat()
        
        if status == "completed":
            request_data['progress_percent'] = 100
            effective_message = message or "处理完成！"
        else:
            effective_message = message or f"处理{status}"
        
        # Send final progress update
        self._send_progress_update(request_id, effective_message, status, progress_percent=100)
        
        logger.info(f"📊 [Progress {request_id}] Request finished with status: {status}")
    
    def get_request_progress(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for a request."""
        return self.active_requests.get(request_id)
    
    def add_progress_callback(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """Add a callback function to receive progress updates."""
        self.progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """Remove a progress callback."""
        if callback in self.progress_callbacks:
            self.progress_callbacks.remove(callback)
    
    def _send_progress_update(
        self, 
        request_id: str, 
        message: str, 
        status: str,
        agent: str = "",
        current_step_name: str = "",
        progress_percent: Optional[int] = None,
        details: Dict[str, Any] = None
    ) -> None:
        """Send progress update to all registered callbacks."""
        if request_id not in self.active_requests:
            return
        
        request_data = self.active_requests[request_id]
        current_step = request_data['steps'][request_data['current_step_index']]
        
        if progress_percent is None:
            progress_percent = request_data['progress_percent']
        
        update = ProgressUpdate(
            request_id=request_id,
            current_step=current_step.step_id,
            current_step_name=current_step_name or current_step.step_name,
            progress_percent=progress_percent,
            status=status,
            message=message,
            agent=agent or current_step.agent,
            details=details or {},
            all_steps=[step for step in request_data['steps']]  # Make a copy
        )
        
        # Send to all callbacks
        for callback in self.progress_callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def cleanup_old_requests(self, max_age_minutes: int = 60) -> int:
        """Clean up old completed requests to prevent memory leaks."""
        current_time = datetime.now()
        removed_count = 0
        
        requests_to_remove = []
        for request_id, request_data in self.active_requests.items():
            if request_data['status'] in ['completed', 'failed']:
                finished_at = request_data.get('finished_at')
                if finished_at:
                    finished_time = datetime.fromisoformat(finished_at)
                    age_minutes = (current_time - finished_time).total_seconds() / 60
                    if age_minutes > max_age_minutes:
                        requests_to_remove.append(request_id)
        
        for request_id in requests_to_remove:
            del self.active_requests[request_id]
            removed_count += 1
        
        if removed_count > 0:
            logger.info(f"📊 [Progress] Cleaned up {removed_count} old request(s)")
        
        return removed_count


# Global progress tracker instance
progress_tracker = ProgressTracker()