"""Workflow Engine for coordinating agent execution."""

from typing import Dict, List, Any
from enum import Enum

class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class WorkflowEngine:
    """Simple workflow engine for agent coordination."""
    
    def __init__(self):
        self.workflows = {}
    
    def create_workflow(self, workflow_id: str, steps: List[Dict[str, Any]]):
        """Create a new workflow."""
        self.workflows[workflow_id] = {
            'id': workflow_id,
            'steps': steps,
            'status': WorkflowStatus.PENDING,
            'results': []
        }
    
    def get_workflow(self, workflow_id: str):
        """Get workflow by ID."""
        return self.workflows.get(workflow_id)