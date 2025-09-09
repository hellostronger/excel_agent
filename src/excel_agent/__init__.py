"""
Excel Intelligent Agent System

A multi-agent collaboration architecture for Excel file processing,
analysis, and intelligent querying based on the Google ADK framework.
"""

__version__ = "0.1.0"
__author__ = "Excel Agent Team"

from .core.orchestrator import Orchestrator
from .models.base import AgentRequest, AgentResponse
from .agents import *

__all__ = [
    "Orchestrator",
    "AgentRequest", 
    "AgentResponse",
]