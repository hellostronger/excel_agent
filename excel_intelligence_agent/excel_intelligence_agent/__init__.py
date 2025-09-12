"""
Excel Intelligence Agent - Multi-Agent Analysis System

A comprehensive Excel file analysis system built with Google ADK (Agent Development Kit).
Implements a four-stage progressive analysis workflow with specialized agents for 
different analysis tasks.

Architecture:
1. File Preparation Stage - Comprehensive metadata and structure extraction
2. Concurrent Analysis Stage - Specialized agents working in parallel
3. Data Integration Stage - Merging results and building relationships
4. Response Generation Stage - Intelligent answer synthesis

Author: Excel Agent Team
Version: 0.1.0
"""

from . import agent

__all__ = ["agent"]

__version__ = "0.1.0"