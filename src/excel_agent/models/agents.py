"""Agent-specific data models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from .base import (
    AgentRequest, AgentResponse, FileMetadata, SheetInfo, 
    ColumnProfile, RelationCandidate, MergeStrategy, DataType
)


# File Ingest Agent Models
class FileIngestRequest(AgentRequest):
    """Request for File Ingest Agent."""
    file_path: str


class FileIngestResponse(AgentResponse):
    """Response from File Ingest Agent."""
    file_id: Optional[str] = None
    sheets: List[str] = Field(default_factory=list)
    metadata: Optional[FileMetadata] = None


# Structure Scan Agent Models
class StructureScanRequest(AgentRequest):
    """Request for Structure Scan Agent."""
    file_id: str
    sheet_name: str


class StructureScanResponse(AgentResponse):
    """Response from Structure Scan Agent."""
    merged_cells: List[str] = Field(default_factory=list)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    images: List[Dict[str, Any]] = Field(default_factory=list)
    formulas: List[str] = Field(default_factory=list)


# Column Profiling Agent Models
class ColumnProfilingRequest(AgentRequest):
    """Request for Column Profiling Agent."""
    file_id: str
    sheet_name: str
    column_name: Optional[str] = None  # If None, profile all columns


class ColumnProfilingResponse(AgentResponse):
    """Response from Column Profiling Agent."""
    profiles: List[ColumnProfile] = Field(default_factory=list)


# Merge Handling Agent Models
class MergeHandlingRequest(AgentRequest):
    """Request for Merge Handling Agent."""
    file_id: str
    sheet_name: str
    merged_cells: List[str]
    strategy: MergeStrategy


class MergeHandlingResponse(AgentResponse):
    """Response from Merge Handling Agent."""
    transformed_sheet: Optional[str] = None  # Path to transformed file
    log: List[str] = Field(default_factory=list)


# Labeling Agent Models
class LabelingRequest(AgentRequest):
    """Request for Labeling Agent."""
    file_id: str
    sheet_name: str
    cell_range: str
    context: Dict[str, Any] = Field(default_factory=dict)


class LabelingResponse(AgentResponse):
    """Response from Labeling Agent."""
    label: Optional[str] = None
    confidence: float = 0.0
    history_match: Optional[Dict[str, Any]] = None


# Code Generation Agent Models
class CodeGenerationRequest(AgentRequest):
    """Request for Code Generation Agent."""
    user_request: str
    context: Dict[str, Any] = Field(default_factory=dict)
    file_info: Optional[Dict[str, Any]] = None


class CodeGenerationResponse(AgentResponse):
    """Response from Code Generation Agent."""
    code: Optional[str] = None
    dry_run_plan: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)


# Execution Agent Models
class ExecutionRequest(AgentRequest):
    """Request for Execution Agent."""
    file_id: str
    code: str
    dry_run: bool = False


class ExecutionResponse(AgentResponse):
    """Response from Execution Agent."""
    result_file: Optional[str] = None
    output: Optional[Any] = None
    execution_log: List[str] = Field(default_factory=list)


# Summarization Agent Models
class SummarizationRequest(AgentRequest):
    """Request for Summarization Agent."""
    file_id: str
    sheet_name: str
    max_rows: int = 100


class SummarizationResponse(AgentResponse):
    """Response from Summarization Agent."""
    summary: Optional[str] = None
    compressed_view: Optional[Dict[str, Any]] = None
    key_insights: List[str] = Field(default_factory=list)


# Memory & Preference Agent Models
class MemoryRequest(AgentRequest):
    """Request for Memory & Preference Agent."""
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    user_feedback: Optional[Dict[str, Any]] = None
    query: Optional[str] = None


class MemoryResponse(AgentResponse):
    """Response from Memory & Preference Agent."""
    stored_preferences: List[Dict[str, Any]] = Field(default_factory=list)
    can_answer_from_memory: bool = False
    memory_match: Optional[Dict[str, Any]] = None


# Relation Discovery Agent Models
class RelationDiscoveryRequest(AgentRequest):
    """Request for Relation Discovery Agent."""
    file_id: str
    sheet_names: List[str]
    profiling_info: Dict[str, List[ColumnProfile]] = Field(default_factory=dict)


class RelationDiscoveryResponse(AgentResponse):
    """Response from Relation Discovery Agent."""
    candidate_relations: List[RelationCandidate] = Field(default_factory=list)
    recommended_keys: List[Dict[str, str]] = Field(default_factory=list)
    need_user_confirmation: bool = False