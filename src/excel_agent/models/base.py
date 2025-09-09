"""Base data models for the Excel Intelligent Agent System."""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class RequestType(str, Enum):
    """Types of user requests."""
    SINGLE_TABLE = "single_table"
    SINGLE_CELL = "single_cell"
    MULTI_TABLE = "multi_table"


class DataType(str, Enum):
    """Data types for columns."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float" 
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class MergeStrategy(str, Enum):
    """Strategies for handling merged cells."""
    PROPAGATE = "propagate"
    KEEP = "keep"
    CLEAR = "clear"


class AgentStatus(str, Enum):
    """Agent execution status."""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    TIMEOUT = "timeout"


class AgentRequest(BaseModel):
    """Base request model for all agents."""
    agent_id: str
    request_id: str = Field(default_factory=lambda: f"req_{datetime.now().isoformat()}")
    timestamp: datetime = Field(default_factory=datetime.now)
    context: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Base response model for all agents."""
    agent_id: str
    request_id: str
    status: AgentStatus
    result: Optional[Dict[str, Any]] = None
    error_log: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time_ms: Optional[int] = None


class FileMetadata(BaseModel):
    """Metadata for uploaded Excel files."""
    file_id: str
    original_filename: str
    file_path: str
    file_size: int
    upload_timestamp: datetime
    sheets: List[str]
    total_rows: int = 0
    total_columns: int = 0
    has_merged_cells: bool = False
    has_formulas: bool = False
    has_charts: bool = False
    has_images: bool = False


class SheetInfo(BaseModel):
    """Information about a specific sheet."""
    sheet_name: str
    rows: int
    columns: int
    merged_cells: List[str] = Field(default_factory=list)
    formulas: List[str] = Field(default_factory=list)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    images: List[Dict[str, Any]] = Field(default_factory=list)


class ColumnProfile(BaseModel):
    """Column profiling information."""
    column_name: str
    data_type: DataType
    value_range: Optional[Dict[str, Any]] = None
    null_ratio: float = 0.0
    unique_values: int = 0
    sample_values: List[Any] = Field(default_factory=list)
    statistics: Optional[Dict[str, float]] = None


class RelationCandidate(BaseModel):
    """Candidate relation between tables."""
    table1: str
    table2: str
    column1: str
    column2: str
    confidence: float
    relation_type: str
    sample_matches: List[Dict[str, Any]] = Field(default_factory=list)


class UserPreference(BaseModel):
    """User preference or confirmation."""
    preference_id: str
    user_id: str
    preference_type: str
    value: Any
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: float = 1.0