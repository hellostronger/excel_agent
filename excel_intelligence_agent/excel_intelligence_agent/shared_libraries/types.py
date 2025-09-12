"""
Shared data types and models for Excel Intelligence Agent System
"""

from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class AnalysisStage(str, Enum):
    """Analysis processing stages"""
    FILE_PREPARATION = "file_preparation"
    CONCURRENT_ANALYSIS = "concurrent_analysis"
    DATA_INTEGRATION = "data_integration"
    RESPONSE_GENERATION = "response_generation"


class AnalysisDepth(str, Enum):
    """Analysis depth levels"""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class DataQualityLevel(str, Enum):
    """Data quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class FileMetadata(BaseModel):
    """Excel file metadata structure"""
    file_path: str = Field(description="Path to the Excel file")
    file_size: int = Field(description="File size in bytes")
    sheets_count: int = Field(description="Number of worksheets")
    creation_time: Optional[datetime] = Field(description="File creation timestamp")
    modification_time: Optional[datetime] = Field(description="Last modification timestamp")
    extraction_timestamp: datetime = Field(description="Metadata extraction timestamp")
    analysis_depth: AnalysisDepth = Field(description="Analysis depth level")


class ColumnProfile(BaseModel):
    """Column data profile"""
    column_name: str = Field(description="Column name")
    column_index: int = Field(description="Column index")
    data_type: str = Field(description="Detected data type")
    data_type_confidence: float = Field(description="Data type detection confidence", ge=0.0, le=1.0)
    null_count: int = Field(description="Number of null values")
    unique_count: int = Field(description="Number of unique values")
    duplicate_count: int = Field(description="Number of duplicate values")
    quality_score: float = Field(description="Data quality score", ge=0.0, le=1.0)
    quality_level: DataQualityLevel = Field(description="Overall quality assessment")
    value_distribution: Dict[str, int] = Field(description="Value frequency distribution")
    sample_values: List[Any] = Field(description="Sample values from the column")
    anomalies: List[str] = Field(description="Detected anomalies or issues")
    business_meaning: Optional[str] = Field(description="Inferred business meaning")


class DataRelationship(BaseModel):
    """Data relationship definition"""
    source_sheet: str = Field(description="Source sheet name")
    source_column: str = Field(description="Source column name")
    target_sheet: str = Field(description="Target sheet name")  
    target_column: str = Field(description="Target column name")
    relationship_type: str = Field(description="Type of relationship (FK, reference, etc.)")
    confidence_score: float = Field(description="Relationship confidence", ge=0.0, le=1.0)
    validation_status: str = Field(description="Validation status")
    description: str = Field(description="Human-readable description")


class AnalysisResult(BaseModel):
    """Individual agent analysis result"""
    agent_name: str = Field(description="Name of the analyzing agent")
    analysis_type: str = Field(description="Type of analysis performed")
    execution_time: float = Field(description="Execution time in seconds")
    success: bool = Field(description="Whether analysis succeeded")
    result_data: Dict[str, Any] = Field(description="Analysis result data")
    confidence_level: float = Field(description="Overall confidence in results", ge=0.0, le=1.0)
    quality_metrics: Dict[str, float] = Field(description="Quality metrics")
    recommendations: List[str] = Field(description="Recommendations based on analysis")
    warnings: List[str] = Field(description="Warnings or limitations")


class IntegratedInsights(BaseModel):
    """Integrated insights from multiple agents"""
    file_summary: str = Field(description="High-level file summary")
    data_quality_overview: str = Field(description="Overall data quality assessment")
    key_relationships: List[DataRelationship] = Field(description="Key data relationships identified")
    business_insights: List[str] = Field(description="Business-relevant insights")
    data_patterns: List[str] = Field(description="Identified data patterns")
    recommendations: List[str] = Field(description="Overall recommendations")
    limitations: List[str] = Field(description="Analysis limitations")


class ExcelIntelligenceRequest(BaseModel):
    """Request structure for Excel intelligence analysis"""
    user_query: str = Field(description="User's question or request")
    file_path: str = Field(description="Path to Excel file")
    analysis_depth: AnalysisDepth = Field(default=AnalysisDepth.COMPREHENSIVE)
    focus_areas: Optional[List[str]] = Field(description="Specific areas to focus on")
    context: Optional[Dict[str, Any]] = Field(description="Additional context")


class ExcelIntelligenceResponse(BaseModel):
    """Response structure for Excel intelligence analysis"""
    success: bool = Field(description="Whether analysis succeeded")
    response: str = Field(description="Natural language response to user query")
    analysis_summary: str = Field(description="Summary of analysis performed")
    insights: IntegratedInsights = Field(description="Integrated insights from analysis")
    column_profiles: List[ColumnProfile] = Field(description="Column analysis results")
    relationships: List[DataRelationship] = Field(description="Discovered relationships")
    performance_metrics: Dict[str, float] = Field(description="Performance metrics")
    processing_stages: List[str] = Field(description="Completed processing stages")
    confidence_score: float = Field(description="Overall confidence in response", ge=0.0, le=1.0)
    timestamp: datetime = Field(description="Response generation timestamp")


class AgentConfiguration(BaseModel):
    """Configuration for individual agents"""
    agent_name: str = Field(description="Agent name")
    model_name: str = Field(description="Model to use for this agent")
    temperature: float = Field(default=0.01, description="Model temperature")
    max_tokens: Optional[int] = Field(description="Maximum tokens for responses")
    timeout_seconds: int = Field(default=300, description="Agent timeout in seconds")
    specialized_tools: List[str] = Field(description="List of specialized tools for this agent")


class SystemConfiguration(BaseModel):
    """Overall system configuration"""
    orchestrator_model: str = Field(default="gemini-2.5-pro")
    worker_model: str = Field(default="gemini-2.5-flash")
    max_parallel_agents: int = Field(default=4)
    file_analysis_depth: AnalysisDepth = Field(default=AnalysisDepth.COMPREHENSIVE)
    enable_concurrent_analysis: bool = Field(default=True)
    memory_threshold_mb: int = Field(default=512)
    stage_timeouts: Dict[str, int] = Field(
        default={
            "file_preparation": 120,
            "parallel_analysis": 300,
            "response_generation": 60
        }
    )
    agent_configurations: List[AgentConfiguration] = Field(description="Individual agent configs")