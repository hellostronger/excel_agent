"""
Shared constants for Excel Intelligence Agent System
"""

# Analysis Context Keys
FILE_METADATA_KEY = "file_metadata"
SHEET_STRUCTURES_KEY = "sheet_structures"
ANALYSIS_RESULTS_KEY = "analysis_results"
INTEGRATION_RESULTS_KEY = "integration_results"
USER_QUERY_KEY = "user_query"
PROCESSING_STAGE_KEY = "processing_stage"

# Agent Names
ROOT_AGENT_NAME = "excel_intelligence_orchestrator"
FILE_ANALYZER_NAME = "file_analyzer"
COLUMN_PROFILER_NAME = "column_profiler"
RELATION_DISCOVERER_NAME = "relation_discoverer"
RESPONSE_SYNTHESIZER_NAME = "response_synthesizer"

# Model Configuration
DEFAULT_TEMPERATURE = 0.01
JSON_RESPONSE_TEMPERATURE = 0.0
CREATIVE_TEMPERATURE = 0.7

# File Processing Limits
MAX_FILE_SIZE_MB = 100
MAX_SHEETS_COUNT = 50
MAX_COLUMNS_PER_SHEET = 500
MAX_ROWS_SAMPLE = 1000

# Data Quality Thresholds
EXCELLENT_QUALITY_THRESHOLD = 0.9
GOOD_QUALITY_THRESHOLD = 0.7
FAIR_QUALITY_THRESHOLD = 0.5
POOR_QUALITY_THRESHOLD = 0.3

# Relationship Confidence Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.8
MEDIUM_CONFIDENCE_THRESHOLD = 0.6
LOW_CONFIDENCE_THRESHOLD = 0.4

# Common Excel Data Types
EXCEL_DATA_TYPES = [
    "text",
    "number", 
    "integer",
    "float",
    "percentage",
    "currency",
    "date",
    "datetime",
    "time",
    "boolean",
    "formula",
    "error",
    "blank"
]

# Common Business Domains
BUSINESS_DOMAINS = [
    "financial",
    "sales",
    "marketing", 
    "operations",
    "hr",
    "inventory",
    "customer",
    "product",
    "geographic",
    "temporal"
]

# Error Messages
ERROR_MESSAGES = {
    "file_not_found": "Excel file not found at specified path",
    "file_too_large": f"File exceeds maximum size limit of {MAX_FILE_SIZE_MB}MB",
    "invalid_format": "File is not a valid Excel format",
    "analysis_timeout": "Analysis timed out - file may be too complex",
    "insufficient_data": "Insufficient data for meaningful analysis",
    "agent_failure": "One or more agents failed during analysis"
}

# Success Messages
SUCCESS_MESSAGES = {
    "file_analyzed": "File structure analysis completed successfully",
    "columns_profiled": "Column profiling completed successfully",
    "relationships_discovered": "Data relationships discovered successfully",
    "response_generated": "Intelligent response generated successfully"
}