"""
File Analyzer Agent - Excel Structure Analysis Specialist

Comprehensive analysis of Excel file structure, metadata, and organizational patterns.
"""

import logging
from typing import Dict, Any, List
import pandas as pd
from pathlib import Path
from openpyxl import load_workbook
import xlrd

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.genai import types

from ...shared_libraries.utils import setup_logging
from ...shared_libraries.constants import FILE_ANALYZER_NAME, DEFAULT_TEMPERATURE
from ...prompts import return_instructions_file_analyzer
from .tools import analyze_file_structure, extract_sheet_metadata, assess_file_complexity

# Setup logging
logger = setup_logging()


async def comprehensive_file_analysis(
    analysis_request: Dict[str, Any],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of Excel file structure and metadata.
    
    This is the main analysis function that coordinates all file-level analysis.
    """
    try:
        file_path = analysis_request["file_path"]
        analysis_depth = analysis_request.get("analysis_depth", "comprehensive")
        metadata = analysis_request.get("metadata", {})
        
        logger.info(f"Starting comprehensive file analysis: {Path(file_path).name}")
        
        # Store request context
        tool_context.state["current_analysis"] = analysis_request
        
        # Perform structural analysis
        structure_result = await analyze_file_structure(file_path, analysis_depth, tool_context)
        
        # Extract detailed sheet metadata
        sheet_metadata_result = await extract_sheet_metadata(file_path, tool_context)
        
        # Assess file complexity and processing requirements
        complexity_assessment = await assess_file_complexity(file_path, tool_context)
        
        # Compile comprehensive results
        analysis_results = {
            "success": True,
            "file_path": file_path,
            "analysis_depth": analysis_depth,
            "basic_metadata": metadata,
            "structural_analysis": structure_result,
            "sheet_metadata": sheet_metadata_result,
            "complexity_assessment": complexity_assessment,
            "recommendations": generate_analysis_recommendations(
                structure_result, sheet_metadata_result, complexity_assessment
            ),
            "confidence": calculate_analysis_confidence(
                structure_result, sheet_metadata_result, complexity_assessment
            ),
            "agent_name": FILE_ANALYZER_NAME
        }
        
        logger.info(f"File analysis completed successfully: {Path(file_path).name}")
        return analysis_results
        
    except Exception as e:
        logger.error(f"File analysis failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "agent_name": FILE_ANALYZER_NAME
        }


def generate_analysis_recommendations(
    structure_result: Dict[str, Any],
    sheet_metadata_result: Dict[str, Any], 
    complexity_assessment: Dict[str, Any]
) -> List[str]:
    """Generate recommendations based on file analysis results"""
    recommendations = []
    
    # Structure-based recommendations
    if structure_result.get("sheet_count", 0) > 10:
        recommendations.append("Consider focused analysis on key worksheets due to high sheet count")
    
    # Complexity-based recommendations
    complexity_level = complexity_assessment.get("complexity_level", "medium")
    if complexity_level == "high":
        recommendations.append("High complexity detected - recommend extended processing time")
    elif complexity_level == "low":
        recommendations.append("Low complexity allows for rapid analysis")
    
    # Data quality recommendations
    if sheet_metadata_result.get("data_quality_indicators", {}).get("has_quality_issues", False):
        recommendations.append("Data quality issues detected - prioritize quality assessment")
    
    # Processing recommendations
    total_cells = structure_result.get("total_data_cells", 0)
    if total_cells > 100000:
        recommendations.append("Large dataset detected - recommend parallel processing")
    
    return recommendations


def calculate_analysis_confidence(
    structure_result: Dict[str, Any],
    sheet_metadata_result: Dict[str, Any],
    complexity_assessment: Dict[str, Any]
) -> float:
    """Calculate overall confidence in the analysis results"""
    confidence_factors = []
    
    # Structure analysis confidence
    if structure_result.get("success", False):
        confidence_factors.append(0.9)
    else:
        confidence_factors.append(0.3)
    
    # Metadata extraction confidence  
    if sheet_metadata_result.get("success", False):
        confidence_factors.append(0.8)
    else:
        confidence_factors.append(0.4)
    
    # Complexity assessment confidence
    complexity_confidence = complexity_assessment.get("confidence", 0.7)
    confidence_factors.append(complexity_confidence)
    
    # Calculate weighted average
    return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5


# Import model adapter
from ...shared_libraries.model_adapter import create_agent_with_best_model

# Create the File Analyzer Agent with best available model
try:
    file_analyzer_agent = create_agent_with_best_model(
        agent_type="worker",
        name=FILE_ANALYZER_NAME,
        description="Excel file structure analysis and metadata extraction specialist (文件结构分析专家)",
        instruction=return_instructions_file_analyzer(),
        tools=[
            comprehensive_file_analysis,
            analyze_file_structure,
            extract_sheet_metadata,
            assess_file_complexity
        ],
        generate_content_config=types.GenerateContentConfig(
            temperature=DEFAULT_TEMPERATURE,
            top_p=0.9,
            max_output_tokens=4096,
            response_mime_type="application/json"
        ),
        disallow_transfer_to_parent=False,
        disallow_transfer_to_peers=True
    )
    logger.info("File Analyzer Agent initialized with best available model")
    
except Exception as e:
    logger.error(f"Failed to create File Analyzer with model adapter: {e}")
    
    # Fallback to basic Agent
    file_analyzer_agent = Agent(
        model="qwen-turbo",  # 使用硅基流动默认模型
        name=FILE_ANALYZER_NAME,
        description="Excel file structure analysis and metadata extraction specialist",
        instruction=return_instructions_file_analyzer(),
        tools=[
            comprehensive_file_analysis,
            analyze_file_structure,
            extract_sheet_metadata,
            assess_file_complexity
        ],
        generate_content_config=types.GenerateContentConfig(
            temperature=DEFAULT_TEMPERATURE,
            top_p=0.9,
            max_output_tokens=4096,
            response_mime_type="application/json"
        ),
        disallow_transfer_to_parent=False,
        disallow_transfer_to_peers=True
    )

logger.info("File Analyzer Agent initialized successfully")