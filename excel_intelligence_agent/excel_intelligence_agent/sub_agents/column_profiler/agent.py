"""
Column Profiler Agent - Data Column Analysis Specialist

Comprehensive analysis of data columns including quality, types, and patterns.
"""

import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.genai import types

from ...shared_libraries.utils import setup_logging
from ...shared_libraries.constants import COLUMN_PROFILER_NAME, DEFAULT_TEMPERATURE
from ...shared_libraries.types import ColumnProfile, DataQualityLevel
from ...prompts import return_instructions_column_profiler
from .tools import analyze_columns_comprehensive, assess_data_quality, generate_column_insights

# Setup logging
logger = setup_logging()


async def comprehensive_column_analysis(
    user_query: str,
    focus_areas: List[str],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of data columns.
    
    This is the main analysis function that coordinates all column-level analysis.
    """
    try:
        logger.info("Starting comprehensive column analysis")
        
        # Get file context from previous analysis
        file_metadata = tool_context.state.get("file_metadata", {})
        structure_analysis = tool_context.state.get("file_structure_analysis", {})
        
        if not file_metadata:
            return {
                "success": False,
                "error": "File metadata not available - file preparation required",
                "agent_name": COLUMN_PROFILER_NAME
            }
        
        file_path = file_metadata.get("file_path")
        if not file_path:
            return {
                "success": False,
                "error": "File path not available in metadata",
                "agent_name": COLUMN_PROFILER_NAME
            }
        
        # Store analysis context
        tool_context.state["current_column_analysis"] = {
            "user_query": user_query,
            "focus_areas": focus_areas,
            "started": True
        }
        
        # Perform comprehensive column analysis
        column_analysis_result = await analyze_columns_comprehensive(file_path, focus_areas, tool_context)
        
        # Assess overall data quality
        quality_assessment_result = await assess_data_quality(column_analysis_result, tool_context)
        
        # Generate insights and recommendations
        insights_result = await generate_column_insights(column_analysis_result, quality_assessment_result, user_query, tool_context)
        
        # Compile comprehensive results
        analysis_results = {
            "success": True,
            "user_query": user_query,
            "focus_areas": focus_areas,
            "column_analysis": column_analysis_result,
            "quality_assessment": quality_assessment_result,
            "insights": insights_result,
            "recommendations": generate_column_recommendations(
                column_analysis_result, quality_assessment_result, insights_result
            ),
            "confidence": calculate_column_analysis_confidence(
                column_analysis_result, quality_assessment_result
            ),
            "agent_name": COLUMN_PROFILER_NAME,
            "analyzed_columns": len(column_analysis_result.get("column_profiles", [])),
            "overall_quality": quality_assessment_result.get("overall_quality_level", "unknown")
        }
        
        # Store results for other agents
        tool_context.state["column_profiling_results"] = analysis_results
        
        logger.info(f"Column analysis completed: {analysis_results['analyzed_columns']} columns analyzed")
        return analysis_results
        
    except Exception as e:
        logger.error(f"Column analysis failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "agent_name": COLUMN_PROFILER_NAME
        }


def generate_column_recommendations(
    column_analysis: Dict[str, Any],
    quality_assessment: Dict[str, Any],
    insights: Dict[str, Any]
) -> List[str]:
    """Generate actionable recommendations based on column analysis"""
    recommendations = []
    
    # Quality-based recommendations
    overall_quality = quality_assessment.get("overall_quality_level", "unknown")
    if overall_quality in ["poor", "critical"]:
        recommendations.append("Immediate data quality improvement required before analysis")
        recommendations.append("Consider data cleansing and validation processes")
    elif overall_quality == "fair":
        recommendations.append("Data quality improvements recommended for better reliability")
    
    # Column-specific recommendations
    column_profiles = column_analysis.get("column_profiles", [])
    
    # Check for high null percentages
    high_null_columns = [col for col in column_profiles if col.get("null_percentage", 0) > 0.3]
    if high_null_columns:
        recommendations.append(f"Address high null percentages in {len(high_null_columns)} columns")
    
    # Check for data type inconsistencies
    low_confidence_columns = [col for col in column_profiles if col.get("type_confidence", 1.0) < 0.7]
    if low_confidence_columns:
        recommendations.append(f"Review data type consistency in {len(low_confidence_columns)} columns")
    
    # Business value recommendations
    business_insights = insights.get("business_value_insights", [])
    if business_insights:
        recommendations.append("Leverage identified business-critical columns for decision making")
    
    # Performance recommendations
    large_columns = [col for col in column_profiles if col.get("unique_count", 0) > 10000]
    if large_columns:
        recommendations.append("Consider indexing strategies for high-cardinality columns")
    
    return recommendations


def calculate_column_analysis_confidence(
    column_analysis: Dict[str, Any],
    quality_assessment: Dict[str, Any]
) -> float:
    """Calculate overall confidence in column analysis results"""
    confidence_factors = []
    
    # Column analysis confidence
    if column_analysis.get("success", False):
        profiles = column_analysis.get("column_profiles", [])
        if profiles:
            # Average of individual column confidences
            individual_confidences = [col.get("analysis_confidence", 0.5) for col in profiles]
            avg_confidence = sum(individual_confidences) / len(individual_confidences)
            confidence_factors.append(avg_confidence)
        else:
            confidence_factors.append(0.3)
    else:
        confidence_factors.append(0.2)
    
    # Quality assessment confidence
    if quality_assessment.get("success", False):
        confidence_factors.append(0.8)
    else:
        confidence_factors.append(0.4)
    
    return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5


# Import model adapter
from ...shared_libraries.model_adapter import create_agent_with_best_model

# Create the Column Profiler Agent with best available model
try:
    column_profiler_agent = create_agent_with_best_model(
        agent_type="worker",
        name=COLUMN_PROFILER_NAME,
        description="Data column analysis and quality assessment specialist (数据列分析专家)",
        instruction=return_instructions_column_profiler(),
        tools=[
            comprehensive_column_analysis,
            analyze_columns_comprehensive,
            assess_data_quality,
            generate_column_insights
        ],
        generate_content_config=types.GenerateContentConfig(
            temperature=DEFAULT_TEMPERATURE,
            top_p=0.9,
            max_output_tokens=6144,
            response_mime_type="application/json"
        ),
        disallow_transfer_to_parent=False,
        disallow_transfer_to_peers=True
    )
    logger.info("Column Profiler Agent initialized with best available model")
    
except Exception as e:
    logger.error(f"Failed to create Column Profiler with model adapter: {e}")
    
    # Fallback to basic Agent
    column_profiler_agent = Agent(
        model="qwen-turbo",  # 使用硅基流动默认模型
        name=COLUMN_PROFILER_NAME,
        description="Data column analysis and quality assessment specialist",
        instruction=return_instructions_column_profiler(),
        tools=[
            comprehensive_column_analysis,
            analyze_columns_comprehensive,
            assess_data_quality,
            generate_column_insights
        ],
        generate_content_config=types.GenerateContentConfig(
            temperature=DEFAULT_TEMPERATURE,
            top_p=0.9,
            max_output_tokens=6144,
            response_mime_type="application/json"
        ),
        disallow_transfer_to_parent=False,
        disallow_transfer_to_peers=True
    )

logger.info("Column Profiler Agent initialized successfully")