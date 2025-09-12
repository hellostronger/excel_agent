"""
Tools for Excel Intelligence Agent System

Core tools used by the orchestrator and sub-agents.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool

from .shared_libraries.types import (
    FileMetadata, 
    AnalysisResult,
    ExcelIntelligenceRequest,
    ColumnProfile
)
from .shared_libraries.utils import (
    validate_excel_file,
    extract_basic_metadata,
    setup_logging,
    run_with_timeout
)
from .shared_libraries.constants import (
    FILE_METADATA_KEY,
    ANALYSIS_RESULTS_KEY,
    USER_QUERY_KEY,
    SUCCESS_MESSAGES,
    ERROR_MESSAGES
)

# Setup logging
logger = setup_logging()


async def prepare_file_analysis(
    file_path: str,
    analysis_depth: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Prepare comprehensive file analysis as foundation for multi-agent processing.
    
    This tool coordinates the File Analyzer agent to extract complete file structure
    and metadata information.
    """
    try:
        # Validate file accessibility
        is_valid, validation_message = await validate_excel_file(file_path)
        if not is_valid:
            return {
                "success": False,
                "error": validation_message,
                "stage": "file_validation"
            }
        
        # Extract basic metadata first
        metadata = await extract_basic_metadata(file_path)
        tool_context.state[FILE_METADATA_KEY] = metadata.dict()
        
        # Import and call File Analyzer agent
        from .sub_agents.file_analyzer.agent import file_analyzer_agent
        
        agent_tool = AgentTool(agent=file_analyzer_agent)
        
        analysis_request = {
            "file_path": file_path,
            "analysis_depth": analysis_depth,
            "metadata": metadata.dict()
        }
        
        # Run File Analyzer with timeout
        file_analysis_result = await run_with_timeout(
            agent_tool.run_async(
                args={"analysis_request": analysis_request},
                tool_context=tool_context
            ),
            timeout_seconds=120
        )
        
        # Store results in context
        tool_context.state["file_structure_analysis"] = file_analysis_result
        
        logger.info(f"File analysis completed for: {Path(file_path).name}")
        
        return {
            "success": True,
            "message": SUCCESS_MESSAGES["file_analyzed"],
            "metadata": metadata.dict(),
            "structure_analysis": file_analysis_result,
            "stage": "file_preparation_complete"
        }
        
    except Exception as e:
        logger.error(f"File analysis failed: {str(e)}")
        return {
            "success": False,
            "error": f"File analysis failed: {str(e)}",
            "stage": "file_analysis_error"
        }


async def execute_concurrent_analysis(
    user_query: str,
    focus_areas: Optional[List[str]],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Execute concurrent multi-agent analysis.
    
    Coordinates Column Profiler and Relation Discoverer agents to work in parallel
    on their specialized analysis tasks.
    """
    try:
        # Ensure file preparation is complete
        if FILE_METADATA_KEY not in tool_context.state:
            return {
                "success": False,
                "error": "File preparation must be completed before concurrent analysis",
                "stage": "prerequisite_missing"
            }
        
        # Prepare concurrent tasks
        analysis_tasks = []
        
        # Task 1: Column Profiling
        from .sub_agents.column_profiler.agent import column_profiler_agent
        
        column_agent_tool = AgentTool(agent=column_profiler_agent)
        column_task = asyncio.create_task(
            run_with_timeout(
                column_agent_tool.run_async(
                    args={
                        "user_query": user_query,
                        "focus_areas": focus_areas or []
                    },
                    tool_context=tool_context
                ),
                timeout_seconds=300
            )
        )
        analysis_tasks.append(("column_profiling", column_task))
        
        # Task 2: Relation Discovery  
        from .sub_agents.relation_discoverer.agent import relation_discoverer_agent
        
        relation_agent_tool = AgentTool(agent=relation_discoverer_agent)
        relation_task = asyncio.create_task(
            run_with_timeout(
                relation_agent_tool.run_async(
                    args={
                        "user_query": user_query,
                        "focus_areas": focus_areas or []
                    },
                    tool_context=tool_context
                ),
                timeout_seconds=300
            )
        )
        analysis_tasks.append(("relation_discovery", relation_task))
        
        # Execute all tasks concurrently
        concurrent_results = {}
        
        for task_name, task in analysis_tasks:
            try:
                result = await task
                concurrent_results[task_name] = {
                    "success": True,
                    "result": result,
                    "agent": task_name
                }
                logger.info(f"Concurrent task completed: {task_name}")
                
            except asyncio.TimeoutError:
                logger.warning(f"Concurrent task timed out: {task_name}")
                concurrent_results[task_name] = {
                    "success": False,
                    "error": "Task timed out",
                    "agent": task_name
                }
            except Exception as e:
                logger.error(f"Concurrent task failed: {task_name} - {str(e)}")
                concurrent_results[task_name] = {
                    "success": False,
                    "error": str(e),
                    "agent": task_name
                }
        
        # Store results in context
        tool_context.state[ANALYSIS_RESULTS_KEY] = concurrent_results
        
        # Calculate success metrics
        successful_tasks = sum(1 for result in concurrent_results.values() if result["success"])
        total_tasks = len(concurrent_results)
        
        return {
            "success": successful_tasks > 0,  # At least one task must succeed
            "message": f"Concurrent analysis completed: {successful_tasks}/{total_tasks} agents successful",
            "results": concurrent_results,
            "stage": "concurrent_analysis_complete",
            "success_rate": successful_tasks / total_tasks
        }
        
    except Exception as e:
        logger.error(f"Concurrent analysis failed: {str(e)}")
        return {
            "success": False,
            "error": f"Concurrent analysis failed: {str(e)}",
            "stage": "concurrent_analysis_error"
        }


async def integrate_analysis_results(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Integrate results from multiple agents into comprehensive insights.
    
    Combines findings from File Analyzer, Column Profiler, and Relation Discoverer
    to create unified insights and data relationship maps.
    """
    try:
        # Verify required analysis results are available
        if ANALYSIS_RESULTS_KEY not in tool_context.state:
            return {
                "success": False,
                "error": "Analysis results not available for integration",
                "stage": "integration_prerequisites_missing"
            }
        
        analysis_results = tool_context.state[ANALYSIS_RESULTS_KEY]
        file_metadata = tool_context.state.get(FILE_METADATA_KEY, {})
        file_structure = tool_context.state.get("file_structure_analysis", {})
        
        # Extract successful results
        successful_analyses = {
            name: result["result"] 
            for name, result in analysis_results.items() 
            if result["success"]
        }
        
        if not successful_analyses:
            return {
                "success": False,
                "error": "No successful analysis results to integrate",
                "stage": "integration_no_data"
            }
        
        # Integration logic
        integrated_insights = {
            "file_overview": {
                "metadata": file_metadata,
                "structure": file_structure
            },
            "data_quality_summary": {},
            "relationship_map": {},
            "business_insights": [],
            "recommendations": [],
            "confidence_assessment": {}
        }
        
        # Integrate column profiling results
        if "column_profiling" in successful_analyses:
            column_results = successful_analyses["column_profiling"]
            integrated_insights["data_quality_summary"] = {
                "overall_quality": column_results.get("overall_quality", "unknown"),
                "column_count": column_results.get("analyzed_columns", 0),
                "quality_issues": column_results.get("quality_issues", []),
                "recommendations": column_results.get("recommendations", [])
            }
        
        # Integrate relationship discovery results
        if "relation_discovery" in successful_analyses:
            relation_results = successful_analyses["relation_discovery"]
            integrated_insights["relationship_map"] = {
                "identified_relationships": relation_results.get("relationships", []),
                "confidence_levels": relation_results.get("confidence_scores", {}),
                "business_connections": relation_results.get("business_relationships", [])
            }
        
        # Generate cross-agent insights
        business_insights = []
        recommendations = []
        
        # Cross-reference data quality with relationships
        if "column_profiling" in successful_analyses and "relation_discovery" in successful_analyses:
            business_insights.append(
                "Cross-analysis reveals data integration opportunities and quality considerations"
            )
            recommendations.append(
                "Consider data quality improvements in key relationship columns for better integration"
            )
        
        integrated_insights["business_insights"] = business_insights
        integrated_insights["recommendations"] = recommendations
        
        # Calculate overall confidence
        confidence_scores = []
        for analysis in successful_analyses.values():
            if isinstance(analysis, dict) and "confidence" in analysis:
                confidence_scores.append(analysis["confidence"])
        
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        integrated_insights["confidence_assessment"] = {
            "overall_confidence": overall_confidence,
            "individual_confidences": {name: result.get("confidence", 0.5) 
                                     for name, result in successful_analyses.items()}
        }
        
        # Store integrated results
        tool_context.state["integrated_insights"] = integrated_insights
        
        logger.info("Analysis results integration completed successfully")
        
        return {
            "success": True,
            "message": "Analysis integration completed successfully",
            "insights": integrated_insights,
            "stage": "integration_complete"
        }
        
    except Exception as e:
        logger.error(f"Analysis integration failed: {str(e)}")
        return {
            "success": False,
            "error": f"Analysis integration failed: {str(e)}",
            "stage": "integration_error"
        }


async def generate_intelligent_response(
    user_query: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Generate comprehensive, intelligent response to user query.
    
    Uses the Response Synthesizer agent to create a thoughtful answer based on
    all available analysis results and integrated insights.
    """
    try:
        # Verify integrated insights are available
        if "integrated_insights" not in tool_context.state:
            return {
                "success": False,
                "error": "Integrated insights not available for response generation",
                "stage": "response_prerequisites_missing"
            }
        
        # Store user query in context
        tool_context.state[USER_QUERY_KEY] = user_query
        
        # Import and call Response Synthesizer agent
        from .sub_agents.response_synthesizer.agent import response_synthesizer_agent
        
        agent_tool = AgentTool(agent=response_synthesizer_agent)
        
        response_request = {
            "user_query": user_query,
            "integrated_insights": tool_context.state["integrated_insights"],
            "analysis_context": {
                "file_metadata": tool_context.state.get(FILE_METADATA_KEY, {}),
                "analysis_results": tool_context.state.get(ANALYSIS_RESULTS_KEY, {}),
                "processing_summary": "Multi-stage analysis completed successfully"
            }
        }
        
        # Generate response with timeout
        response_result = await run_with_timeout(
            agent_tool.run_async(
                args={"response_request": response_request},
                tool_context=tool_context
            ),
            timeout_seconds=60
        )
        
        logger.info("Intelligent response generation completed")
        
        return {
            "success": True,
            "message": SUCCESS_MESSAGES["response_generated"],
            "response": response_result,
            "stage": "response_generation_complete"
        }
        
    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}")
        return {
            "success": False,
            "error": f"Response generation failed: {str(e)}",
            "stage": "response_generation_error"
        }


# State management and utility tools

def store_analysis_state(
    key: str,
    value: Any,
    tool_context: ToolContext
) -> Dict[str, str]:
    """Store information in the analysis session state"""
    tool_context.state[key] = value
    return {"status": f"Stored '{key}' in analysis state"}


def retrieve_analysis_state(
    key: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """Retrieve information from the analysis session state"""
    value = tool_context.state.get(key)
    if value is not None:
        return {"status": "found", "value": value}
    else:
        return {"status": "not_found", "message": f"Key '{key}' not found in analysis state"}


def get_analysis_summary(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """Get summary of current analysis state and progress"""
    state_keys = list(tool_context.state.keys())
    
    # Categorize state information
    file_info_keys = [k for k in state_keys if k.startswith("file") or "metadata" in k]
    analysis_keys = [k for k in state_keys if "analysis" in k or "result" in k]
    integration_keys = [k for k in state_keys if "integrated" in k or "insight" in k]
    
    summary = {
        "total_state_items": len(state_keys),
        "file_information": file_info_keys,
        "analysis_results": analysis_keys,
        "integration_data": integration_keys,
        "processing_stage": "determined_from_available_data"
    }
    
    # Determine current processing stage
    if "integrated_insights" in state_keys:
        summary["processing_stage"] = "ready_for_response_generation"
    elif ANALYSIS_RESULTS_KEY in state_keys:
        summary["processing_stage"] = "ready_for_integration"
    elif FILE_METADATA_KEY in state_keys:
        summary["processing_stage"] = "ready_for_concurrent_analysis"
    else:
        summary["processing_stage"] = "ready_for_file_preparation"
    
    return summary