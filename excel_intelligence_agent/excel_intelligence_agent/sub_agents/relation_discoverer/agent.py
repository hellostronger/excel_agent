"""
Relation Discoverer Agent - Data Relationship Analysis Specialist

Discovers and analyzes relationships, dependencies, and connections across Excel data.
"""

import logging
from typing import Dict, Any, List

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.genai import types

from ...shared_libraries.utils import setup_logging
from ...shared_libraries.constants import RELATION_DISCOVERER_NAME, DEFAULT_TEMPERATURE
from ...prompts import return_instructions_relation_discoverer
from .tools import discover_relationships_comprehensive, validate_relationships, generate_relationship_insights

# Setup logging
logger = setup_logging()


async def comprehensive_relationship_analysis(
    user_query: str,
    focus_areas: List[str],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of data relationships.
    
    This is the main analysis function for discovering relationships across the Excel file.
    """
    try:
        logger.info("Starting comprehensive relationship analysis")
        
        # Get context from previous analyses
        file_metadata = tool_context.state.get("file_metadata", {})
        column_profiling_results = tool_context.state.get("column_profiling_results", {})
        
        if not file_metadata or not column_profiling_results:
            return {
                "success": False,
                "error": "Required analysis context not available - file and column analysis must be completed first",
                "agent_name": RELATION_DISCOVERER_NAME
            }
        
        file_path = file_metadata.get("file_path")
        if not file_path:
            return {
                "success": False,
                "error": "File path not available in metadata",
                "agent_name": RELATION_DISCOVERER_NAME
            }
        
        # Store analysis context
        tool_context.state["current_relationship_analysis"] = {
            "user_query": user_query,
            "focus_areas": focus_areas,
            "started": True
        }
        
        # Discover relationships across data
        relationship_discovery_result = await discover_relationships_comprehensive(
            file_path, column_profiling_results, focus_areas, tool_context
        )
        
        # Validate discovered relationships
        validation_result = await validate_relationships(relationship_discovery_result, tool_context)
        
        # Generate insights and business implications
        insights_result = await generate_relationship_insights(
            relationship_discovery_result, validation_result, user_query, tool_context
        )
        
        # Compile comprehensive results
        analysis_results = {
            "success": True,
            "user_query": user_query,
            "focus_areas": focus_areas,
            "relationship_discovery": relationship_discovery_result,
            "relationship_validation": validation_result,
            "insights": insights_result,
            "recommendations": generate_relationship_recommendations(
                relationship_discovery_result, validation_result, insights_result
            ),
            "confidence": calculate_relationship_analysis_confidence(
                relationship_discovery_result, validation_result
            ),
            "agent_name": RELATION_DISCOVERER_NAME,
            "relationships_found": len(relationship_discovery_result.get("discovered_relationships", [])),
            "validated_relationships": len(validation_result.get("validated_relationships", []))
        }
        
        # Store results for other agents
        tool_context.state["relationship_discovery_results"] = analysis_results
        
        logger.info(f"Relationship analysis completed: {analysis_results['relationships_found']} relationships discovered")
        return analysis_results
        
    except Exception as e:
        logger.error(f"Relationship analysis failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "agent_name": RELATION_DISCOVERER_NAME
        }


def generate_relationship_recommendations(
    discovery_result: Dict[str, Any],
    validation_result: Dict[str, Any],
    insights_result: Dict[str, Any]
) -> List[str]:
    """Generate actionable recommendations based on relationship analysis"""
    recommendations = []
    
    # Relationship discovery recommendations
    discovered_count = len(discovery_result.get("discovered_relationships", []))
    validated_count = len(validation_result.get("validated_relationships", []))
    
    if validated_count > 0:
        recommendations.append(f"Leverage {validated_count} validated relationships for data integration and analysis")
    
    if discovered_count > validated_count:
        unvalidated = discovered_count - validated_count
        recommendations.append(f"Investigate {unvalidated} potential relationships for data quality improvement")
    
    # Data integrity recommendations
    integrity_issues = validation_result.get("integrity_issues", [])
    if integrity_issues:
        recommendations.append("Address referential integrity issues before using relationships for critical analysis")
    
    # Business process recommendations
    business_relationships = insights_result.get("business_relationships", [])
    if business_relationships:
        recommendations.append("Document discovered business relationships for process optimization")
    
    # Integration recommendations
    integration_opportunities = insights_result.get("integration_opportunities", [])
    if integration_opportunities:
        recommendations.append("Explore data integration opportunities to reduce redundancy and improve consistency")
    
    return recommendations


def calculate_relationship_analysis_confidence(
    discovery_result: Dict[str, Any],
    validation_result: Dict[str, Any]
) -> float:
    """Calculate overall confidence in relationship analysis results"""
    confidence_factors = []
    
    # Discovery confidence
    if discovery_result.get("success", False):
        discovery_confidence = discovery_result.get("analysis_confidence", 0.7)
        confidence_factors.append(discovery_confidence)
    else:
        confidence_factors.append(0.3)
    
    # Validation confidence
    if validation_result.get("success", False):
        validation_confidence = validation_result.get("validation_confidence", 0.8)
        confidence_factors.append(validation_confidence)
    else:
        confidence_factors.append(0.4)
    
    # Relationship quality factor
    discovered_relationships = discovery_result.get("discovered_relationships", [])
    if discovered_relationships:
        avg_relationship_confidence = sum(
            rel.get("confidence_score", 0.5) for rel in discovered_relationships
        ) / len(discovered_relationships)
        confidence_factors.append(avg_relationship_confidence)
    else:
        confidence_factors.append(0.5)
    
    return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5


# Import model adapter
from ...shared_libraries.model_adapter import create_agent_with_best_model

# Create the Relation Discoverer Agent with best available model
try:
    relation_discoverer_agent = create_agent_with_best_model(
        agent_type="worker",
        name=RELATION_DISCOVERER_NAME,
        description="Data relationship discovery and dependency analysis specialist (数据关系发现专家)",
        instruction=return_instructions_relation_discoverer(),
        tools=[
            comprehensive_relationship_analysis,
            discover_relationships_comprehensive,
            validate_relationships,
            generate_relationship_insights
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
    logger.info("Relation Discoverer Agent initialized with best available model")
    
except Exception as e:
    logger.error(f"Failed to create Relation Discoverer with model adapter: {e}")
    
    # Fallback to basic Agent
    relation_discoverer_agent = Agent(
        model="qwen-turbo",  # 使用硅基流动默认模型
        name=RELATION_DISCOVERER_NAME,
        description="Data relationship discovery and dependency analysis specialist",
        instruction=return_instructions_relation_discoverer(),
        tools=[
            comprehensive_relationship_analysis,
            discover_relationships_comprehensive,
            validate_relationships,
            generate_relationship_insights
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

logger.info("Relation Discoverer Agent initialized successfully")