"""
Response Synthesizer Agent - Intelligent Response Generation Specialist

Synthesizes multi-agent analysis results into comprehensive, user-focused responses.
"""

import logging
from typing import Dict, Any

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.genai import types

from ...shared_libraries.utils import setup_logging
from ...shared_libraries.constants import DEFAULT_TEMPERATURE
from ...prompts import return_instructions_response_synthesizer

# Setup logging
logger = setup_logging()


async def synthesize_intelligent_response(
    response_request: Dict[str, Any],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Synthesize comprehensive response based on multi-agent analysis results.
    
    Main function for generating intelligent, contextual responses to user queries.
    """
    try:
        logger.info("Starting intelligent response synthesis")
        
        user_query = response_request.get("user_query", "")
        integrated_insights = response_request.get("integrated_insights", {})
        analysis_context = response_request.get("analysis_context", {})
        
        if not user_query:
            return {
                "success": False,
                "error": "User query not provided for response synthesis"
            }
        
        # Extract key information from analysis results
        file_overview = integrated_insights.get("file_overview", {})
        data_quality_summary = integrated_insights.get("data_quality_summary", {})
        relationship_map = integrated_insights.get("relationship_map", {})
        business_insights = integrated_insights.get("business_insights", [])
        recommendations = integrated_insights.get("recommendations", [])
        
        # Generate structured response
        response_parts = []
        
        # 1. Direct answer section
        response_parts.append("## Analysis Results\n")
        
        if file_overview:
            metadata = file_overview.get("metadata", {})
            sheets_count = metadata.get("sheets_count", 0)
            file_size = metadata.get("file_size", 0)
            response_parts.append(f"Analyzed Excel file with {sheets_count} worksheets ({file_size:,} bytes).\n")
        
        # 2. Data quality insights
        if data_quality_summary:
            overall_quality = data_quality_summary.get("overall_quality", "unknown")
            column_count = data_quality_summary.get("column_count", 0)
            response_parts.append(f"### Data Quality Assessment\n")
            response_parts.append(f"Overall data quality: **{overall_quality.title()}** across {column_count} columns analyzed.\n")
            
            quality_issues = data_quality_summary.get("quality_issues", [])
            if quality_issues:
                response_parts.append(f"Key quality considerations: {', '.join(quality_issues[:3])}\n")
        
        # 3. Relationship insights
        if relationship_map:
            relationships = relationship_map.get("identified_relationships", [])
            if relationships:
                response_parts.append(f"### Data Relationships\n")
                response_parts.append(f"Discovered {len(relationships)} data relationships enabling integrated analysis.\n")
        
        # 4. Business insights
        if business_insights:
            response_parts.append(f"### Key Insights\n")
            for insight in business_insights[:3]:  # Top 3 insights
                response_parts.append(f"• {insight}\n")
        
        # 5. Recommendations
        if recommendations:
            response_parts.append(f"### Recommendations\n")
            for rec in recommendations[:3]:  # Top 3 recommendations
                response_parts.append(f"• {rec}\n")
        
        # Compile final response
        final_response = "".join(response_parts)
        
        # Add confidence and limitations
        confidence_assessment = integrated_insights.get("confidence_assessment", {})
        overall_confidence = confidence_assessment.get("overall_confidence", 0.5)
        
        if overall_confidence < 0.7:
            final_response += f"\n*Note: Analysis confidence is {overall_confidence:.1%}. Results should be validated for critical decisions.*"
        
        logger.info("Response synthesis completed successfully")
        
        return {
            "success": True,
            "response": final_response,
            "confidence": overall_confidence,
            "response_length": len(final_response),
            "sections_included": len([p for p in response_parts if p.startswith("###")]),
            "agent_name": "response_synthesizer"
        }
        
    except Exception as e:
        logger.error(f"Response synthesis failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "response": f"I encountered an error while analyzing your Excel file: {str(e)}. Please try again or contact support if the issue persists.",
            "agent_name": "response_synthesizer"
        }


# Import model adapter
from ...shared_libraries.model_adapter import create_agent_with_best_model
from ...shared_libraries.constants import RESPONSE_SYNTHESIZER_NAME

# Create the Response Synthesizer Agent with best available model
try:
    response_synthesizer_agent = create_agent_with_best_model(
        agent_type="worker",
        name=RESPONSE_SYNTHESIZER_NAME,
        description="Intelligent response generation specialist for multi-agent analysis results (智能回复生成专家)",
        instruction=return_instructions_response_synthesizer(),
        tools=[
            synthesize_intelligent_response
        ],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.3,  # Slightly higher for more natural language generation
            top_p=0.95,
            max_output_tokens=4096,
            response_mime_type="text/plain"
        ),
        disallow_transfer_to_parent=False,
        disallow_transfer_to_peers=True
    )
    logger.info("Response Synthesizer Agent initialized with best available model")
    
except Exception as e:
    logger.error(f"Failed to create Response Synthesizer with model adapter: {e}")
    
    # Fallback to basic Agent
    response_synthesizer_agent = Agent(
        model="qwen-turbo",  # 使用硅基流动默认模型
        name=RESPONSE_SYNTHESIZER_NAME,
        description="Intelligent response generation specialist for multi-agent analysis results",
        instruction=return_instructions_response_synthesizer(),
        tools=[
            synthesize_intelligent_response
        ],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.3,  # Slightly higher for more natural language generation
            top_p=0.95,
            max_output_tokens=4096,
            response_mime_type="text/plain"
        ),
        disallow_transfer_to_parent=False,
        disallow_transfer_to_peers=True
    )

logger.info("Response Synthesizer Agent initialized successfully")