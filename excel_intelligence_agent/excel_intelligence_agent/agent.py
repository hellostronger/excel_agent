"""
Main Excel Intelligence Agent - Multi-Agent Orchestrator

This is the root agent that coordinates specialized sub-agents for comprehensive
Excel file analysis and intelligent query response.
"""

import os
import logging
from typing import List, Optional

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from .prompts import return_instructions_root
from .tools import (
    prepare_file_analysis,
    execute_concurrent_analysis, 
    integrate_analysis_results,
    generate_intelligent_response,
    store_analysis_state,
    retrieve_analysis_state,
    get_analysis_summary
)
from .shared_libraries.utils import setup_logging, load_environment_variables
from .shared_libraries.constants import ROOT_AGENT_NAME, DEFAULT_TEMPERATURE

# Setup logging
logger = setup_logging()

# Load environment configuration
try:
    env_config = load_environment_variables()
    logger.info(f"Environment configuration loaded successfully - Provider: {env_config.get('MODEL_PROVIDER')}")
except Exception as e:
    logger.warning(f"Failed to load full environment config: {e}")
    # Fallback configuration
    env_config = {
        "MODEL_PROVIDER": "siliconflow",
        "ROOT_AGENT_MODEL": "qwen-plus",
        "ORCHESTRATOR_MODEL": "qwen-max",
        "WORKER_MODEL": "qwen-turbo"
    }


def setup_before_agent_call(callback_context: CallbackContext):
    """
    Setup function called before the root agent processes a request.
    
    Initializes the analysis context and prepares the system for multi-stage processing.
    """
    try:
        # Initialize analysis tracking
        if "analysis_session" not in callback_context.state:
            callback_context.state["analysis_session"] = {
                "started": True,
                "stages_completed": [],
                "current_stage": "initialization"
            }
        
        # Set up logging context
        callback_context.state["session_id"] = f"excel_analysis_{id(callback_context) % 10000}"
        
        # Enhance agent instructions with dynamic context
        base_instruction = callback_context._invocation_context.agent.instruction
        
        # Add dynamic context about available tools and current state
        dynamic_context = f"""
        
        # Current Session Context:
        - Session ID: {callback_context.state.get("session_id", "unknown")}
        - Available specialized tools for file analysis, concurrent processing, integration, and response generation
        - Multi-stage processing capability with state persistence
        - Error handling and recovery mechanisms active
        
        # Processing State:
        {callback_context.state.get("analysis_session", {})}
        """
        
        callback_context._invocation_context.agent.instruction = base_instruction + dynamic_context
        
        logger.info(f"Root agent setup completed for session: {callback_context.state.get('session_id')}")
        
    except Exception as e:
        logger.error(f"Root agent setup failed: {e}")
        # Continue with basic setup
        callback_context.state["setup_error"] = str(e)


def cleanup_after_agent_call(
    callback_context: CallbackContext,
    llm_response: "LlmResponse",  # Type hint as string to avoid import issues
) -> "LlmResponse":
    """
    Cleanup function called after the root agent completes processing.
    
    Performs final processing and cleanup of the analysis session.
    """
    try:
        # Update session completion status
        if "analysis_session" in callback_context.state:
            callback_context.state["analysis_session"]["completed"] = True
            callback_context.state["analysis_session"]["final_stage"] = "response_delivered"
        
        # Log session completion
        session_id = callback_context.state.get("session_id", "unknown")
        stages_completed = callback_context.state.get("analysis_session", {}).get("stages_completed", [])
        
        logger.info(f"Analysis session {session_id} completed. Stages: {', '.join(stages_completed)}")
        
        # Add session summary to response if helpful
        if hasattr(llm_response, 'content') and hasattr(llm_response.content, 'parts'):
            # Optional: Add processing summary to response
            summary_text = f"\n\n---\n*Analysis completed using {len(stages_completed)} processing stages*"
            llm_response.content.parts.append(types.Part(text=summary_text))
        
    except Exception as e:
        logger.error(f"Root agent cleanup failed: {e}")
    
    return llm_response


# Import model adapter
from .shared_libraries.model_adapter import create_agent_with_best_model

# Create the main Excel Intelligence Agent with best available model
try:
    # 尝试使用模型适配器创建Agent
    excel_intelligence_agent = create_agent_with_best_model(
        agent_type="root",
        name=ROOT_AGENT_NAME,
        description="Multi-agent Excel Intelligence Analysis System coordinator (智能Excel分析系统)",
        instruction=return_instructions_root(),
        tools=[
            prepare_file_analysis,
            execute_concurrent_analysis,
            integrate_analysis_results, 
            generate_intelligent_response,
            store_analysis_state,
            retrieve_analysis_state,
            get_analysis_summary
        ],
        generate_content_config=types.GenerateContentConfig(
            temperature=DEFAULT_TEMPERATURE,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="text/plain"
        ),
        before_agent_callback=setup_before_agent_call,
        after_model_callback=cleanup_after_agent_call,
        global_instruction=f"""
        You are part of an advanced Excel Intelligence System that provides comprehensive
        analysis through coordinated specialized agents. Always use your available tools
        to provide accurate, evidence-based responses to user questions about Excel files.
        
        Current Configuration:
        - Provider: {env_config.get('MODEL_PROVIDER', 'unknown')}
        - Model: {env_config.get('ROOT_AGENT_MODEL', 'unknown')}
        - Language: Support both English and Chinese
        """
    )
    
    model_info = f"{env_config.get('MODEL_PROVIDER', 'unknown')}:{env_config.get('ROOT_AGENT_MODEL', 'unknown')}"
    logger.info(f"Excel Intelligence Agent (root) initialized successfully with model: {model_info}")
    
except Exception as e:
    logger.error(f"Failed to create agent with model adapter: {e}")
    logger.info("Falling back to basic ADK Agent")
    
    # Fallback to basic ADK Agent
    excel_intelligence_agent = Agent(
        model=env_config.get("ROOT_AGENT_MODEL", "qwen-plus"),
        name=ROOT_AGENT_NAME,
        description="Multi-agent Excel Intelligence Analysis System coordinator",
        instruction=return_instructions_root(),
        tools=[
            prepare_file_analysis,
            execute_concurrent_analysis,
            integrate_analysis_results, 
            generate_intelligent_response,
            store_analysis_state,
            retrieve_analysis_state,
            get_analysis_summary
        ],
        generate_content_config=types.GenerateContentConfig(
            temperature=DEFAULT_TEMPERATURE,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="text/plain"
        ),
        before_agent_callback=setup_before_agent_call,
        after_model_callback=cleanup_after_agent_call,
    )

logger.info("Excel Intelligence Agent (root) initialized successfully")


# Alternative configurations for different use cases

def create_focused_analysis_agent(focus_type: str = "quality") -> Agent:
    """
    Create a focused analysis agent for specific analysis types.
    
    Args:
        focus_type: Type of focused analysis ("quality", "relationships", "business")
    """
    
    focused_instructions = {
        "quality": return_instructions_root() + "\n\nFOCUS: Prioritize data quality assessment and improvement recommendations.",
        "relationships": return_instructions_root() + "\n\nFOCUS: Emphasize data relationships and integration opportunities.",
        "business": return_instructions_root() + "\n\nFOCUS: Highlight business insights and decision-making support."
    }
    
    return Agent(
        model=env_config.get("ROOT_AGENT_MODEL", "gemini-2.5-flash"),
        name=f"{ROOT_AGENT_NAME}_{focus_type}_focused",
        description=f"Excel Intelligence Agent focused on {focus_type} analysis",
        instruction=focused_instructions.get(focus_type, return_instructions_root()),
        tools=[
            prepare_file_analysis,
            execute_concurrent_analysis,
            integrate_analysis_results,
            generate_intelligent_response,
            get_analysis_summary
        ],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.1 if focus_type == "quality" else 0.3,
            max_output_tokens=6144
        ),
        before_agent_callback=setup_before_agent_call
    )


def create_high_performance_agent() -> Agent:
    """
    Create a high-performance version using the premium model.
    """
    return Agent(
        model=env_config.get("ORCHESTRATOR_MODEL", "gemini-2.5-pro"),
        name=f"{ROOT_AGENT_NAME}_premium",
        description="High-performance Excel Intelligence Agent using premium models",
        instruction=return_instructions_root(),
        tools=[
            prepare_file_analysis,
            execute_concurrent_analysis,
            integrate_analysis_results,
            generate_intelligent_response,
            store_analysis_state,
            retrieve_analysis_state,
            get_analysis_summary
        ],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.01,
            top_p=0.98,
            top_k=50,
            max_output_tokens=12288
        ),
        before_agent_callback=setup_before_agent_call,
        after_model_callback=cleanup_after_agent_call
    )