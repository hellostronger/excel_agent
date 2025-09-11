"""Intelligent orchestrator that can choose between original and ST-Raptor processing pipelines."""

import os
import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

from ..models.base import AgentRequest, AgentResponse
from ..utils.config import get_config
from ..utils.logging import setup_logger

# Import both processing pipelines
from ..agents.file_ingest import FileIngestAgent  # Original pipeline
from ..agents.st_raptor_agent import STRaptorAgent  # Enhanced pipeline
from ..agents.query_decomposer import QueryDecomposer
from ..agents.code_generation import CodeGenerationAgent
from ..agents.execution import ExecutionAgent
from ..agents.verification_agent import VerificationAgent


class IntelligentOrchestrator:
    """Orchestrator that intelligently chooses between processing pipelines."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logger(__name__)
        
        # Initialize both processing pipelines
        self.original_file_agent = FileIngestAgent()
        self.st_raptor_agent = STRaptorAgent()
        
        # Initialize other agents
        self.query_decomposer = QueryDecomposer()
        self.code_generation_agent = CodeGenerationAgent()
        self.execution_agent = ExecutionAgent()
        self.verification_agent = VerificationAgent()
        
        # Processing statistics
        self.stats = {
            "total_requests": 0,
            "original_pipeline_used": 0,
            "st_raptor_pipeline_used": 0,
            "success_rate": 0.0,
            "average_processing_time": 0.0
        }
    
    async def process_user_request(
        self, 
        user_request: str, 
        file_path: str, 
        context: Optional[Dict[str, Any]] = None,
        processing_mode: str = "auto"  # "auto", "original", "st_raptor"
    ) -> Dict[str, Any]:
        """
        Main entry point for processing user requests.
        
        Args:
            user_request: Natural language query
            file_path: Path to Excel file
            context: Optional context information
            processing_mode: "auto" (intelligent choice), "original", or "st_raptor"
        """
        start_time = datetime.now()
        self.stats["total_requests"] += 1
        
        try:
            self.logger.info(f"Processing request: {user_request[:100]}...")
            
            # Step 1: Choose processing pipeline
            chosen_mode = self._choose_processing_pipeline(user_request, file_path, processing_mode)
            
            # Step 2: Process file with chosen pipeline
            file_result = await self._process_file_with_pipeline(file_path, chosen_mode)
            if file_result["status"] != "success":
                return file_result
            
            # Step 3: Process query based on pipeline capabilities
            if chosen_mode == "st_raptor":
                result = await self._process_with_st_raptor_pipeline(
                    user_request, file_result, context
                )
            else:
                result = await self._process_with_original_pipeline(
                    user_request, file_result, context
                )
            
            # Step 4: Compile final result
            processing_time = (datetime.now() - start_time).total_seconds()
            final_result = self._compile_final_result(result, processing_time, chosen_mode)
            
            # Update statistics
            self._update_statistics(chosen_mode, final_result.get("status") == "success", processing_time)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in intelligent orchestrator: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _choose_processing_pipeline(self, query: str, file_path: str, mode: str) -> str:
        """Intelligently choose between original and ST-Raptor pipeline."""
        if mode in ["original", "st_raptor"]:
            return mode
        
        # Auto mode - make intelligent choice based on various factors
        factors = {
            "query_complexity": self._assess_query_complexity(query),
            "file_complexity": self._assess_file_complexity(file_path),
            "performance_requirements": self._assess_performance_requirements(query),
            "feature_requirements": self._assess_feature_requirements(query)
        }
        
        score = 0
        
        # Complex queries benefit from ST-Raptor
        if factors["query_complexity"] in ["medium", "high"]:
            score += 2
        
        # Large or complex files benefit from ST-Raptor caching
        if factors["file_complexity"] in ["medium", "high"]:
            score += 1
        
        # Semantic search requirements
        if factors["feature_requirements"].get("needs_semantic_search", False):
            score += 2
        
        # Multi-step queries benefit from decomposition
        if factors["feature_requirements"].get("needs_decomposition", False):
            score += 1
        
        # Verification requirements
        if factors["feature_requirements"].get("needs_verification", False):
            score += 1
        
        # Choose pipeline based on score
        if score >= 3:
            chosen_mode = "st_raptor"
            self.logger.info(f"Chose ST-Raptor pipeline (score: {score})")
        else:
            chosen_mode = "original"
            self.logger.info(f"Chose original pipeline (score: {score})")
        
        return chosen_mode
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity."""
        query_lower = query.lower()
        
        # High complexity indicators
        high_complexity_keywords = [
            "join", "merge", "combine", "compare", "correlate", "relationship",
            "across multiple", "between tables", "complex analysis"
        ]
        
        # Medium complexity indicators
        medium_complexity_keywords = [
            "group by", "aggregate", "summarize", "pivot", "trend",
            "calculate", "find all", "list all"
        ]
        
        if any(kw in query_lower for kw in high_complexity_keywords):
            return "high"
        elif any(kw in query_lower for kw in medium_complexity_keywords):
            return "medium"
        elif len(query.split()) > 15:
            return "medium"
        else:
            return "low"
    
    def _assess_file_complexity(self, file_path: str) -> str:
        """Assess file complexity based on basic file information."""
        try:
            path = Path(file_path)
            if not path.exists():
                return "unknown"
            
            file_size = path.stat().st_size / (1024 * 1024)  # MB
            
            if file_size > 50:
                return "high"
            elif file_size > 10:
                return "medium"
            else:
                return "low"
        except Exception:
            return "medium"  # Default assumption
    
    def _assess_performance_requirements(self, query: str) -> str:
        """Assess performance requirements from query."""
        urgent_keywords = ["quickly", "fast", "immediate", "urgent"]
        detailed_keywords = ["detailed", "comprehensive", "thorough", "complete"]
        
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in urgent_keywords):
            return "fast"
        elif any(kw in query_lower for kw in detailed_keywords):
            return "comprehensive"
        else:
            return "balanced"
    
    def _assess_feature_requirements(self, query: str) -> Dict[str, bool]:
        """Assess what features are needed based on query."""
        query_lower = query.lower()
        
        return {
            "needs_semantic_search": any(kw in query_lower for kw in [
                "similar", "like", "related", "find", "search", "match"
            ]),
            "needs_decomposition": any(kw in query_lower for kw in [
                " and ", " then ", "step by step", "multiple", "also"
            ]) and len(query.split()) > 20,
            "needs_verification": any(kw in query_lower for kw in [
                "accurate", "verify", "confirm", "check", "validate"
            ]),
            "needs_caching": "repeat" in query_lower or "again" in query_lower
        }
    
    async def _process_file_with_pipeline(self, file_path: str, pipeline: str) -> Dict[str, Any]:
        """Process file with the chosen pipeline."""
        try:
            request = AgentRequest(
                request_id=f"file_process_{datetime.now().timestamp()}",
                agent_name=f"{pipeline}_file_processing",
                data={"file_path": file_path}
            )
            
            if pipeline == "st_raptor":
                response = await self.st_raptor_agent.process(request)
            else:
                # For original pipeline, we need to use FileIngestRequest
                from ..models.agents import FileIngestRequest
                original_request = FileIngestRequest(
                    request_id=request.request_id,
                    file_path=file_path
                )
                response = await self.original_file_agent.process(original_request)
            
            if hasattr(response, 'status') and response.status in ["error", "failed"]:
                return {
                    "status": "error",
                    "error_message": getattr(response, 'error_message', 'Unknown error'),
                    "pipeline": pipeline
                }
            
            return {
                "status": "success",
                "pipeline": pipeline,
                "response": response,
                "data": response.data if hasattr(response, 'data') else response.result
            }
            
        except Exception as e:
            self.logger.error(f"Error processing file with {pipeline} pipeline: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "pipeline": pipeline
            }
    
    async def _process_with_st_raptor_pipeline(
        self, 
        query: str, 
        file_result: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process query using ST-Raptor enhanced pipeline."""
        try:
            # Get ST-Raptor processed data
            response = file_result["response"]
            file_id = response.data["file_id"]
            
            # Get enhanced data from ST-Raptor agent
            processed_data = self.st_raptor_agent.get_processed_file(file_id)
            if not processed_data:
                return {"status": "error", "error_message": "Failed to get ST-Raptor processed data"}
            
            feature_tree = processed_data["feature_tree"]
            embedding_dict = processed_data.get("embedding_dict")
            
            # Step 1: Query decomposition
            subqueries = self.query_decomposer.decompose_query(query, feature_tree)
            
            # Step 2: Semantic matching (if available)
            semantic_matches = []
            if embedding_dict:
                from .embedding_agent import EmbeddingAgent
                embedding_agent = EmbeddingAgent()
                semantic_matches = embedding_agent.match_query_to_content(query, embedding_dict, top_k=5)
            
            # Step 3: Process queries
            if len(subqueries) == 1:
                # Simple query processing
                result = await self._process_single_query(
                    query, feature_tree, processed_data["metadata"], semantic_matches, context
                )
            else:
                # Complex query processing
                result = await self._process_multiple_queries(
                    query, subqueries, feature_tree, processed_data["metadata"], context
                )
            
            # Step 4: Verification
            if result.get("status") == "success":
                verification_result = await self._verify_result(
                    query, result.get("generated_code", ""), result.get("result", ""), context or {}
                )
                result["verification"] = verification_result
                result["reliability_score"] = verification_result.get("overall_reliability", 0.5)
            
            result["processing_pipeline"] = "st_raptor"
            result["semantic_matches"] = [{"text": text, "similarity": sim} for text, sim, _ in semantic_matches]
            result["query_decomposition"] = len(subqueries) > 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in ST-Raptor pipeline: {e}")
            return {"status": "error", "error_message": str(e)}
    
    async def _process_with_original_pipeline(
        self, 
        query: str, 
        file_result: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process query using original pipeline."""
        try:
            response = file_result["response"]
            
            # Extract file information
            if hasattr(response, 'result'):
                file_data = response.result
            else:
                file_data = response.data
            
            file_id = file_data["file_id"]
            metadata = file_data["metadata"]
            
            # Simple processing context
            processing_context = {
                "file_id": file_id,
                "metadata": metadata,
                "query": query,
                "user_context": context or {}
            }
            
            # Generate code
            code_result = await self._generate_code_simple(query, processing_context)
            if code_result.get("status") != "success":
                return code_result
            
            # Execute code
            execution_result = await self._execute_code_simple(
                code_result["code"], processing_context
            )
            if execution_result.get("status") != "success":
                return execution_result
            
            return {
                "status": "success",
                "query": query,
                "generated_code": code_result["code"],
                "result": execution_result["result"],
                "processing_pipeline": "original",
                "semantic_matches": [],
                "query_decomposition": False
            }
            
        except Exception as e:
            self.logger.error(f"Error in original pipeline: {e}")
            return {"status": "error", "error_message": str(e)}
    
    async def _process_single_query(
        self, query: str, feature_tree, metadata, semantic_matches, context
    ) -> Dict[str, Any]:
        """Process a single query with ST-Raptor features."""
        # Implementation similar to your original _process_simple_query
        processing_context = {
            "metadata": metadata,
            "feature_tree": feature_tree,
            "semantic_matches": semantic_matches,
            "user_context": context or {}
        }
        
        # Generate and execute code
        code_result = await self._generate_code_enhanced(query, processing_context)
        if code_result.get("status") != "success":
            return code_result
        
        execution_result = await self._execute_code_simple(code_result["code"], processing_context)
        if execution_result.get("status") != "success":
            return execution_result
        
        return {
            "status": "success",
            "generated_code": code_result["code"],
            "result": execution_result["result"]
        }
    
    async def _process_multiple_queries(
        self, original_query: str, subqueries, feature_tree, metadata, context
    ) -> Dict[str, Any]:
        """Process multiple sub-queries."""
        results = []
        
        for subquery in subqueries:
            result = await self._process_single_query(
                subquery.query_text, feature_tree, metadata, [], context
            )
            results.append({
                "query": subquery.query_text,
                "result": result
            })
        
        # Combine results (simplified)
        successful_results = [r for r in results if r["result"].get("status") == "success"]
        
        if not successful_results:
            return {"status": "error", "error_message": "All sub-queries failed"}
        
        # Simple result combination
        combined_result = successful_results[-1]["result"]["result"]  # Take last result as primary
        
        return {
            "status": "success",
            "generated_code": "Multiple code blocks executed",
            "result": combined_result,
            "subquery_results": results
        }
    
    async def _generate_code_simple(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code using original method."""
        try:
            request = AgentRequest(
                request_id=f"code_gen_{datetime.now().timestamp()}",
                agent_name="code_generation",
                data={
                    "query": query,
                    "context": context
                }
            )
            
            response = await self.code_generation_agent.process(request)
            
            if hasattr(response, 'status') and response.status == "error":
                return {"status": "error", "error_message": response.error_message}
            
            return {
                "status": "success",
                "code": response.data.get("generated_code", "")
            }
            
        except Exception as e:
            return {"status": "error", "error_message": str(e)}
    
    async def _generate_code_enhanced(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code using enhanced ST-Raptor context."""
        # Use semantic matches and feature tree for better context
        enhanced_context = context.copy()
        enhanced_context["use_enhanced_prompts"] = True
        
        return await self._generate_code_simple(query, enhanced_context)
    
    async def _execute_code_simple(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code safely."""
        try:
            request = AgentRequest(
                request_id=f"execution_{datetime.now().timestamp()}",
                agent_name="execution",
                data={
                    "code": code,
                    "context": context
                }
            )
            
            response = await self.execution_agent.process(request)
            
            if hasattr(response, 'status') and response.status == "error":
                return {"status": "error", "error_message": response.error_message}
            
            return {
                "status": "success",
                "result": response.data.get("result", "")
            }
            
        except Exception as e:
            return {"status": "error", "error_message": str(e)}
    
    async def _verify_result(self, query: str, code: str, result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify result using verification agent."""
        try:
            request = AgentRequest(
                request_id=f"verification_{datetime.now().timestamp()}",
                agent_name="verification",
                data={
                    "query": query,
                    "code": code,
                    "result": result,
                    "context": context
                }
            )
            
            response = await self.verification_agent.process(request)
            
            if hasattr(response, 'data'):
                return response.data
            else:
                return {"overall_reliability": 0.5, "passed_verification": False}
                
        except Exception as e:
            self.logger.warning(f"Verification failed: {e}")
            return {"overall_reliability": 0.0, "passed_verification": False}
    
    def _compile_final_result(
        self, result: Dict[str, Any], processing_time: float, pipeline: str
    ) -> Dict[str, Any]:
        """Compile final result with metadata."""
        return {
            "status": result.get("status", "unknown"),
            "answer": result.get("result", result.get("answer")),
            "generated_code": result.get("generated_code", ""),
            "processing_time_seconds": round(processing_time, 2),
            "pipeline_used": pipeline,
            "semantic_matches": result.get("semantic_matches", []),
            "query_decomposed": result.get("query_decomposition", False),
            "reliability_score": result.get("reliability_score", 0.5),
            "verification_passed": result.get("verification", {}).get("passed_verification", False),
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_statistics(self, pipeline: str, success: bool, processing_time: float):
        """Update processing statistics."""
        if pipeline == "st_raptor":
            self.stats["st_raptor_pipeline_used"] += 1
        else:
            self.stats["original_pipeline_used"] += 1
        
        # Update success rate
        if success:
            current_successes = self.stats["success_rate"] * (self.stats["total_requests"] - 1)
            self.stats["success_rate"] = (current_successes + 1) / self.stats["total_requests"]
        else:
            current_successes = self.stats["success_rate"] * (self.stats["total_requests"] - 1)
            self.stats["success_rate"] = current_successes / self.stats["total_requests"]
        
        # Update average processing time
        current_avg = self.stats["average_processing_time"]
        if self.stats["total_requests"] > 1:
            self.stats["average_processing_time"] = (
                (current_avg * (self.stats["total_requests"] - 1) + processing_time) 
                / self.stats["total_requests"]
            )
        else:
            self.stats["average_processing_time"] = processing_time
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        total = self.stats["total_requests"]
        
        return {
            "total_requests": total,
            "pipeline_usage": {
                "original": self.stats["original_pipeline_used"],
                "st_raptor": self.stats["st_raptor_pipeline_used"],
                "original_percentage": (self.stats["original_pipeline_used"] / max(total, 1)) * 100,
                "st_raptor_percentage": (self.stats["st_raptor_pipeline_used"] / max(total, 1)) * 100
            },
            "performance": {
                "success_rate": self.stats["success_rate"] * 100,
                "average_processing_time": self.stats["average_processing_time"]
            },
            "recommendations": self._get_pipeline_recommendations()
        }
    
    def _get_pipeline_recommendations(self) -> List[str]:
        """Get recommendations based on usage patterns."""
        recommendations = []
        
        if self.stats["st_raptor_pipeline_used"] > self.stats["original_pipeline_used"]:
            recommendations.append("Consider enabling ST-Raptor features by default for better performance")
        
        if self.stats["average_processing_time"] > 10:
            recommendations.append("Enable caching to improve processing times")
        
        if self.stats["success_rate"] < 0.8:
            recommendations.append("Consider using ST-Raptor verification for better accuracy")
        
        return recommendations