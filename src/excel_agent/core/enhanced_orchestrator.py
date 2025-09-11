"""Enhanced orchestrator with ST-Raptor optimizations for Excel intelligent analysis."""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from ..models.base import AgentRequest, AgentResponse
from ..models.feature_tree import FeatureTree
from ..utils.config import get_config
from ..utils.cache_manager import get_cache_manager
from ..utils.prompt_templates import ContextualPrompts, prompt_optimizer
from ..agents.file_ingest import FileIngestAgent
from ..agents.embedding_agent import EmbeddingAgent
from ..agents.query_decomposer import QueryDecomposer
from ..agents.code_generation import CodeGenerationAgent
from ..agents.execution import ExecutionAgent
from ..agents.verification_agent import VerificationAgent
from ..utils.logging import setup_logger


class EnhancedOrchestrator:
    """Enhanced orchestrator implementing ST-Raptor optimizations."""
    
    def __init__(self):
        self.config = get_config()
        self.cache_manager = get_cache_manager()
        self.logger = setup_logger(__name__)
        
        # Initialize agents
        self.file_ingest_agent = FileIngestAgent()
        self.embedding_agent = EmbeddingAgent()
        self.query_decomposer = QueryDecomposer()
        self.code_generation_agent = CodeGenerationAgent()
        self.execution_agent = ExecutionAgent()
        self.verification_agent = VerificationAgent()
        
        # Processing statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "cache_hits": 0,
            "average_processing_time": 0.0,
            "verification_pass_rate": 0.0
        }
    
    async def process_user_request(self, user_request: str, file_path: str, context: Optional[Dict[str, Any]] = None, use_st_raptor: bool = True) -> Dict[str, Any]:
        """
        Main entry point for processing user requests (ST-Raptor style pipeline).
        
        This implements the complete pipeline:
        1. File ingestion with feature tree creation
        2. Query decomposition (if needed)
        3. Embedding-based semantic matching
        4. Code generation with optimized prompts
        5. Safe code execution
        6. Two-stage verification
        7. Result compilation and caching
        """
        start_time = datetime.now()
        self.stats["total_requests"] += 1
        
        try:
            self.logger.info(f"Processing request: {user_request[:100]}...")
            
            # Step 1: File ingestion and feature tree creation
            file_result = await self._process_file(file_path)
            if file_result["status"] != "success":
                return file_result
            
            file_id = file_result["file_id"]
            feature_tree = file_result["feature_tree"]
            metadata = file_result["metadata"]
            
            # Step 2: Query analysis and decomposition
            query_analysis = await self._analyze_query(user_request, feature_tree)
            
            # Step 3: Process queries (single or multiple)
            if query_analysis["complexity"] == "simple":
                result = await self._process_simple_query(
                    user_request, feature_tree, file_id, metadata, context
                )
            else:
                result = await self._process_complex_query(
                    user_request, feature_tree, file_id, metadata, query_analysis, context
                )
            
            # Step 4: Compile final result
            processing_time = (datetime.now() - start_time).total_seconds()
            final_result = self._compile_final_result(result, processing_time, query_analysis)
            
            # Update statistics
            if final_result["status"] == "success":
                self.stats["successful_requests"] += 1
            
            self._update_processing_stats(processing_time, final_result.get("reliability_score", 0.0))
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in orchestrator processing: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _process_file(self, file_path: str) -> Dict[str, Any]:
        """Process file with caching and feature tree creation."""
        try:
            path = Path(file_path)
            if not path.exists():
                return {"status": "error", "error_message": f"File not found: {file_path}"}
            
            # Create file ingestion request
            request = AgentRequest(
                request_id=f"file_ingest_{datetime.now().timestamp()}",
                agent_name="file_ingest",
                data={"file_path": str(path.absolute())}
            )
            
            # Process file with enhanced agent
            response = await self.file_ingest_agent.process(request)
            
            if response.status == "error":
                return {"status": "error", "error_message": response.error_message}
            
            # Extract processed data
            file_id = response.data["file_id"]
            stored_data = self.file_ingest_agent.file_store[file_id]
            
            return {
                "status": "success",
                "file_id": file_id,
                "feature_tree": stored_data["feature_tree"],
                "metadata": stored_data["metadata"],
                "embedding_dict": stored_data.get("embedding_dict"),
                "cached": response.data.get("cached", False)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing file: {e}")
            return {"status": "error", "error_message": str(e)}
    
    async def _analyze_query(self, query: str, feature_tree: FeatureTree) -> Dict[str, Any]:
        """Analyze query complexity and prepare for processing."""
        try:
            complexity = self.query_decomposer.classify_query_complexity(query)
            
            analysis = {
                "original_query": query,
                "complexity": complexity,
                "subqueries": [],
                "processing_strategy": "single"
            }
            
            # Decompose if complex
            if complexity in ["medium", "complex"] and self.config.enable_query_decomposition:
                subqueries = self.query_decomposer.decompose_query(query, feature_tree)
                optimized_queries = self.query_decomposer.optimize_execution_order(subqueries)
                
                analysis.update({
                    "subqueries": [
                        {
                            "text": sq.query_text,
                            "needs_retrieval": sq.needs_retrieval,
                            "type": sq.query_type,
                            "priority": sq.priority
                        } for sq in optimized_queries
                    ],
                    "processing_strategy": "decomposed" if len(optimized_queries) > 1 else "single"
                })
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Query analysis failed, using simple strategy: {e}")
            return {
                "original_query": query,
                "complexity": "simple",
                "subqueries": [],
                "processing_strategy": "single"
            }
    
    async def _process_simple_query(self, query: str, feature_tree: FeatureTree, file_id: str, 
                                  metadata: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a simple query with ST-Raptor optimizations."""
        try:
            # Step 1: Semantic matching (if enabled)
            relevant_content = []
            if self.config.enable_embedding_cache:
                stored_data = self.file_ingest_agent.file_store.get(file_id, {})
                embedding_dict = stored_data.get("embedding_dict")
                
                if embedding_dict:
                    matches = self.embedding_agent.match_query_to_content(query, embedding_dict, top_k=5)
                    relevant_content = [{"text": text, "similarity": sim} for text, sim, _ in matches]
            
            # Step 2: Generate optimized context
            processing_context = self._create_processing_context(
                metadata, feature_tree, relevant_content, context
            )
            
            # Step 3: Generate code with optimized prompts
            code_result = await self._generate_code(query, processing_context, "simple")
            if code_result["status"] != "success":
                return code_result
            
            # Step 4: Execute code safely
            execution_result = await self._execute_code(code_result["code"], processing_context)
            if execution_result["status"] != "success":
                return execution_result
            
            # Step 5: Verify results
            verification_result = await self._verify_results(
                query, code_result["code"], execution_result["result"], processing_context
            )
            
            return {
                "status": "success",
                "query": query,
                "generated_code": code_result["code"],
                "execution_result": execution_result["result"],
                "verification": verification_result,
                "reliability_score": verification_result["overall_reliability"],
                "relevant_content": relevant_content,
                "processing_method": "simple_pipeline"
            }
            
        except Exception as e:
            self.logger.error(f"Error in simple query processing: {e}")
            return {"status": "error", "error_message": str(e)}
    
    async def _process_complex_query(self, query: str, feature_tree: FeatureTree, file_id: str,
                                   metadata: Dict[str, Any], query_analysis: Dict[str, Any],
                                   context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a complex query using decomposition and sub-query processing."""
        try:
            subqueries = query_analysis["subqueries"]
            subquery_results = []
            
            # Process each sub-query
            for i, subquery_info in enumerate(subqueries):
                subquery_text = subquery_info["text"]
                
                self.logger.info(f"Processing sub-query {i+1}/{len(subqueries)}: {subquery_text[:50]}...")
                
                # Process sub-query using simple pipeline
                subresult = await self._process_simple_query(
                    subquery_text, feature_tree, file_id, metadata, context
                )
                
                subquery_results.append({
                    "query": subquery_text,
                    "type": subquery_info["type"],
                    "result": subresult,
                    "order": i + 1
                })
                
                # Break on critical failure
                if subresult["status"] != "success" and subquery_info.get("critical", False):
                    return {
                        "status": "error",
                        "error_message": f"Critical sub-query failed: {subresult.get('error_message', 'Unknown error')}",
                        "partial_results": subquery_results
                    }
            
            # Merge sub-query results
            merged_result = self._merge_subquery_results(query, subquery_results, query_analysis)
            
            # Verify merged result
            final_verification = await self._verify_merged_results(query, merged_result)
            
            return {
                "status": "success",
                "query": query,
                "processing_method": "decomposed_pipeline",
                "subquery_count": len(subqueries),
                "subquery_results": subquery_results,
                "merged_result": merged_result,
                "final_verification": final_verification,
                "reliability_score": final_verification.get("overall_reliability", 0.5)
            }
            
        except Exception as e:
            self.logger.error(f"Error in complex query processing: {e}")
            return {"status": "error", "error_message": str(e)}
    
    def _create_processing_context(self, metadata: Dict[str, Any], feature_tree: FeatureTree,
                                 relevant_content: List[Dict], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create optimized processing context."""
        # Compress metadata for token efficiency
        essential_metadata = {
            "file_path": metadata.get("file_path", "unknown"),
            "sheets": metadata.get("sheets", [])[:5],  # Limit to first 5 sheets
            "total_rows": metadata.get("total_rows", 0),
            "total_columns": metadata.get("total_columns", 0),
            "structure_complexity": metadata.get("structure_complexity", "unknown")
        }
        
        # Get tree statistics
        tree_stats = feature_tree.get_statistics()
        
        # Get compressed schema
        schema = prompt_optimizer.truncate_schema(feature_tree.__index__(), max_lines=15)
        
        processing_context = {
            "metadata": essential_metadata,
            "tree_stats": tree_stats,
            "schema": schema,
            "relevant_content": relevant_content[:3],  # Top 3 matches only
            "user_context": context or {}
        }
        
        return processing_context
    
    async def _generate_code(self, query: str, context: Dict[str, Any], query_type: str = "general") -> Dict[str, Any]:
        """Generate code using optimized prompts."""
        try:
            request = AgentRequest(
                request_id=f"code_gen_{datetime.now().timestamp()}",
                agent_name="code_generation",
                data={
                    "query": query,
                    "context": context,
                    "query_type": query_type,
                    "optimize_tokens": True,
                    "max_prompt_tokens": self.config.max_prompt_tokens
                }
            )
            
            response = await self.code_generation_agent.process(request)
            
            if response.status == "error":
                return {"status": "error", "error_message": response.error_message}
            
            return {
                "status": "success",
                "code": response.data["generated_code"],
                "explanation": response.data.get("explanation", ""),
                "token_usage": response.data.get("token_usage", {})
            }
            
        except Exception as e:
            return {"status": "error", "error_message": str(e)}
    
    async def _execute_code(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code safely with context."""
        try:
            request = AgentRequest(
                request_id=f"execution_{datetime.now().timestamp()}",
                agent_name="execution",
                data={
                    "code": code,
                    "context": context,
                    "timeout": self.config.agent_timeout_seconds
                }
            )
            
            response = await self.execution_agent.process(request)
            
            if response.status == "error":
                return {"status": "error", "error_message": response.error_message}
            
            return {
                "status": "success",
                "result": response.data["result"],
                "execution_time": response.data.get("execution_time", 0),
                "warnings": response.data.get("warnings", [])
            }
            
        except Exception as e:
            return {"status": "error", "error_message": str(e)}
    
    async def _verify_results(self, query: str, code: str, result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform two-stage verification."""
        try:
            request = AgentRequest(
                request_id=f"verification_{datetime.now().timestamp()}",
                agent_name="verification",
                data={
                    "query": query,
                    "code": code,
                    "result": result,
                    "context": context,
                    "stage": "comprehensive"
                }
            )
            
            response = await self.verification_agent.process(request)
            
            if response.status == "error":
                self.logger.warning(f"Verification failed: {response.error_message}")
                return {
                    "verification_results": {},
                    "overall_reliability": 0.0,
                    "passed_verification": False,
                    "error": response.error_message
                }
            
            return response.data
            
        except Exception as e:
            self.logger.warning(f"Verification error: {e}")
            return {
                "verification_results": {},
                "overall_reliability": 0.0,
                "passed_verification": False,
                "error": str(e)
            }
    
    def _merge_subquery_results(self, original_query: str, subquery_results: List[Dict], 
                              query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from multiple sub-queries."""
        successful_results = [sr for sr in subquery_results if sr["result"]["status"] == "success"]
        
        if not successful_results:
            return {"status": "error", "error_message": "No successful sub-queries"}
        
        # Extract results
        results = [sr["result"]["execution_result"] for sr in successful_results]
        codes = [sr["result"]["generated_code"] for sr in successful_results]
        
        merged = {
            "status": "success",
            "original_query": original_query,
            "successful_subqueries": len(successful_results),
            "total_subqueries": len(subquery_results),
            "individual_results": results,
            "combined_codes": codes
        }
        
        # Try to combine results intelligently
        try:
            if len(results) == 1:
                merged["final_answer"] = results[0]
            elif all(isinstance(r, (int, float)) for r in results):
                merged["final_answer"] = sum(results)
            elif all(isinstance(r, list) for r in results):
                combined_list = []
                for r in results:
                    combined_list.extend(r if isinstance(r, list) else [r])
                merged["final_answer"] = combined_list
            else:
                merged["final_answer"] = results  # Return all results
        except Exception:
            merged["final_answer"] = results
        
        return merged
    
    async def _verify_merged_results(self, query: str, merged_result: Dict[str, Any]) -> Dict[str, Any]:
        """Verify the merged results of complex queries."""
        try:
            # Simple verification for merged results
            final_answer = merged_result.get("final_answer")
            
            verification = {
                "overall_reliability": 0.7,  # Default for merged results
                "passed_verification": True,
                "confidence_score": 0.7,
                "issues": []
            }
            
            # Check if we have a reasonable answer
            if final_answer is None:
                verification["passed_verification"] = False
                verification["issues"].append("No final answer generated")
                verification["overall_reliability"] = 0.0
            elif merged_result.get("successful_subqueries", 0) == 0:
                verification["passed_verification"] = False
                verification["issues"].append("No successful sub-queries")
                verification["overall_reliability"] = 0.0
            
            return verification
            
        except Exception as e:
            return {
                "overall_reliability": 0.0,
                "passed_verification": False,
                "error": str(e)
            }
    
    def _compile_final_result(self, result: Dict[str, Any], processing_time: float, 
                            query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compile the final result with metadata."""
        final_result = {
            "status": result.get("status", "unknown"),
            "processing_time_seconds": round(processing_time, 2),
            "query_analysis": query_analysis,
            "cache_hits": self.stats.get("cache_hits", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        if result.get("status") == "success":
            final_result.update({
                "answer": result.get("execution_result") or result.get("merged_result", {}).get("final_answer"),
                "generated_code": result.get("generated_code") or "Multiple code blocks (decomposed query)",
                "reliability_score": result.get("reliability_score", 0.5),
                "verification_passed": result.get("verification", {}).get("passed_verification", False),
                "processing_method": result.get("processing_method", "unknown")
            })
            
            # Add relevant content if available
            if "relevant_content" in result:
                final_result["semantic_matches"] = result["relevant_content"]
        
        else:
            final_result.update({
                "error_message": result.get("error_message", "Unknown error"),
                "partial_results": result.get("partial_results", [])
            })
        
        return final_result
    
    def _update_processing_stats(self, processing_time: float, reliability_score: float):
        """Update processing statistics."""
        # Update average processing time
        current_avg = self.stats["average_processing_time"]
        total_requests = self.stats["total_requests"]
        
        if total_requests > 1:
            self.stats["average_processing_time"] = (
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            )
        else:
            self.stats["average_processing_time"] = processing_time
        
        # Update verification pass rate
        current_pass_rate = self.stats["verification_pass_rate"]
        if reliability_score >= 0.7:  # Consider 0.7+ as passing
            if total_requests > 1:
                self.stats["verification_pass_rate"] = (
                    (current_pass_rate * (total_requests - 1) + 1.0) / total_requests
                )
            else:
                self.stats["verification_pass_rate"] = 1.0
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get comprehensive workflow statistics."""
        cache_stats = self.cache_manager.get_cache_stats()
        
        return {
            "processing_stats": self.stats,
            "cache_stats": cache_stats,
            "configuration": {
                "enable_cache": self.config.enable_cache,
                "enable_embedding_cache": self.config.enable_embedding_cache,
                "enable_query_decomposition": self.config.enable_query_decomposition,
                "max_prompt_tokens": self.config.max_prompt_tokens
            },
            "performance_metrics": {
                "success_rate": (self.stats["successful_requests"] / max(self.stats["total_requests"], 1)) * 100,
                "average_processing_time": self.stats["average_processing_time"],
                "verification_pass_rate": self.stats["verification_pass_rate"] * 100,
                "cache_hit_rate": (cache_stats.get("total_files", 0) / max(self.stats["total_requests"], 1)) * 100
            }
        }