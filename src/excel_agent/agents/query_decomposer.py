"""Query decomposition agent for breaking complex queries into simpler sub-queries (ST-Raptor inspired)."""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from ..models.base import AgentRequest, AgentResponse
from ..models.feature_tree import FeatureTree
from ..utils.prompt_templates import ContextualPrompts
from ..utils.siliconflow_client import SiliconFlowClient
from .base import BaseAgent


@dataclass
class SubQuery:
    """Represents a decomposed sub-query."""
    query_text: str
    needs_retrieval: bool
    priority: int = 1
    dependencies: List[int] = None
    query_type: str = "general"  # general, aggregation, filter, join
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class QueryDecomposer(BaseAgent):
    """Agent for decomposing complex queries into manageable sub-queries."""
    
    def __init__(self):
        super().__init__("query_decomposer")
        self.llm_client = SiliconFlowClient()
    
    def classify_query_complexity(self, query: str) -> str:
        """Classify query complexity level."""
        query_lower = query.lower()
        
        # Complex indicators
        complex_keywords = [
            'join', 'merge', 'combine', 'compare', 'correlate',
            'multiple', 'across', 'between', 'relationship',
            'and then', 'after that', 'followed by'
        ]
        
        # Aggregation indicators
        agg_keywords = [
            'sum', 'count', 'average', 'max', 'min', 'total',
            'group by', 'having', 'distinct'
        ]
        
        # Filter indicators  
        filter_keywords = [
            'where', 'filter', 'select', 'find', 'search',
            'contains', 'like', 'match'
        ]
        
        complex_score = sum(1 for kw in complex_keywords if kw in query_lower)
        agg_score = sum(1 for kw in agg_keywords if kw in query_lower)
        filter_score = sum(1 for kw in filter_keywords if kw in query_lower)
        
        # Word count as complexity indicator
        word_count = len(query.split())
        
        if complex_score > 0 or word_count > 20:
            return "complex"
        elif agg_score > 1 or (agg_score > 0 and filter_score > 0):
            return "medium"
        else:
            return "simple"
    
    def extract_subqueries_from_response(self, response: str) -> List[SubQuery]:
        """Extract sub-queries from LLM response."""
        subqueries = []
        lines = response.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for pattern: [Query] <text> [Retrieve] <true/false>
            query_match = re.search(r'\\[Query\\]\\s*(.+?)\\s*\\[Retrieve\\]\\s*(true|false)', line, re.IGNORECASE)
            
            if query_match:
                query_text = query_match.group(1).strip()
                needs_retrieval = query_match.group(2).lower() == 'true'
                
                # Determine query type
                query_type = self._classify_subquery_type(query_text)
                
                subquery = SubQuery(
                    query_text=query_text,
                    needs_retrieval=needs_retrieval,
                    priority=i + 1,
                    query_type=query_type
                )
                
                subqueries.append(subquery)
        
        return subqueries
    
    def _classify_subquery_type(self, query: str) -> str:
        """Classify the type of sub-query."""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ['sum', 'count', 'average', 'max', 'min', 'total']):
            return "aggregation"
        elif any(kw in query_lower for kw in ['join', 'merge', 'combine']):
            return "join"
        elif any(kw in query_lower for kw in ['where', 'filter', 'find', 'select']):
            return "filter"
        else:
            return "general"
    
    def decompose_query(self, query: str, feature_tree: FeatureTree) -> List[SubQuery]:
        """Decompose a complex query into sub-queries."""
        # Check if decomposition is needed
        complexity = self.classify_query_complexity(query)
        
        if complexity == "simple":
            return [SubQuery(
                query_text=query,
                needs_retrieval=True,
                priority=1,
                query_type="general"
            )]
        
        # Generate decomposition prompt
        schema = feature_tree.__index__()
        prompt = ContextualPrompts.generate_decomposition_prompt(query, schema)
        
        try:
            # Call LLM for decomposition
            response = self.llm_client.generate_response(
                prompt=prompt,
                temperature=0.3,
                max_tokens=800
            )
            
            # Extract sub-queries
            subqueries = self.extract_subqueries_from_response(response)
            
            if not subqueries:
                # Fallback: create simple decomposition
                subqueries = self._create_fallback_decomposition(query)
            
            # Add dependencies based on query types
            self._add_dependencies(subqueries)
            
            return subqueries
            
        except Exception as e:
            self.logger.error(f"Failed to decompose query: {e}")
            # Return original query as single sub-query
            return [SubQuery(
                query_text=query,
                needs_retrieval=True,
                priority=1,
                query_type="general"
            )]
    
    def _create_fallback_decomposition(self, query: str) -> List[SubQuery]:
        """Create simple fallback decomposition."""
        query_lower = query.lower()
        subqueries = []
        
        # Split by common conjunctions
        if ' and ' in query_lower:
            parts = query.split(' and ')
            for i, part in enumerate(parts):
                if part.strip():
                    subqueries.append(SubQuery(
                        query_text=part.strip(),
                        needs_retrieval=True,
                        priority=i + 1,
                        query_type=self._classify_subquery_type(part)
                    ))
        else:
            # Single query
            subqueries.append(SubQuery(
                query_text=query,
                needs_retrieval=True,
                priority=1,
                query_type=self._classify_subquery_type(query)
            ))
        
        return subqueries
    
    def _add_dependencies(self, subqueries: List[SubQuery]):
        """Add dependencies between sub-queries."""
        for i, subquery in enumerate(subqueries):
            # Aggregation queries usually depend on filter queries
            if subquery.query_type == "aggregation":
                for j, other in enumerate(subqueries):
                    if j < i and other.query_type == "filter":
                        subquery.dependencies.append(j)
            
            # Join queries depend on individual table queries
            elif subquery.query_type == "join":
                for j, other in enumerate(subqueries):
                    if j < i and other.query_type in ["filter", "general"]:
                        subquery.dependencies.append(j)
    
    def optimize_execution_order(self, subqueries: List[SubQuery]) -> List[SubQuery]:
        """Optimize the execution order of sub-queries."""
        # Topological sort based on dependencies
        sorted_queries = []
        remaining = subqueries.copy()
        
        while remaining:
            # Find queries with no unfulfilled dependencies
            ready = []
            for query in remaining:
                if all(dep in [sq.priority - 1 for sq in sorted_queries] for dep in query.dependencies):
                    ready.append(query)
            
            if not ready:
                # Break circular dependencies by taking the first one
                ready = [remaining[0]]
            
            # Sort ready queries by priority
            ready.sort(key=lambda x: x.priority)
            
            # Add the first ready query
            next_query = ready[0]
            sorted_queries.append(next_query)
            remaining.remove(next_query)
        
        return sorted_queries
    
    def merge_subquery_results(self, subqueries: List[SubQuery], results: List[Any]) -> Dict[str, Any]:
        """Merge results from multiple sub-queries."""
        if len(results) == 1:
            return {"final_result": results[0], "subquery_count": 1}
        
        merged_result = {
            "subquery_results": [],
            "subquery_count": len(results),
            "execution_order": [sq.priority for sq in subqueries]
        }
        
        for i, (subquery, result) in enumerate(zip(subqueries, results)):
            merged_result["subquery_results"].append({
                "query": subquery.query_text,
                "type": subquery.query_type,
                "result": result,
                "order": i + 1
            })
        
        # Try to combine results for final answer
        try:
            if all(isinstance(r, (int, float)) for r in results):
                # Numeric results - sum them
                merged_result["final_result"] = sum(results)
            elif all(isinstance(r, list) for r in results):
                # List results - combine them
                final_list = []
                for r in results:
                    final_list.extend(r)
                merged_result["final_result"] = final_list
            else:
                # Mixed results - use the last one as primary
                merged_result["final_result"] = results[-1]
        except Exception:
            merged_result["final_result"] = results[-1]
        
        return merged_result
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process query decomposition request."""
        try:
            query = request.data.get("query", "")
            feature_tree = request.data.get("feature_tree")
            
            if not query or not feature_tree:
                return AgentResponse(
                    request_id=request.request_id,
                    agent_name=self.name,
                    status="error",
                    error_message="Missing query or feature_tree in request"
                )
            
            # Decompose query
            subqueries = self.decompose_query(query, feature_tree)
            
            # Optimize execution order
            optimized_queries = self.optimize_execution_order(subqueries)
            
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status="completed",
                data={
                    "original_query": query,
                    "complexity": self.classify_query_complexity(query),
                    "subqueries": [
                        {
                            "text": sq.query_text,
                            "needs_retrieval": sq.needs_retrieval,
                            "priority": sq.priority,
                            "type": sq.query_type,
                            "dependencies": sq.dependencies
                        }
                        for sq in optimized_queries
                    ],
                    "execution_order": [sq.priority for sq in optimized_queries]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in query decomposition: {e}")
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status="error",
                error_message=str(e)
            )


def query_decompose(feature_tree: FeatureTree, query: str) -> List[SubQuery]:
    """Standalone function for query decomposition (ST-Raptor style)."""
    decomposer = QueryDecomposer()
    return decomposer.decompose_query(query, feature_tree)


def flatten_nested_list(value: List[Any]) -> List[Any]:
    """Utility function to flatten nested lists (from ST-Raptor)."""
    res = []
    for x in value:
        if isinstance(x, list):
            res.extend(flatten_nested_list(x))
        else:
            res.append(x)
    return res


def bool_string(s: str) -> bool:
    """Convert string to boolean (from ST-Raptor)."""
    s = s.lower()
    if s in ["true", "t", "yes", "y", "1", "on"]:
        return True
    return False