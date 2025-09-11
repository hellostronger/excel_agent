"""Two-stage verification agent for validating query results (ST-Raptor inspired)."""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.base import AgentRequest, AgentResponse
from ..models.feature_tree import FeatureTree
from ..utils.prompt_templates import ContextualPrompts, PromptTemplates
from ..utils.siliconflow_client import SiliconFlowClient
from .base import BaseAgent


class VerificationStage(Enum):
    """Verification stages."""
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class VerificationResult:
    """Result of verification process."""
    stage: VerificationStage
    passed: bool
    confidence_score: float
    issues: List[str]
    explanation: str
    suggested_fixes: List[str] = None
    
    def __post_init__(self):
        if self.suggested_fixes is None:
            self.suggested_fixes = []


class VerificationAgent(BaseAgent):
    """Agent for two-stage verification of query results."""
    
    def __init__(self):
        super().__init__("verification_agent")
        self.llm_client = SiliconFlowClient()
        self.max_retries = 3
    
    def forward_verification(self, query: str, code: str, result: Any, context: Dict[str, Any]) -> VerificationResult:
        """
        Forward verification: Check if code execution result matches the query intent.
        This validates the logical correctness of the generated code.
        """
        try:
            # Generate forward verification prompt
            prompt = PromptTemplates.FORWARD_VERIFICATION.format(
                query=query,
                code=self._truncate_code(code, 500),
                result=self._format_result(result, 200)
            )
            
            # Call LLM for verification
            response = self.llm_client.generate_response(
                prompt=prompt,
                temperature=0.1,  # Low temperature for consistent verification
                max_tokens=300
            )
            
            # Parse verification result
            return self._parse_forward_verification_response(response, query, code, result)
            
        except Exception as e:
            self.logger.error(f"Forward verification failed: {e}")
            return VerificationResult(
                stage=VerificationStage.FORWARD,
                passed=False,
                confidence_score=0.0,
                issues=[f"Verification failed: {str(e)}"],
                explanation="Failed to perform forward verification due to technical error"
            )
    
    def backward_verification(self, original_query: str, answer: Any, context: Dict[str, Any]) -> VerificationResult:
        """
        Backward verification: Check if the answer satisfies the original question.
        This validates the semantic correctness of the final result.
        """
        try:
            # Generate backward verification prompt
            prompt = PromptTemplates.BACKWARD_VERIFICATION.format(
                original_query=original_query,
                answer=self._format_result(answer, 300),
                context=self._format_context(context)
            )
            
            # Call LLM for verification
            response = self.llm_client.generate_response(
                prompt=prompt,
                temperature=0.1,
                max_tokens=400
            )
            
            # Parse verification result
            return self._parse_backward_verification_response(response, original_query, answer)
            
        except Exception as e:
            self.logger.error(f"Backward verification failed: {e}")
            return VerificationResult(
                stage=VerificationStage.BACKWARD,
                passed=False,
                confidence_score=0.0,
                issues=[f"Verification failed: {str(e)}"],
                explanation="Failed to perform backward verification due to technical error"
            )
    
    def comprehensive_verification(self, query: str, code: str, result: Any, context: Dict[str, Any]) -> Dict[str, VerificationResult]:
        """Perform both forward and backward verification."""
        results = {}
        
        # Forward verification
        forward_result = self.forward_verification(query, code, result, context)
        results["forward"] = forward_result
        
        # Backward verification
        backward_result = self.backward_verification(query, result, context)
        results["backward"] = backward_result
        
        return results
    
    def _parse_forward_verification_response(self, response: str, query: str, code: str, result: Any) -> VerificationResult:
        """Parse forward verification response from LLM."""
        response_lower = response.lower()
        
        # Extract validity
        if "valid? yes" in response_lower or "yes" in response_lower.split("\\n")[0]:
            passed = True
        elif "valid? no" in response_lower or "no" in response_lower.split("\\n")[0]:
            passed = False
        else:
            # Parse based on keywords
            positive_keywords = ["correct", "valid", "matches", "appropriate"]
            negative_keywords = ["incorrect", "invalid", "wrong", "mismatch", "error"]
            
            pos_score = sum(1 for kw in positive_keywords if kw in response_lower)
            neg_score = sum(1 for kw in negative_keywords if kw in response_lower)
            
            passed = pos_score > neg_score
        
        # Extract reason/explanation
        reason_match = re.search(r"reason:?\\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        explanation = reason_match.group(1).strip() if reason_match else response
        
        # Calculate confidence based on response clarity
        confidence_score = self._calculate_confidence(response, passed)
        
        # Extract issues
        issues = self._extract_issues_from_response(response)
        
        return VerificationResult(
            stage=VerificationStage.FORWARD,
            passed=passed,
            confidence_score=confidence_score,
            issues=issues,
            explanation=explanation
        )
    
    def _parse_backward_verification_response(self, response: str, query: str, answer: Any) -> VerificationResult:
        """Parse backward verification response from LLM."""
        # Extract confidence score
        confidence_match = re.search(r"confidence\\s*(?:score)?:?\\s*(\\d+)", response, re.IGNORECASE)
        confidence_score = float(confidence_match.group(1)) / 100.0 if confidence_match else 0.5
        
        # Determine if verification passed
        passed = confidence_score >= 0.7
        
        # Extract issues
        issues_match = re.search(r"issues?\\s*\\(if any\\):?\\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        issues_text = issues_match.group(1).strip() if issues_match else ""
        
        issues = []
        if issues_text and issues_text.lower() not in ["none", "no issues", "n/a"]:
            issues = [issue.strip() for issue in issues_text.split(",") if issue.strip()]
        
        # Use the response as explanation
        explanation = response.strip()
        
        return VerificationResult(
            stage=VerificationStage.BACKWARD,
            passed=passed,
            confidence_score=confidence_score,
            issues=issues,
            explanation=explanation
        )
    
    def _calculate_confidence(self, response: str, passed: bool) -> float:
        """Calculate confidence score based on response characteristics."""
        base_confidence = 0.8 if passed else 0.3
        
        # Adjust based on response length and clarity
        if len(response) > 100:  # Detailed explanation
            base_confidence += 0.1
        
        # Check for uncertainty indicators
        uncertainty_words = ["maybe", "might", "possibly", "unclear", "uncertain"]
        if any(word in response.lower() for word in uncertainty_words):
            base_confidence -= 0.2
        
        # Check for strong confidence indicators
        strong_words = ["definitely", "clearly", "obviously", "certainly"]
        if any(word in response.lower() for word in strong_words):
            base_confidence += 0.1
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _extract_issues_from_response(self, response: str) -> List[str]:
        """Extract issues from verification response."""
        issues = []
        
        # Look for common error patterns
        error_patterns = [
            r"error:?\\s*(.+)",
            r"problem:?\\s*(.+)",
            r"issue:?\\s*(.+)",
            r"incorrect:?\\s*(.+)",
            r"wrong:?\\s*(.+)"
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            issues.extend([match.strip() for match in matches if match.strip()])
        
        return issues[:3]  # Limit to top 3 issues
    
    def _truncate_code(self, code: str, max_length: int) -> str:
        """Truncate code to fit within token limits."""
        if len(code) <= max_length:
            return code
        
        lines = code.split("\\n")
        truncated_lines = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) > max_length - 20:  # Reserve space for "..."
                truncated_lines.append("...")
                break
            truncated_lines.append(line)
            current_length += len(line) + 1
        
        return "\\n".join(truncated_lines)
    
    def _format_result(self, result: Any, max_length: int) -> str:
        """Format result for display in prompts."""
        try:
            if isinstance(result, (dict, list)):
                formatted = json.dumps(result, indent=2, ensure_ascii=False)
            else:
                formatted = str(result)
            
            if len(formatted) > max_length:
                return formatted[:max_length-3] + "..."
            
            return formatted
        except Exception:
            return str(result)[:max_length]
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information."""
        essential_keys = ["file_path", "columns", "data_types", "row_count"]
        formatted_context = {}
        
        for key in essential_keys:
            if key in context:
                formatted_context[key] = context[key]
        
        try:
            return json.dumps(formatted_context, indent=2, ensure_ascii=False)[:300]
        except Exception:
            return str(formatted_context)[:300]
    
    def generate_improvement_suggestions(self, verification_results: Dict[str, VerificationResult], original_query: str, code: str) -> List[str]:
        """Generate suggestions for improving failed verifications."""
        suggestions = []
        
        forward_result = verification_results.get("forward")
        backward_result = verification_results.get("backward")
        
        if forward_result and not forward_result.passed:
            suggestions.extend([
                "Review the generated code logic",
                "Check if all query requirements are addressed",
                "Verify data filtering and aggregation steps"
            ])
        
        if backward_result and not backward_result.passed:
            suggestions.extend([
                "Ensure the answer format matches the question",
                "Check if the result scope covers the full query",
                "Verify numerical accuracy and units"
            ])
        
        # Add specific suggestions based on common issues
        if any("join" in issue.lower() for result in verification_results.values() for issue in result.issues):
            suggestions.append("Consider reviewing table join conditions")
        
        if any("aggregation" in issue.lower() for result in verification_results.values() for issue in result.issues):
            suggestions.append("Double-check aggregation functions and grouping")
        
        return list(set(suggestions))  # Remove duplicates
    
    def create_reliability_score(self, verification_results: Dict[str, VerificationResult]) -> float:
        """Create overall reliability score based on verification results."""
        if not verification_results:
            return 0.0
        
        total_score = 0.0
        weights = {"forward": 0.4, "backward": 0.6}  # Backward verification is slightly more important
        
        for stage, result in verification_results.items():
            weight = weights.get(stage, 0.5)
            stage_score = result.confidence_score if result.passed else 0.0
            total_score += stage_score * weight
        
        return total_score
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process verification request."""
        try:
            query = request.data.get("query", "")
            code = request.data.get("code", "")
            result = request.data.get("result")
            context = request.data.get("context", {})
            stage = request.data.get("stage", "comprehensive")
            
            if not query:
                return AgentResponse(
                    request_id=request.request_id,
                    agent_name=self.name,
                    status="error",
                    error_message="Missing query in request"
                )
            
            verification_results = {}
            
            if stage == "forward":
                verification_results["forward"] = self.forward_verification(query, code, result, context)
            elif stage == "backward":
                verification_results["backward"] = self.backward_verification(query, result, context)
            else:  # comprehensive
                verification_results = self.comprehensive_verification(query, code, result, context)
            
            # Generate reliability score and suggestions
            reliability_score = self.create_reliability_score(verification_results)
            suggestions = self.generate_improvement_suggestions(verification_results, query, code)
            
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status="completed",
                data={
                    "verification_results": {
                        stage: {
                            "passed": result.passed,
                            "confidence_score": result.confidence_score,
                            "issues": result.issues,
                            "explanation": result.explanation,
                            "suggested_fixes": result.suggested_fixes
                        }
                        for stage, result in verification_results.items()
                    },
                    "overall_reliability": reliability_score,
                    "improvement_suggestions": suggestions,
                    "passed_verification": all(result.passed for result in verification_results.values())
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in verification process: {e}")
            return AgentResponse(
                request_id=request.request_id,
                agent_name=self.name,
                status="error",
                error_message=str(e)
            )


def verify_forward(query: str, code: str, result: Any) -> Tuple[bool, float]:
    """Standalone forward verification function (ST-Raptor style)."""
    agent = VerificationAgent()
    verification_result = agent.forward_verification(query, code, result, {})
    return verification_result.passed, verification_result.confidence_score


def verify_backward(query: str, answer: Any, context: Dict[str, Any] = None) -> Tuple[bool, float]:
    """Standalone backward verification function (ST-Raptor style)."""
    agent = VerificationAgent()
    if context is None:
        context = {}
    verification_result = agent.backward_verification(query, answer, context)
    return verification_result.passed, verification_result.confidence_score