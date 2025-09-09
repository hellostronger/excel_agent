"""Memory & Preference Agent - placeholder implementation."""

from .base import BaseAgent
from ..models.agents import MemoryRequest, MemoryResponse
from ..models.base import AgentRequest, AgentResponse, AgentStatus


class MemoryAgent(BaseAgent):
    """Agent responsible for recording user preferences/confirmations for future reuse."""
    
    def __init__(self):
        super().__init__(
            name="MemoryAgent",
            description="Records user preferences/confirmations for future reuse"
        )
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process memory request - placeholder implementation."""
        if not isinstance(request, MemoryRequest):
            return self.create_error_response(
                request,
                f"Invalid request type. Expected MemoryRequest, got {type(request)}"
            )
        
        # Placeholder implementation
        return MemoryResponse(
            agent_id=self.name,
            request_id=request.request_id,
            status=AgentStatus.SUCCESS,
            stored_preferences=[],
            can_answer_from_memory=False,
            memory_match=None,
            result={"status": "placeholder", "message": "Not yet implemented"}
        )