"""Summarization Agent - generates summaries and answers based on text content."""

from .base import BaseAgent
from ..models.agents import SummarizationRequest, SummarizationResponse
from ..models.base import AgentRequest, AgentResponse, AgentStatus
from ..utils.llm import get_llm_client


class SummarizationAgent(BaseAgent):
    """Agent responsible for generating sheet summaries and text-based analysis."""
    
    def __init__(self):
        super().__init__(
            name="SummarizationAgent",
            description="Generates sheet summaries, analyzes text content, and provides contextual answers"
        )
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process summarization request."""
        if not isinstance(request, SummarizationRequest):
            return self.create_error_response(
                request,
                f"Invalid request type. Expected SummarizationRequest, got {type(request)}"
            )
        
        try:
            # 如果提供了直接的文本内容，使用文本分析模式
            if request.text_content:
                return await self._process_text_content(request)
            
            # 否则使用传统的文件处理模式
            return await self._process_file_content(request)
            
        except Exception as e:
            return self.create_error_response(request, str(e))
    
    async def _process_text_content(self, request: SummarizationRequest) -> SummarizationResponse:
        """处理直接提供的文本内容"""
        try:
            llm_client = await get_llm_client()
            
            # 构建提示
            system_prompt = """你是一个专业的Excel数据分析助手。用户会提供Excel工作表的markdown格式内容和一个查询问题。
请基于提供的内容，回答用户的问题。请：
1. 仔细分析提供的数据内容
2. 针对用户的具体问题给出准确、有用的回答
3. 如果可能，提供数据洞察和建议
4. 回答要专业、简洁、易懂
5. 用中文回答"""
            
            user_prompt = f"""用户查询: {request.query or "请分析这些数据"}

Excel数据内容 (Markdown格式):
{request.text_content}

请基于以上内容回答用户的问题。"""
            
            # 调用LLM
            response = await llm_client.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            llm_answer = response.choices[0].message.content if response.choices else "无法生成回答"
            
            return SummarizationResponse(
                agent_id=self.name,
                request_id=request.request_id,
                status=AgentStatus.SUCCESS,
                summary=llm_answer,
                compressed_view={
                    "processing_mode": "text_content",
                    "content_length": len(request.text_content),
                    "query": request.query
                },
                key_insights=["基于文本内容的智能分析"],
                result={
                    "analysis": llm_answer,
                    "processing_mode": "text_content",
                    "success": True
                }
            )
            
        except Exception as e:
            return self.create_error_response(request, f"文本内容处理失败: {str(e)}")
    
    async def _process_file_content(self, request: SummarizationRequest) -> SummarizationResponse:
        """处理传统的文件内容（保持向后兼容）"""
        # 传统的占位符实现
        return SummarizationResponse(
            agent_id=self.name,
            request_id=request.request_id,
            status=AgentStatus.SUCCESS,
            summary=f"Summary of sheet {request.sheet_name}",
            compressed_view={"rows": 100, "columns": 10, "compressed": True},
            key_insights=["传统文件处理模式"],
            result={"status": "placeholder", "message": "Traditional file processing mode"}
        )