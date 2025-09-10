"""Response Generation Agent - Synthesizes all agent results and generates user-friendly responses."""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base import BaseAgent
from ..models.base import AgentRequest, AgentResponse, AgentStatus
from ..models.agents import ResponseGenerationRequest, ResponseGenerationResponse
from ..utils.siliconflow_client import SiliconFlowClient


class ResponseGenerationAgent(BaseAgent):
    """Agent responsible for generating comprehensive responses based on all agent results."""
    
    def __init__(self):
        super().__init__(
            name="ResponseGenerationAgent",
            description="Synthesizes all agent results and generates natural language responses for users"
        )
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process response generation request."""
        if not isinstance(request, ResponseGenerationRequest):
            return self.create_error_response(
                request,
                f"Invalid request type. Expected ResponseGenerationRequest, got {type(request)}"
            )
        
        try:
            self.logger.info(f"Generating response for user query: {request.user_query[:100]}...")
            
            # Generate response based on workflow results
            response_content = await self._generate_comprehensive_response(
                user_query=request.user_query,
                workflow_results=request.workflow_results,
                file_info=request.file_info,
                context=request.context or {}
            )
            
            # Format the response
            formatted_response = await self._format_response(
                response_content=response_content,
                workflow_results=request.workflow_results,
                user_query=request.user_query
            )
            
            self.logger.info(f"Response generation completed successfully")
            
            return ResponseGenerationResponse(
                agent_id=self.name,
                request_id=request.request_id,
                status=AgentStatus.SUCCESS,
                response=formatted_response['main_response'],
                summary=formatted_response['summary'],
                recommendations=formatted_response['recommendations'],
                technical_details=formatted_response['technical_details'],
                result={
                    'response_generated': True,
                    'response_length': len(formatted_response['main_response']),
                    'includes_recommendations': len(formatted_response['recommendations']) > 0,
                    'includes_technical_details': bool(formatted_response['technical_details'])
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error during response generation: {e}")
            return self.create_error_response(request, str(e))
    
    async def _generate_comprehensive_response(
        self,
        user_query: str,
        workflow_results: Dict[str, Any],
        file_info: Optional[Dict[str, Any]] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive response based on all available data."""
        
        # Prepare context for the LLM
        serializable_workflow = self._make_json_serializable(workflow_results)
        serializable_file_info = self._make_json_serializable(file_info or {})
        serializable_context = self._make_json_serializable(context or {})
        
        # Build the prompt for response generation
        response_prompt = f"""
你是一个专业的Excel数据分析助手。请基于以下信息为用户生成一个全面、准确、易懂的回答。

用户问题: "{user_query}"

文件信息:
{json.dumps(serializable_file_info, indent=2, ensure_ascii=False)}

工作流执行结果:
{json.dumps(serializable_workflow, indent=2, ensure_ascii=False)}

上下文信息:
{json.dumps(serializable_context, indent=2, ensure_ascii=False)}

请生成一个包含以下内容的响应:

1. **主要回答**: 直接回答用户的问题，使用自然、友好的语言
2. **数据洞察**: 基于分析结果提供有价值的数据洞察
3. **具体数值**: 包含相关的统计数据、计算结果等
4. **建议**: 基于分析结果给出实用的建议
5. **技术说明**: 简要说明使用的分析方法（如果适用）

要求:
- 使用中文回答
- 语言自然流畅，避免过于技术化
- 突出关键数据和发现
- 如果有错误或警告，要明确说明
- 如果分析不完整，要诚实说明限制

请以JSON格式返回:
{{
    "main_response": "主要回答内容",
    "data_insights": ["洞察1", "洞察2"],
    "key_numbers": {{"指标1": "数值1", "指标2": "数值2"}},
    "recommendations": ["建议1", "建议2"],
    "technical_notes": "技术说明",
    "confidence_level": "high|medium|low",
    "limitations": ["限制1", "限制2"]
}}
"""

        try:
            import time
            from ..utils.logging import get_logger
            
            logger = get_logger(__name__)
            request_id = context.get('request_id', 'response_generation')
            
            # Log request start
            start_time = time.time()
            logger.info(f"🤖 [ResponseGeneration] Starting LLM request {request_id}")
            
            async with SiliconFlowClient() as client:
                try:
                    response = await client.chat_completion(
                        messages=[{"role": "user", "content": response_prompt}],
                        temperature=0.7,
                        request_id=request_id
                    )
                    
                    # Log successful completion
                    end_time = time.time()
                    duration = end_time - start_time
                    logger.info(f"🤖 [ResponseGeneration] LLM request {request_id} completed in {duration:.2f}s")
                    
                except Exception as e:
                    # Log error
                    end_time = time.time()
                    duration = end_time - start_time
                    logger.error(f"❌ [ResponseGeneration] LLM request {request_id} failed after {duration:.2f}s: {type(e).__name__}: {e}")
                    raise
            
            # Parse the response
            response_text = response['choices'][0]['message']['content']
            
            # Extract JSON from response if wrapped in code blocks
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end]
            
            response_data = json.loads(response_text.strip())
            
            return response_data
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback to simple response generation
            return self._generate_fallback_response(user_query, workflow_results)
    
    async def _format_response(
        self,
        response_content: Dict[str, Any],
        workflow_results: Dict[str, Any],
        user_query: str
    ) -> Dict[str, Any]:
        """Format the response for better presentation."""
        
        # Extract key components
        main_response = response_content.get('main_response', '抱歉，我无法生成完整的回答。')
        data_insights = response_content.get('data_insights', [])
        key_numbers = response_content.get('key_numbers', {})
        recommendations = response_content.get('recommendations', [])
        technical_notes = response_content.get('technical_notes', '')
        confidence_level = response_content.get('confidence_level', 'medium')
        limitations = response_content.get('limitations', [])
        
        # Build formatted response
        formatted_main = self._build_formatted_main_response(
            main_response, data_insights, key_numbers, confidence_level
        )
        
        # Generate summary
        summary = self._generate_summary(workflow_results, key_numbers)
        
        # Format technical details
        technical_details = self._format_technical_details(
            workflow_results, technical_notes, limitations
        )
        
        return {
            'main_response': formatted_main,
            'summary': summary,
            'recommendations': recommendations,
            'technical_details': technical_details
        }
    
    def _build_formatted_main_response(
        self,
        main_response: str,
        data_insights: List[str],
        key_numbers: Dict[str, Any],
        confidence_level: str
    ) -> str:
        """Build a well-formatted main response."""
        
        formatted_parts = [main_response]
        
        # Add key numbers if available
        if key_numbers:
            formatted_parts.append("\n📊 **关键数据:**")
            for key, value in key_numbers.items():
                formatted_parts.append(f"• {key}: {value}")
        
        # Add insights if available
        if data_insights:
            formatted_parts.append("\n💡 **数据洞察:**")
            for insight in data_insights:
                formatted_parts.append(f"• {insight}")
        
        # Add confidence indicator
        confidence_emoji = {
            'high': '🟢',
            'medium': '🟡', 
            'low': '🟠'
        }
        emoji = confidence_emoji.get(confidence_level, '🟡')
        confidence_text = {
            'high': '高度可信',
            'medium': '中等可信',
            'low': '需要进一步验证'
        }
        text = confidence_text.get(confidence_level, '中等可信')
        
        formatted_parts.append(f"\n{emoji} **可信度**: {text}")
        
        return '\n'.join(formatted_parts)
    
    def _generate_summary(
        self,
        workflow_results: Dict[str, Any],
        key_numbers: Dict[str, Any]
    ) -> str:
        """Generate a concise summary of the analysis."""
        
        summary_parts = []
        
        # Workflow status
        status = workflow_results.get('status', 'unknown')
        if status == 'success':
            summary_parts.append("✅ 数据分析成功完成")
        else:
            summary_parts.append("⚠️ 数据分析部分完成")
        
        # Processing info
        if workflow_results.get('file_processed'):
            processed_file = workflow_results.get('processed_file', '未知文件')
            summary_parts.append(f"📁 已处理文件: {processed_file}")
        
        # Key metrics count
        if key_numbers:
            summary_parts.append(f"📈 生成了 {len(key_numbers)} 项关键指标")
        
        # Workflow steps
        steps = workflow_results.get('steps', [])
        if steps:
            completed_steps = [s for s in steps if s.get('status') == 'success']
            summary_parts.append(f"⚙️ 完成了 {len(completed_steps)}/{len(steps)} 个处理步骤")
        
        return '\n'.join(summary_parts)
    
    def _format_technical_details(
        self,
        workflow_results: Dict[str, Any],
        technical_notes: str,
        limitations: List[str]
    ) -> Dict[str, Any]:
        """Format technical details for advanced users."""
        
        details = {}
        
        # Workflow information
        details['workflow_type'] = workflow_results.get('workflow_type', 'unknown')
        details['execution_time'] = workflow_results.get('execution_time', 0)
        
        # Processing steps
        steps = workflow_results.get('steps', [])
        details['processing_steps'] = [
            {
                'step': step.get('step', 'unknown'),
                'status': step.get('status', 'unknown')
            }
            for step in steps
        ]
        
        # Generated code
        if workflow_results.get('generated_code'):
            details['code_generated'] = True
            details['code_length'] = len(workflow_results['generated_code'])
        
        # Technical notes
        if technical_notes:
            details['technical_notes'] = technical_notes
        
        # Limitations
        if limitations:
            details['limitations'] = limitations
        
        # MCP usage
        details['mcp_enhanced'] = any(
            'mcp' in str(step).lower() 
            for step in steps
        )
        
        return details
    
    def _generate_fallback_response(
        self,
        user_query: str,
        workflow_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a fallback response when LLM fails."""
        
        status = workflow_results.get('status', 'unknown')
        
        if status == 'success':
            main_response = f"已成功处理您的问题：{user_query}\n\n基于数据分析结果，我已完成相关处理。"
            
            # Add execution output if available
            if workflow_results.get('output'):
                main_response += f"\n\n执行结果：\n{workflow_results['output']}"
                
        else:
            main_response = f"在处理您的问题时遇到了一些困难：{user_query}\n\n请检查数据格式或重新表述您的问题。"
        
        return {
            'main_response': main_response,
            'data_insights': [],
            'key_numbers': {},
            'recommendations': ["建议检查数据格式", "如需帮助请重新表述问题"],
            'technical_notes': '使用了备用响应生成方式',
            'confidence_level': 'low',
            'limitations': ['LLM响应生成失败，使用简化回答']
        }