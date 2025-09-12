"""
Model Adapter for ADK Integration with SiliconFlow

Provides a bridge between ADK Agent system and SiliconFlow models.
"""

import os
import logging
from typing import Dict, Any, Optional, List
import asyncio

try:
    from google.adk.agents import Agent
    from google.genai import types
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    Agent = object
    types = object

from .siliconflow_client import SiliconFlowClient, create_siliconflow_client
from .utils import load_environment_variables

logger = logging.getLogger(__name__)


class ModelAdapter:
    """模型适配器 - 统一不同模型提供商的接口"""
    
    def __init__(self):
        self.env_config = load_environment_variables()
        self.provider = self.env_config.get("MODEL_PROVIDER", "siliconflow")
        self.siliconflow_client = None
        
        if self.provider == "siliconflow":
            self.siliconflow_client = create_siliconflow_client()
            if not self.siliconflow_client:
                logger.warning("Failed to initialize SiliconFlow client, falling back to Google models")
                self.provider = "google"
    
    def get_model_for_agent(self, agent_type: str = "worker") -> str:
        """获取指定agent类型的模型名称"""
        if agent_type == "orchestrator":
            return self.env_config.get("ORCHESTRATOR_MODEL", "qwen-max")
        elif agent_type == "root":
            return self.env_config.get("ROOT_AGENT_MODEL", "qwen-plus")
        else:  # worker
            return self.env_config.get("WORKER_MODEL", "qwen-turbo")
    
    def create_agent_with_model(
        self,
        agent_type: str,
        name: str,
        description: str,
        instruction: str,
        tools: List = None,
        **kwargs
    ) -> Agent:
        """创建配置了合适模型的Agent"""
        if not ADK_AVAILABLE:
            raise ImportError("ADK not available. Please install google-adk.")
        
        model_name = self.get_model_for_agent(agent_type)
        tools = tools or []
        
        # 根据提供商创建不同的Agent配置
        if self.provider == "siliconflow":
            return self._create_siliconflow_agent(
                model_name, name, description, instruction, tools, **kwargs
            )
        else:
            return self._create_google_agent(
                model_name, name, description, instruction, tools, **kwargs
            )
    
    def _create_siliconflow_agent(
        self,
        model_name: str,
        name: str,
        description: str,
        instruction: str,
        tools: List,
        **kwargs
    ) -> Agent:
        """创建使用硅基流动模型的Agent"""
        
        # 为了与ADK兼容，我们需要创建一个自定义的Agent
        # 这里使用一个包装器来处理硅基流动API调用
        
        # 获取生成配置
        generate_config = kwargs.get('generate_content_config', types.GenerateContentConfig(
            temperature=0.01,
            top_p=0.9,
            max_output_tokens=6144
        ))
        
        # 由于ADK原生不支持硅基流动，我们创建一个包装的Agent
        # 在实际项目中，可能需要直接使用硅基流动客户端而不是ADK Agent
        
        agent = Agent(
            model=f"siliconflow:{model_name}",  # 标记这是硅基流动模型
            name=name,
            description=f"{description} (使用硅基流动 {model_name} 模型)",
            instruction=instruction,
            tools=tools,
            generate_content_config=generate_config,
            **{k: v for k, v in kwargs.items() if k != 'generate_content_config'}
        )
        
        # 添加硅基流动客户端的引用
        agent._siliconflow_client = self.siliconflow_client
        agent._siliconflow_model = model_name
        
        return agent
    
    def _create_google_agent(
        self,
        model_name: str,
        name: str,
        description: str,
        instruction: str,
        tools: List,
        **kwargs
    ) -> Agent:
        """创建使用Google模型的Agent"""
        
        generate_config = kwargs.get('generate_content_config', types.GenerateContentConfig(
            temperature=0.01,
            top_p=0.9,
            max_output_tokens=6144
        ))
        
        return Agent(
            model=model_name,
            name=name,
            description=description,
            instruction=instruction,
            tools=tools,
            generate_content_config=generate_config,
            **{k: v for k, v in kwargs.items() if k != 'generate_content_config'}
        )


# 创建全局模型适配器实例
model_adapter = ModelAdapter()


def create_agent_with_best_model(
    agent_type: str,
    name: str,
    description: str,
    instruction: str,
    tools: List = None,
    **kwargs
) -> Agent:
    """使用最佳可用模型创建Agent的便捷函数"""
    return model_adapter.create_agent_with_model(
        agent_type=agent_type,
        name=name,
        description=description,
        instruction=instruction,
        tools=tools,
        **kwargs
    )


# SiliconFlow 直接调用的辅助类
class SiliconFlowAgentExecutor:
    """
    硅基流动Agent执行器
    
    当ADK无法直接支持硅基流动模型时，使用这个类来直接调用硅基流动API
    """
    
    def __init__(self, model_name: str, client: SiliconFlowClient = None):
        self.model_name = model_name
        self.client = client or create_siliconflow_client()
        if not self.client:
            raise ValueError("SiliconFlow client not available")
    
    async def execute_task(
        self,
        instruction: str,
        user_input: str,
        context: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """执行Agent任务"""
        try:
            # 构建提示词
            prompt = self._build_prompt(instruction, user_input, context)
            
            # 调用硅基流动API
            response = await self.client.generate_response(
                prompt=prompt,
                model=self.model_name,
                temperature=kwargs.get('temperature', 0.01),
                max_tokens=kwargs.get('max_tokens')
            )
            
            return {
                "success": True,
                "response": response,
                "model": self.model_name,
                "provider": "siliconflow"
            }
            
        except Exception as e:
            logger.error(f"SiliconFlow agent execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model_name,
                "provider": "siliconflow"
            }
    
    def _build_prompt(self, instruction: str, user_input: str, context: Dict[str, Any] = None) -> str:
        """构建完整的提示词"""
        prompt_parts = []
        
        # 添加系统指令
        if instruction:
            prompt_parts.append(f"系统指令：\n{instruction}\n")
        
        # 添加上下文信息
        if context:
            context_str = self._format_context(context)
            if context_str:
                prompt_parts.append(f"上下文信息：\n{context_str}\n")
        
        # 添加用户输入
        if user_input:
            prompt_parts.append(f"用户请求：\n{user_input}")
        
        return "\n".join(prompt_parts)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """格式化上下文信息"""
        formatted_parts = []
        
        # 格式化文件信息
        if "file_metadata" in context:
            metadata = context["file_metadata"]
            formatted_parts.append(f"文件信息：{metadata.get('file_path', 'Unknown')}")
        
        # 格式化分析结果
        if "analysis_results" in context:
            formatted_parts.append("前期分析结果可用")
        
        # 格式化其他关键信息
        for key, value in context.items():
            if key not in ["file_metadata", "analysis_results"] and isinstance(value, (str, int, float)):
                formatted_parts.append(f"{key}: {value}")
        
        return "\n".join(formatted_parts)


def create_siliconflow_executor(model_name: str) -> SiliconFlowAgentExecutor:
    """创建硅基流动执行器的便捷函数"""
    return SiliconFlowAgentExecutor(model_name=model_name)