"""
渐进式多Agent协作编排器 - 基于ADK架构

实现多阶段Excel文件智能分析系统，采用渐进式解释策略:
1. 文件解释阶段 - 充分准备所有上下文信息
2. 并发分析阶段 - 多Agent协作处理不同分析任务
3. 数据整合阶段 - 合并分析结果和构建关系
4. 响应生成阶段 - 基于完整分析结果生成智能回答

基于Google ADK (Agent Development Kit) 多Agent协作架构
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from ..models.base import RequestType, AgentRequest, AgentResponse, AgentStatus
from ..utils.config import config
from ..utils.logging import get_logger
from ..utils.siliconflow_client import SiliconFlowClient
from .orchestrator import Orchestrator, WorkflowType

# ADK相关导入
try:
    from adk import Agent, AgentTool, ToolContext, CallbackContext, Tool
    from adk.models import Model
    from adk.agents import SequentialAgent, LoopAgent
    ADK_AVAILABLE = True
except ImportError:
    # 兼容性导入，用于开发阶段
    ADK_AVAILABLE = False
    Agent = object
    AgentTool = object
    ToolContext = object  
    CallbackContext = object
    Tool = object
    SequentialAgent = object
    LoopAgent = object


class ProcessingStage(str, Enum):
    """渐进式多Agent处理阶段"""
    # 文件解释充分准备阶段
    FILE_PREPARATION = "file_preparation"
    METADATA_EXTRACTION = "metadata_extraction"
    STRUCTURE_ANALYSIS = "structure_analysis"
    
    # 并发分析阶段
    CONCURRENT_ANALYSIS = "concurrent_analysis"
    COLUMN_PROFILING = "column_profiling"
    RELATION_DISCOVERY = "relation_discovery"
    CONTENT_ANALYSIS = "content_analysis"
    
    # 数据整合阶段
    DATA_INTEGRATION = "data_integration"
    RELATIONSHIP_BUILDING = "relationship_building"
    
    # 响应生成阶段
    RESPONSE_GENERATION = "response_generation"
    
    # 原有阶段（保持兼容性）
    INITIAL_SCOPE = "initial_scope"
    SHEET_REDUCTION = "sheet_reduction" 
    REGION_REDUCTION = "region_reduction"
    COLUMN_REDUCTION = "column_reduction"
    ROW_REDUCTION = "row_reduction"
    FINAL_PROCESSING = "final_processing"


class ProgressiveOrchestrator(Orchestrator):
    """渐进式智能协调器"""
    
    def __init__(self):
        super().__init__()
        self.scope_reducer = None
        self.processing_stages = []
        
        # 导入智能范围缩小器
        try:
            import sys
            from pathlib import Path
            backend_path = Path(__file__).parent.parent.parent.parent / "backend"
            sys.path.insert(0, str(backend_path))
            
            from utils.smart_scope_reducer import smart_scope_reducer
            self.scope_reducer = smart_scope_reducer
            self.logger.info("智能范围缩小器加载成功")
        except ImportError as e:
            self.logger.warning(f"无法加载智能范围缩小器: {e}")
    
    async def process_user_request(
        self, 
        user_request: str, 
        file_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        progress_tracker=None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        渐进式处理用户请求，在后台不断缩小查询范围
        """
        context = context or {}
        
        if request_id is None:
            request_id = f"prog_req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self) % 10000}"
        context['request_id'] = request_id
        
        try:
            self.logger.info(f"🚀 [ProgressiveOrchestrator {request_id}] 开始渐进式处理")
            start_time = datetime.now()
            
            # 初始化处理阶段
            self.processing_stages = []
            
            # Step 0: Excel相关性检测
            if progress_tracker:
                progress_tracker.update_step(request_id, "excel_relevance_check", "in_progress", "检查Excel相关性...")
            
            excel_relevance = await self._check_excel_relevance(user_request, file_path, context)
            
            if not excel_relevance['is_excel_related']:
                if progress_tracker:
                    progress_tracker.update_step(request_id, "excel_relevance_check", "completed", "非Excel相关请求")
                return await self._handle_non_excel_request(user_request, excel_relevance, request_id, start_time, progress_tracker)
            
            if progress_tracker:
                progress_tracker.update_step(request_id, "excel_relevance_check", "completed", "确认为Excel相关请求")
            
            # Step 1: 获取文件分析数据
            if progress_tracker:
                progress_tracker.update_step(request_id, "file_analysis", "in_progress", "分析文件结构...")
            
            file_analysis = await self._get_file_analysis(file_path, context)
            
            if progress_tracker:
                progress_tracker.update_step(request_id, "file_analysis", "completed", "文件分析完成")
            
            # Step 2: 智能范围缩小（多阶段）
            if self.scope_reducer and file_analysis:
                scope_reductions = await self._progressive_scope_reduction(
                    user_request, file_analysis, progress_tracker, request_id
                )
                context['scope_reductions'] = scope_reductions
            
            # Step 3: 意图解析（基于缩小后的范围）
            if progress_tracker:
                progress_tracker.update_step(request_id, "intent_parsing", "in_progress", "解析用户意图...")
            
            intent_result = await self._parse_intent_with_scope(user_request, context)
            workflow_type = intent_result['workflow_type']
            
            if progress_tracker:
                progress_tracker.update_step(request_id, "intent_parsing", "completed", f"意图解析完成 - {workflow_type}")
            
            # Step 4: 执行优化后的工作流
            if progress_tracker:
                progress_tracker.update_step(request_id, "workflow_execution", "in_progress", f"执行{workflow_type}工作流...")
            
            result = await self._execute_progressive_workflow(
                user_request, file_path, intent_result, request_id, progress_tracker, context
            )
            
            if progress_tracker:
                progress_tracker.update_step(request_id, "workflow_execution", "completed", "工作流执行完成")
            
            # 记录处理阶段
            result['processing_stages'] = self.processing_stages
            result['scope_optimization'] = True
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            self.logger.info(f"✅ [ProgressiveOrchestrator {request_id}] 渐进式处理完成，耗时 {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ [ProgressiveOrchestrator {request_id}] 渐进式处理失败: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'processing_stages': self.processing_stages,
                'scope_optimization': False
            }
    
    async def _progressive_scope_reduction(
        self, 
        user_request: str, 
        file_analysis: Dict[str, Any],
        progress_tracker=None,
        request_id: str = "unknown"
    ) -> List[Dict[str, Any]]:
        """
        渐进式范围缩小过程
        """
        if not self.scope_reducer:
            return []
        
        self.logger.info(f"🔍 [ProgressiveOrchestrator {request_id}] 开始智能范围缩小")
        
        scope_reductions = []
        
        try:
            # 开始范围缩小过程
            if progress_tracker:
                progress_tracker.update_step(request_id, "scope_reduction", "in_progress", "开始智能范围缩小...")
            
            # 阶段1: 工作表级别缩小
            await self._stage_sheet_reduction(user_request, file_analysis, scope_reductions, progress_tracker, request_id)
            
            # 阶段2: 区域级别缩小
            await self._stage_region_reduction(user_request, file_analysis, scope_reductions, progress_tracker, request_id)
            
            # 阶段3: 列级别缩小
            await self._stage_column_reduction(user_request, file_analysis, scope_reductions, progress_tracker, request_id)
            
            # 阶段4: 行级别缩小
            await self._stage_row_reduction(user_request, file_analysis, scope_reductions, progress_tracker, request_id)
            
            if progress_tracker:
                progress_tracker.update_step(request_id, "scope_reduction", "completed", 
                                           f"范围缩小完成，共{len(scope_reductions)}个优化步骤")
            
            # 生成范围缩小摘要
            reduction_summary = self.scope_reducer.get_final_scope_summary(scope_reductions)
            self.logger.info(f"🎯 [ProgressiveOrchestrator {request_id}] 范围缩小摘要: {reduction_summary}")
            
            return scope_reductions
            
        except Exception as e:
            self.logger.error(f"❌ [ProgressiveOrchestrator {request_id}] 范围缩小失败: {e}")
            if progress_tracker:
                progress_tracker.update_step(request_id, "scope_reduction", "warning", f"范围缩小失败: {str(e)}")
            return []
    
    async def _stage_sheet_reduction(self, user_request: str, file_analysis: Dict, 
                                   scope_reductions: List, progress_tracker, request_id: str):
        """工作表级别缩小阶段"""
        stage_start = datetime.now()
        
        if progress_tracker:
            progress_tracker.update_step(request_id, "sheet_scope_reduction", "in_progress", "分析工作表相关性...")
        
        # 执行工作表级别的范围缩小
        sheet_reduction = self.scope_reducer._reduce_by_sheets(
            user_request, 
            file_analysis, 
            self.scope_reducer._create_full_scope(file_analysis)
        )
        
        if sheet_reduction:
            scope_reductions.append(sheet_reduction)
            stage_time = (datetime.now() - stage_start).total_seconds()
            
            self.processing_stages.append({
                'stage': ProcessingStage.SHEET_REDUCTION,
                'duration': stage_time,
                'result': 'success',
                'details': {
                    'target_sheets': sheet_reduction.target_sheets,
                    'confidence': sheet_reduction.confidence_score
                }
            })
            
            if progress_tracker:
                progress_tracker.update_step(request_id, "sheet_scope_reduction", "completed", 
                                           f"工作表范围缩小完成，聚焦到{len(sheet_reduction.target_sheets)}个工作表")
        else:
            if progress_tracker:
                progress_tracker.update_step(request_id, "sheet_scope_reduction", "completed", "未发现需要缩小的工作表范围")
    
    async def _stage_region_reduction(self, user_request: str, file_analysis: Dict,
                                    scope_reductions: List, progress_tracker, request_id: str):
        """区域级别缩小阶段"""
        stage_start = datetime.now()
        
        if progress_tracker:
            progress_tracker.update_step(request_id, "region_scope_reduction", "in_progress", "分析数据区域...")
        
        # 分析查询意图
        intent_analysis = self.scope_reducer.analyze_query_intent(user_request)
        
        # 执行区域级别的范围缩小
        current_scope = self.scope_reducer._create_full_scope(file_analysis)
        if scope_reductions:
            # 使用前一阶段的结果更新范围
            last_reduction = scope_reductions[-1]
            current_scope = self.scope_reducer._update_scope_from_reduction(current_scope, last_reduction)
        
        region_reduction = self.scope_reducer._reduce_by_regions(
            user_request, file_analysis, current_scope, intent_analysis
        )
        
        if region_reduction:
            scope_reductions.append(region_reduction)
            stage_time = (datetime.now() - stage_start).total_seconds()
            
            self.processing_stages.append({
                'stage': ProcessingStage.REGION_REDUCTION,
                'duration': stage_time,
                'result': 'success',
                'details': {
                    'regions_found': len(region_reduction.target_regions),
                    'confidence': region_reduction.confidence_score,
                    'method': region_reduction.reduction_method
                }
            })
            
            if progress_tracker:
                progress_tracker.update_step(request_id, "region_scope_reduction", "completed", 
                                           f"数据区域缩小完成，识别到{len(region_reduction.target_regions)}个相关区域")
        else:
            if progress_tracker:
                progress_tracker.update_step(request_id, "region_scope_reduction", "completed", "未发现需要缩小的数据区域")
    
    async def _stage_column_reduction(self, user_request: str, file_analysis: Dict,
                                    scope_reductions: List, progress_tracker, request_id: str):
        """列级别缩小阶段"""
        stage_start = datetime.now()
        
        if progress_tracker:
            progress_tracker.update_step(request_id, "column_scope_reduction", "in_progress", "分析相关列...")
        
        # 执行列级别的范围缩小
        current_scope = self.scope_reducer._create_full_scope(file_analysis)
        if scope_reductions:
            for reduction in scope_reductions:
                current_scope = self.scope_reducer._update_scope_from_reduction(current_scope, reduction)
        
        intent_analysis = self.scope_reducer.analyze_query_intent(user_request)
        column_reduction = self.scope_reducer._reduce_by_columns(
            user_request, file_analysis, current_scope, intent_analysis
        )
        
        if column_reduction:
            scope_reductions.append(column_reduction)
            stage_time = (datetime.now() - stage_start).total_seconds()
            
            relevant_columns = []
            for region in column_reduction.target_regions:
                if 'columns' in region:
                    relevant_columns.extend(region['columns'])
            
            self.processing_stages.append({
                'stage': ProcessingStage.COLUMN_REDUCTION,
                'duration': stage_time,
                'result': 'success',
                'details': {
                    'relevant_columns': relevant_columns,
                    'column_count': len(relevant_columns),
                    'confidence': column_reduction.confidence_score
                }
            })
            
            if progress_tracker:
                progress_tracker.update_step(request_id, "column_scope_reduction", "completed", 
                                           f"列范围缩小完成，聚焦到{len(relevant_columns)}个相关列")
        else:
            if progress_tracker:
                progress_tracker.update_step(request_id, "column_scope_reduction", "completed", "未发现需要缩小的列范围")
    
    async def _stage_row_reduction(self, user_request: str, file_analysis: Dict,
                                 scope_reductions: List, progress_tracker, request_id: str):
        """行级别缩小阶段"""
        stage_start = datetime.now()
        
        if progress_tracker:
            progress_tracker.update_step(request_id, "row_scope_reduction", "in_progress", "分析行数据条件...")
        
        # 执行行级别的范围缩小
        current_scope = self.scope_reducer._create_full_scope(file_analysis)
        if scope_reductions:
            for reduction in scope_reductions:
                current_scope = self.scope_reducer._update_scope_from_reduction(current_scope, reduction)
        
        intent_analysis = self.scope_reducer.analyze_query_intent(user_request)
        row_reduction = self.scope_reducer._reduce_by_rows(
            user_request, file_analysis, current_scope, intent_analysis
        )
        
        if row_reduction:
            scope_reductions.append(row_reduction)
            stage_time = (datetime.now() - stage_start).total_seconds()
            
            self.processing_stages.append({
                'stage': ProcessingStage.ROW_REDUCTION,
                'duration': stage_time,
                'result': 'success',
                'details': {
                    'method': row_reduction.reduction_method,
                    'confidence': row_reduction.confidence_score
                }
            })
            
            if progress_tracker:
                progress_tracker.update_step(request_id, "row_scope_reduction", "completed", 
                                           f"行范围缩小完成，应用{row_reduction.reduction_method}过滤")
        else:
            if progress_tracker:
                progress_tracker.update_step(request_id, "row_scope_reduction", "completed", "未发现需要缩小的行范围")
    
    async def _get_file_analysis(self, file_path: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取文件分析数据"""
        try:
            # 尝试从上下文中获取文件分析数据
            if 'relevance_analysis' in context:
                relevance_analysis = context['relevance_analysis']
                # 这里需要从relevance_analysis中提取或重构文件分析数据
                # 暂时返回一个基本结构
                return {
                    'keywords_by_sheet': {},
                    'sheet_details': {},
                    'top_words': {}
                }
            
            # 如果没有现成的分析数据，执行文件摄取来获取
            # 这里应该调用FileIngestAgent
            self.logger.warning("文件分析数据不可用，使用基本处理模式")
            return None
            
        except Exception as e:
            self.logger.error(f"获取文件分析数据失败: {e}")
            return None
    
    async def _parse_intent_with_scope(self, user_request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """基于缩小后的范围解析意图"""
        # 调用父类的意图解析方法
        intent_result = await self._parse_intent(user_request, context)
        
        # 如果有范围缩小信息，将其加入意图分析
        if 'scope_reductions' in context:
            scope_reductions = context['scope_reductions']
            intent_result['scope_optimizations'] = {
                'total_reductions': len(scope_reductions),
                'reduction_levels': [r.level for r in scope_reductions],
                'optimization_applied': True
            }
        
        return intent_result
    
    async def _execute_progressive_workflow(
        self,
        user_request: str,
        file_path: str,
        intent_result: Dict[str, Any],
        request_id: str,
        progress_tracker=None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行经过范围优化的工作流"""
        
        workflow_type = intent_result['workflow_type']
        
        # 根据范围缩小结果优化上下文
        optimized_context = self._create_optimized_context(context or {})
        
        # 执行对应的工作流（使用父类方法，但传入优化后的上下文）
        if workflow_type == WorkflowType.SINGLE_TABLE:
            result = await self._execute_optimized_single_table_workflow(
                user_request, file_path, intent_result, request_id, progress_tracker, optimized_context
            )
        elif workflow_type == WorkflowType.SINGLE_CELL:
            result = await self._execute_single_cell_workflow(
                user_request, file_path, intent_result, request_id, progress_tracker
            )
        elif workflow_type == WorkflowType.MULTI_TABLE:
            result = await self._execute_multi_table_workflow(
                user_request, file_path, intent_result, request_id, progress_tracker
            )
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        # 添加范围优化信息到结果
        if 'scope_reductions' in context:
            result['scope_optimizations'] = {
                'reductions_applied': len(context['scope_reductions']),
                'optimization_summary': self.scope_reducer.get_final_scope_summary(context['scope_reductions']) if self.scope_reducer else {}
            }
        
        return result
    
    def _create_optimized_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """创建经过范围优化的上下文"""
        optimized_context = context.copy()
        
        # 如果有范围缩小信息，添加到上下文中
        if 'scope_reductions' in context:
            scope_reductions = context['scope_reductions']
            
            # 提取优化后的范围信息
            optimization_hints = {
                'focused_sheets': [],
                'focused_columns': [],
                'focused_regions': [],
                'filter_conditions': []
            }
            
            for reduction in scope_reductions:
                if reduction.level == 'sheet' and reduction.target_sheets:
                    optimization_hints['focused_sheets'].extend(reduction.target_sheets)
                elif reduction.level == 'column' and reduction.target_regions:
                    for region in reduction.target_regions:
                        if 'columns' in region:
                            optimization_hints['focused_columns'].extend(region['columns'])
                elif reduction.level == 'region' and reduction.target_regions:
                    optimization_hints['focused_regions'].extend(reduction.target_regions)
                elif reduction.level == 'row' and reduction.target_regions:
                    for region in reduction.target_regions:
                        if 'time_ranges' in region:
                            optimization_hints['filter_conditions'].append(f"时间范围: {region['time_ranges']}")
                        if 'numeric_filters' in region:
                            optimization_hints['filter_conditions'].append(f"数值过滤: {region['numeric_filters']}")
            
            optimized_context['optimization_hints'] = optimization_hints
            optimized_context['scope_optimized'] = True
        
        return optimized_context
    
    async def _execute_optimized_single_table_workflow(
        self,
        user_request: str,
        file_path: str,
        intent_result: Dict[str, Any],
        request_id: str,
        progress_tracker=None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行经过范围优化的单表工作流"""
        
        # 使用父类的单表工作流，但传入优化后的上下文
        result = await super()._execute_single_table_workflow(
            user_request, file_path, intent_result, request_id, progress_tracker
        )
        
        # 在结果中添加优化信息
        if context and 'optimization_hints' in context:
            hints = context['optimization_hints']
            optimization_info = []
            
            if hints['focused_sheets']:
                optimization_info.append(f"聚焦工作表: {', '.join(hints['focused_sheets'])}")
            if hints['focused_columns']:
                optimization_info.append(f"关注列: {', '.join(hints['focused_columns'][:5])}")
            if hints['filter_conditions']:
                optimization_info.append(f"应用条件: {'; '.join(hints['filter_conditions'])}")
            
            if optimization_info:
                # 在结果中添加优化说明
                if 'workflow_steps' in result:
                    result['workflow_steps'].insert(0, {
                        'step': 'scope_optimization',
                        'status': 'success',
                        'agent': 'SmartScopeReducer',
                        'description': f"智能范围优化: {' | '.join(optimization_info)}"
                    })
        
        return result