"""
智能范围缩小器

在Excel问答处理过程中，动态缩小查询范围，逐步聚焦到最相关的数据区域。
实现多阶段的范围缩小策略：
1. 工作表级别缩小
2. 数据区域缩小  
3. 列范围缩小
4. 行范围缩小
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import re
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScopeReduction:
    """范围缩小结果"""
    level: str  # 'sheet', 'region', 'column', 'row'
    target_sheets: List[str]
    target_regions: List[Dict[str, Any]]  # 包含具体的行列范围
    confidence_score: float
    reduction_method: str
    details: Dict[str, Any]


@dataclass
class QueryScope:
    """查询范围定义"""
    sheets: List[str]
    columns: List[str]
    rows: Optional[Tuple[int, int]]  # (start_row, end_row)
    regions: List[Dict[str, Any]]
    confidence: float


class SmartScopeReducer:
    """智能范围缩小器"""
    
    def __init__(self):
        """初始化范围缩小器"""
        self.logger = logging.getLogger(__name__)
        
        # 定义不同类型查询的关键词模式
        self.query_patterns = {
            'summary': ['总计', '汇总', '合计', '统计', '概览', '总体', '整体'],
            'filter': ['筛选', '过滤', '满足', '条件', '符合', '包含', '等于'],
            'calculation': ['计算', '求和', '平均', '最大', '最小', '累计', '比例'],
            'comparison': ['对比', '比较', '差异', '相对', '增长', '下降', '变化'],
            'search': ['查找', '搜索', '定位', '寻找', '包含', '匹配'],
            'analysis': ['分析', '趋势', '模式', '规律', '关联', '相关性']
        }
        
        # 时间相关关键词（用于时间范围缩小）
        self.time_keywords = {
            'year': ['年', '年度', '年份'],
            'month': ['月', '月份', '月度'],
            'quarter': ['季度', '季', 'Q1', 'Q2', 'Q3', 'Q4'],
            'recent': ['最近', '近期', '当前', '本', '今'],
            'specific_time': r'\d{4}年|\d{1,2}月|\d{1,2}日'
        }
        
        self.logger.info("智能范围缩小器初始化完成")
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        分析查询意图，确定范围缩小策略
        
        Args:
            query: 用户查询
            
        Returns:
            意图分析结果
        """
        query_lower = query.lower()
        intent_scores = {}
        
        # 计算各类意图的匹配度
        for intent, keywords in self.query_patterns.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            if matches > 0:
                intent_scores[intent] = matches / len(keywords)
        
        # 确定主要意图
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else 'general'
        
        # 检测时间相关信息
        time_info = self._extract_time_info(query)
        
        # 检测数值范围信息
        numeric_info = self._extract_numeric_info(query)
        
        return {
            'primary_intent': primary_intent,
            'intent_scores': intent_scores,
            'time_info': time_info,
            'numeric_info': numeric_info,
            'specificity_level': self._calculate_specificity(query)
        }
    
    def reduce_scope_iteratively(
        self, 
        query: str, 
        file_analysis: Dict[str, Any],
        initial_scope: Optional[QueryScope] = None
    ) -> List[ScopeReduction]:
        """
        迭代式范围缩小
        
        Args:
            query: 用户查询
            file_analysis: 文件分析结果
            initial_scope: 初始查询范围
            
        Returns:
            范围缩小步骤列表
        """
        reductions = []
        current_scope = initial_scope or self._create_full_scope(file_analysis)
        
        # 分析查询意图
        intent_analysis = self.analyze_query_intent(query)
        
        # 第一步：工作表级别缩小
        sheet_reduction = self._reduce_by_sheets(query, file_analysis, current_scope)
        if sheet_reduction:
            reductions.append(sheet_reduction)
            current_scope = self._update_scope_from_reduction(current_scope, sheet_reduction)
        
        # 第二步：数据区域缩小
        region_reduction = self._reduce_by_regions(query, file_analysis, current_scope, intent_analysis)
        if region_reduction:
            reductions.append(region_reduction)
            current_scope = self._update_scope_from_reduction(current_scope, region_reduction)
        
        # 第三步：列范围缩小
        column_reduction = self._reduce_by_columns(query, file_analysis, current_scope, intent_analysis)
        if column_reduction:
            reductions.append(column_reduction)
            current_scope = self._update_scope_from_reduction(current_scope, column_reduction)
        
        # 第四步：行范围缩小（基于时间或条件）
        row_reduction = self._reduce_by_rows(query, file_analysis, current_scope, intent_analysis)
        if row_reduction:
            reductions.append(row_reduction)
        
        return reductions
    
    def _reduce_by_sheets(self, query: str, file_analysis: Dict, scope: QueryScope) -> Optional[ScopeReduction]:
        """基于工作表名称和内容相关性缩小范围"""
        keywords_by_sheet = file_analysis.get('keywords_by_sheet', {})
        query_words = set(query.lower().split())
        
        sheet_scores = {}
        
        for sheet_name, sheet_keywords in keywords_by_sheet.items():
            # 计算工作表名称匹配度
            name_score = self._calculate_name_similarity(query, sheet_name)
            
            # 计算内容匹配度
            sheet_words = set()
            for kw in sheet_keywords:
                if isinstance(kw, tuple) and len(kw) >= 1:
                    sheet_words.add(kw[0].lower())
            
            content_score = len(query_words.intersection(sheet_words)) / len(query_words) if query_words else 0
            
            # 综合评分
            total_score = name_score * 0.4 + content_score * 0.6
            if total_score > 0.1:  # 最低相关性阈值
                sheet_scores[sheet_name] = total_score
        
        if sheet_scores:
            # 选择评分最高的工作表
            best_sheets = sorted(sheet_scores.items(), key=lambda x: x[1], reverse=True)
            relevant_sheets = [sheet for sheet, score in best_sheets if score >= 0.3]
            
            if len(relevant_sheets) < len(scope.sheets):  # 确实缩小了范围
                return ScopeReduction(
                    level='sheet',
                    target_sheets=relevant_sheets,
                    target_regions=[],
                    confidence_score=max(sheet_scores.values()),
                    reduction_method='content_and_name_matching',
                    details={'sheet_scores': sheet_scores, 'selected_sheets': relevant_sheets}
                )
        
        return None
    
    def _reduce_by_regions(self, query: str, file_analysis: Dict, scope: QueryScope, intent: Dict) -> Optional[ScopeReduction]:
        """基于数据区域特征缩小范围"""
        regions = []
        
        # 根据查询意图确定目标区域
        if intent['primary_intent'] == 'summary':
            # 汇总查询：寻找包含合计、总计的区域
            regions = self._find_summary_regions(file_analysis, scope.sheets)
        elif intent['primary_intent'] == 'filter':
            # 筛选查询：寻找数据密集区域
            regions = self._find_data_dense_regions(file_analysis, scope.sheets)
        elif intent['time_info']['has_time']:
            # 时间相关查询：寻找包含时间列的区域
            regions = self._find_time_related_regions(file_analysis, scope.sheets, intent['time_info'])
        
        if regions:
            return ScopeReduction(
                level='region',
                target_sheets=scope.sheets,
                target_regions=regions,
                confidence_score=0.7,
                reduction_method=f"{intent['primary_intent']}_based",
                details={'found_regions': len(regions), 'intent': intent['primary_intent']}
            )
        
        return None
    
    def _reduce_by_columns(self, query: str, file_analysis: Dict, scope: QueryScope, intent: Dict) -> Optional[ScopeReduction]:
        """基于列名和查询关键词缩小列范围"""
        query_words = set(query.lower().split())
        relevant_columns = set()
        
        # 从文件分析中提取列信息
        for sheet in scope.sheets:
            sheet_info = file_analysis.get('sheet_details', {}).get(sheet, {})
            columns = sheet_info.get('columns', [])
            
            for col in columns:
                col_lower = col.lower()
                # 检查列名是否与查询词匹配
                if any(word in col_lower for word in query_words):
                    relevant_columns.add(col)
        
        # 根据查询意图添加相关列
        intent_columns = self._get_intent_related_columns(intent, file_analysis, scope.sheets)
        relevant_columns.update(intent_columns)
        
        if relevant_columns:
            return ScopeReduction(
                level='column',
                target_sheets=scope.sheets,
                target_regions=[{'columns': list(relevant_columns)}],
                confidence_score=0.6,
                reduction_method='keyword_and_intent_matching',
                details={'relevant_columns': list(relevant_columns), 'column_count': len(relevant_columns)}
            )
        
        return None
    
    def _reduce_by_rows(self, query: str, file_analysis: Dict, scope: QueryScope, intent: Dict) -> Optional[ScopeReduction]:
        """基于行数据特征缩小行范围"""
        if intent['time_info']['has_time']:
            # 基于时间范围缩小
            time_ranges = self._extract_time_ranges(intent['time_info'], file_analysis, scope.sheets)
            if time_ranges:
                return ScopeReduction(
                    level='row',
                    target_sheets=scope.sheets,
                    target_regions=[{'time_ranges': time_ranges}],
                    confidence_score=0.8,
                    reduction_method='time_based_filtering',
                    details={'time_ranges': time_ranges}
                )
        
        if intent['numeric_info']['has_numeric']:
            # 基于数值条件缩小
            numeric_filters = self._create_numeric_filters(intent['numeric_info'])
            if numeric_filters:
                return ScopeReduction(
                    level='row',
                    target_sheets=scope.sheets,
                    target_regions=[{'numeric_filters': numeric_filters}],
                    confidence_score=0.7,
                    reduction_method='numeric_condition_filtering',
                    details={'filters': numeric_filters}
                )
        
        return None
    
    def _extract_time_info(self, query: str) -> Dict[str, Any]:
        """提取查询中的时间信息"""
        time_info = {
            'has_time': False,
            'time_type': None,
            'specific_values': [],
            'relative_terms': []
        }
        
        query_lower = query.lower()
        
        # 检查时间关键词
        for time_type, keywords in self.time_keywords.items():
            if time_type == 'specific_time':
                matches = re.findall(keywords, query)
                if matches:
                    time_info['has_time'] = True
                    time_info['specific_values'].extend(matches)
            else:
                for keyword in keywords:
                    if keyword in query_lower:
                        time_info['has_time'] = True
                        time_info['time_type'] = time_type
                        time_info['relative_terms'].append(keyword)
        
        return time_info
    
    def _extract_numeric_info(self, query: str) -> Dict[str, Any]:
        """提取查询中的数值信息"""
        numeric_info = {
            'has_numeric': False,
            'numbers': [],
            'operators': [],
            'comparisons': []
        }
        
        # 提取数字
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        if numbers:
            numeric_info['has_numeric'] = True
            numeric_info['numbers'] = [float(n) for n in numbers]
        
        # 提取比较操作符
        operators = ['大于', '小于', '等于', '超过', '低于', '高于', '>=', '<=', '=', '>', '<']
        for op in operators:
            if op in query:
                numeric_info['operators'].append(op)
        
        return numeric_info
    
    def _calculate_specificity(self, query: str) -> float:
        """计算查询的具体程度"""
        specificity_indicators = [
            len(re.findall(r'\d+', query)) * 0.2,  # 数字数量
            len([w for w in query.split() if len(w) > 3]) * 0.1,  # 长词数量
            len(re.findall(r'["""].*?["""]', query)) * 0.3,  # 引用的具体内容
            1 if any(op in query for op in ['=', '>', '<', '>=', '<=']) else 0  # 比较操作符
        ]
        
        return min(sum(specificity_indicators), 1.0)
    
    def _calculate_name_similarity(self, query: str, sheet_name: str) -> float:
        """计算查询与工作表名称的相似度"""
        query_words = set(query.lower().split())
        sheet_words = set(sheet_name.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(sheet_words)
        return len(intersection) / len(query_words)
    
    def _find_summary_regions(self, file_analysis: Dict, sheets: List[str]) -> List[Dict]:
        """寻找包含汇总数据的区域"""
        regions = []
        summary_keywords = ['总计', '合计', '小计', '汇总', 'total', 'sum']
        
        for sheet in sheets:
            # 这里需要实际的数据分析逻辑
            # 暂时返回一个示例区域
            regions.append({
                'sheet': sheet,
                'type': 'summary',
                'description': '检测到可能的汇总区域'
            })
        
        return regions
    
    def _find_data_dense_regions(self, file_analysis: Dict, sheets: List[str]) -> List[Dict]:
        """寻找数据密集区域"""
        regions = []
        
        for sheet in sheets:
            # 实际实现中，这里会分析数据密度
            regions.append({
                'sheet': sheet,
                'type': 'data_dense',
                'description': '检测到数据密集区域'
            })
        
        return regions
    
    def _find_time_related_regions(self, file_analysis: Dict, sheets: List[str], time_info: Dict) -> List[Dict]:
        """寻找时间相关的数据区域"""
        regions = []
        
        for sheet in sheets:
            regions.append({
                'sheet': sheet,
                'type': 'time_related',
                'time_info': time_info,
                'description': '检测到时间相关数据区域'
            })
        
        return regions
    
    def _get_intent_related_columns(self, intent: Dict, file_analysis: Dict, sheets: List[str]) -> Set[str]:
        """根据意图获取相关列"""
        relevant_columns = set()
        
        intent_column_mapping = {
            'calculation': ['金额', '数量', '价格', '总计', 'amount', 'quantity', 'price'],
            'comparison': ['对比', '比较', '增长率', '变化', 'rate', 'change'],
            'summary': ['总计', '合计', '汇总', 'total', 'summary', 'sum']
        }
        
        primary_intent = intent.get('primary_intent', 'general')
        if primary_intent in intent_column_mapping:
            relevant_columns.update(intent_column_mapping[primary_intent])
        
        return relevant_columns
    
    def _extract_time_ranges(self, time_info: Dict, file_analysis: Dict, sheets: List[str]) -> List[Dict]:
        """基于时间信息提取时间范围"""
        time_ranges = []
        
        if time_info.get('specific_values'):
            for value in time_info['specific_values']:
                time_ranges.append({
                    'type': 'specific',
                    'value': value,
                    'description': f'特定时间: {value}'
                })
        
        if time_info.get('relative_terms'):
            for term in time_info['relative_terms']:
                time_ranges.append({
                    'type': 'relative',
                    'term': term,
                    'description': f'相对时间: {term}'
                })
        
        return time_ranges
    
    def _create_numeric_filters(self, numeric_info: Dict) -> List[Dict]:
        """创建数值过滤条件"""
        filters = []
        
        numbers = numeric_info.get('numbers', [])
        operators = numeric_info.get('operators', [])
        
        for i, number in enumerate(numbers):
            op = operators[i] if i < len(operators) else '='
            filters.append({
                'operator': op,
                'value': number,
                'description': f'{op} {number}'
            })
        
        return filters
    
    def _create_full_scope(self, file_analysis: Dict) -> QueryScope:
        """创建包含所有数据的完整范围"""
        all_sheets = list(file_analysis.get('keywords_by_sheet', {}).keys())
        
        return QueryScope(
            sheets=all_sheets,
            columns=[],
            rows=None,
            regions=[],
            confidence=1.0
        )
    
    def _update_scope_from_reduction(self, current_scope: QueryScope, reduction: ScopeReduction) -> QueryScope:
        """根据范围缩小结果更新当前范围"""
        new_scope = QueryScope(
            sheets=reduction.target_sheets if reduction.target_sheets else current_scope.sheets,
            columns=current_scope.columns,
            rows=current_scope.rows,
            regions=reduction.target_regions if reduction.target_regions else current_scope.regions,
            confidence=min(current_scope.confidence, reduction.confidence_score)
        )
        
        return new_scope
    
    def get_final_scope_summary(self, reductions: List[ScopeReduction]) -> Dict[str, Any]:
        """获取最终范围缩小的摘要"""
        if not reductions:
            return {'message': '未进行范围缩小', 'coverage': '100%'}
        
        summary = {
            'total_reductions': len(reductions),
            'reduction_levels': [r.level for r in reductions],
            'final_confidence': min(r.confidence_score for r in reductions),
            'steps': []
        }
        
        for reduction in reductions:
            step_summary = {
                'level': reduction.level,
                'method': reduction.reduction_method,
                'confidence': reduction.confidence_score,
                'description': f"{reduction.level}级别缩小 - {reduction.reduction_method}"
            }
            
            if reduction.target_sheets:
                step_summary['target_sheets'] = reduction.target_sheets
            if reduction.target_regions:
                step_summary['regions_count'] = len(reduction.target_regions)
            
            summary['steps'].append(step_summary)
        
        # 估算覆盖范围缩小程度
        coverage_reduction = sum(0.3 for r in reductions if r.level == 'sheet') + \
                           sum(0.2 for r in reductions if r.level == 'region') + \
                           sum(0.1 for r in reductions if r.level in ['column', 'row'])
        
        final_coverage = max(10, 100 - coverage_reduction * 100)
        summary['estimated_coverage'] = f"{final_coverage:.0f}%"
        
        return summary


# 全局实例
smart_scope_reducer = SmartScopeReducer()