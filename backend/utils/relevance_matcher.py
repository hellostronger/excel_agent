"""
Excel相关性匹配器

基于jieba分词实现用户查询与Excel内容的智能匹配，
在调用LLM之前先通过关键词匹配进行快速筛选。
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

# Import text processor for segmentation
try:
    from .text_processor import text_processor
    TEXT_PROCESSOR_AVAILABLE = True
except ImportError:
    logger.warning("Text processor not available")
    TEXT_PROCESSOR_AVAILABLE = False


@dataclass
class RelevanceResult:
    """相关性匹配结果"""
    is_relevant: bool
    confidence_score: float
    matched_sheets: List[str]
    matched_keywords: List[str]
    method: str  # 'keyword_match' or 'llm_fallback'
    details: Dict[str, Any]


class ExcelRelevanceMatcher:
    """Excel内容相关性匹配器"""
    
    def __init__(self):
        """初始化相关性匹配器"""
        self.excel_domain_keywords = {
            # 数据分析相关
            '数据', '分析', '统计', '计算', '汇总', '求和', '平均', '最大', '最小',
            '图表', '可视化', '趋势', '对比', '筛选', '排序', '查找', '透视表',
            
            # Excel功能相关
            '表格', '工作表', '单元格', '行', '列', '公式', '函数', '格式',
            '导入', '导出', '保存', '打开', '编辑', '修改', '删除', '添加',
            
            # 业务数据相关
            '销售', '收入', '支出', '利润', '成本', '预算', '财务', '报表',
            '产品', '客户', '订单', '库存', '价格', '数量', '金额', '百分比',
            '日期', '时间', '年份', '月份', '季度', '期间', '周期',
            
            # 数据操作相关
            '插入', '更新', '复制', '粘贴', '移动', '替换', '合并', '拆分',
            '去重', '验证', '清洗', '转换', '格式化', '标准化'
        }
        
        logger.info(f"Excel相关性匹配器初始化完成，内置{len(self.excel_domain_keywords)}个领域关键词")
    
    def segment_query(self, query: str) -> List[str]:
        """
        对用户查询进行分词
        
        Args:
            query: 用户查询文本
            
        Returns:
            分词结果列表
        """
        if not TEXT_PROCESSOR_AVAILABLE:
            # 简单分词fallback
            words = re.findall(r'[\u4e00-\u9fff]+|\w+', query.lower())
            return [word for word in words if len(word) > 1]
        
        try:
            # 使用text_processor进行分词
            words = text_processor.segment_text(query)
            return words
        except Exception as e:
            logger.error(f"分词失败: {e}")
            # fallback到简单分词
            words = re.findall(r'[\u4e00-\u9fff]+|\w+', query.lower())
            return [word for word in words if len(word) > 1]
    
    def is_excel_related_query(self, query: str) -> Tuple[bool, float, List[str]]:
        """
        判断查询是否与Excel相关
        
        Args:
            query: 用户查询文本
            
        Returns:
            (是否相关, 相关度评分, 匹配的关键词)
        """
        query_words = self.segment_query(query)
        if not query_words:
            return False, 0.0, []
        
        # 查找匹配的领域关键词
        matched_keywords = []
        for word in query_words:
            if word.lower() in self.excel_domain_keywords:
                matched_keywords.append(word)
        
        # 计算相关度评分
        if matched_keywords:
            score = len(matched_keywords) / len(query_words)
            # 调整评分：至少有一个匹配词就认为可能相关
            score = max(score, 0.3)  # 最低相关度
            return True, min(score, 1.0), matched_keywords
        
        # 检查是否包含数字（可能是数据查询）
        has_numbers = any(re.search(r'\d+', word) for word in query_words)
        if has_numbers and len(query_words) <= 10:  # 短查询且包含数字
            return True, 0.2, ['数字查询']
        
        return False, 0.0, []
    
    def match_query_to_sheets(self, query: str, file_text_analysis: Dict[str, Any]) -> RelevanceResult:
        """
        将用户查询与Excel工作表内容进行匹配
        
        Args:
            query: 用户查询
            file_text_analysis: 文件的文本分析结果
            
        Returns:
            相关性匹配结果
        """
        # 分词用户查询
        query_words = self.segment_query(query)
        if not query_words:
            return RelevanceResult(
                is_relevant=False,
                confidence_score=0.0,
                matched_sheets=[],
                matched_keywords=[],
                method='keyword_match',
                details={'error': 'Empty query after segmentation'}
            )
        
        query_words_set = set(word.lower() for word in query_words)
        
        # 获取各工作表的关键词
        keywords_by_sheet = file_text_analysis.get('keywords_by_sheet', {})
        top_words = file_text_analysis.get('top_words', {})
        
        # 创建全局关键词集合（用于快速检查）
        all_file_keywords = set()
        for sheet_keywords in keywords_by_sheet.values():
            for kw in sheet_keywords:
                if isinstance(kw, tuple) and len(kw) >= 1:
                    all_file_keywords.add(kw[0].lower())
        
        # 添加高频词
        for word in top_words.keys():
            all_file_keywords.add(word.lower())
        
        # 计算每个工作表的匹配度
        sheet_matches = {}
        total_matches = 0
        
        for sheet_name, sheet_keywords in keywords_by_sheet.items():
            sheet_words = set()
            
            # 提取工作表关键词
            for kw in sheet_keywords:
                if isinstance(kw, tuple) and len(kw) >= 1:
                    sheet_words.add(kw[0].lower())
            
            # 计算交集
            intersection = query_words_set.intersection(sheet_words)
            if intersection:
                match_score = len(intersection) / len(query_words_set)
                sheet_matches[sheet_name] = {
                    'score': match_score,
                    'matched_words': list(intersection),
                    'total_query_words': len(query_words_set),
                    'matched_count': len(intersection)
                }
                total_matches += len(intersection)
        
        # 检查是否有工作表匹配
        if sheet_matches:
            # 按匹配度排序
            sorted_sheets = sorted(
                sheet_matches.items(), 
                key=lambda x: x[1]['score'], 
                reverse=True
            )
            
            best_match = sorted_sheets[0]
            best_sheet_name = best_match[0]
            best_score = best_match[1]['score']
            
            all_matched_keywords = set()
            matched_sheet_names = []
            
            # 收集所有匹配的工作表和关键词
            for sheet_name, match_info in sheet_matches.items():
                if match_info['score'] >= 0.1:  # 最低匹配阈值
                    matched_sheet_names.append(sheet_name)
                    all_matched_keywords.update(match_info['matched_words'])
            
            return RelevanceResult(
                is_relevant=True,
                confidence_score=min(best_score * 1.2, 1.0),  # 稍微提升置信度
                matched_sheets=matched_sheet_names,
                matched_keywords=list(all_matched_keywords),
                method='keyword_match',
                details={
                    'sheet_matches': sheet_matches,
                    'best_match': {
                        'sheet': best_sheet_name,
                        'score': best_score
                    },
                    'query_words': query_words,
                    'total_file_keywords': len(all_file_keywords)
                }
            )
        
        # 检查是否是通用Excel相关查询
        is_excel_related, excel_score, excel_keywords = self.is_excel_related_query(query)
        if is_excel_related:
            return RelevanceResult(
                is_relevant=True,
                confidence_score=excel_score,
                matched_sheets=[],  # 没有特定工作表匹配
                matched_keywords=excel_keywords,
                method='keyword_match',
                details={
                    'type': 'general_excel_query',
                    'query_words': query_words,
                    'excel_keywords': excel_keywords
                }
            )
        
        # 没有找到匹配，需要LLM判断
        return RelevanceResult(
            is_relevant=False,  # 暂时标记为不相关，等待LLM判断
            confidence_score=0.0,
            matched_sheets=[],
            matched_keywords=[],
            method='needs_llm_fallback',
            details={
                'query_words': query_words,
                'file_keywords_count': len(all_file_keywords),
                'reason': 'No keyword matches found'
            }
        )
    
    def enhance_query_with_context(self, query: str, matched_sheets: List[str], matched_keywords: List[str]) -> str:
        """
        根据匹配结果增强查询上下文
        
        Args:
            query: 原始查询
            matched_sheets: 匹配的工作表
            matched_keywords: 匹配的关键词
            
        Returns:
            增强后的查询
        """
        if not matched_sheets and not matched_keywords:
            return query
        
        context_parts = []
        
        if matched_sheets:
            if len(matched_sheets) == 1:
                context_parts.append(f"请重点关注工作表'{matched_sheets[0]}'")
            else:
                sheets_str = "、".join(matched_sheets[:3])  # 最多显示3个
                context_parts.append(f"请重点关注工作表：{sheets_str}")
        
        if matched_keywords:
            keywords_str = "、".join(matched_keywords[:5])  # 最多显示5个关键词
            context_parts.append(f"查询涉及以下关键词：{keywords_str}")
        
        if context_parts:
            enhanced_query = f"{query}\n\n上下文提示：{' | '.join(context_parts)}"
            return enhanced_query
        
        return query
    
    def get_relevance_summary(self, result: RelevanceResult) -> str:
        """
        获取相关性匹配结果的摘要
        
        Args:
            result: 相关性匹配结果
            
        Returns:
            结果摘要文本
        """
        if not result.is_relevant:
            return f"查询与文件内容无明显关联 (方法: {result.method})"
        
        summary_parts = []
        
        if result.matched_sheets:
            sheets_str = "、".join(result.matched_sheets)
            summary_parts.append(f"匹配工作表: {sheets_str}")
        
        if result.matched_keywords:
            keywords_str = "、".join(result.matched_keywords[:5])
            summary_parts.append(f"关键词: {keywords_str}")
        
        confidence_pct = int(result.confidence_score * 100)
        summary_parts.append(f"置信度: {confidence_pct}%")
        summary_parts.append(f"方法: {result.method}")
        
        return " | ".join(summary_parts)


# 全局实例
relevance_matcher = ExcelRelevanceMatcher()