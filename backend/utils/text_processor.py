"""
Text Processing Utilities for Excel Content Analysis.

This module provides functions to extract text from Excel files, 
perform Chinese word segmentation using jieba, and filter stop words.
"""

import re
import logging
from typing import List, Dict, Set, Any, Optional
from pathlib import Path
import pandas as pd
import jieba
import jieba.analyse

logger = logging.getLogger(__name__)

# 默认中文停用词列表
DEFAULT_STOPWORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
    '个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
    '看', '好', '自己', '这', '那', '来', '能', '对', '时', '地', '们', '出',
    '为', '子', '中', '年', '从', '同', '三', '两', '些', '看到', '由于',
    '但是', '因为', '所以', '如果', '虽然', '然而', '而且', '并且', '或者',
    '另外', '首先', '其次', '最后', '总之', '因此', '可以', '应该', '需要',
    '进行', '实现', '完成', '开始', '结束', '包括', '具有', '存在', '发生',
    '产生', '形成', '成为', '作为', '通过', '根据', '按照', '依据', '关于',
    '对于', '由于', '用于', '便于', '有利于', '有助于'
}


class TextProcessor:
    """Excel文本处理器，用于提取和分析Excel文件中的文本内容。"""
    
    def __init__(self, custom_stopwords: Optional[Set[str]] = None):
        """
        初始化文本处理器。
        
        Args:
            custom_stopwords: 自定义停用词集合，如果为None则使用默认停用词
        """
        self.stopwords = custom_stopwords or DEFAULT_STOPWORDS.copy()
        
        # 初始化jieba
        jieba.initialize()
        
        # 设置jieba日志级别
        jieba.setLogLevel(logging.WARNING)
        
        logger.info(f"TextProcessor initialized with {len(self.stopwords)} stopwords")
    
    def add_stopwords(self, words: List[str]):
        """添加停用词。"""
        self.stopwords.update(words)
        logger.debug(f"Added {len(words)} stopwords")
    
    def remove_stopwords(self, words: List[str]):
        """移除停用词。"""
        for word in words:
            self.stopwords.discard(word)
        logger.debug(f"Removed {len(words)} stopwords")
    
    def extract_text_from_excel(self, file_path: str, max_rows: int = None) -> Dict[str, List[str]]:
        """
        从Excel文件中提取所有文本内容。
        
        Args:
            file_path: Excel文件路径
            max_rows: 最大读取行数，None表示读取所有行
            
        Returns:
            字典，键为sheet名称，值为该sheet中的所有文本内容列表
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        sheet_texts = {}
        
        try:
            # 读取所有sheet
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                logger.debug(f"Processing sheet: {sheet_name}")
                
                # 读取sheet数据
                try:
                    df = pd.read_excel(
                        file_path, 
                        sheet_name=sheet_name, 
                        nrows=max_rows,
                        header=None,  # 不使用头部行作为列名
                        dtype=str     # 所有数据都作为字符串处理
                    )
                    
                    # 提取所有文本
                    all_texts = []
                    
                    # 遍历所有单元格
                    for row_idx in range(len(df)):
                        for col_idx in range(len(df.columns)):
                            cell_value = df.iloc[row_idx, col_idx]
                            
                            # 跳过空值和NaN
                            if pd.isna(cell_value) or cell_value == '':
                                continue
                            
                            # 转换为字符串并清理
                            text = str(cell_value).strip()
                            if text and text.lower() not in ['nan', 'none', 'null']:
                                all_texts.append(text)
                    
                    sheet_texts[sheet_name] = all_texts
                    logger.info(f"Extracted {len(all_texts)} text items from sheet '{sheet_name}'")
                
                except Exception as e:
                    logger.error(f"Error processing sheet '{sheet_name}': {e}")
                    sheet_texts[sheet_name] = []
        
        except Exception as e:
            logger.error(f"Error reading Excel file {file_path}: {e}")
            raise
        
        return sheet_texts
    
    def clean_text(self, text: str) -> str:
        """
        清理文本，移除特殊字符和多余空白。
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除邮箱
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # 移除特殊字符，保留中文、英文、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\w\s\.,!?;:()（），。！？；：""''【】\[\]]+', ' ', text)
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def segment_text(self, text: str, use_hmm: bool = True) -> List[str]:
        """
        使用jieba进行中文分词。
        
        Args:
            text: 待分词的文本
            use_hmm: 是否使用HMM模型
            
        Returns:
            分词结果列表
        """
        if not text:
            return []
        
        # 清理文本
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []
        
        # 使用jieba分词
        words = jieba.lcut(cleaned_text, HMM=use_hmm)
        
        # 过滤结果
        filtered_words = []
        for word in words:
            word = word.strip()
            # 跳过空字符串、单个字符、纯数字和停用词
            if (len(word) > 1 and 
                not word.isdigit() and 
                word not in self.stopwords and
                not re.match(r'^[^\u4e00-\u9fff\w]+$', word)):  # 跳过纯标点符号
                filtered_words.append(word)
        
        return filtered_words
    
    def extract_keywords(self, text: str, top_k: int = 20) -> List[tuple]:
        """
        提取关键词。
        
        Args:
            text: 待分析的文本
            top_k: 返回前k个关键词
            
        Returns:
            关键词列表，每个元素为(词, 权重)元组
        """
        if not text:
            return []
        
        # 清理文本
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []
        
        try:
            # 使用TF-IDF提取关键词
            keywords = jieba.analyse.extract_tags(
                cleaned_text, 
                topK=top_k, 
                withWeight=True,
                allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'vd', 'vn', 'a', 'ad', 'an')
            )
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def process_excel_file(self, file_path: str, max_rows: int = None) -> Dict[str, Any]:
        """
        处理Excel文件，提取文本并进行分词分析。
        
        Args:
            file_path: Excel文件路径
            max_rows: 最大读取行数
            
        Returns:
            包含分析结果的字典
        """
        logger.info(f"Processing Excel file: {file_path}")
        
        result = {
            'file_path': str(file_path),
            'sheets': {},
            'total_texts': 0,
            'total_words': 0,
            'unique_words': set(),
            'all_keywords': []
        }
        
        try:
            # 提取文本
            sheet_texts = self.extract_text_from_excel(file_path, max_rows)
            
            for sheet_name, texts in sheet_texts.items():
                logger.debug(f"Processing {len(texts)} texts from sheet '{sheet_name}'")
                
                sheet_result = {
                    'name': sheet_name,
                    'text_count': len(texts),
                    'raw_texts': texts,
                    'segmented_words': [],
                    'word_count': 0,
                    'unique_words': set(),
                    'keywords': [],
                    'word_frequency': {}
                }
                
                # 对每个文本进行分词
                all_words = []
                for text in texts:
                    words = self.segment_text(text)
                    all_words.extend(words)
                    sheet_result['segmented_words'].append(words)
                
                sheet_result['word_count'] = len(all_words)
                sheet_result['unique_words'] = set(all_words)
                
                # 计算词频
                word_freq = {}
                for word in all_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                
                sheet_result['word_frequency'] = dict(sorted(
                    word_freq.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ))
                
                # 提取关键词（合并所有文本）
                combined_text = ' '.join(texts)
                keywords = self.extract_keywords(combined_text)
                sheet_result['keywords'] = keywords
                
                # 更新总体统计
                result['total_texts'] += len(texts)
                result['total_words'] += len(all_words)
                result['unique_words'].update(all_words)
                result['all_keywords'].extend(keywords)
                
                # 转换set为list以便JSON序列化
                sheet_result['unique_words'] = list(sheet_result['unique_words'])
                result['sheets'][sheet_name] = sheet_result
                
                logger.info(f"Sheet '{sheet_name}': {len(texts)} texts, {len(all_words)} words, {len(sheet_result['unique_words'])} unique words")
        
        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            raise
        
        # 转换set为list以便JSON序列化
        result['unique_words'] = list(result['unique_words'])
        result['unique_word_count'] = len(result['unique_words'])
        
        logger.info(f"Processing complete: {result['total_texts']} texts, {result['total_words']} words, {result['unique_word_count']} unique words")
        
        return result
    
    def get_text_metadata(self, file_path: str, max_rows: int = None) -> Dict[str, Any]:
        """
        获取Excel文件的文本元数据（简化版本，用于存储）。
        
        Args:
            file_path: Excel文件路径
            max_rows: 最大读取行数
            
        Returns:
            文本元数据字典
        """
        result = self.process_excel_file(file_path, max_rows)
        
        # 简化结果，只保留关键信息
        metadata = {
            'text_analysis': {
                'total_texts': result['total_texts'],
                'total_words': result['total_words'],
                'unique_word_count': result['unique_word_count'],
                'top_words': dict(list(
                    sorted(
                        {word: sum(1 for sheet in result['sheets'].values() 
                                 for w in sheet['unique_words'] if w == word)
                         for word in result['unique_words']}.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:50]  # 只保留前50个高频词
                )),
                'keywords_by_sheet': {
                    sheet_name: sheet_data['keywords'][:10]  # 每个sheet只保留前10个关键词
                    for sheet_name, sheet_data in result['sheets'].items()
                },
                'sheet_summary': {
                    sheet_name: {
                        'text_count': sheet_data['text_count'],
                        'word_count': sheet_data['word_count'],
                        'unique_word_count': len(sheet_data['unique_words'])
                    }
                    for sheet_name, sheet_data in result['sheets'].items()
                }
            }
        }
        
        return metadata


# 全局文本处理器实例
text_processor = TextProcessor()