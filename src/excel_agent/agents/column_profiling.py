"""Column Profiling Agent for analyzing column data types and statistics."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import pandas as pd
import numpy as np

from .base import BaseAgent
from ..models.agents import ColumnProfilingRequest, ColumnProfilingResponse
from ..models.base import (
    AgentRequest, AgentResponse, AgentStatus, 
    ColumnProfile, DataType
)


class ColumnProfilingAgent(BaseAgent):
    """Agent responsible for analyzing column data types and statistics."""
    
    def __init__(self):
        super().__init__(
            name="ColumnProfilingAgent",
            description="Analyzes column ranges, infers data types, and computes statistics",
            mcp_capabilities=["data_analysis", "excel_tools"]
        )
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process column profiling request."""
        if not isinstance(request, ColumnProfilingRequest):
            return self.create_error_response(
                request,
                f"Invalid request type. Expected ColumnProfilingRequest, got {type(request)}"
            )
        
        try:
            # Get file metadata from context
            file_metadata = request.context.get('file_metadata')
            if not file_metadata:
                return self.create_error_response(
                    request,
                    "File metadata not found in request context"
                )
            
            file_path = Path(file_metadata['file_path'])
            
            # Profile columns
            profiles = await self._profile_columns(
                file_path,
                request.sheet_name,
                request.column_name
            )
            
            # Enhance with MCP data analysis if available
            enhanced_profiles = await self._enhance_with_mcp_analysis(
                file_path, request.sheet_name, profiles
            )
            if enhanced_profiles:
                profiles = enhanced_profiles
            
            self.logger.info(
                f"Column profiling completed for sheet '{request.sheet_name}': "
                f"{len(profiles)} columns analyzed"
            )
            
            return ColumnProfilingResponse(
                agent_id=self.name,
                request_id=request.request_id,
                status=AgentStatus.SUCCESS,
                profiles=profiles,
                result={
                    "profiles": [p.model_dump() for p in profiles],
                    "sheet_name": request.sheet_name
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error during column profiling: {e}")
            return self.create_error_response(request, str(e))
    
    async def _profile_columns(
        self,
        file_path: Path,
        sheet_name: str,
        target_column: Optional[str] = None
    ) -> List[ColumnProfile]:
        """Profile columns in the specified sheet."""
        try:
            # Read sheet data
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            profiles = []
            columns_to_profile = [target_column] if target_column else df.columns.tolist()
            
            for column_name in columns_to_profile:
                if column_name not in df.columns:
                    self.logger.warning(f"Column '{column_name}' not found in sheet '{sheet_name}'")
                    continue
                
                profile = self._analyze_column(df[column_name], str(column_name))
                profiles.append(profile)
            
            return profiles
            
        except Exception as e:
            self.logger.error(f"Error profiling columns: {e}")
            raise
    
    def _analyze_column(self, series: pd.Series, column_name: str) -> ColumnProfile:
        """Analyze a single column and create profile."""
        # Basic statistics
        total_count = len(series)
        null_count = series.isnull().sum()
        null_ratio = null_count / total_count if total_count > 0 else 0.0
        unique_values = series.nunique()
        
        # Get non-null values for analysis
        non_null_series = series.dropna()
        
        # Infer data type
        data_type = self._infer_data_type(non_null_series)
        
        # Get sample values (up to 10 unique values)
        sample_values = non_null_series.drop_duplicates().head(10).tolist()
        
        # Calculate value range and statistics
        value_range = self._calculate_value_range(non_null_series, data_type)
        statistics = self._calculate_statistics(non_null_series, data_type)
        
        return ColumnProfile(
            column_name=column_name,
            data_type=data_type,
            value_range=value_range,
            null_ratio=null_ratio,
            unique_values=unique_values,
            sample_values=sample_values,
            statistics=statistics
        )
    
    def _infer_data_type(self, series: pd.Series) -> DataType:
        """Infer the data type of a pandas Series."""
        if len(series) == 0:
            return DataType.UNKNOWN
        
        # Check pandas dtype first
        if pd.api.types.is_integer_dtype(series):
            return DataType.INTEGER
        elif pd.api.types.is_float_dtype(series):
            return DataType.FLOAT
        elif pd.api.types.is_bool_dtype(series):
            return DataType.BOOLEAN
        elif pd.api.types.is_datetime64_any_dtype(series):
            return DataType.DATETIME
        
        # For object dtype, analyze content
        if pd.api.types.is_object_dtype(series):
            return self._infer_object_type(series)
        
        return DataType.UNKNOWN
    
    def _infer_object_type(self, series: pd.Series) -> DataType:
        """Infer data type for object dtype series."""
        sample_size = min(100, len(series))
        sample = series.sample(n=sample_size) if len(series) > sample_size else series
        
        # Count different type patterns
        type_counts = {
            DataType.INTEGER: 0,
            DataType.FLOAT: 0,
            DataType.BOOLEAN: 0,
            DataType.DATE: 0,
            DataType.DATETIME: 0,
            DataType.STRING: 0
        }
        
        for value in sample:
            if pd.isna(value):
                continue
                
            value_str = str(value).strip()
            if not value_str:
                continue
            
            # Check for integer
            if self._is_integer_string(value_str):
                type_counts[DataType.INTEGER] += 1
            # Check for float
            elif self._is_float_string(value_str):
                type_counts[DataType.FLOAT] += 1
            # Check for boolean
            elif self._is_boolean_string(value_str):
                type_counts[DataType.BOOLEAN] += 1
            # Check for date/datetime
            elif self._is_date_string(value_str):
                if self._is_datetime_string(value_str):
                    type_counts[DataType.DATETIME] += 1
                else:
                    type_counts[DataType.DATE] += 1
            # Default to string
            else:
                type_counts[DataType.STRING] += 1
        
        # Determine dominant type
        total_analyzed = sum(type_counts.values())
        if total_analyzed == 0:
            return DataType.UNKNOWN
        
        # Find type with highest proportion
        max_count = max(type_counts.values())
        dominant_type = next(
            dtype for dtype, count in type_counts.items() 
            if count == max_count
        )
        
        # If dominant type is less than 80%, consider it mixed
        if max_count / total_analyzed < 0.8:
            return DataType.MIXED
        
        return dominant_type
    
    def _is_integer_string(self, value: str) -> bool:
        """Check if string represents an integer."""
        try:
            int(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_float_string(self, value: str) -> bool:
        """Check if string represents a float."""
        try:
            float(value)
            return '.' in value or 'e' in value.lower()
        except (ValueError, TypeError):
            return False
    
    def _is_boolean_string(self, value: str) -> bool:
        """Check if string represents a boolean."""
        return value.lower() in ['true', 'false', 'yes', 'no', '1', '0', 'y', 'n']
    
    def _is_date_string(self, value: str) -> bool:
        """Check if string represents a date."""
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
            r'^\d{1,2}/\d{1,2}/\d{4}$',  # M/D/YYYY
        ]
        
        return any(re.match(pattern, value) for pattern in date_patterns)
    
    def _is_datetime_string(self, value: str) -> bool:
        """Check if string represents a datetime."""
        return ' ' in value and ':' in value
    
    def _calculate_value_range(
        self, 
        series: pd.Series, 
        data_type: DataType
    ) -> Optional[Dict[str, Any]]:
        """Calculate value range based on data type."""
        if len(series) == 0:
            return None
        
        try:
            if data_type in [DataType.INTEGER, DataType.FLOAT]:
                numeric_series = pd.to_numeric(series, errors='coerce').dropna()
                if len(numeric_series) > 0:
                    return {
                        'min': float(numeric_series.min()),
                        'max': float(numeric_series.max()),
                        'median': float(numeric_series.median())
                    }
            
            elif data_type in [DataType.DATE, DataType.DATETIME]:
                date_series = pd.to_datetime(series, errors='coerce').dropna()
                if len(date_series) > 0:
                    return {
                        'min': date_series.min().isoformat(),
                        'max': date_series.max().isoformat()
                    }
            
            elif data_type == DataType.STRING:
                str_lengths = series.astype(str).str.len()
                return {
                    'min_length': int(str_lengths.min()),
                    'max_length': int(str_lengths.max()),
                    'avg_length': float(str_lengths.mean())
                }
            
        except Exception as e:
            self.logger.warning(f"Error calculating value range: {e}")
        
        return None
    
    def _calculate_statistics(
        self, 
        series: pd.Series, 
        data_type: DataType
    ) -> Optional[Dict[str, float]]:
        """Calculate statistics based on data type."""
        if len(series) == 0:
            return None
        
        try:
            if data_type in [DataType.INTEGER, DataType.FLOAT]:
                numeric_series = pd.to_numeric(series, errors='coerce').dropna()
                if len(numeric_series) > 0:
                    return {
                        'mean': float(numeric_series.mean()),
                        'std': float(numeric_series.std()),
                        'var': float(numeric_series.var()),
                        'skewness': float(numeric_series.skew()),
                        'kurtosis': float(numeric_series.kurtosis()),
                        'q25': float(numeric_series.quantile(0.25)),
                        'q75': float(numeric_series.quantile(0.75))
                    }
            
            elif data_type == DataType.STRING:
                str_lengths = series.astype(str).str.len()
                return {
                    'avg_length': float(str_lengths.mean()),
                    'std_length': float(str_lengths.std()),
                    'diversity': float(series.nunique() / len(series))
                }
            
            elif data_type == DataType.BOOLEAN:
                bool_series = series.astype(str).str.lower()
                true_count = bool_series.isin(['true', 'yes', '1', 'y']).sum()
                return {
                    'true_ratio': float(true_count / len(series)),
                    'false_ratio': float(1 - true_count / len(series))
                }
                
        except Exception as e:
            self.logger.warning(f"Error calculating statistics: {e}")
        
        return None
    
    async def _enhance_with_mcp_analysis(
        self, 
        file_path: Path, 
        sheet_name: str, 
        profiles: List[ColumnProfile]
    ) -> Optional[List[ColumnProfile]]:
        """Enhance column profiles using MCP data analysis capabilities."""
        try:
            # Use MCP to read Excel data for detailed analysis
            excel_data = await self.call_mcp_tool("read_excel_file", {
                "file_path": str(file_path),
                "sheet_name": sheet_name
            })
            
            if not excel_data or excel_data.get("error"):
                return None
            
            # Convert sheet data to DataFrame format for MCP analysis
            sheet_data = excel_data.get("head", {})
            if not sheet_data:
                return None
            
            # Perform advanced analysis using MCP
            analysis_result = await self.call_mcp_tool("analyze_dataset", {
                "file_path": str(file_path),
                "sheet_name": sheet_name,
                "analysis_type": "statistical"
            })
            
            if analysis_result and not analysis_result.get("error"):
                describe_data = analysis_result.get("describe", {})
                skewness_data = analysis_result.get("skewness", {})
                kurtosis_data = analysis_result.get("kurtosis", {})
                
                # Detect outliers for numeric columns
                outlier_result = await self.call_mcp_tool("detect_outliers", {
                    "data": sheet_data,
                    "method": "iqr"
                })
                
                outlier_info = {}
                if outlier_result and not outlier_result.get("error"):
                    outlier_info = outlier_result.get("outliers", {})
                
                # Enhance existing profiles with MCP analysis
                enhanced_profiles = []
                for profile in profiles:
                    enhanced_profile = profile.model_copy()
                    
                    # Add enhanced statistics from MCP
                    if profile.column_name in describe_data:
                        stats = describe_data[profile.column_name]
                        enhanced_stats = enhanced_profile.statistics or {}
                        enhanced_stats.update({
                            'count': stats.get('count', 0),
                            'mean': stats.get('mean', 0),
                            'std': stats.get('std', 0),
                            'min': stats.get('min', 0),
                            'max': stats.get('max', 0),
                            '25%': stats.get('25%', 0),
                            '50%': stats.get('50%', 0),
                            '75%': stats.get('75%', 0)
                        })
                        
                        # Add advanced statistical measures
                        if profile.column_name in skewness_data:
                            enhanced_stats['skewness'] = skewness_data[profile.column_name]
                        if profile.column_name in kurtosis_data:
                            enhanced_stats['kurtosis'] = kurtosis_data[profile.column_name]
                        
                        enhanced_profile.statistics = enhanced_stats
                    
                    # Add outlier information
                    if profile.column_name in outlier_info:
                        outlier_data = outlier_info[profile.column_name]
                        enhanced_profile.outlier_count = outlier_data.get('count', 0)
                        enhanced_profile.outlier_ratio = outlier_data['count'] / profile.total_count if profile.total_count > 0 else 0
                    
                    enhanced_profiles.append(enhanced_profile)
                
                self.logger.info("Enhanced column profiles with MCP data analysis")
                return enhanced_profiles
            
            return None
            
        except Exception as e:
            self.logger.warning(f"MCP analysis enhancement failed: {e}")
            return None