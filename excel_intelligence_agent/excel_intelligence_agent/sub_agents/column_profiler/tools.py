"""
Tools for Column Profiler Agent

Specialized tools for comprehensive data column analysis and quality assessment.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re

from google.adk.tools import ToolContext

from ...shared_libraries.utils import (
    setup_logging, infer_data_type, calculate_quality_score, 
    detect_column_anomalies, sample_column_values
)
from ...shared_libraries.types import ColumnProfile, DataQualityLevel
from ...shared_libraries.constants import EXCEL_DATA_TYPES, BUSINESS_DOMAINS

# Setup logging
logger = setup_logging()


async def analyze_columns_comprehensive(
    file_path: str,
    focus_areas: List[str],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of all columns in Excel file.
    
    Analyzes each column for data types, quality, patterns, and business meaning.
    """
    try:
        logger.info(f"Starting comprehensive column analysis for: {Path(file_path).name}")
        
        # Read Excel file
        excel_data = await _read_excel_file(file_path)
        
        if not excel_data:
            return {
                "success": False,
                "error": "Unable to read Excel file or no data sheets found"
            }
        
        all_column_profiles = []
        sheet_analyses = {}
        
        # Analyze each sheet
        for sheet_name, df in excel_data.items():
            if df.empty:
                continue
                
            logger.info(f"Analyzing columns in sheet: {sheet_name}")
            
            sheet_column_profiles = []
            for col_idx, column_name in enumerate(df.columns):
                try:
                    column_profile = await _analyze_single_column(
                        df[column_name], column_name, col_idx, sheet_name, focus_areas
                    )
                    sheet_column_profiles.append(column_profile)
                    all_column_profiles.append(column_profile)
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze column '{column_name}': {str(e)}")
                    continue
            
            sheet_analyses[sheet_name] = {
                "columns_analyzed": len(sheet_column_profiles),
                "column_profiles": sheet_column_profiles
            }
        
        # Calculate summary statistics
        analysis_summary = _calculate_analysis_summary(all_column_profiles)
        
        return {
            "success": True,
            "total_columns_analyzed": len(all_column_profiles),
            "sheets_analyzed": list(sheet_analyses.keys()),
            "sheet_analyses": sheet_analyses,
            "column_profiles": all_column_profiles,
            "analysis_summary": analysis_summary,
            "focus_areas_applied": focus_areas
        }
        
    except Exception as e:
        logger.error(f"Comprehensive column analysis failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


async def _read_excel_file(file_path: str) -> Dict[str, pd.DataFrame]:
    """Read Excel file and return dictionary of DataFrames"""
    try:
        # Read all sheets
        excel_data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
        
        # Filter out empty sheets and sheets with insufficient data
        filtered_data = {}
        for sheet_name, df in excel_data.items():
            if not df.empty and len(df.columns) > 0 and len(df) > 1:
                # Clean column names
                df.columns = [str(col).strip() if pd.notna(col) else f"Column_{i}" 
                             for i, col in enumerate(df.columns)]
                filtered_data[sheet_name] = df
        
        return filtered_data
        
    except Exception as e:
        logger.error(f"Failed to read Excel file: {str(e)}")
        return {}


async def _analyze_single_column(
    column: pd.Series,
    column_name: str,
    column_index: int,
    sheet_name: str,
    focus_areas: List[str]
) -> Dict[str, Any]:
    """Analyze a single column comprehensively"""
    
    # Basic metrics
    total_count = len(column)
    null_count = column.isnull().sum()
    non_null_count = total_count - null_count
    
    if non_null_count == 0:
        return _create_empty_column_profile(column_name, column_index, sheet_name, total_count)
    
    # Data type analysis
    data_type, type_confidence = infer_data_type(column)
    
    # Quality analysis
    quality_score, quality_level = calculate_quality_score(column)
    
    # Uniqueness analysis
    unique_count = column.nunique()
    duplicate_count = non_null_count - unique_count
    
    # Value distribution analysis
    value_distribution = _analyze_value_distribution(column, data_type)
    
    # Anomaly detection
    anomalies = detect_column_anomalies(column, data_type)
    
    # Sample values
    sample_values = sample_column_values(column, max_samples=10)
    
    # Business meaning inference
    business_meaning = _infer_business_meaning(column_name, column, data_type)
    
    # Statistical analysis
    statistical_summary = _calculate_statistical_summary(column, data_type)
    
    # Format pattern analysis
    format_patterns = _analyze_format_patterns(column, data_type)
    
    # Focus area specific analysis
    focus_analysis = _apply_focus_area_analysis(column, data_type, focus_areas)
    
    return {
        "column_name": column_name,
        "column_index": column_index,
        "sheet_name": sheet_name,
        "data_type": data_type,
        "type_confidence": type_confidence,
        "total_count": total_count,
        "null_count": null_count,
        "null_percentage": null_count / total_count,
        "non_null_count": non_null_count,
        "unique_count": unique_count,
        "duplicate_count": duplicate_count,
        "uniqueness_ratio": unique_count / non_null_count if non_null_count > 0 else 0,
        "quality_score": quality_score,
        "quality_level": quality_level.value,
        "value_distribution": value_distribution,
        "sample_values": sample_values,
        "anomalies": anomalies,
        "business_meaning": business_meaning,
        "statistical_summary": statistical_summary,
        "format_patterns": format_patterns,
        "focus_analysis": focus_analysis,
        "analysis_confidence": _calculate_column_confidence(
            type_confidence, quality_score, len(anomalies), non_null_count
        )
    }


def _create_empty_column_profile(column_name: str, column_index: int, sheet_name: str, total_count: int) -> Dict[str, Any]:
    """Create profile for empty column"""
    return {
        "column_name": column_name,
        "column_index": column_index,
        "sheet_name": sheet_name,
        "data_type": "blank",
        "type_confidence": 1.0,
        "total_count": total_count,
        "null_count": total_count,
        "null_percentage": 1.0,
        "non_null_count": 0,
        "unique_count": 0,
        "duplicate_count": 0,
        "uniqueness_ratio": 0.0,
        "quality_score": 0.0,
        "quality_level": DataQualityLevel.CRITICAL.value,
        "value_distribution": {},
        "sample_values": [],
        "anomalies": ["Column is completely empty"],
        "business_meaning": "unknown",
        "statistical_summary": {},
        "format_patterns": {},
        "focus_analysis": {},
        "analysis_confidence": 1.0  # We're confident it's empty
    }


def _analyze_value_distribution(column: pd.Series, data_type: str) -> Dict[str, Any]:
    """Analyze the distribution of values in a column"""
    non_null_column = column.dropna()
    
    if non_null_column.empty:
        return {"distribution_type": "empty"}
    
    # Calculate value frequency
    value_counts = non_null_column.value_counts()
    
    # Distribution characteristics
    distribution_info = {
        "most_frequent_value": value_counts.index[0] if len(value_counts) > 0 else None,
        "most_frequent_count": value_counts.iloc[0] if len(value_counts) > 0 else 0,
        "frequency_distribution": dict(value_counts.head(10)),  # Top 10 values
        "distribution_evenness": _calculate_distribution_evenness(value_counts),
        "has_dominant_value": value_counts.iloc[0] / len(non_null_column) > 0.5 if len(value_counts) > 0 else False
    }
    
    # Type-specific distribution analysis
    if data_type in ["integer", "float"]:
        distribution_info.update(_analyze_numeric_distribution(non_null_column))
    elif data_type == "text":
        distribution_info.update(_analyze_text_distribution(non_null_column))
    elif data_type == "date":
        distribution_info.update(_analyze_date_distribution(non_null_column))
    
    return distribution_info


def _calculate_distribution_evenness(value_counts: pd.Series) -> float:
    """Calculate how evenly values are distributed (closer to 1 = more even)"""
    if len(value_counts) <= 1:
        return 1.0
    
    # Calculate entropy-based evenness
    proportions = value_counts / value_counts.sum()
    entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
    max_entropy = np.log2(len(value_counts))
    
    return entropy / max_entropy if max_entropy > 0 else 1.0


def _analyze_numeric_distribution(column: pd.Series) -> Dict[str, Any]:
    """Analyze distribution characteristics for numeric columns"""
    numeric_column = pd.to_numeric(column, errors='coerce').dropna()
    
    if numeric_column.empty:
        return {"numeric_analysis": "no_valid_numbers"}
    
    return {
        "min_value": float(numeric_column.min()),
        "max_value": float(numeric_column.max()),
        "mean_value": float(numeric_column.mean()),
        "median_value": float(numeric_column.median()),
        "std_deviation": float(numeric_column.std()),
        "skewness": float(numeric_column.skew()) if len(numeric_column) > 1 else 0,
        "has_negative_values": (numeric_column < 0).any(),
        "has_zero_values": (numeric_column == 0).any(),
        "range_span": float(numeric_column.max() - numeric_column.min())
    }


def _analyze_text_distribution(column: pd.Series) -> Dict[str, Any]:
    """Analyze distribution characteristics for text columns"""
    text_column = column.astype(str)
    
    # Length analysis
    lengths = text_column.str.len()
    
    # Common patterns
    patterns = {
        "email_like": text_column.str.contains(r'@', na=False).sum(),
        "phone_like": text_column.str.contains(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', na=False).sum(),
        "url_like": text_column.str.contains(r'http[s]?://', na=False).sum(),
        "all_caps": text_column.str.isupper().sum(),
        "all_lower": text_column.str.islower().sum(),
        "numeric_strings": text_column.str.isnumeric().sum()
    }
    
    return {
        "avg_text_length": float(lengths.mean()),
        "min_text_length": int(lengths.min()),
        "max_text_length": int(lengths.max()),
        "text_patterns": patterns,
        "consistent_case": patterns["all_caps"] > len(column) * 0.8 or patterns["all_lower"] > len(column) * 0.8
    }


def _analyze_date_distribution(column: pd.Series) -> Dict[str, Any]:
    """Analyze distribution characteristics for date columns"""
    try:
        date_column = pd.to_datetime(column, errors='coerce').dropna()
        
        if date_column.empty:
            return {"date_analysis": "no_valid_dates"}
        
        return {
            "earliest_date": date_column.min().strftime('%Y-%m-%d'),
            "latest_date": date_column.max().strftime('%Y-%m-%d'),
            "date_range_days": (date_column.max() - date_column.min()).days,
            "has_future_dates": (date_column > pd.Timestamp.now()).any(),
            "year_range": date_column.dt.year.max() - date_column.dt.year.min()
        }
    except:
        return {"date_analysis": "date_parsing_failed"}


def _infer_business_meaning(column_name: str, column: pd.Series, data_type: str) -> str:
    """Infer the likely business meaning of a column"""
    name_lower = column_name.lower()
    
    # Common business patterns
    business_patterns = {
        "identifier": ["id", "key", "code", "number", "ref"],
        "name": ["name", "title", "description", "label"],
        "financial": ["price", "cost", "amount", "value", "revenue", "sales", "profit"],
        "temporal": ["date", "time", "created", "updated", "modified"],
        "contact": ["email", "phone", "address", "contact"],
        "status": ["status", "state", "condition", "flag"],
        "quantity": ["count", "quantity", "num", "total", "sum"],
        "geographic": ["location", "city", "country", "region", "zip", "postal"],
        "category": ["type", "category", "class", "group", "segment"]
    }
    
    # Check name patterns
    for meaning, keywords in business_patterns.items():
        if any(keyword in name_lower for keyword in keywords):
            return meaning
    
    # Check data patterns
    if data_type == "text" and column.nunique() / len(column.dropna()) < 0.1:
        return "category"
    elif data_type in ["integer", "float"] and column_name.lower().endswith(("_id", "id")):
        return "identifier"
    elif data_type == "date":
        return "temporal"
    
    return "unknown"


def _calculate_statistical_summary(column: pd.Series, data_type: str) -> Dict[str, Any]:
    """Calculate relevant statistical summary based on data type"""
    summary = {}
    
    if data_type in ["integer", "float"]:
        numeric_data = pd.to_numeric(column, errors='coerce').dropna()
        if len(numeric_data) > 0:
            summary.update({
                "count": len(numeric_data),
                "mean": float(numeric_data.mean()),
                "std": float(numeric_data.std()),
                "min": float(numeric_data.min()),
                "25%": float(numeric_data.quantile(0.25)),
                "50%": float(numeric_data.quantile(0.5)),
                "75%": float(numeric_data.quantile(0.75)),
                "max": float(numeric_data.max())
            })
    
    elif data_type == "text":
        non_null_data = column.dropna()
        if len(non_null_data) > 0:
            text_lengths = non_null_data.astype(str).str.len()
            summary.update({
                "count": len(non_null_data),
                "unique": non_null_data.nunique(),
                "top": non_null_data.mode().iloc[0] if len(non_null_data.mode()) > 0 else None,
                "freq": non_null_data.value_counts().iloc[0] if len(non_null_data) > 0 else 0,
                "mean_length": float(text_lengths.mean()),
                "max_length": int(text_lengths.max())
            })
    
    return summary


def _analyze_format_patterns(column: pd.Series, data_type: str) -> Dict[str, Any]:
    """Analyze format patterns in the data"""
    non_null_column = column.dropna().astype(str)
    
    if non_null_column.empty:
        return {"pattern_analysis": "no_data"}
    
    patterns = {}
    
    # Common format patterns
    if data_type == "text":
        # Check for consistent formatting patterns
        sample_values = non_null_column.head(min(100, len(non_null_column)))
        
        # Length consistency
        lengths = sample_values.str.len()
        patterns["consistent_length"] = lengths.nunique() == 1
        patterns["common_length"] = int(lengths.mode().iloc[0]) if len(lengths.mode()) > 0 else None
        
        # Character patterns
        patterns["alpha_only"] = sample_values.str.isalpha().sum()
        patterns["numeric_only"] = sample_values.str.isnumeric().sum()
        patterns["alphanumeric"] = sample_values.str.isalnum().sum()
        
        # Special patterns
        patterns["contains_spaces"] = sample_values.str.contains(' ').sum()
        patterns["contains_dashes"] = sample_values.str.contains('-').sum()
        patterns["contains_underscores"] = sample_values.str.contains('_').sum()
    
    return patterns


def _apply_focus_area_analysis(column: pd.Series, data_type: str, focus_areas: List[str]) -> Dict[str, Any]:
    """Apply analysis specific to user's focus areas"""
    focus_results = {}
    
    if "data_quality" in focus_areas:
        focus_results["quality_focus"] = {
            "completeness_score": 1 - (column.isnull().sum() / len(column)),
            "consistency_issues": len(detect_column_anomalies(column, data_type)),
            "standardization_needed": _needs_standardization(column, data_type)
        }
    
    if "business_value" in focus_areas:
        focus_results["business_focus"] = {
            "likely_key_column": _is_likely_key_column(column),
            "business_critical": _assess_business_criticality(column),
            "reporting_suitable": _assess_reporting_suitability(column, data_type)
        }
    
    if "relationships" in focus_areas:
        focus_results["relationship_focus"] = {
            "potential_foreign_key": _is_potential_foreign_key(column),
            "lookup_candidate": _is_lookup_candidate(column),
            "join_suitability": _assess_join_suitability(column, data_type)
        }
    
    return focus_results


def _needs_standardization(column: pd.Series, data_type: str) -> bool:
    """Determine if column values need standardization"""
    if data_type != "text":
        return False
    
    non_null_column = column.dropna().astype(str)
    if len(non_null_column) < 2:
        return False
    
    # Check case consistency
    all_upper = non_null_column.str.isupper().all()
    all_lower = non_null_column.str.islower().all()
    
    # Check spacing consistency
    has_leading_spaces = non_null_column.str.startswith(' ').any()
    has_trailing_spaces = non_null_column.str.endswith(' ').any()
    
    return not (all_upper or all_lower) or has_leading_spaces or has_trailing_spaces


def _is_likely_key_column(column: pd.Series) -> bool:
    """Determine if column is likely a key column"""
    if column.empty:
        return False
    
    # High uniqueness suggests key column
    uniqueness_ratio = column.nunique() / len(column.dropna())
    return uniqueness_ratio > 0.95 and column.isnull().sum() == 0


def _assess_business_criticality(column: pd.Series) -> str:
    """Assess business criticality of a column"""
    null_percentage = column.isnull().sum() / len(column)
    uniqueness = column.nunique() / len(column.dropna()) if len(column.dropna()) > 0 else 0
    
    if null_percentage < 0.05 and uniqueness > 0.8:
        return "high"
    elif null_percentage < 0.2 and uniqueness > 0.3:
        return "medium"
    else:
        return "low"


def _assess_reporting_suitability(column: pd.Series, data_type: str) -> str:
    """Assess suitability for reporting and analytics"""
    if data_type in ["integer", "float"]:
        return "high"  # Numeric data is good for reporting
    elif data_type == "date":
        return "high"  # Dates are excellent for time-based reporting
    elif data_type == "text":
        uniqueness = column.nunique() / len(column.dropna()) if len(column.dropna()) > 0 else 0
        if uniqueness < 0.1:  # Low uniqueness suggests categorical
            return "medium"
        else:
            return "low"
    else:
        return "low"


def _is_potential_foreign_key(column: pd.Series) -> bool:
    """Determine if column could be a foreign key"""
    if column.empty:
        return False
    
    # Check for patterns typical of foreign keys
    non_null_column = column.dropna()
    
    # Should have moderate uniqueness (not as high as primary key)
    uniqueness_ratio = non_null_column.nunique() / len(non_null_column)
    
    # Should have some repeated values (referencing same records)
    has_duplicates = len(non_null_column) > non_null_column.nunique()
    
    return 0.1 < uniqueness_ratio < 0.9 and has_duplicates


def _is_lookup_candidate(column: pd.Series) -> bool:
    """Determine if column is good for lookup operations"""
    if column.empty:
        return False
    
    # Low null percentage and reasonable uniqueness
    null_percentage = column.isnull().sum() / len(column)
    uniqueness_ratio = column.nunique() / len(column.dropna()) if len(column.dropna()) > 0 else 0
    
    return null_percentage < 0.1 and uniqueness_ratio > 0.5


def _assess_join_suitability(column: pd.Series, data_type: str) -> str:
    """Assess suitability for joining operations"""
    if column.empty:
        return "poor"
    
    null_percentage = column.isnull().sum() / len(column)
    uniqueness_ratio = column.nunique() / len(column.dropna()) if len(column.dropna()) > 0 else 0
    
    # Ideal join columns have low nulls and good uniqueness
    if null_percentage < 0.05 and uniqueness_ratio > 0.8:
        return "excellent"
    elif null_percentage < 0.2 and uniqueness_ratio > 0.5:
        return "good"
    elif null_percentage < 0.5:
        return "fair"
    else:
        return "poor"


def _calculate_column_confidence(type_confidence: float, quality_score: float, anomaly_count: int, data_points: int) -> float:
    """Calculate overall confidence in column analysis"""
    base_confidence = (type_confidence + quality_score) / 2
    
    # Reduce confidence based on anomalies
    anomaly_penalty = min(anomaly_count * 0.1, 0.3)
    
    # Reduce confidence for small sample sizes
    sample_penalty = max(0, (100 - data_points) / 1000) if data_points < 100 else 0
    
    final_confidence = max(0.1, base_confidence - anomaly_penalty - sample_penalty)
    return min(1.0, final_confidence)


def _calculate_analysis_summary(column_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics across all analyzed columns"""
    if not column_profiles:
        return {"summary": "no_columns_analyzed"}
    
    # Quality distribution
    quality_levels = [col.get("quality_level", "unknown") for col in column_profiles]
    quality_distribution = dict(Counter(quality_levels))
    
    # Data type distribution
    data_types = [col.get("data_type", "unknown") for col in column_profiles]
    type_distribution = dict(Counter(data_types))
    
    # Average metrics
    quality_scores = [col.get("quality_score", 0) for col in column_profiles]
    avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    null_percentages = [col.get("null_percentage", 0) for col in column_profiles]
    avg_null_percentage = sum(null_percentages) / len(null_percentages) if null_percentages else 0
    
    # Business meaning distribution
    business_meanings = [col.get("business_meaning", "unknown") for col in column_profiles]
    business_distribution = dict(Counter(business_meanings))
    
    return {
        "total_columns": len(column_profiles),
        "quality_distribution": quality_distribution,
        "data_type_distribution": type_distribution,
        "business_meaning_distribution": business_distribution,
        "average_quality_score": avg_quality_score,
        "average_null_percentage": avg_null_percentage,
        "columns_with_anomalies": sum(1 for col in column_profiles if col.get("anomalies", [])),
        "high_quality_columns": sum(1 for col in column_profiles if col.get("quality_score", 0) > 0.8)
    }


async def assess_data_quality(
    column_analysis_result: Dict[str, Any],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Assess overall data quality based on column analysis results.
    
    Provides comprehensive quality assessment and improvement recommendations.
    """
    try:
        if not column_analysis_result.get("success", False):
            return {
                "success": False,
                "error": "Column analysis results not available"
            }
        
        column_profiles = column_analysis_result.get("column_profiles", [])
        if not column_profiles:
            return {
                "success": False,
                "error": "No column profiles available for quality assessment"
            }
        
        # Overall quality metrics
        quality_metrics = _calculate_overall_quality_metrics(column_profiles)
        
        # Quality issues identification
        quality_issues = _identify_quality_issues(column_profiles)
        
        # Quality recommendations
        quality_recommendations = _generate_quality_recommendations(quality_metrics, quality_issues)
        
        # Determine overall quality level
        overall_quality_level = _determine_overall_quality_level(quality_metrics)
        
        return {
            "success": True,
            "overall_quality_level": overall_quality_level,
            "quality_metrics": quality_metrics,
            "quality_issues": quality_issues,
            "recommendations": quality_recommendations,
            "assessment_confidence": 0.9
        }
        
    except Exception as e:
        logger.error(f"Data quality assessment failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def _calculate_overall_quality_metrics(column_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall quality metrics across all columns"""
    total_columns = len(column_profiles)
    
    # Quality score distribution
    quality_scores = [col.get("quality_score", 0) for col in column_profiles]
    avg_quality_score = sum(quality_scores) / total_columns if quality_scores else 0
    
    # Completeness metrics
    null_percentages = [col.get("null_percentage", 0) for col in column_profiles]
    avg_completeness = 1 - (sum(null_percentages) / total_columns) if null_percentages else 1
    
    # Consistency metrics
    high_confidence_columns = sum(1 for col in column_profiles if col.get("type_confidence", 0) > 0.8)
    consistency_rate = high_confidence_columns / total_columns if total_columns > 0 else 0
    
    # Anomaly metrics
    columns_with_anomalies = sum(1 for col in column_profiles if col.get("anomalies", []))
    anomaly_rate = columns_with_anomalies / total_columns if total_columns > 0 else 0
    
    return {
        "average_quality_score": avg_quality_score,
        "average_completeness": avg_completeness,
        "consistency_rate": consistency_rate,
        "anomaly_rate": anomaly_rate,
        "total_columns_assessed": total_columns,
        "high_quality_columns": sum(1 for score in quality_scores if score > 0.8),
        "problematic_columns": sum(1 for score in quality_scores if score < 0.5)
    }


def _identify_quality_issues(column_profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify specific data quality issues"""
    issues = []
    
    for col in column_profiles:
        col_issues = []
        
        # High null percentage
        if col.get("null_percentage", 0) > 0.3:
            col_issues.append("high_null_percentage")
        
        # Low type confidence
        if col.get("type_confidence", 1.0) < 0.7:
            col_issues.append("inconsistent_data_types")
        
        # Quality score issues
        if col.get("quality_score", 1.0) < 0.5:
            col_issues.append("low_overall_quality")
        
        # Anomalies present
        if col.get("anomalies", []):
            col_issues.append("data_anomalies_detected")
        
        if col_issues:
            issues.append({
                "column": f"{col.get('sheet_name')}.{col.get('column_name')}",
                "issues": col_issues,
                "severity": _assess_issue_severity(col_issues, col)
            })
    
    return issues


def _assess_issue_severity(issues: List[str], column_profile: Dict[str, Any]) -> str:
    """Assess the severity of quality issues for a column"""
    quality_score = column_profile.get("quality_score", 1.0)
    null_percentage = column_profile.get("null_percentage", 0)
    
    if quality_score < 0.3 or null_percentage > 0.7:
        return "critical"
    elif quality_score < 0.6 or null_percentage > 0.4:
        return "high"
    elif quality_score < 0.8 or null_percentage > 0.2:
        return "medium"
    else:
        return "low"


def _generate_quality_recommendations(quality_metrics: Dict[str, Any], quality_issues: List[Dict[str, Any]]) -> List[str]:
    """Generate actionable quality improvement recommendations"""
    recommendations = []
    
    # Overall recommendations
    avg_quality = quality_metrics.get("average_quality_score", 0)
    if avg_quality < 0.6:
        recommendations.append("Implement comprehensive data cleansing procedures")
        recommendations.append("Establish data validation rules and constraints")
    
    avg_completeness = quality_metrics.get("average_completeness", 1.0)
    if avg_completeness < 0.8:
        recommendations.append("Address missing data through imputation or collection procedures")
    
    anomaly_rate = quality_metrics.get("anomaly_rate", 0)
    if anomaly_rate > 0.3:
        recommendations.append("Investigate and resolve data anomalies and outliers")
    
    # Issue-specific recommendations
    critical_issues = [issue for issue in quality_issues if issue.get("severity") == "critical"]
    if critical_issues:
        recommendations.append(f"Immediately address {len(critical_issues)} critical data quality issues")
    
    high_issues = [issue for issue in quality_issues if issue.get("severity") == "high"]
    if high_issues:
        recommendations.append(f"Prioritize resolution of {len(high_issues)} high-severity quality issues")
    
    # Preventive recommendations
    consistency_rate = quality_metrics.get("consistency_rate", 1.0)
    if consistency_rate < 0.8:
        recommendations.append("Implement data type validation and standardization procedures")
    
    return recommendations


def _determine_overall_quality_level(quality_metrics: Dict[str, Any]) -> str:
    """Determine overall quality level based on metrics"""
    avg_quality = quality_metrics.get("average_quality_score", 0)
    avg_completeness = quality_metrics.get("average_completeness", 1.0)
    anomaly_rate = quality_metrics.get("anomaly_rate", 0)
    
    # Weighted assessment
    quality_weight = 0.4
    completeness_weight = 0.3
    anomaly_weight = 0.3
    
    overall_score = (
        avg_quality * quality_weight +
        avg_completeness * completeness_weight +
        (1 - anomaly_rate) * anomaly_weight
    )
    
    if overall_score >= 0.85:
        return "excellent"
    elif overall_score >= 0.7:
        return "good"
    elif overall_score >= 0.5:
        return "fair"
    elif overall_score >= 0.3:
        return "poor"
    else:
        return "critical"


async def generate_column_insights(
    column_analysis_result: Dict[str, Any],
    quality_assessment_result: Dict[str, Any],
    user_query: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Generate business insights and actionable recommendations from column analysis.
    
    Creates valuable insights that help answer the user's specific questions.
    """
    try:
        if not column_analysis_result.get("success", False):
            return {
                "success": False,
                "error": "Column analysis results not available"
            }
        
        column_profiles = column_analysis_result.get("column_profiles", [])
        quality_metrics = quality_assessment_result.get("quality_metrics", {})
        
        # Generate different types of insights
        business_insights = _generate_business_insights(column_profiles, user_query)
        data_insights = _generate_data_insights(column_profiles, quality_metrics)
        relationship_insights = _generate_relationship_insights(column_profiles)
        optimization_insights = _generate_optimization_insights(column_profiles, quality_metrics)
        
        return {
            "success": True,
            "business_insights": business_insights,
            "data_insights": data_insights,
            "relationship_insights": relationship_insights,
            "optimization_insights": optimization_insights,
            "query_specific_insights": _generate_query_specific_insights(column_profiles, user_query),
            "insight_confidence": 0.8
        }
        
    except Exception as e:
        logger.error(f"Column insights generation failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def _generate_business_insights(column_profiles: List[Dict[str, Any]], user_query: str) -> List[str]:
    """Generate business-focused insights"""
    insights = []
    
    # Identify business-critical columns
    critical_columns = [col for col in column_profiles if col.get("focus_analysis", {}).get("business_focus", {}).get("business_critical") == "high"]
    if critical_columns:
        insights.append(f"Identified {len(critical_columns)} business-critical columns requiring special attention")
    
    # Financial data insights
    financial_columns = [col for col in column_profiles if col.get("business_meaning") == "financial"]
    if financial_columns:
        insights.append(f"Found {len(financial_columns)} financial data columns suitable for revenue and cost analysis")
    
    # Identifier insights
    id_columns = [col for col in column_profiles if col.get("business_meaning") == "identifier"]
    if len(id_columns) > 1:
        insights.append(f"Multiple identifier columns detected - potential for data linking and integration")
    
    # Category insights
    category_columns = [col for col in column_profiles if col.get("business_meaning") == "category"]
    if category_columns:
        insights.append(f"Categorical data in {len(category_columns)} columns enables segmentation and grouping analysis")
    
    return insights


def _generate_data_insights(column_profiles: List[Dict[str, Any]], quality_metrics: Dict[str, Any]) -> List[str]:
    """Generate data-focused insights"""
    insights = []
    
    # Data type insights
    type_distribution = {}
    for col in column_profiles:
        data_type = col.get("data_type", "unknown")
        type_distribution[data_type] = type_distribution.get(data_type, 0) + 1
    
    insights.append(f"Data composition: {dict(type_distribution)}")
    
    # Quality insights
    avg_quality = quality_metrics.get("average_quality_score", 0)
    if avg_quality > 0.8:
        insights.append("High overall data quality enables reliable analysis and reporting")
    elif avg_quality < 0.6:
        insights.append("Data quality issues may impact analysis reliability - cleansing recommended")
    
    # Completeness insights
    high_null_columns = [col for col in column_profiles if col.get("null_percentage", 0) > 0.3]
    if high_null_columns:
        insights.append(f"{len(high_null_columns)} columns have significant missing data affecting completeness")
    
    return insights


def _generate_relationship_insights(column_profiles: List[Dict[str, Any]]) -> List[str]:
    """Generate relationship-focused insights"""
    insights = []
    
    # Key column insights
    key_candidates = [col for col in column_profiles if col.get("focus_analysis", {}).get("relationship_focus", {}).get("potential_foreign_key")]
    if key_candidates:
        insights.append(f"Identified {len(key_candidates)} potential key columns for data relationships")
    
    # Join suitability insights
    excellent_join_columns = [col for col in column_profiles if col.get("focus_analysis", {}).get("relationship_focus", {}).get("join_suitability") == "excellent"]
    if excellent_join_columns:
        insights.append(f"{len(excellent_join_columns)} columns are excellent candidates for joining operations")
    
    return insights


def _generate_optimization_insights(column_profiles: List[Dict[str, Any]], quality_metrics: Dict[str, Any]) -> List[str]:
    """Generate optimization-focused insights"""
    insights = []
    
    # Performance insights
    high_cardinality_columns = [col for col in column_profiles if col.get("unique_count", 0) > 10000]
    if high_cardinality_columns:
        insights.append(f"{len(high_cardinality_columns)} high-cardinality columns may benefit from indexing")
    
    # Storage insights
    text_columns = [col for col in column_profiles if col.get("data_type") == "text"]
    long_text_columns = [col for col in text_columns if col.get("statistical_summary", {}).get("max_length", 0) > 255]
    if long_text_columns:
        insights.append(f"{len(long_text_columns)} text columns with long values may impact storage efficiency")
    
    return insights


def _generate_query_specific_insights(column_profiles: List[Dict[str, Any]], user_query: str) -> List[str]:
    """Generate insights specific to the user's query"""
    insights = []
    query_lower = user_query.lower()
    
    # Query intent detection
    if any(word in query_lower for word in ["quality", "clean", "valid"]):
        quality_issues = sum(1 for col in column_profiles if col.get("anomalies", []))
        insights.append(f"Data quality analysis reveals issues in {quality_issues} columns requiring attention")
    
    if any(word in query_lower for word in ["analyze", "analysis", "insight"]):
        numeric_columns = [col for col in column_profiles if col.get("data_type") in ["integer", "float"]]
        insights.append(f"{len(numeric_columns)} numeric columns available for quantitative analysis")
    
    if any(word in query_lower for word in ["relationship", "connect", "link"]):
        key_columns = [col for col in column_profiles if col.get("business_meaning") == "identifier"]
        insights.append(f"{len(key_columns)} identifier columns can be used for establishing data relationships")
    
    return insights