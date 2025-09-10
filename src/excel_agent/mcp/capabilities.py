"""MCP Capabilities for Excel Intelligent Agent System."""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional
import tempfile
from pathlib import Path

from .base import MCPCapability, MCPTool, MCPResource, MCPPrompt


class ExcelToolsCapability(MCPCapability):
    """MCP capability for Excel file operations."""
    
    def __init__(self):
        super().__init__("excel_tools", "Excel file reading, writing, and manipulation tools")
    
    async def initialize(self) -> None:
        """Initialize Excel tools capability."""
        
        # Register Excel manipulation tools
        self.register_tool(
            MCPTool(
                name="read_excel_file",
                description="Read Excel file and return sheet information",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to Excel file"},
                        "sheet_name": {"type": "string", "description": "Sheet name (optional)"}
                    },
                    "required": ["file_path"]
                }
            ),
            self._read_excel_file
        )
        
        self.register_tool(
            MCPTool(
                name="write_excel_file",
                description="Write data to Excel file",
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Data to write (dict or DataFrame)"},
                        "file_path": {"type": "string", "description": "Output file path"},
                        "sheet_name": {"type": "string", "description": "Sheet name"}
                    },
                    "required": ["data", "file_path"]
                }
            ),
            self._write_excel_file
        )
        
        self.register_tool(
            MCPTool(
                name="get_sheet_names",
                description="Get all sheet names from Excel file",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to Excel file"}
                    },
                    "required": ["file_path"]
                }
            ),
            self._get_sheet_names
        )
        
        # Register Excel file resources
        self.register_resource(
            MCPResource(
                uri="excel://workbook_structure",
                name="Excel Workbook Structure",
                description="Current workbook structure and metadata",
                mime_type="application/json"
            ),
            self._get_workbook_structure
        )
        
        # Register Excel prompts
        self.register_prompt(
            MCPPrompt(
                name="excel_analysis_prompt",
                description="Generate analysis prompt for Excel data",
                arguments=[
                    {"name": "sheet_name", "description": "Name of the sheet to analyze"},
                    {"name": "columns", "description": "List of column names"}
                ]
            ),
            self._generate_excel_analysis_prompt
        )
    
    async def _read_excel_file(self, file_path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        """Read Excel file and return data."""
        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                return {
                    "sheet_name": sheet_name,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict(),
                    "head": df.head().to_dict(),
                    "info": {
                        "memory_usage": df.memory_usage(deep=True).sum(),
                        "null_counts": df.isnull().sum().to_dict()
                    }
                }
            else:
                # Read all sheets
                excel_file = pd.ExcelFile(file_path)
                sheets_info = {}
                for sheet in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    sheets_info[sheet] = {
                        "shape": df.shape,
                        "columns": df.columns.tolist(),
                        "dtypes": df.dtypes.to_dict()
                    }
                return {
                    "file_path": file_path,
                    "sheets": sheets_info,
                    "total_sheets": len(excel_file.sheet_names)
                }
        except Exception as e:
            return {"error": str(e)}
    
    async def _write_excel_file(self, data: Dict[str, Any], file_path: str, sheet_name: str = "Sheet1") -> Dict[str, Any]:
        """Write data to Excel file."""
        try:
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
            
            df.to_excel(file_path, sheet_name=sheet_name, index=False)
            return {
                "status": "success",
                "file_path": file_path,
                "sheet_name": sheet_name,
                "rows_written": len(df),
                "columns_written": len(df.columns)
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_sheet_names(self, file_path: str) -> List[str]:
        """Get all sheet names from Excel file."""
        try:
            excel_file = pd.ExcelFile(file_path)
            return excel_file.sheet_names
        except Exception as e:
            return [f"Error: {str(e)}"]
    
    async def _get_workbook_structure(self) -> str:
        """Get current workbook structure resource."""
        # This would be populated by the agent context
        return json.dumps({
            "current_workbook": None,
            "available_sheets": [],
            "last_operation": None,
            "timestamp": pd.Timestamp.now().isoformat()
        })
    
    async def _generate_excel_analysis_prompt(self, sheet_name: str, columns: List[str]) -> str:
        """Generate analysis prompt for Excel data."""
        return f"""
        Analyze the Excel sheet '{sheet_name}' with the following columns: {', '.join(columns)}.
        
        Please provide:
        1. Data quality assessment
        2. Statistical summary
        3. Potential data insights
        4. Recommended analysis approaches
        5. Any data cleaning suggestions
        
        Focus on practical business insights and actionable recommendations.
        """


class DataAnalysisCapability(MCPCapability):
    """MCP capability for data analysis operations."""
    
    def __init__(self):
        super().__init__("data_analysis", "Advanced data analysis and statistical operations")
    
    async def initialize(self) -> None:
        """Initialize data analysis capability."""
        
        self.register_tool(
            MCPTool(
                name="analyze_dataset",
                description="Perform comprehensive dataset analysis",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to data file"},
                        "sheet_name": {"type": "string", "description": "Sheet name for Excel files"},
                        "analysis_type": {
                            "type": "string", 
                            "enum": ["basic", "statistical", "correlation", "distribution"],
                            "description": "Type of analysis to perform"
                        }
                    },
                    "required": ["file_path", "analysis_type"]
                }
            ),
            self._analyze_dataset
        )
        
        self.register_tool(
            MCPTool(
                name="detect_outliers",
                description="Detect outliers in numeric columns",
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Dataset as dict"},
                        "columns": {"type": "array", "description": "Columns to analyze"},
                        "method": {
                            "type": "string",
                            "enum": ["iqr", "zscore", "isolation_forest"],
                            "description": "Outlier detection method"
                        }
                    },
                    "required": ["data", "method"]
                }
            ),
            self._detect_outliers
        )
        
        self.register_tool(
            MCPTool(
                name="calculate_correlation",
                description="Calculate correlation matrix for numeric columns",
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Dataset as dict"},
                        "method": {
                            "type": "string",
                            "enum": ["pearson", "spearman", "kendall"],
                            "description": "Correlation method"
                        }
                    },
                    "required": ["data"]
                }
            ),
            self._calculate_correlation
        )
        
        # Analysis results resource
        self.register_resource(
            MCPResource(
                uri="analysis://latest_results",
                name="Latest Analysis Results",
                description="Results from the most recent analysis",
                mime_type="application/json"
            ),
            self._get_latest_analysis_results
        )
    
    async def _analyze_dataset(self, file_path: str, analysis_type: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        """Perform dataset analysis."""
        try:
            # Load data
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_csv(file_path)
            
            results = {"analysis_type": analysis_type, "file_path": file_path}
            
            if analysis_type == "basic":
                results.update({
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict(),
                    "missing_values": df.isnull().sum().to_dict(),
                    "memory_usage": df.memory_usage(deep=True).to_dict()
                })
            
            elif analysis_type == "statistical":
                numeric_df = df.select_dtypes(include=['number'])
                results.update({
                    "describe": numeric_df.describe().to_dict(),
                    "skewness": numeric_df.skew().to_dict(),
                    "kurtosis": numeric_df.kurtosis().to_dict()
                })
            
            elif analysis_type == "correlation":
                numeric_df = df.select_dtypes(include=['number'])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    results["correlation_matrix"] = corr_matrix.to_dict()
                else:
                    results["error"] = "Need at least 2 numeric columns for correlation"
            
            elif analysis_type == "distribution":
                numeric_df = df.select_dtypes(include=['number'])
                distribution_info = {}
                for col in numeric_df.columns:
                    distribution_info[col] = {
                        "histogram": numeric_df[col].value_counts().to_dict(),
                        "percentiles": numeric_df[col].quantile([0.25, 0.5, 0.75]).to_dict()
                    }
                results["distributions"] = distribution_info
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _detect_outliers(self, data: Dict[str, Any], method: str, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect outliers in data."""
        try:
            df = pd.DataFrame(data)
            numeric_df = df.select_dtypes(include=['number'])
            
            if columns:
                numeric_df = numeric_df[columns]
            
            outliers = {}
            
            if method == "iqr":
                for col in numeric_df.columns:
                    Q1 = numeric_df[col].quantile(0.25)
                    Q3 = numeric_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_indices = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)].index.tolist()
                    outliers[col] = {
                        "count": len(outlier_indices),
                        "indices": outlier_indices,
                        "bounds": {"lower": lower_bound, "upper": upper_bound}
                    }
            
            elif method == "zscore":
                from scipy import stats
                for col in numeric_df.columns:
                    z_scores = stats.zscore(numeric_df[col].dropna())
                    outlier_indices = numeric_df[col].dropna().index[abs(z_scores) > 3].tolist()
                    outliers[col] = {
                        "count": len(outlier_indices),
                        "indices": outlier_indices,
                        "threshold": 3
                    }
            
            return {
                "method": method,
                "outliers": outliers,
                "total_outliers": sum(info["count"] for info in outliers.values())
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _calculate_correlation(self, data: Dict[str, Any], method: str = "pearson") -> Dict[str, Any]:
        """Calculate correlation matrix."""
        try:
            df = pd.DataFrame(data)
            numeric_df = df.select_dtypes(include=['number'])
            
            if len(numeric_df.columns) < 2:
                return {"error": "Need at least 2 numeric columns for correlation"}
            
            corr_matrix = numeric_df.corr(method=method)
            
            return {
                "method": method,
                "correlation_matrix": corr_matrix.to_dict(),
                "high_correlations": self._find_high_correlations(corr_matrix)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find high correlations in correlation matrix."""
        high_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_corrs.append({
                        "column1": corr_matrix.columns[i],
                        "column2": corr_matrix.columns[j],
                        "correlation": corr_value
                    })
        return high_corrs
    
    async def _get_latest_analysis_results(self) -> str:
        """Get latest analysis results resource."""
        # This would store the most recent analysis results
        return json.dumps({
            "last_analysis": None,
            "timestamp": pd.Timestamp.now().isoformat(),
            "status": "No analysis performed yet"
        })


class FileManagementCapability(MCPCapability):
    """MCP capability for file management operations."""
    
    def __init__(self):
        super().__init__("file_management", "File system operations and management")
    
    async def initialize(self) -> None:
        """Initialize file management capability."""
        
        self.register_tool(
            MCPTool(
                name="list_files",
                description="List files in directory",
                input_schema={
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "Directory path"},
                        "pattern": {"type": "string", "description": "File pattern (e.g., '*.xlsx')"}
                    },
                    "required": ["directory"]
                }
            ),
            self._list_files
        )
        
        self.register_tool(
            MCPTool(
                name="create_backup",
                description="Create backup copy of file",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "File to backup"},
                        "backup_dir": {"type": "string", "description": "Backup directory"}
                    },
                    "required": ["file_path"]
                }
            ),
            self._create_backup
        )
        
        self.register_resource(
            MCPResource(
                uri="files://recent_files",
                name="Recent Files",
                description="List of recently accessed files",
                mime_type="application/json"
            ),
            self._get_recent_files
        )
    
    async def _list_files(self, directory: str, pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """List files in directory."""
        try:
            import glob
            
            if pattern:
                search_pattern = os.path.join(directory, pattern)
                files = glob.glob(search_pattern)
            else:
                files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            
            file_info = []
            for file_path in files:
                stat = os.stat(file_path)
                file_info.append({
                    "path": file_path,
                    "name": os.path.basename(file_path),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "extension": os.path.splitext(file_path)[1]
                })
            
            return file_info
            
        except Exception as e:
            return [{"error": str(e)}]
    
    async def _create_backup(self, file_path: str, backup_dir: Optional[str] = None) -> Dict[str, Any]:
        """Create backup copy of file."""
        try:
            import shutil
            from datetime import datetime
            
            if not backup_dir:
                backup_dir = os.path.dirname(file_path)
            
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{name}_backup_{timestamp}{ext}"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            shutil.copy2(file_path, backup_path)
            
            return {
                "status": "success",
                "original_file": file_path,
                "backup_file": backup_path,
                "backup_size": os.path.getsize(backup_path)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_recent_files(self) -> str:
        """Get recent files resource."""
        return json.dumps({
            "recent_files": [],
            "last_updated": pd.Timestamp.now().isoformat()
        })


class VisualizationCapability(MCPCapability):
    """MCP capability for data visualization."""
    
    def __init__(self):
        super().__init__("visualization", "Data visualization and chart generation")
    
    async def initialize(self) -> None:
        """Initialize visualization capability."""
        
        self.register_tool(
            MCPTool(
                name="create_chart",
                description="Create chart from data",
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Data to visualize"},
                        "chart_type": {
                            "type": "string",
                            "enum": ["line", "bar", "scatter", "histogram", "heatmap"],
                            "description": "Type of chart to create"
                        },
                        "x_column": {"type": "string", "description": "X-axis column"},
                        "y_column": {"type": "string", "description": "Y-axis column"},
                        "title": {"type": "string", "description": "Chart title"}
                    },
                    "required": ["data", "chart_type"]
                }
            ),
            self._create_chart
        )
        
        self.register_tool(
            MCPTool(
                name="create_correlation_heatmap",
                description="Create correlation heatmap",
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "Data for correlation analysis"},
                        "title": {"type": "string", "description": "Heatmap title"}
                    },
                    "required": ["data"]
                }
            ),
            self._create_correlation_heatmap
        )
    
    async def _create_chart(
        self, 
        data: Dict[str, Any], 
        chart_type: str, 
        x_column: Optional[str] = None, 
        y_column: Optional[str] = None,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create chart from data."""
        try:
            df = pd.DataFrame(data)
            
            # Create temporary file for chart
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                chart_path = tmp_file.name
            
            plt.figure(figsize=(10, 6))
            
            if chart_type == "line" and x_column and y_column:
                plt.plot(df[x_column], df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
            
            elif chart_type == "bar" and x_column and y_column:
                plt.bar(df[x_column], df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
            
            elif chart_type == "scatter" and x_column and y_column:
                plt.scatter(df[x_column], df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
            
            elif chart_type == "histogram" and x_column:
                plt.hist(df[x_column], bins=20)
                plt.xlabel(x_column)
                plt.ylabel("Frequency")
            
            elif chart_type == "heatmap":
                numeric_df = df.select_dtypes(include=['number'])
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
            
            if title:
                plt.title(title)
            
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "status": "success",
                "chart_path": chart_path,
                "chart_type": chart_type,
                "title": title or f"{chart_type.title()} Chart"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _create_correlation_heatmap(self, data: Dict[str, Any], title: Optional[str] = None) -> Dict[str, Any]:
        """Create correlation heatmap."""
        try:
            df = pd.DataFrame(data)
            numeric_df = df.select_dtypes(include=['number'])
            
            if len(numeric_df.columns) < 2:
                return {"error": "Need at least 2 numeric columns for correlation heatmap"}
            
            # Create temporary file for heatmap
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                heatmap_path = tmp_file.name
            
            plt.figure(figsize=(12, 8))
            correlation_matrix = numeric_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            
            if title:
                plt.title(title)
            else:
                plt.title("Correlation Heatmap")
            
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "status": "success",
                "heatmap_path": heatmap_path,
                "correlation_matrix": correlation_matrix.to_dict()
            }
            
        except Exception as e:
            return {"error": str(e)}