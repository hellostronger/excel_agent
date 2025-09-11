"""
Markdown Converter for Excel and CSV files.

This module provides functionality to convert Excel and CSV files to markdown format
for better text representation and storage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MarkdownConverter:
    """Converts Excel and CSV files to markdown format."""
    
    def __init__(self, max_rows: int = 1000, max_cols: int = 50, preview_rows: int = 5):
        """
        Initialize MarkdownConverter.
        
        Args:
            max_rows: Maximum number of rows to convert per sheet
            max_cols: Maximum number of columns to convert per sheet
            preview_rows: Number of rows to show in preview tables
        """
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.preview_rows = preview_rows
    
    def convert_file_to_markdown(self, file_path: str, file_id: str = None) -> Dict[str, Any]:
        """
        Convert a file (Excel or CSV) to markdown format.
        
        Args:
            file_path: Path to the file to convert
            file_id: Optional file identifier
            
        Returns:
            Dictionary containing markdown content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension in ['.xlsx', '.xls', '.xlsm']:
                return self._convert_excel_to_markdown(file_path, file_id)
            elif file_extension == '.csv':
                return self._convert_csv_to_markdown(file_path, file_id)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error converting file {file_path} to markdown: {e}")
            raise
    
    def _convert_excel_to_markdown(self, file_path: Path, file_id: str = None) -> Dict[str, Any]:
        """Convert Excel file to markdown format."""
        markdown_content = []
        metadata = {
            'file_path': str(file_path),
            'file_id': file_id,
            'file_type': 'excel',
            'conversion_timestamp': datetime.now().isoformat(),
            'sheets': {},
            'total_data_rows': 0,
            'total_data_cols': 0
        }
        
        # Add file header
        markdown_content.append(f"# Excel File: {file_path.name}")
        markdown_content.append(f"**File Path:** `{file_path}`")
        markdown_content.append(f"**File ID:** {file_id or 'N/A'}")
        markdown_content.append(f"**Converted:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append("")
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            markdown_content.append(f"**Total Sheets:** {len(sheet_names)}")
            markdown_content.append("")
            
            for sheet_name in sheet_names:
                logger.info(f"Converting sheet: {sheet_name}")
                sheet_markdown, sheet_meta = self._convert_sheet_to_markdown(
                    excel_file, sheet_name
                )
                
                markdown_content.extend(sheet_markdown)
                metadata['sheets'][sheet_name] = sheet_meta
                metadata['total_data_rows'] += sheet_meta.get('data_rows', 0)
                metadata['total_data_cols'] += sheet_meta.get('data_cols', 0)
            
        except Exception as e:
            logger.error(f"Error reading Excel file {file_path}: {e}")
            markdown_content.append(f"**Error:** Could not read Excel file - {str(e)}")
        
        return {
            'markdown_content': '\n'.join(markdown_content),
            'metadata': metadata
        }
    
    def _convert_csv_to_markdown(self, file_path: Path, file_id: str = None) -> Dict[str, Any]:
        """Convert CSV file to markdown format."""
        markdown_content = []
        metadata = {
            'file_path': str(file_path),
            'file_id': file_id,
            'file_type': 'csv',
            'conversion_timestamp': datetime.now().isoformat(),
            'data_rows': 0,
            'data_cols': 0
        }
        
        # Add file header
        markdown_content.append(f"# CSV File: {file_path.name}")
        markdown_content.append(f"**File Path:** `{file_path}`")
        markdown_content.append(f"**File ID:** {file_id or 'N/A'}")
        markdown_content.append(f"**Converted:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append("")
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path, nrows=self.max_rows)
            
            # Limit columns if necessary
            if len(df.columns) > self.max_cols:
                df = df.iloc[:, :self.max_cols]
                markdown_content.append(f"⚠️ **Note:** Only showing first {self.max_cols} columns (total: {len(pd.read_csv(file_path, nrows=1).columns)})")
                markdown_content.append("")
            
            sheet_markdown, sheet_meta = self._dataframe_to_markdown(df, "CSV Data")
            markdown_content.extend(sheet_markdown)
            
            metadata.update(sheet_meta)
            
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            markdown_content.append(f"**Error:** Could not read CSV file - {str(e)}")
        
        return {
            'markdown_content': '\n'.join(markdown_content),
            'metadata': metadata
        }
    
    def _convert_sheet_to_markdown(self, excel_file: pd.ExcelFile, sheet_name: str) -> tuple:
        """Convert a single Excel sheet to markdown."""
        markdown_content = []
        sheet_metadata = {'sheet_name': sheet_name}
        
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=self.max_rows)
            
            # Limit columns if necessary
            original_cols = len(df.columns)
            if original_cols > self.max_cols:
                df = df.iloc[:, :self.max_cols]
                markdown_content.append(f"⚠️ **Note:** Only showing first {self.max_cols} columns (total: {original_cols})")
                markdown_content.append("")
            
            sheet_markdown, sheet_meta = self._dataframe_to_markdown(df, sheet_name)
            markdown_content.extend(sheet_markdown)
            sheet_metadata.update(sheet_meta)
            
        except Exception as e:
            logger.error(f"Error reading sheet {sheet_name}: {e}")
            markdown_content.append(f"**Error reading sheet {sheet_name}:** {str(e)}")
            markdown_content.append("")
        
        return markdown_content, sheet_metadata
    
    def _dataframe_to_markdown(self, df: pd.DataFrame, section_title: str) -> tuple:
        """Convert DataFrame to markdown format."""
        markdown_content = []
        metadata = {
            'data_rows': len(df),
            'data_cols': len(df.columns),
            'columns': list(df.columns),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # Section header
        markdown_content.append(f"## {section_title}")
        markdown_content.append("")
        
        # Basic statistics
        markdown_content.append(f"**Dimensions:** {len(df)} rows × {len(df.columns)} columns")
        markdown_content.append("")
        
        # Column information
        if len(df.columns) > 0:
            markdown_content.append("### Column Information")
            markdown_content.append("| Column | Data Type | Non-Null Count | Sample Values |")
            markdown_content.append("|--------|-----------|----------------|---------------|")
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null_count = df[col].count()
                
                # Get sample values
                sample_values = []
                non_null_values = df[col].dropna().head(3)
                for val in non_null_values:
                    if pd.isna(val):
                        continue
                    val_str = str(val).replace('|', '\\|').replace('\n', ' ')[:50]
                    sample_values.append(val_str)
                
                sample_str = ', '.join(sample_values) if sample_values else 'N/A'
                markdown_content.append(f"| {col} | {dtype} | {non_null_count}/{len(df)} | {sample_str} |")
            
            markdown_content.append("")
        
        # Data preview
        if len(df) > 0:
            markdown_content.append("### Data Preview")
            preview_df = df.head(self.preview_rows)
            
            # Clean data for markdown display
            clean_df = preview_df.copy()
            for col in clean_df.columns:
                clean_df[col] = clean_df[col].apply(self._clean_cell_value)
            
            # Convert to markdown table
            try:
                md_table = clean_df.to_markdown(index=True, tablefmt='github')
                markdown_content.append(md_table)
            except Exception as e:
                logger.warning(f"Could not create markdown table: {e}")
                markdown_content.append("*Could not display data table*")
            
            markdown_content.append("")
            
            if len(df) > self.preview_rows:
                markdown_content.append(f"*... and {len(df) - self.preview_rows} more rows*")
                markdown_content.append("")
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            markdown_content.append("### Numeric Summary")
            summary_df = df[numeric_cols].describe()
            
            # Clean summary data
            clean_summary = summary_df.copy()
            for col in clean_summary.columns:
                clean_summary[col] = clean_summary[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            
            try:
                summary_table = clean_summary.to_markdown(tablefmt='github')
                markdown_content.append(summary_table)
            except Exception as e:
                logger.warning(f"Could not create summary table: {e}")
                markdown_content.append("*Could not display summary statistics*")
            
            markdown_content.append("")
        
        # Value counts for text columns (top 5 values)
        text_cols = df.select_dtypes(include=['object', 'string']).columns
        if len(text_cols) > 0 and len(text_cols) <= 5:  # Limit to avoid too much output
            markdown_content.append("### Text Column Value Counts (Top 5)")
            
            for col in text_cols:
                value_counts = df[col].value_counts().head(5)
                if len(value_counts) > 0:
                    markdown_content.append(f"**{col}:**")
                    for value, count in value_counts.items():
                        clean_value = self._clean_cell_value(value)
                        markdown_content.append(f"- {clean_value}: {count}")
                    markdown_content.append("")
        
        markdown_content.append("---")
        markdown_content.append("")
        
        return markdown_content, metadata
    
    def _clean_cell_value(self, value) -> str:
        """Clean cell value for markdown display."""
        if pd.isna(value):
            return ""
        
        # Convert to string and handle special characters
        str_val = str(value)
        
        # Replace problematic characters for markdown
        str_val = str_val.replace('|', '\\|')  # Escape pipe characters
        str_val = str_val.replace('\n', ' ')   # Replace newlines with spaces
        str_val = str_val.replace('\r', ' ')   # Replace carriage returns
        str_val = str_val.replace('\t', ' ')   # Replace tabs with spaces
        
        # Limit length
        if len(str_val) > 100:
            str_val = str_val[:97] + "..."
        
        return str_val
    
    def save_markdown(self, markdown_data: Dict[str, Any], output_path: str) -> str:
        """
        Save markdown content to file.
        
        Args:
            markdown_data: Dictionary containing markdown content and metadata
            output_path: Path to save the markdown file
            
        Returns:
            Path to the saved markdown file
        """
        output_path = Path(output_path)
        
        try:
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write markdown content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_data['markdown_content'])
            
            # Save metadata as JSON sidecar file
            metadata_path = output_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(markdown_data['metadata'], f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved markdown file: {output_path}")
            logger.info(f"Saved metadata file: {metadata_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving markdown to {output_path}: {e}")
            raise


# Global markdown converter instance
markdown_converter = MarkdownConverter()