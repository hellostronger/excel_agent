"""
HTML Converter for Excel and CSV files.

This module provides functionality to convert Excel and CSV files to HTML format
for web display and better visualization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class HTMLConverter:
    """Converts Excel and CSV files to HTML format."""
    
    def __init__(self, max_rows: int = 1000, max_cols: int = 50, preview_rows: int = 20):
        """
        Initialize HTMLConverter.
        
        Args:
            max_rows: Maximum number of rows to convert per sheet
            max_cols: Maximum number of columns to convert per sheet
            preview_rows: Number of rows to show in preview tables
        """
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.preview_rows = preview_rows
    
    def convert_file_to_html(self, file_path: str, file_id: str = None) -> Dict[str, Any]:
        """
        Convert a file (Excel or CSV) to HTML format.
        
        Args:
            file_path: Path to the file to convert
            file_id: Optional file identifier
            
        Returns:
            Dictionary containing HTML content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension in ['.xlsx', '.xls', '.xlsm']:
                return self._convert_excel_to_html(file_path, file_id)
            elif file_extension == '.csv':
                return self._convert_csv_to_html(file_path, file_id)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error converting file {file_path} to HTML: {e}")
            raise
    
    def _convert_excel_to_html(self, file_path: Path, file_id: str = None) -> Dict[str, Any]:
        """Convert Excel file to HTML format."""
        html_parts = []
        metadata = {
            'file_path': str(file_path),
            'file_id': file_id,
            'file_type': 'excel',
            'conversion_timestamp': datetime.now().isoformat(),
            'sheets': {},
            'total_data_rows': 0,
            'total_data_cols': 0
        }
        
        # HTML header
        html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Excel File Preview</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .file-header { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .sheet-section { margin-bottom: 30px; border: 1px solid #ddd; border-radius: 5px; }
        .sheet-header { background-color: #e9ecef; padding: 10px; font-weight: bold; }
        .sheet-stats { padding: 10px; background-color: #f8f9fa; font-size: 0.9em; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .truncated { color: #666; font-style: italic; }
        .numeric { text-align: right; }
        .error { color: red; font-style: italic; }
    </style>
</head>
<body>
""")
        
        # File information
        html_parts.append(f"""
    <div class="file-header">
        <h1>Excel File: {file_path.name}</h1>
        <p><strong>File Path:</strong> {file_path}</p>
        <p><strong>File ID:</strong> {file_id or 'N/A'}</p>
        <p><strong>Converted:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
""")
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            html_parts.append(f"<p><strong>Total Sheets:</strong> {len(sheet_names)}</p>")
            
            for sheet_name in sheet_names:
                logger.info(f"Converting sheet to HTML: {sheet_name}")
                sheet_html, sheet_meta = self._convert_sheet_to_html(
                    excel_file, sheet_name
                )
                
                html_parts.append(sheet_html)
                metadata['sheets'][sheet_name] = sheet_meta
                metadata['total_data_rows'] += sheet_meta.get('data_rows', 0)
                metadata['total_data_cols'] += sheet_meta.get('data_cols', 0)
            
        except Exception as e:
            logger.error(f"Error reading Excel file {file_path}: {e}")
            html_parts.append(f'<div class="error">Error: Could not read Excel file - {str(e)}</div>')
        
        # HTML footer
        html_parts.append("</body></html>")
        
        return {
            'html_content': '\n'.join(html_parts),
            'metadata': metadata
        }
    
    def _convert_csv_to_html(self, file_path: Path, file_id: str = None) -> Dict[str, Any]:
        """Convert CSV file to HTML format."""
        html_parts = []
        metadata = {
            'file_path': str(file_path),
            'file_id': file_id,
            'file_type': 'csv',
            'conversion_timestamp': datetime.now().isoformat(),
            'data_rows': 0,
            'data_cols': 0
        }
        
        # HTML header
        html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CSV File Preview</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .file-header { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .data-section { margin-bottom: 30px; border: 1px solid #ddd; border-radius: 5px; }
        .data-header { background-color: #e9ecef; padding: 10px; font-weight: bold; }
        .data-stats { padding: 10px; background-color: #f8f9fa; font-size: 0.9em; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .truncated { color: #666; font-style: italic; }
        .numeric { text-align: right; }
        .error { color: red; font-style: italic; }
    </style>
</head>
<body>
""")
        
        # File information
        html_parts.append(f"""
    <div class="file-header">
        <h1>CSV File: {file_path.name}</h1>
        <p><strong>File Path:</strong> {file_path}</p>
        <p><strong>File ID:</strong> {file_id or 'N/A'}</p>
        <p><strong>Converted:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
""")
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path, nrows=self.max_rows)
            
            # Limit columns if necessary
            original_cols = len(df.columns)
            if original_cols > self.max_cols:
                df = df.iloc[:, :self.max_cols]
                html_parts.append(f'<p class="truncated">Note: Only showing first {self.max_cols} columns (total: {original_cols})</p>')
            
            sheet_html, sheet_meta = self._dataframe_to_html(df, "CSV Data")
            html_parts.append(sheet_html)
            
            metadata.update(sheet_meta)
            
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            html_parts.append(f'<div class="error">Error: Could not read CSV file - {str(e)}</div>')
        
        # HTML footer
        html_parts.append("</body></html>")
        
        return {
            'html_content': '\n'.join(html_parts),
            'metadata': metadata
        }
    
    def _convert_sheet_to_html(self, excel_file: pd.ExcelFile, sheet_name: str) -> tuple:
        """Convert a single Excel sheet to HTML."""
        sheet_metadata = {'sheet_name': sheet_name}
        
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=self.max_rows)
            
            # Limit columns if necessary
            original_cols = len(df.columns)
            truncated_note = ""
            if original_cols > self.max_cols:
                df = df.iloc[:, :self.max_cols]
                truncated_note = f'<p class="truncated">Note: Only showing first {self.max_cols} columns (total: {original_cols})</p>'
            
            sheet_html, sheet_meta = self._dataframe_to_html(df, sheet_name, truncated_note)
            sheet_metadata.update(sheet_meta)
            
        except Exception as e:
            logger.error(f"Error reading sheet {sheet_name}: {e}")
            sheet_html = f'''
    <div class="sheet-section">
        <div class="sheet-header">{sheet_name}</div>
        <div class="error">Error reading sheet: {str(e)}</div>
    </div>
'''
        
        return sheet_html, sheet_metadata
    
    def _dataframe_to_html(self, df: pd.DataFrame, section_title: str, truncated_note: str = "") -> tuple:
        """Convert DataFrame to HTML format."""
        metadata = {
            'data_rows': len(df),
            'data_cols': len(df.columns),
            'columns': list(df.columns),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        html_parts = [f'''
    <div class="sheet-section">
        <div class="sheet-header">{section_title}</div>
        <div class="sheet-stats">
            <strong>Dimensions:</strong> {len(df)} rows × {len(df.columns)} columns
        </div>
        {truncated_note}
''']
        
        if len(df) > 0:
            # Convert DataFrame to HTML table
            try:
                # Clean data for HTML display
                clean_df = df.copy()
                for col in clean_df.columns:
                    clean_df[col] = clean_df[col].apply(self._clean_cell_value)
                
                # Convert to HTML with styling
                table_html = clean_df.head(self.preview_rows).to_html(
                    classes='data-table',
                    table_id=f'table-{section_title.replace(" ", "-").lower()}',
                    escape=False,
                    index=True,
                    border=0
                )
                
                html_parts.append(table_html)
                
                if len(df) > self.preview_rows:
                    html_parts.append(f'<p class="truncated">... and {len(df) - self.preview_rows} more rows</p>')
                    
            except Exception as e:
                logger.warning(f"Could not create HTML table: {e}")
                html_parts.append('<div class="error">Could not display data table</div>')
        else:
            html_parts.append('<p>No data to display</p>')
        
        html_parts.append('    </div>')
        
        return '\n'.join(html_parts), metadata
    
    def _clean_cell_value(self, value) -> str:
        """Clean cell value for HTML display."""
        if pd.isna(value):
            return ""
        
        # Convert to string and handle special characters
        str_val = str(value)
        
        # Escape HTML special characters
        str_val = str_val.replace('&', '&amp;')
        str_val = str_val.replace('<', '&lt;')
        str_val = str_val.replace('>', '&gt;')
        str_val = str_val.replace('"', '&quot;')
        str_val = str_val.replace("'", '&#x27;')
        
        # Replace line breaks with <br>
        str_val = str_val.replace('\n', '<br>')
        str_val = str_val.replace('\r', '<br>')
        
        # Limit length
        if len(str_val) > 200:
            str_val = str_val[:197] + "..."
        
        return str_val
    
    def save_html(self, html_data: Dict[str, Any], output_path: str) -> str:
        """
        Save HTML content to file.
        
        Args:
            html_data: Dictionary containing HTML content and metadata
            output_path: Path to save the HTML file
            
        Returns:
            Path to the saved HTML file
        """
        output_path = Path(output_path)
        
        try:
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write HTML content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_data['html_content'])
            
            # Save metadata as JSON sidecar file
            metadata_path = output_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(html_data['metadata'], f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved HTML file: {output_path}")
            logger.info(f"Saved metadata file: {metadata_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving HTML to {output_path}: {e}")
            raise


# Global HTML converter instance
html_converter = HTMLConverter()