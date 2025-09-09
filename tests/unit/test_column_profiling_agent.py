"""Unit tests for Column Profiling Agent."""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from excel_agent.agents.column_profiling import ColumnProfilingAgent
from excel_agent.models.agents import ColumnProfilingRequest
from excel_agent.models.base import AgentStatus, DataType


class TestColumnProfilingAgent:
    """Test suite for Column Profiling Agent."""
    
    @pytest.fixture
    def agent(self):
        """Create Column Profiling Agent instance."""
        return ColumnProfilingAgent()
    
    @pytest.fixture
    def sample_excel_file(self):
        """Create a temporary Excel file with diverse data types."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            # Create sample data with different types
            data = {
                'integers': [1, 2, 3, 4, 5],
                'floats': [1.1, 2.2, 3.3, 4.4, 5.5],
                'strings': ['apple', 'banana', 'cherry', 'date', 'elderberry'],
                'booleans': [True, False, True, False, True],
                'mixed': [1, 'text', 3.14, True, None],
                'dates': pd.date_range('2024-01-01', periods=5),
                'with_nulls': [1, None, 3, None, 5]
            }
            df = pd.DataFrame(data)
            df.to_excel(tmp.name, index=False, sheet_name='TestData')
            
            return Path(tmp.name)
    
    def test_infer_data_type_integer(self, agent):
        """Test integer data type inference."""
        series = pd.Series([1, 2, 3, 4, 5])
        data_type = agent._infer_data_type(series)
        assert data_type == DataType.INTEGER
    
    def test_infer_data_type_float(self, agent):
        """Test float data type inference."""
        series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
        data_type = agent._infer_data_type(series)
        assert data_type == DataType.FLOAT
    
    def test_infer_data_type_string(self, agent):
        """Test string data type inference."""
        series = pd.Series(['apple', 'banana', 'cherry'])
        data_type = agent._infer_data_type(series)
        assert data_type == DataType.STRING
    
    def test_infer_data_type_boolean(self, agent):
        """Test boolean data type inference."""
        series = pd.Series([True, False, True])
        data_type = agent._infer_data_type(series)
        assert data_type == DataType.BOOLEAN
    
    def test_infer_data_type_datetime(self, agent):
        """Test datetime data type inference."""
        series = pd.Series(pd.date_range('2024-01-01', periods=3))
        data_type = agent._infer_data_type(series)
        assert data_type == DataType.DATETIME
    
    def test_infer_data_type_empty(self, agent):
        """Test empty series data type inference."""
        series = pd.Series([])
        data_type = agent._infer_data_type(series)
        assert data_type == DataType.UNKNOWN
    
    def test_is_integer_string(self, agent):
        """Test integer string detection."""
        assert agent._is_integer_string("123") == True
        assert agent._is_integer_string("-456") == True
        assert agent._is_integer_string("12.3") == False
        assert agent._is_integer_string("abc") == False
    
    def test_is_float_string(self, agent):
        """Test float string detection."""
        assert agent._is_float_string("12.34") == True
        assert agent._is_float_string("1.23e-4") == True
        assert agent._is_float_string("123") == False  # Integer, not float
        assert agent._is_float_string("abc") == False
    
    def test_is_boolean_string(self, agent):
        """Test boolean string detection."""
        assert agent._is_boolean_string("true") == True
        assert agent._is_boolean_string("false") == True
        assert agent._is_boolean_string("yes") == True
        assert agent._is_boolean_string("no") == True
        assert agent._is_boolean_string("1") == True
        assert agent._is_boolean_string("0") == True
        assert agent._is_boolean_string("maybe") == False
    
    def test_is_date_string(self, agent):
        """Test date string detection."""
        assert agent._is_date_string("2024-01-01") == True
        assert agent._is_date_string("12/25/2024") == True
        assert agent._is_date_string("01-15-2024") == True
        assert agent._is_date_string("not-a-date") == False
    
    def test_analyze_column_integer(self, agent):
        """Test column analysis for integer data."""
        series = pd.Series([1, 2, 3, 4, 5], name='test_column')
        profile = agent._analyze_column(series, 'test_column')
        
        assert profile.column_name == 'test_column'
        assert profile.data_type == DataType.INTEGER
        assert profile.null_ratio == 0.0
        assert profile.unique_values == 5
        assert profile.value_range is not None
        assert profile.value_range['min'] == 1.0
        assert profile.value_range['max'] == 5.0
        assert profile.statistics is not None
    
    def test_analyze_column_with_nulls(self, agent):
        """Test column analysis with null values."""
        series = pd.Series([1, None, 3, None, 5], name='with_nulls')
        profile = agent._analyze_column(series, 'with_nulls')
        
        assert profile.column_name == 'with_nulls'
        assert profile.null_ratio == 0.4  # 2 out of 5 are null
        assert profile.unique_values == 3  # Only non-null unique values
    
    def test_analyze_column_string(self, agent):
        """Test column analysis for string data."""
        series = pd.Series(['apple', 'banana', 'cherry'], name='fruits')
        profile = agent._analyze_column(series, 'fruits')
        
        assert profile.column_name == 'fruits'
        assert profile.data_type == DataType.STRING
        assert profile.value_range is not None
        assert 'min_length' in profile.value_range
        assert 'max_length' in profile.value_range
        assert 'avg_length' in profile.value_range
    
    @pytest.mark.asyncio
    async def test_successful_column_profiling(self, agent, sample_excel_file):
        """Test successful column profiling."""
        # Mock file metadata in context
        file_metadata = {
            'file_path': str(sample_excel_file),
            'file_size': sample_excel_file.stat().st_size
        }
        
        request = ColumnProfilingRequest(
            agent_id="ColumnProfilingAgent",
            file_id="test_file_id",
            sheet_name="TestData",
            context={'file_metadata': file_metadata}
        )
        
        async with agent:
            response = await agent.process(request)
        
        assert response.status == AgentStatus.SUCCESS
        assert len(response.profiles) > 0
        
        # Check that different data types are correctly identified
        profile_dict = {p.column_name: p for p in response.profiles}
        
        # Verify some expected profiles exist
        expected_columns = ['integers', 'floats', 'strings', 'booleans', 'mixed', 'with_nulls']
        for col in expected_columns:
            assert col in profile_dict, f"Column {col} not found in profiles"
        
        # Verify data types
        assert profile_dict['integers'].data_type == DataType.INTEGER
        assert profile_dict['floats'].data_type == DataType.FLOAT  
        assert profile_dict['strings'].data_type == DataType.STRING
        assert profile_dict['booleans'].data_type == DataType.BOOLEAN
        
        # Check null ratio calculation
        assert profile_dict['with_nulls'].null_ratio > 0
        
        # Clean up
        sample_excel_file.unlink()
    
    @pytest.mark.asyncio
    async def test_missing_file_metadata(self, agent):
        """Test handling of missing file metadata."""
        request = ColumnProfilingRequest(
            agent_id="ColumnProfilingAgent", 
            file_id="test_file_id",
            sheet_name="TestData",
            context={}  # Missing file_metadata
        )
        
        async with agent:
            response = await agent.process(request)
        
        assert response.status == AgentStatus.FAILED
        assert "File metadata not found" in response.error_log
    
    @pytest.mark.asyncio
    async def test_specific_column_profiling(self, agent, sample_excel_file):
        """Test profiling of a specific column."""
        file_metadata = {
            'file_path': str(sample_excel_file),
            'file_size': sample_excel_file.stat().st_size
        }
        
        request = ColumnProfilingRequest(
            agent_id="ColumnProfilingAgent",
            file_id="test_file_id", 
            sheet_name="TestData",
            column_name="integers",  # Specific column
            context={'file_metadata': file_metadata}
        )
        
        async with agent:
            response = await agent.process(request)
        
        assert response.status == AgentStatus.SUCCESS
        assert len(response.profiles) == 1
        assert response.profiles[0].column_name == "integers"
        assert response.profiles[0].data_type == DataType.INTEGER
        
        # Clean up
        sample_excel_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__])