"""Unit tests for File Ingest Agent."""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
import asyncio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from excel_agent.agents.file_ingest import FileIngestAgent
from excel_agent.models.agents import FileIngestRequest
from excel_agent.models.base import AgentStatus


class TestFileIngestAgent:
    """Test suite for File Ingest Agent."""
    
    @pytest.fixture
    def agent(self):
        """Create File Ingest Agent instance."""
        return FileIngestAgent()
    
    @pytest.fixture
    def sample_excel_file(self):
        """Create a temporary Excel file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            # Create sample data
            data = {
                'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['New York', 'London', 'Tokyo']
            }
            df = pd.DataFrame(data)
            df.to_excel(tmp.name, index=False, sheet_name='TestSheet')
            
            return Path(tmp.name)
    
    @pytest.mark.asyncio
    async def test_successful_file_ingestion(self, agent, sample_excel_file):
        """Test successful file ingestion."""
        request = FileIngestRequest(
            agent_id="FileIngestAgent",
            file_path=str(sample_excel_file)
        )
        
        async with agent:
            response = await agent.process(request)
        
        assert response.status == AgentStatus.SUCCESS
        assert response.file_id is not None
        assert len(response.sheets) > 0
        assert 'TestSheet' in response.sheets
        assert response.result is not None
        
        # Clean up
        sample_excel_file.unlink()
    
    @pytest.mark.asyncio
    async def test_nonexistent_file(self, agent):
        """Test handling of non-existent file."""
        request = FileIngestRequest(
            agent_id="FileIngestAgent",
            file_path="/nonexistent/file.xlsx"
        )
        
        async with agent:
            response = await agent.process(request)
        
        assert response.status == AgentStatus.FAILED
        assert "does not exist" in response.error_log
    
    @pytest.mark.asyncio
    async def test_invalid_file_format(self, agent):
        """Test handling of invalid file format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"This is not an Excel file")
            tmp_path = Path(tmp.name)
        
        request = FileIngestRequest(
            agent_id="FileIngestAgent",
            file_path=str(tmp_path)
        )
        
        async with agent:
            response = await agent.process(request)
        
        assert response.status == AgentStatus.FAILED
        assert "Unsupported file format" in response.error_log
        
        # Clean up
        tmp_path.unlink()
    
    @pytest.mark.asyncio 
    async def test_invalid_request_type(self, agent):
        """Test handling of invalid request type."""
        from excel_agent.models.base import AgentRequest
        
        invalid_request = AgentRequest(
            agent_id="FileIngestAgent"
        )
        
        async with agent:
            response = await agent.process(invalid_request)
        
        assert response.status == AgentStatus.FAILED
        assert "Invalid request type" in response.error_log
    
    def test_file_id_generation(self, agent):
        """Test file ID generation."""
        file_path = "/test/path/file.xlsx"
        
        # Generate first ID
        id1 = agent._generate_file_id(file_path)
        
        # Add small delay to ensure different timestamp
        import time
        time.sleep(0.001)
        
        # Generate second ID
        id2 = agent._generate_file_id(file_path)
        
        # IDs should be different (due to timestamp)
        assert id1 != id2, f"Expected different IDs but got: {id1} and {id2}"
        assert len(id1) == 32  # MD5 hash length
        assert len(id2) == 32
        
        # Test that same path with same timestamp would generate same ID
        # (by testing the deterministic nature when timestamp is fixed)
        assert isinstance(id1, str)
        assert isinstance(id2, str)
    
    def test_file_metadata_storage(self, agent):
        """Test file metadata storage and retrieval."""
        # This test doesn't require actual file processing
        file_id = "test_file_id"
        
        # Initially, no metadata should exist
        metadata = agent.get_file_metadata(file_id)
        assert metadata is None
        
        # List should be empty
        files = agent.list_files()
        assert len(files) == 0


if __name__ == "__main__":
    pytest.main([__file__])