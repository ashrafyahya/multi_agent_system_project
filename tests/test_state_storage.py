"""Tests for state storage utility.

This module tests the state storage functionality for managing large
state data externally, including storage, retrieval, cleanup, and
reference handling.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.config import Config
from src.utils.state_storage import (cleanup_old_data, get_storage_stats,
                                     is_storage_reference, retrieve_large_data,
                                     store_large_data)


class TestStoreLargeData:
    """Tests for store_large_data function."""
    
    def test_store_large_data_creates_reference(self, tmp_path: Path) -> None:
        """Test storing data creates a valid reference."""
        data = {"key": "value", "number": 42}
        
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            reference = store_large_data(data, data_type="test")
        
        assert reference.startswith("__storage_ref:test:")
        assert len(reference.split(":")) == 3
    
    def test_store_large_data_creates_file(self, tmp_path: Path) -> None:
        """Test storing data creates a JSON file."""
        data = {"key": "value", "nested": {"inner": "data"}}
        
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            reference = store_large_data(data, data_type="test")
        
        # Extract hash from reference
        _, _, data_hash = reference.split(":", 2)
        file_path = tmp_path / f"test_{data_hash}.json"
        
        assert file_path.exists()
        with open(file_path, "r", encoding="utf-8") as f:
            stored_data = json.load(f)
        assert stored_data == data
    
    def test_store_large_data_deduplication(self, tmp_path: Path) -> None:
        """Test storing same data twice returns same reference."""
        data = {"key": "value"}
        
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            ref1 = store_large_data(data, data_type="test")
            ref2 = store_large_data(data, data_type="test")
        
        assert ref1 == ref2
    
    def test_store_large_data_different_types(self, tmp_path: Path) -> None:
        """Test storing same data with different types creates different files."""
        data = {"key": "value"}
        
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            ref1 = store_large_data(data, data_type="type1")
            ref2 = store_large_data(data, data_type="type2")
        
        # Should have different references (different types)
        assert ref1 != ref2
        assert "type1" in ref1
        assert "type2" in ref2
    
    def test_store_large_data_creates_directory(self, tmp_path: Path) -> None:
        """Test storing data creates storage directory if it doesn't exist."""
        storage_dir = tmp_path / "new_storage"
        data = {"key": "value"}
        
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = storage_dir
            store_large_data(data, data_type="test")
        
        assert storage_dir.exists()
        assert storage_dir.is_dir()
    
    def test_store_large_data_string_data(self, tmp_path: Path) -> None:
        """Test storing string data."""
        large_string = "A" * 10000
        
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            reference = store_large_data(large_string, data_type="report")
        
        assert reference.startswith("__storage_ref:report:")
    
    def test_store_large_data_list_data(self, tmp_path: Path) -> None:
        """Test storing list data."""
        large_list = [{"item": i} for i in range(100)]
        
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            reference = store_large_data(large_list, data_type="collected_data")
        
        assert reference.startswith("__storage_ref:collected_data:")
    
    def test_store_large_data_custom_storage_dir(self, tmp_path: Path) -> None:
        """Test storing data with custom storage directory."""
        custom_dir = tmp_path / "custom"
        data = {"key": "value"}
        
        reference = store_large_data(data, data_type="test", storage_dir=custom_dir)
        
        assert custom_dir.exists()
        assert reference.startswith("__storage_ref:test:")


class TestRetrieveLargeData:
    """Tests for retrieve_large_data function."""
    
    def test_retrieve_large_data_returns_stored_data(self, tmp_path: Path) -> None:
        """Test retrieving stored data returns original data."""
        original_data = {"key": "value", "number": 42}
        
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            reference = store_large_data(original_data, data_type="test")
            retrieved_data = retrieve_large_data(reference)
        
        assert retrieved_data == original_data
    
    def test_retrieve_large_data_returns_non_reference_as_is(self) -> None:
        """Test retrieving non-reference returns value as-is."""
        data = "This is not a storage reference"
        
        result = retrieve_large_data(data)
        
        assert result == data
    
    def test_retrieve_large_data_handles_dict_as_non_reference(self) -> None:
        """Test retrieving dict (not a reference) returns as-is."""
        data = {"key": "value"}
        
        result = retrieve_large_data(data)
        
        assert result == data
    
    def test_retrieve_large_data_invalid_reference_format(self, tmp_path: Path) -> None:
        """Test retrieving invalid reference format raises error."""
        invalid_ref = "__storage_ref:invalid"
        
        with pytest.raises(ValueError, match="Invalid storage reference format"):
            retrieve_large_data(invalid_ref, storage_dir=tmp_path)
    
    def test_retrieve_large_data_missing_file(self, tmp_path: Path) -> None:
        """Test retrieving non-existent reference raises error."""
        missing_ref = "__storage_ref:test:nonexistent123"
        
        with pytest.raises(FileNotFoundError):
            retrieve_large_data(missing_ref, storage_dir=tmp_path)
    
    def test_retrieve_large_data_custom_storage_dir(self, tmp_path: Path) -> None:
        """Test retrieving data with custom storage directory."""
        custom_dir = tmp_path / "custom"
        data = {"key": "value"}
        
        reference = store_large_data(data, data_type="test", storage_dir=custom_dir)
        retrieved = retrieve_large_data(reference, storage_dir=custom_dir)
        
        assert retrieved == data


class TestCleanupOldData:
    """Tests for cleanup_old_data function."""
    
    def test_cleanup_old_data_removes_old_files(self, tmp_path: Path) -> None:
        """Test cleanup removes files older than TTL."""
        # Store data
        data = {"key": "value"}
        
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            mock_config.return_value.state_storage_ttl = 1  # 1 second TTL
            reference = store_large_data(data, data_type="test")
        
        # Extract hash and verify file exists
        _, _, data_hash = reference.split(":", 2)
        file_path = tmp_path / f"test_{data_hash}.json"
        assert file_path.exists()
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Cleanup
        cleaned = cleanup_old_data(storage_dir=tmp_path, ttl=1)
        
        assert cleaned > 0
        assert not file_path.exists()
    
    def test_cleanup_old_data_keeps_recent_files(self, tmp_path: Path) -> None:
        """Test cleanup keeps files newer than TTL."""
        data = {"key": "value"}
        
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            mock_config.return_value.state_storage_ttl = 3600  # 1 hour TTL
            reference = store_large_data(data, data_type="test")
        
        # Extract hash and verify file exists
        _, _, data_hash = reference.split(":", 2)
        file_path = tmp_path / f"test_{data_hash}.json"
        assert file_path.exists()
        
        # Cleanup immediately (should not remove recent file)
        cleaned = cleanup_old_data(storage_dir=tmp_path, ttl=3600)
        
        assert cleaned == 0
        assert file_path.exists()
    
    def test_cleanup_old_data_removes_orphaned_files(self, tmp_path: Path) -> None:
        """Test cleanup removes orphaned files without metadata."""
        # Create an orphaned file
        orphaned_file = tmp_path / "test_orphan123.json"
        with open(orphaned_file, "w", encoding="utf-8") as f:
            json.dump({"orphan": True}, f)
        
        # Cleanup should remove orphaned file
        cleaned = cleanup_old_data(storage_dir=tmp_path, ttl=3600)
        
        assert cleaned > 0
        assert not orphaned_file.exists()
    
    def test_cleanup_old_data_returns_count(self, tmp_path: Path) -> None:
        """Test cleanup returns number of files cleaned."""
        # Store multiple old files
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            mock_config.return_value.state_storage_ttl = 1
            
            for i in range(3):
                store_large_data({"key": i}, data_type="test")
        
        # Wait for TTL
        time.sleep(1.1)
        
        # Cleanup
        cleaned = cleanup_old_data(storage_dir=tmp_path, ttl=1)
        
        assert cleaned >= 3
    
    def test_cleanup_old_data_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test cleanup handles nonexistent directory gracefully."""
        nonexistent_dir = tmp_path / "nonexistent"
        
        cleaned = cleanup_old_data(storage_dir=nonexistent_dir, ttl=3600)
        
        assert cleaned == 0


class TestIsStorageReference:
    """Tests for is_storage_reference function."""
    
    def test_is_storage_reference_true(self) -> None:
        """Test is_storage_reference returns True for valid reference."""
        ref = "__storage_ref:test:abc123"
        
        assert is_storage_reference(ref) is True
    
    def test_is_storage_reference_false_for_string(self) -> None:
        """Test is_storage_reference returns False for regular string."""
        value = "This is not a reference"
        
        assert is_storage_reference(value) is False
    
    def test_is_storage_reference_false_for_dict(self) -> None:
        """Test is_storage_reference returns False for dict."""
        value = {"key": "value"}
        
        assert is_storage_reference(value) is False
    
    def test_is_storage_reference_false_for_none(self) -> None:
        """Test is_storage_reference returns False for None."""
        assert is_storage_reference(None) is False


class TestGetStorageStats:
    """Tests for get_storage_stats function."""
    
    def test_get_storage_stats_empty_directory(self, tmp_path: Path) -> None:
        """Test get_storage_stats for empty directory."""
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            stats = get_storage_stats(storage_dir=tmp_path)
        
        assert stats["total_files"] == 0
        assert stats["total_size"] == 0
        assert stats["by_type"] == {}
    
    def test_get_storage_stats_with_data(self, tmp_path: Path) -> None:
        """Test get_storage_stats with stored data."""
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            
            # Store multiple files
            store_large_data({"key": 1}, data_type="type1")
            store_large_data({"key": 2}, data_type="type1")
            store_large_data({"key": 3}, data_type="type2")
            
            stats = get_storage_stats(storage_dir=tmp_path)
        
        assert stats["total_files"] == 3
        assert stats["total_size"] > 0
        assert stats["by_type"]["type1"] == 2
        assert stats["by_type"]["type2"] == 1
    
    def test_get_storage_stats_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test get_storage_stats for nonexistent directory."""
        nonexistent_dir = tmp_path / "nonexistent"
        
        stats = get_storage_stats(storage_dir=nonexistent_dir)
        
        assert stats["total_files"] == 0
        assert stats["total_size"] == 0
        assert stats["by_type"] == {}


class TestStateStorageIntegration:
    """Integration tests for state storage."""
    
    def test_store_and_retrieve_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow of storing and retrieving data."""
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            
            # Store large report
            large_report = "A" * 10000
            ref = store_large_data(large_report, data_type="report")
            
            # Store in state (simulated)
            state_report = ref
            
            # Retrieve when needed
            retrieved_report = retrieve_large_data(state_report)
            
            assert retrieved_report == large_report
            assert len(retrieved_report) == 10000
    
    def test_multiple_data_types(self, tmp_path: Path) -> None:
        """Test storing and retrieving multiple data types."""
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            
            # Store different types
            report_ref = store_large_data("Report content", data_type="report")
            data_ref = store_large_data({"competitors": []}, data_type="collected_data")
            insights_ref = store_large_data({"swot": {}}, data_type="insights")
            
            # Retrieve all
            report = retrieve_large_data(report_ref)
            data = retrieve_large_data(data_ref)
            insights = retrieve_large_data(insights_ref)
            
            assert report == "Report content"
            assert data == {"competitors": []}
            assert insights == {"swot": {}}
    
    def test_cleanup_preserves_recent_data(self, tmp_path: Path) -> None:
        """Test cleanup preserves recently accessed data."""
        with patch("src.utils.state_storage.get_config") as mock_config:
            mock_config.return_value.state_storage_dir = tmp_path
            mock_config.return_value.state_storage_ttl = 1
            
            # Store data
            ref = store_large_data({"key": "value"}, data_type="test")
            
            # Wait a bit
            time.sleep(0.6)
            
            # Access data (updates timestamp)
            retrieve_large_data(ref)
            
            # Wait a bit more
            time.sleep(0.6)
            
            # Cleanup (should not remove recently accessed data)
            cleaned = cleanup_old_data(storage_dir=tmp_path, ttl=1)
            
            # Data should still be retrievable
            data = retrieve_large_data(ref)
            assert data == {"key": "value"}
            assert cleaned == 0  # Should not clean recently accessed data

