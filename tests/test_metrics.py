"""Tests for metrics tracking functionality.

This module tests the metrics system for execution time, token usage,
and API call tracking.
"""

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.config import Config
from src.utils.metrics import (
    AggregatedMetrics,
    ExecutionMetrics,
    MetricsCollector,
    get_metrics_collector,
    track_execution_time,
)


class TestExecutionMetrics:
    """Test ExecutionMetrics dataclass."""
    
    def test_execution_metrics_creation(self):
        """Test creating ExecutionMetrics."""
        metrics = ExecutionMetrics(
            name="test_component",
            execution_time=1.5,
            timestamp=1234567890.0,
        )
        
        assert metrics.name == "test_component"
        assert metrics.execution_time == 1.5
        assert metrics.timestamp == 1234567890.0
        assert metrics.token_usage is None
        assert metrics.api_calls == 0
        assert metrics.success is True
        assert metrics.error is None
    
    def test_execution_metrics_with_token_usage(self):
        """Test ExecutionMetrics with token usage."""
        token_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        metrics = ExecutionMetrics(
            name="test_component",
            execution_time=2.0,
            timestamp=1234567890.0,
            token_usage=token_usage,
            api_calls=1,
        )
        
        assert metrics.token_usage == token_usage
        assert metrics.api_calls == 1


class TestMetricsCollector:
    """Test MetricsCollector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector()
    
    def test_record_metrics(self):
        """Test recording metrics."""
        metrics = ExecutionMetrics(
            name="test_component",
            execution_time=1.0,
            timestamp=1234567890.0,
        )
        
        self.collector.record_metrics(metrics)
        
        aggregated = self.collector.get_aggregated_metrics("test_component")
        assert "test_component" in aggregated
        assert aggregated["test_component"].total_executions == 1
    
    def test_get_aggregated_metrics_single_component(self):
        """Test getting aggregated metrics for a single component."""
        # Record multiple metrics
        for i in range(3):
            metrics = ExecutionMetrics(
                name="test_component",
                execution_time=float(i + 1),
                timestamp=1234567890.0 + i,
            )
            self.collector.record_metrics(metrics)
        
        aggregated = self.collector.get_aggregated_metrics("test_component")
        
        assert "test_component" in aggregated
        agg = aggregated["test_component"]
        assert agg.total_executions == 3
        assert agg.successful_executions == 3
        assert agg.failed_executions == 0
        assert agg.total_execution_time == 6.0
        assert agg.average_execution_time == 2.0
        assert agg.min_execution_time == 1.0
        assert agg.max_execution_time == 3.0
    
    def test_get_aggregated_metrics_all_components(self):
        """Test getting aggregated metrics for all components."""
        # Record metrics for multiple components
        self.collector.record_metrics(ExecutionMetrics(
            name="component1",
            execution_time=1.0,
            timestamp=1234567890.0,
        ))
        self.collector.record_metrics(ExecutionMetrics(
            name="component2",
            execution_time=2.0,
            timestamp=1234567890.0,
        ))
        
        aggregated = self.collector.get_aggregated_metrics()
        
        assert "component1" in aggregated
        assert "component2" in aggregated
        assert len(aggregated) == 2
    
    def test_get_aggregated_metrics_with_failures(self):
        """Test aggregated metrics with failed executions."""
        # Record successful execution
        self.collector.record_metrics(ExecutionMetrics(
            name="test_component",
            execution_time=1.0,
            timestamp=1234567890.0,
            success=True,
        ))
        
        # Record failed execution
        self.collector.record_metrics(ExecutionMetrics(
            name="test_component",
            execution_time=0.5,
            timestamp=1234567891.0,
            success=False,
            error="Test error",
        ))
        
        aggregated = self.collector.get_aggregated_metrics("test_component")
        agg = aggregated["test_component"]
        
        assert agg.total_executions == 2
        assert agg.successful_executions == 1
        assert agg.failed_executions == 1
    
    def test_get_aggregated_metrics_with_tokens(self):
        """Test aggregated metrics with token usage."""
        self.collector.record_metrics(ExecutionMetrics(
            name="test_component",
            execution_time=1.0,
            timestamp=1234567890.0,
            token_usage={"total_tokens": 100},
        ))
        self.collector.record_metrics(ExecutionMetrics(
            name="test_component",
            execution_time=2.0,
            timestamp=1234567891.0,
            token_usage={"total_tokens": 200},
        ))
        
        aggregated = self.collector.get_aggregated_metrics("test_component")
        agg = aggregated["test_component"]
        
        assert agg.total_tokens == 300
    
    def test_clear_metrics_single_component(self):
        """Test clearing metrics for a single component."""
        self.collector.record_metrics(ExecutionMetrics(
            name="component1",
            execution_time=1.0,
            timestamp=1234567890.0,
        ))
        self.collector.record_metrics(ExecutionMetrics(
            name="component2",
            execution_time=2.0,
            timestamp=1234567890.0,
        ))
        
        self.collector.clear_metrics("component1")
        
        aggregated = self.collector.get_aggregated_metrics()
        assert "component1" not in aggregated
        assert "component2" in aggregated
    
    def test_clear_metrics_all(self):
        """Test clearing all metrics."""
        self.collector.record_metrics(ExecutionMetrics(
            name="component1",
            execution_time=1.0,
            timestamp=1234567890.0,
        ))
        
        self.collector.clear_metrics()
        
        aggregated = self.collector.get_aggregated_metrics()
        assert len(aggregated) == 0
    
    def test_export_metrics(self, tmp_path: Path):
        """Test exporting metrics to JSON file."""
        # Record some metrics
        self.collector.record_metrics(ExecutionMetrics(
            name="test_component",
            execution_time=1.0,
            timestamp=1234567890.0,
            token_usage={"total_tokens": 100},
        ))
        
        # Export metrics
        export_path = tmp_path / "metrics"
        file_path = self.collector.export_metrics(export_path, "test_metrics.json")
        
        # Verify file was created
        assert file_path.exists()
        assert file_path.name == "test_metrics.json"
        
        # Verify file contents
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "export_timestamp" in data
        assert "components" in data
        assert "test_component" in data["components"]
        assert data["components"]["test_component"]["total_executions"] == 1
    
    def test_export_metrics_default_filename(self, tmp_path: Path):
        """Test exporting metrics with default filename."""
        self.collector.record_metrics(ExecutionMetrics(
            name="test_component",
            execution_time=1.0,
            timestamp=1234567890.0,
        ))
        
        export_path = tmp_path / "metrics"
        file_path = self.collector.export_metrics(export_path)
        
        assert file_path.exists()
        assert file_path.name.startswith("metrics_")
        assert file_path.name.endswith(".json")


class TestTrackExecutionTime:
    """Test track_execution_time decorator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear metrics collector
        collector = get_metrics_collector()
        collector.clear_metrics()
    
    @patch("src.utils.metrics.get_config")
    def test_track_execution_time_disabled(self, mock_get_config):
        """Test that tracking is bypassed when metrics are disabled."""
        mock_config = Mock()
        mock_config.metrics_enabled = False
        mock_get_config.return_value = mock_config
        
        @track_execution_time("test_function")
        def test_func():
            return "result"
        
        result = test_func()
        assert result == "result"
        
        # Verify no metrics were recorded
        collector = get_metrics_collector()
        aggregated = collector.get_aggregated_metrics()
        assert len(aggregated) == 0
    
    @patch("src.utils.metrics.get_config")
    def test_track_execution_time_basic(self, mock_get_config):
        """Test basic execution time tracking."""
        mock_config = Mock()
        mock_config.metrics_enabled = True
        mock_get_config.return_value = mock_config
        
        @track_execution_time("test_function")
        def test_func():
            time.sleep(0.1)  # Sleep to ensure measurable time
            return "result"
        
        result = test_func()
        assert result == "result"
        
        # Verify metrics were recorded
        collector = get_metrics_collector()
        aggregated = collector.get_aggregated_metrics("test_function")
        assert "test_function" in aggregated
        agg = aggregated["test_function"]
        assert agg.total_executions == 1
        assert agg.total_execution_time > 0.0
        assert agg.successful_executions == 1
    
    @patch("src.utils.metrics.get_config")
    def test_track_execution_time_with_error(self, mock_get_config):
        """Test execution time tracking with error."""
        mock_config = Mock()
        mock_config.metrics_enabled = True
        mock_get_config.return_value = mock_config
        
        @track_execution_time("test_function")
        def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_func()
        
        # Verify metrics were recorded with error
        collector = get_metrics_collector()
        aggregated = collector.get_aggregated_metrics("test_function")
        assert "test_function" in aggregated
        agg = aggregated["test_function"]
        assert agg.total_executions == 1
        assert agg.failed_executions == 1
        assert agg.successful_executions == 0
    
    @patch("src.utils.metrics.get_config")
    def test_track_execution_time_uses_function_name(self, mock_get_config):
        """Test that decorator uses function name when component_name not provided."""
        mock_config = Mock()
        mock_config.metrics_enabled = True
        mock_get_config.return_value = mock_config
        
        @track_execution_time()
        def my_test_function():
            return "result"
        
        my_test_function()
        
        # Verify metrics were recorded with function name
        collector = get_metrics_collector()
        aggregated = collector.get_aggregated_metrics("my_test_function")
        assert "my_test_function" in aggregated
    
    @patch("src.utils.metrics.get_config")
    def test_track_execution_time_multiple_calls(self, mock_get_config):
        """Test tracking multiple executions."""
        mock_config = Mock()
        mock_config.metrics_enabled = True
        mock_get_config.return_value = mock_config
        
        @track_execution_time("test_function")
        def test_func():
            return "result"
        
        # Call multiple times
        test_func()
        test_func()
        test_func()
        
        # Verify metrics were aggregated
        collector = get_metrics_collector()
        aggregated = collector.get_aggregated_metrics("test_function")
        assert "test_function" in aggregated
        agg = aggregated["test_function"]
        assert agg.total_executions == 3


class TestTokenUsageExtraction:
    """Test token usage extraction from LLM responses."""
    
    @patch("src.utils.metrics.get_config")
    def test_extract_token_usage_from_response_metadata(self, mock_get_config):
        """Test extracting token usage from response_metadata."""
        mock_config = Mock()
        mock_config.metrics_enabled = True
        mock_get_config.return_value = mock_config
        
        # Create mock response with response_metadata
        mock_response = Mock()
        mock_response.response_metadata = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }
        
        @track_execution_time("test_function", track_tokens=True)
        def test_func():
            return mock_response
        
        test_func()
        
        # Verify token usage was recorded
        collector = get_metrics_collector()
        aggregated = collector.get_aggregated_metrics("test_function")
        agg = aggregated["test_function"]
        assert agg.total_tokens == 150


class TestGetMetricsCollector:
    """Test get_metrics_collector function."""
    
    def test_get_metrics_collector_singleton(self):
        """Test that get_metrics_collector returns singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2
        assert isinstance(collector1, MetricsCollector)

