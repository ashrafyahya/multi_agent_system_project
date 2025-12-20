"""Tests for agent logger utility.

This module contains unit tests for the AgentLogger class to verify
log file creation, output formatting, error handling, and disabled state.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from src.graph.state import WorkflowState, create_initial_state
from src.utils.agent_logger import AgentLogger


class TestAgentLogger:
    """Tests for AgentLogger class."""
    
    def test_logger_initialization_enabled(self) -> None:
        """Test logger initialization with logging enabled."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            
            assert logger.log_dir == log_dir
            assert logger.enabled is True
            assert log_dir.exists()
    
    def test_logger_initialization_disabled(self) -> None:
        """Test logger initialization with logging disabled."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=False)
            
            assert logger.log_dir == log_dir
            assert logger.enabled is False
            # Directory should NOT be created when disabled
            assert not log_dir.exists()
    
    def test_logger_creates_directory(self) -> None:
        """Test that logger creates log directory if it doesn't exist."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "nonexistent" / "logs"
            assert not log_dir.exists()
            
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            assert log_dir.exists()
    
    def test_log_agent_output_creates_file(self) -> None:
        """Test that logging creates a log file."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            
            output = {"plan": {"tasks": ["task1", "task2"]}}
            state = create_initial_state("Test query")
            
            logger.log_agent_output("planner_agent", output, state)
            
            # Check that a log file was created
            log_files = list(log_dir.glob("planner_agent_*.log"))
            assert len(log_files) == 1
            assert log_files[0].exists()
    
    def test_log_agent_output_file_content(self) -> None:
        """Test that log file contains expected content."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            
            output = {"plan": {"tasks": ["task1", "task2"], "minimum_results": 4}}
            state = create_initial_state("Test query")
            state["retry_count"] = 0
            state["current_task"] = "Planning completed"
            
            logger.log_agent_output("planner_agent", output, state)
            
            log_files = list(log_dir.glob("planner_agent_*.log"))
            assert len(log_files) == 1
            
            content = log_files[0].read_text(encoding="utf-8")
            assert "=== Agent Output Log ===" in content
            assert "Agent: planner_agent" in content
            assert "Workflow Stage: Planning" in content
            assert "Retry Count: 0" in content
            assert "Current Task: Planning completed" in content
            assert "task1" in content
            assert "task2" in content
    
    def test_log_agent_output_disabled_no_file(self) -> None:
        """Test that logging doesn't create files when disabled."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=False)
            
            output = {"plan": {"tasks": ["task1"]}}
            state = create_initial_state("Test query")
            
            logger.log_agent_output("planner_agent", output, state)
            
            # Check that no log file was created
            log_files = list(log_dir.glob("*.log"))
            assert len(log_files) == 0
    
    def test_log_agent_output_string_output(self) -> None:
        """Test logging string output (e.g., report)."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            
            report = "# Report\n## Summary\nThis is a test report."
            state = create_initial_state("Test query")
            
            logger.log_agent_output("report_agent", report, state)
            
            log_files = list(log_dir.glob("report_agent_*.log"))
            assert len(log_files) == 1
            
            content = log_files[0].read_text(encoding="utf-8")
            assert "Report Length:" in content
            assert report in content
    
    def test_log_agent_output_none_output(self) -> None:
        """Test logging None output."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            
            state = create_initial_state("Test query")
            
            logger.log_agent_output("test_agent", None, state)
            
            log_files = list(log_dir.glob("test_agent_*.log"))
            assert len(log_files) == 1
            
            content = log_files[0].read_text(encoding="utf-8")
            assert "No output generated" in content
    
    def test_log_agent_output_handles_errors_gracefully(self) -> None:
        """Test that logging errors don't raise exceptions."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            
            # Create a read-only directory to cause permission error
            log_dir.chmod(0o444)
            
            try:
                output = {"plan": {"tasks": ["task1"]}}
                state = create_initial_state("Test query")
                
                # Should not raise exception
                logger.log_agent_output("planner_agent", output, state)
            finally:
                # Restore permissions for cleanup
                log_dir.chmod(0o755)
    
    def test_format_output_structured_data(self) -> None:
        """Test formatting of structured data output."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            
            output = {
                "competitors": [
                    {"name": "Competitor A", "url": "https://example.com/a"},
                    {"name": "Competitor B", "url": "https://example.com/b"},
                ]
            }
            state = create_initial_state("Test query")
            
            logger.log_agent_output("data_collector_agent", output, state)
            
            log_files = list(log_dir.glob("data_collector_agent_*.log"))
            assert len(log_files) == 1
            
            content = log_files[0].read_text(encoding="utf-8")
            # Should contain JSON-formatted output
            assert "Competitor A" in content
            assert "Competitor B" in content
            assert "Competitors Collected: 2" in content
    
    def test_format_output_insights_with_summary(self) -> None:
        """Test formatting of insights output with summary statistics."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            
            output = {
                "swot": {
                    "strengths": ["strength1", "strength2"],
                    "weaknesses": ["weakness1"],
                    "opportunities": ["opp1", "opp2", "opp3"],
                    "threats": ["threat1"],
                },
                "trends": ["trend1", "trend2"],
            }
            state = create_initial_state("Test query")
            
            logger.log_agent_output("insight_agent", output, state)
            
            log_files = list(log_dir.glob("insight_agent_*.log"))
            assert len(log_files) == 1
            
            content = log_files[0].read_text(encoding="utf-8")
            assert "SWOT Analysis:" in content
            assert "2 strengths" in content
            assert "Market Trends Identified: 2" in content
    
    def test_format_output_export_paths(self) -> None:
        """Test formatting of export paths output."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            
            output = {
                "pdf": "/path/to/report.pdf",
                "swot_diagram": "/path/to/swot.png",
            }
            state = create_initial_state("Test query")
            
            logger.log_agent_output("export_agent", output, state)
            
            log_files = list(log_dir.glob("export_agent_*.log"))
            assert len(log_files) == 1
            
            content = log_files[0].read_text(encoding="utf-8")
            assert "Export Files Generated: 2" in content
            assert "pdf: /path/to/report.pdf" in content
            assert "swot_diagram: /path/to/swot.png" in content
    
    def test_format_output_supervisor_decisions(self) -> None:
        """Test formatting of supervisor output."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            
            output = {
                "current_task": "Validation passed",
                "validation_errors": [],
                "retry_count": 0,
            }
            state = create_initial_state("Test query")
            state["current_task"] = "Validation passed"
            
            logger.log_agent_output("supervisor_agent", output, state)
            
            log_files = list(log_dir.glob("supervisor_agent_*.log"))
            assert len(log_files) == 1
            
            content = log_files[0].read_text(encoding="utf-8")
            assert "Validation passed" in content
    
    def test_log_file_naming_convention(self) -> None:
        """Test that log files follow naming convention."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            
            output = {"test": "data"}
            state = create_initial_state("Test query")
            
            logger.log_agent_output("planner_agent", output, state)
            
            log_files = list(log_dir.glob("planner_agent_*.log"))
            assert len(log_files) == 1
            
            filename = log_files[0].name
            assert filename.startswith("planner_agent_")
            assert filename.endswith(".log")
            # Check timestamp format: YYYYMMDD_HHMMSS
            timestamp_part = filename.replace("planner_agent_", "").replace(".log", "")
            assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS
            assert "_" in timestamp_part
    
    def test_log_file_contains_validation_errors(self) -> None:
        """Test that validation errors are included in log file."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            
            output = {"plan": {"tasks": ["task1"]}}
            state = create_initial_state("Test query")
            state["validation_errors"] = [
                "Error 1: Missing data",
                "Error 2: Invalid format",
            ]
            
            logger.log_agent_output("planner_agent", output, state)
            
            log_files = list(log_dir.glob("planner_agent_*.log"))
            assert len(log_files) == 1
            
            content = log_files[0].read_text(encoding="utf-8")
            assert "Validation Errors: 2" in content
            assert "Validation Error Details:" in content
            assert "Error 1: Missing data" in content
            assert "Error 2: Invalid format" in content
    
    def test_multiple_agents_create_separate_files(self) -> None:
        """Test that multiple agents create separate log files."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            
            state = create_initial_state("Test query")
            
            logger.log_agent_output("planner_agent", {"plan": {}}, state)
            logger.log_agent_output("insight_agent", {"insights": {}}, state)
            logger.log_agent_output("report_agent", "Report text", state)
            
            planner_files = list(log_dir.glob("planner_agent_*.log"))
            insight_files = list(log_dir.glob("insight_agent_*.log"))
            report_files = list(log_dir.glob("report_agent_*.log"))
            
            assert len(planner_files) == 1
            assert len(insight_files) == 1
            assert len(report_files) == 1
    
    def test_non_serializable_output_handled(self) -> None:
        """Test that non-serializable output is handled gracefully."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = AgentLogger(log_dir=log_dir, enabled=True)
            
            # Create output with non-serializable object
            class NonSerializable:
                def __str__(self) -> str:
                    return "NonSerializable object"
            
            output = {"data": NonSerializable()}
            state = create_initial_state("Test query")
            
            # Should not raise exception
            logger.log_agent_output("test_agent", output, state)
            
            log_files = list(log_dir.glob("test_agent_*.log"))
            assert len(log_files) == 1

