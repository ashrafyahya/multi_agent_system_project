"""Tests for main entry point.

This module contains unit tests for the main entry point to verify
configuration loading, LLM initialization, workflow creation, and execution.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from langchain_groq import ChatGroq

from src.config import Config
from src.graph.state import WorkflowState
from src.main import initialize_llm, run_analysis, main


class TestInitializeLLM:
    """Tests for initialize_llm function."""
    
    def test_initialize_llm_success(self) -> None:
        """Test LLM initialization succeeds."""
        config = Mock(spec=Config)
        config.groq_api_key = "test_api_key"
        config.groq_model = "llama-3.1-8b-instant"
        
        with patch("src.main.ChatGroq") as mock_chatgroq:
            mock_llm = Mock(spec=ChatGroq)
            mock_chatgroq.return_value = mock_llm
            
            llm = initialize_llm(config)
            
            assert llm == mock_llm
            mock_chatgroq.assert_called_once_with(
                api_key="test_api_key",
                model="llama-3.1-8b-instant",
                temperature=0,
            )
    
    def test_initialize_llm_uses_config_values(self) -> None:
        """Test LLM initialization uses config values."""
        config = Mock(spec=Config)
        config.groq_api_key = "custom_api_key"
        config.groq_model = "custom-model"
        
        with patch("src.main.ChatGroq") as mock_chatgroq:
            mock_llm = Mock(spec=ChatGroq)
            mock_chatgroq.return_value = mock_llm
            
            initialize_llm(config)
            
            mock_chatgroq.assert_called_once_with(
                api_key="custom_api_key",
                model="custom-model",
                temperature=0,
            )


class TestRunAnalysis:
    """Tests for run_analysis function."""
    
    def test_run_analysis_success(self) -> None:
        """Test run_analysis executes successfully."""
        user_query = "Analyze competitors"
        
        mock_config = Mock(spec=Config)
        mock_config.max_retries = 3
        
        mock_llm = Mock(spec=ChatGroq)
        
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = {
            "messages": [],
            "plan": {"tasks": ["Find competitors"]},
            "collected_data": {"competitors": []},
            "insights": {"swot": {}},
            "report": "Final report",
            "retry_count": 0,
            "current_task": "Complete",
            "validation_errors": [],
        }
        
        with patch("src.main.get_config", return_value=mock_config):
            with patch("src.main.initialize_llm", return_value=mock_llm):
                with patch("src.main.create_workflow", return_value=mock_workflow):
                    with patch("src.main.create_initial_state") as mock_create_state:
                        mock_create_state.return_value = {
                            "messages": [],
                            "plan": None,
                            "collected_data": None,
                            "insights": None,
                            "report": None,
                            "retry_count": 0,
                            "current_task": None,
                            "validation_errors": [],
                        }
                        
                        result = run_analysis(user_query, config=mock_config, llm=mock_llm)
                        
                        assert result["report"] == "Final report"
                        mock_workflow.invoke.assert_called_once()
    
    def test_run_analysis_loads_config_if_not_provided(self) -> None:
        """Test run_analysis loads config if not provided."""
        user_query = "Analyze competitors"
        
        mock_config = Mock(spec=Config)
        mock_config.max_retries = 3
        
        mock_llm = Mock(spec=ChatGroq)
        
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = {
            "report": "Final report",
            "validation_errors": [],
        }
        
        with patch("src.main.get_config", return_value=mock_config) as mock_get_config:
            with patch("src.main.initialize_llm", return_value=mock_llm):
                with patch("src.main.create_workflow", return_value=mock_workflow):
                    with patch("src.main.create_initial_state"):
                        run_analysis(user_query)
                        
                        mock_get_config.assert_called_once()
    
    def test_run_analysis_initializes_llm_if_not_provided(self) -> None:
        """Test run_analysis initializes LLM if not provided."""
        user_query = "Analyze competitors"
        
        mock_config = Mock(spec=Config)
        mock_config.max_retries = 3
        
        mock_llm = Mock(spec=ChatGroq)
        
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = {"report": "Final report"}
        
        with patch("src.main.get_config", return_value=mock_config):
            with patch("src.main.initialize_llm", return_value=mock_llm) as mock_init_llm:
                with patch("src.main.create_workflow", return_value=mock_workflow):
                    with patch("src.main.create_initial_state"):
                        run_analysis(user_query, config=mock_config)
                        
                        mock_init_llm.assert_called_once_with(mock_config)
    
    def test_run_analysis_raises_error_on_empty_query(self) -> None:
        """Test run_analysis raises error on empty query."""
        with pytest.raises(ValueError, match="User query cannot be empty"):
            run_analysis("")
        
        with pytest.raises(ValueError, match="User query cannot be empty"):
            run_analysis("   ")
    
    def test_run_analysis_handles_config_loading_error(self) -> None:
        """Test run_analysis handles config loading error."""
        user_query = "Analyze competitors"
        
        with patch("src.main.get_config", side_effect=Exception("Config error")):
            with pytest.raises(RuntimeError, match="Configuration loading failed"):
                run_analysis(user_query)
    
    def test_run_analysis_handles_llm_initialization_error(self) -> None:
        """Test run_analysis handles LLM initialization error."""
        user_query = "Analyze competitors"
        
        mock_config = Mock(spec=Config)
        mock_config.max_retries = 3
        
        with patch("src.main.get_config", return_value=mock_config):
            with patch("src.main.initialize_llm", side_effect=Exception("LLM error")):
                with pytest.raises(RuntimeError, match="LLM initialization failed"):
                    run_analysis(user_query)
    
    def test_run_analysis_handles_workflow_creation_error(self) -> None:
        """Test run_analysis handles workflow creation error."""
        user_query = "Analyze competitors"
        
        mock_config = Mock(spec=Config)
        mock_config.max_retries = 3
        
        mock_llm = Mock(spec=ChatGroq)
        
        with patch("src.main.get_config", return_value=mock_config):
            with patch("src.main.initialize_llm", return_value=mock_llm):
                with patch("src.main.create_workflow", side_effect=Exception("Workflow error")):
                    with pytest.raises(RuntimeError, match="Workflow creation failed"):
                        run_analysis(user_query, config=mock_config, llm=mock_llm)
    
    def test_run_analysis_handles_workflow_execution_error(self) -> None:
        """Test run_analysis handles workflow execution error."""
        user_query = "Analyze competitors"
        
        mock_config = Mock(spec=Config)
        mock_config.max_retries = 3
        
        mock_llm = Mock(spec=ChatGroq)
        
        mock_workflow = MagicMock()
        mock_workflow.invoke.side_effect = Exception("Execution error")
        
        with patch("src.main.get_config", return_value=mock_config):
            with patch("src.main.initialize_llm", return_value=mock_llm):
                with patch("src.main.create_workflow", return_value=mock_workflow):
                    with patch("src.main.create_initial_state"):
                        with pytest.raises(RuntimeError, match="Workflow execution failed"):
                            run_analysis(user_query, config=mock_config, llm=mock_llm)
    
    def test_run_analysis_uses_provided_config_and_llm(self) -> None:
        """Test run_analysis uses provided config and LLM."""
        user_query = "Analyze competitors"
        
        mock_config = Mock(spec=Config)
        mock_config.max_retries = 3
        
        mock_llm = Mock(spec=ChatGroq)
        
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = {"report": "Final report"}
        
        with patch("src.main.get_config") as mock_get_config:
            with patch("src.main.initialize_llm") as mock_init_llm:
                with patch("src.main.create_workflow", return_value=mock_workflow):
                    with patch("src.main.create_initial_state"):
                        run_analysis(user_query, config=mock_config, llm=mock_llm)
                        
                        # Should not call get_config or initialize_llm
                        mock_get_config.assert_not_called()
                        mock_init_llm.assert_not_called()


class TestMain:
    """Tests for main function."""
    
    def test_main_success(self) -> None:
        """Test main function executes successfully."""
        with patch("sys.argv", ["main.py", "Analyze competitors"]):
            with patch("src.main.run_analysis") as mock_run:
                mock_run.return_value = {
                    "report": "Final report",
                    "validation_errors": [],
                }
                with patch("builtins.print"):
                    exit_code = main()
                    
                    assert exit_code == 0
                    mock_run.assert_called_once_with("Analyze competitors")
    
    def test_main_no_report(self) -> None:
        """Test main function handles missing report."""
        with patch("sys.argv", ["main.py", "Analyze competitors"]):
            with patch("src.main.run_analysis") as mock_run:
                mock_run.return_value = {
                    "report": None,
                    "validation_errors": ["Error 1", "Error 2"],
                }
                with patch("builtins.print"):
                    exit_code = main()
                    
                    assert exit_code == 1
    
    def test_main_handles_value_error(self) -> None:
        """Test main function handles ValueError."""
        with patch("sys.argv", ["main.py", "Analyze competitors"]):
            with patch("src.main.run_analysis", side_effect=ValueError("Invalid input")):
                with patch("builtins.print"):
                    exit_code = main()
                    
                    assert exit_code == 1
    
    def test_main_handles_runtime_error(self) -> None:
        """Test main function handles RuntimeError."""
        with patch("sys.argv", ["main.py", "Analyze competitors"]):
            with patch("src.main.run_analysis", side_effect=RuntimeError("Runtime error")):
                with patch("builtins.print"):
                    exit_code = main()
                    
                    assert exit_code == 1
    
    def test_main_handles_keyboard_interrupt(self) -> None:
        """Test main function handles KeyboardInterrupt."""
        with patch("sys.argv", ["main.py", "Analyze competitors"]):
            with patch("src.main.run_analysis", side_effect=KeyboardInterrupt()):
                with patch("builtins.print"):
                    exit_code = main()
                    
                    assert exit_code == 130
    
    def test_main_handles_unexpected_error(self) -> None:
        """Test main function handles unexpected errors."""
        with patch("sys.argv", ["main.py", "Analyze competitors"]):
            with patch("src.main.run_analysis", side_effect=Exception("Unexpected error")):
                with patch("builtins.print"):
                    exit_code = main()
                    
                    assert exit_code == 1
    
    def test_main_verbose_flag(self) -> None:
        """Test main function handles verbose flag."""
        with patch("sys.argv", ["main.py", "--verbose", "Analyze competitors"]):
            with patch("src.main.run_analysis") as mock_run:
                mock_run.return_value = {"report": "Final report"}
                with patch("builtins.print"):
                    with patch("logging.getLogger") as mock_get_logger:
                        mock_logger = Mock()
                        mock_get_logger.return_value = mock_logger
                        
                        main()
                        
                        mock_logger.setLevel.assert_called_once_with(10)  # DEBUG level


