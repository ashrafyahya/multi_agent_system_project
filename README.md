# Competitor Analysis Multi-Agent System

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A robust, scalable multi-agent system using LangGraph for automated competitor analysis with validation gates, retry mechanisms, and quality assurance.

## Overview

This system leverages multiple AI agents working together to perform comprehensive competitor analysis:  

1. **Planner Agent**: Breaks down user requests into actionable tasks
2. **Supervisor Agent**: Controls workflow flow and applies business rules
3. **Data Collector Agent**: Gathers competitor data using web search and scraping
4. **Insight Agent**: Transforms raw data into business insights and SWOT analysis
5. **Report Agent**: Generates comprehensive formatted reports
6. **Export Agent**: Exports reports to PDF and generates visualizations (SWOT diagrams, charts)

The system uses LangGraph to orchestrate these agents through a stateful workflow with validation gates at each stage and automatic retry mechanisms for error recovery.

## Features

- 🤖 **Multi-Agent Architecture**: Specialized agents for each workflow stage
- 🔄 **Retry Logic**: Automatic retry with exponential backoff for transient failures
- ✅ **Validation Gates**: Quality checks at each workflow stage
- 📊 **Structured Output**: Pydantic models ensure type safety and validation
- 📄 **Professional PDF Export**: Advanced PDF generation with branding, layout customization, and professional features
  - Custom branding (logos, colors, fonts)
  - Multiple cover page templates (default, executive, minimal)
  - Branded headers and footers
  - PDF metadata and bookmarks
  - Flexible page layouts (size, orientation, margins)
- 📈 **Visualizations**: SWOT diagrams, trends charts, opportunities charts, and advanced competitor comparisons
- 🛡️ **Error Handling**: Comprehensive error handling with custom exception hierarchy
- 📝 **Comprehensive Testing**: Unit and integration tests with 80%+ coverage
- 🔧 **Type Safety**: Full type hints throughout the codebase
- 📚 **Well Documented**: Google-style docstrings with usage examples
- 📋 **Agent Output Logging**: Automatic logging of agent outputs to timestamped files for debugging and analysis

## Architecture

![](images/system_overview.png)

### Key Components

- **Agents**: Self-contained units following the Agent Pattern
- **Tools**: Stateless functions for web search, scraping, and text processing
- **Validators**: Quality gates ensuring output meets standards
- **Nodes**: Pure functions wrapping agent execution
- **Workflow**: LangGraph StateGraph with conditional edges

## Installation

### Prerequisites

- Python 3.10 or higher
- Groq API key (for LLM)
- Optional: Tavily API key (for enhanced web search)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ashrafyahya/multi_agent_system_project.git
   cd multi_agent_system_project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   LLM_MODEL=llama-3.1-8b-instant  # Primary fallback for all agents (recommended)
   
   # Optional: Override models for specific agents
   LLM_MODEL_PLANNER=llama-3.1-8b-instant
   LLM_MODEL_SUPERVISOR=llama-3.1-8b-instant
   LLM_MODEL_INSIGHT=llama-3.3-70b-versatile
   LLM_MODEL_REPORT=llama-3.3-70b-versatile
   LLM_MODEL_COLLECTOR=llama-3.1-8b-instant
   LLM_MODEL_EXPORT=llama-3.3-70b-versatile
   
   MAX_RETRIES=3
   LOG_LEVEL=INFO
   DATA_DIR=./data
   TAVILY_API_KEY=your_tavily_api_key_here 
   ```

## Configuration

Configuration is managed through a centralized configuration system using Pydantic Settings. The system automatically loads configuration from environment variables and a `.env` file in the project root.

### Configuration System

The application uses a centralized `Config` class that:
- Automatically loads from `.env` file (if present)
- Falls back to environment variables
- Provides type-safe access to configuration values
- Validates configuration on load

**Accessing Configuration:**
```python
from src.config import get_config

config = get_config()
api_key = config.groq_api_key  # Type-safe access
max_retries = config.max_retries
```

### Required Configuration

- `GROQ_API_KEY`: Your Groq API key (required)
  - Must be set in `.env` file or environment variables
  - Automatically loaded by the configuration system

### Optional Configuration

- `LLM_MODEL`: Default LLM model for all agents (default: `llama-3.1-8b-instant`, fallback)
- `LLM_MODEL_PLANNER`: Model for Planner agent (optional, defaults to `llama-3.1-8b-instant`)
- `LLM_MODEL_SUPERVISOR`: Model for Supervisor agent (optional, defaults to `llama-3.1-8b-instant`)
- `LLM_MODEL_INSIGHT`: Model for Insight agent (optional, defaults to `llama-3.3-70b-versatile`)
- `LLM_MODEL_REPORT`: Model for Report agent (optional, defaults to `llama-3.3-70b-versatile`)
- `LLM_MODEL_COLLECTOR`: Model for Data Collector agent (optional, defaults to `llama-3.1-8b-instant`)
- `LLM_MODEL_EXPORT`: Model for Export agent (optional, defaults to `llama-3.3-70b-versatile`)
- `MAX_RETRIES`: Maximum retry attempts (default: 3, range: 1-10)
- `LOG_LEVEL`: Logging level (default: `INFO`, options: DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `DATA_DIR`: Directory for temporary data (default: `./data`)
- `TAVILY_API_KEY`: Tavily API key for enhanced web search (optional)
- `AGENT_LOG_DIR`: Directory for agent output log files (default: `./data/agent_logs`)
- `AGENT_LOG_ENABLED`: Enable/disable agent output logging (default: `true`)
  - Automatically loaded from `.env` file or environment variables
  - Accessed via `config.tavily_api_key` in code

### Flexible Model Configuration

The system supports a tiered model approach where different agents can use different models based on task complexity. You can configure models per agent in your `.env` file without modifying any code:

- **Complex Analysis Agents** (Insight, Report, Export): Use larger models like `llama-3.3-70b-versatile` for better analysis quality (default)
- **Planning/Coordination Agents** (Planner, Supervisor, Collector): Use faster models like `llama-3.1-8b-instant` for speed (default)

**Fallback Priority:**
1. Agent-specific model (`LLM_MODEL_*`) if set
2. `LLM_MODEL` (fallback, default: `llama-3.1-8b-instant`)
3. Agent-specific default

This allows you to:
- Use a single model for all agents (set only `LLM_MODEL`)
- Use different models for different agents (set agent-specific `LLM_MODEL_*` variables)
- Change models anytime by updating `.env` file - no code changes needed

## Usage

### Command-Line Interface

```bash
python -m src.main "Analyze competitors in the SaaS market"
```

With verbose logging:
```bash
python -m src.main --verbose "Compare pricing strategies of top 5 competitors"
```

### PDF Export Configuration

The system supports professional PDF generation with customizable branding and layout.


**PDF Features:**
- **Cover Pages**: Professional cover pages with three template styles
- **Headers & Footers**: Branded headers and footers on all pages
- **PDF Metadata**: Document properties (title, author, keywords)
- **Custom Branding**: Company logos, colors, fonts
- **Flexible Layout**: Page size, orientation, margins, columns


## Project Structure

```
multi_agent_system/
├── src/
│   ├── main.py                 # Main entry point
│   ├── config.py               # Configuration management
│   │
│   ├── agents/                 # Agent implementations
│   │   ├── base_agent.py       # Base agent class
│   │   ├── planner_agent.py    # Plan generation
│   │   ├── supervisor_agent.py # Workflow control
│   │   ├── data_collector.py   # Data collection
│   │   ├── insight_agent.py    # Insight generation
│   │   ├── report_agent.py     # Report generation
│   │   └── export_agent.py      # PDF export and visualizations
│   │
│   ├── template/               # PDF template utilities
│   │   ├── template_engine.py  # PDF template engine
│   │   ├── pdf_generator.py    # PDF generation
│   │   ├── pdf_formatter.py    # PDF formatting utilities
│   │   ├── markdown_parser.py  # Markdown parsing
│   │   ├── markdown_converter.py # Markdown to PDF conversion
│   │   ├── cover_page.py        # Cover page generation
│   │   ├── header_footer.py     # Header and footer utilities
│   │   ├── pdf_styles.py       # PDF styling utilities
│   │   ├── pdf_utils.py        # PDF utility functions
│   │   └── style_utils.py      # Style utility functions
│   │
│   ├── graph/                  # Workflow components
│   │   ├── workflow.py         # LangGraph workflow builder
│   │   ├── state.py            # WorkflowState TypedDict
│   │   ├── nodes/              # Pure function nodes
│   │   │   ├── planner_node.py
│   │   │   ├── supervisor_node.py
│   │   │   ├── data_collector_node.py
│   │   │   ├── insight_node.py
│   │   │   ├── report_node.py
│   │   │   ├── export_node.py
│   │   │   └── retry_node.py
│   │   └── validators/         # Validation gates
│   │       ├── base_validator.py
│   │       ├── collector_validator.py
│   │       ├── data_consistency_validator.py
│   │       ├── insight_validator.py
│   │       └── report_validator.py
│   │
│   ├── tools/                   # Stateless tools
│   │   ├── base_tool.py        # Base tool class
│   │   ├── web_search.py       # Web search tool
│   │   ├── scraper.py          # Web scraping tool
│   │   ├── query_generator.py   # Query optimization
│   │   └── text_utils.py       # Text processing utilities
│   │
│   ├── models/                  # Pydantic data models
│   │   ├── plan_model.py       # Execution plan model
│   │   ├── competitor_profile.py # Competitor data model
│   │   ├── insight_model.py    # Business insights model
│   │   ├── report_model.py     # Report model
│   │   ├── pdf_branding_config.py # PDF branding configuration
│   │   └── pdf_layout_config.py   # PDF layout configuration
│   │
│   ├── utils/                   # Utility modules
│   │   └── agent_logger.py      # Agent output logging utility
│   │
│   └── exceptions/              # Custom exception hierarchy
│       ├── base.py              # Base exception class
│       ├── collector_error.py   # Data collector exceptions
│       ├── validation_error.py  # Validation exceptions
│       └── workflow_error.py    # Workflow exceptions
│
├── tests/                       # Test suite
│   ├── conftest.py             # Pytest configuration and fixtures
│   ├── test_agents.py          # Agent tests
│   ├── test_agent_logger.py    # Agent logger tests
│   ├── test_config.py          # Configuration tests
│   ├── test_exceptions.py      # Exception tests
│   ├── test_main.py           # Main entry point tests
│   ├── test_models.py          # Model tests
│   ├── test_nodes.py           # Node tests
│   ├── test_state.py           # State management tests
│   ├── test_template.py        # Template tests
│   ├── test_tools.py           # Tool tests
│   ├── test_validators.py      # Validator tests
│   ├── test_workflow.py        # Workflow tests
│   ├── fixtures/               # Test fixtures
│   │   └── sample_data.py      # Sample test data
│   └── integration/           # Integration tests
│       ├── test_full_workflow.py # Full workflow integration tests
│       └── test_pdf_export.py   # PDF export integration tests
│
├── docs/                       # Documentation
│   └── pdf_configuration_examples.md # PDF configuration guide
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project configuration
├── Makefile                    # Development commands
└── README.md                   # This file
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_agents.py -v
```

### Code Quality Checks

```bash
make lint          # Run ruff and bandit
make format        # Format with black and ruff
make type-check    # Run mypy type checking
make test-cov      # Run tests with coverage
```

### Pre-commit Hooks

Install pre-commit hooks to automatically run quality checks:
```bash
make pre-commit-install
```

## Architecture Overview

### Agent Pattern

All agents follow the Agent Pattern:
- **Self-contained**: Clear inputs/outputs
- **Stateless**: State passed in, not stored
- **Dependency Injection**: LLM and config injected via constructor
- **Communication**: Through state objects, not direct method calls

### Node Pattern

All nodes are pure functions:
- **Pure Functions**: `State -> State` with no side effects
- **Wrappers**: Wrap agent execution
- **Error Handling**: Graceful error handling

### Validator Pattern

All validators follow the Validator Pattern:
- **Composable**: Return `ValidationResult` objects
- **Non-throwing**: Don't raise exceptions for business rule violations
- **Structured**: Return errors and warnings

### Tool Pattern

All tools are stateless functions:
- **Stateless**: No internal state
- **Decorated**: Use `@tool` decorator from LangChain
- **Structured Output**: Return dictionaries with success/error information

## Workflow Flow

1. **User Query** → Initial state created
2. **Planner Agent** → Generates execution plan
   - Output logged to `planner_agent_YYYYMMDD_HHMMSS.log`
3. **Supervisor Agent** → Validates plan and routes to collector
   - Output logged to `supervisor_agent_YYYYMMDD_HHMMSS.log`
4. **Data Collector Agent** → Performs web search and scraping
   - Output logged to `data_collector_agent_YYYYMMDD_HHMMSS.log`
5. **Collector Validator** → Validates collected data quality
6. **Insight Agent** → Generates SWOT analysis and insights
   - Output logged to `insight_agent_YYYYMMDD_HHMMSS.log`
7. **Insight Validator** → Validates insight quality
8. **Report Agent** → Generates formatted report
   - Output logged to `report_agent_YYYYMMDD_HHMMSS.log`
9. **Report Validator** → Validates report completeness
10. **Export Agent** → Generates PDF and visualizations
    - Output logged to `export_agent_YYYYMMDD_HHMMSS.log`
11. **Final Report + Exports** → Returned to user

**Agent Output Logging**: Each agent's output is automatically logged to a separate timestamped file in `./data/agent_logs/` (configurable via `AGENT_LOG_DIR`). This makes it easy to track what each agent produced during workflow execution. Logging can be disabled by setting `AGENT_LOG_ENABLED=false` in your `.env` file.

If validation fails at any stage:
- Retry node modifies queries and retries (if retries available)
- Supervisor agent re-evaluates and routes accordingly
- Workflow ends if max retries exceeded

### Supervisor Agent Flow

The Supervisor Agent acts as the quality control and workflow coordinator:

![](images/supervisor_agent_flow.png)

**Supervisor Responsibilities:**
- ✅ Validates outputs from Collector, Insight, and Report agents
- ✅ Controls workflow flow and routing decisions
- ✅ Enforces business rules (minimum sources, data quality, etc.)
- ✅ Triggers retry logic when validation fails
- ✅ Manages retry count and decides when to end workflow

## Testing

The project includes comprehensive test coverage:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test complete workflow execution
- **Coverage**: 80%+ code coverage requirement

Run tests:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Troubleshooting

### Common Issues

**Issue**: `GROQ_API_KEY not found`
- **Solution**: Ensure `.env` file exists in the project root with `GROQ_API_KEY` set
- The configuration system automatically loads from `.env` file
- Verify the key is set: `cat .env | grep GROQ_API_KEY` (Linux/Mac) or `type .env | findstr GROQ_API_KEY` (Windows)

**Issue**: `ModuleNotFoundError`
- **Solution**: Install dependencies: `pip install -r requirements.txt`

**Issue**: Workflow fails with validation errors
- **Solution**: Check `validation_errors` in result. May need to adjust query or increase `MAX_RETRIES`

**Issue**: Web search returns no results
- **Solution**: Ensure `TAVILY_API_KEY` is set in `.env` file (optional but recommended) or check network connectivity
- The configuration system automatically loads the key from `.env` file or environment variables
- Verify: `config.tavily_api_key` should not be `None` when accessed via `get_config()`

**Issue**: LLM rate limit errors
- **Solution**: Reduce concurrent requests or upgrade Groq API plan

**Issue**: Agent log files not being created
- **Solution**: Check that `AGENT_LOG_ENABLED=true` in `.env` and that `AGENT_LOG_DIR` is writable
- The log directory is automatically created if it doesn't exist
- Verify the path is correct and has write permissions

### Agent Output Logging

The system automatically logs the output of each agent to separate timestamped log files. This makes it easy to track what each agent produced during workflow execution.

**Log File Location**: By default, log files are stored in `./data/agent_logs/`

**Log File Format**: Each agent creates a separate log file with the naming pattern:
- `planner_agent_YYYYMMDD_HHMMSS.log`
- `supervisor_agent_YYYYMMDD_HHMMSS.log`
- `data_collector_agent_YYYYMMDD_HHMMSS.log`
- `insight_agent_YYYYMMDD_HHMMSS.log`
- `report_agent_YYYYMMDD_HHMMSS.log`
- `export_agent_YYYYMMDD_HHMMSS.log`

**Log File Contents**: Each log file contains:
- Agent name and timestamp
- Workflow stage and retry count
- Formatted agent output (JSON for structured data, plain text for reports)
- Execution context (current task, validation errors)

**Configuration**: Control logging via environment variables:
```env
AGENT_LOG_DIR=./data/agent_logs
AGENT_LOG_ENABLED=true
```

**Disabling Logging**: Set `AGENT_LOG_ENABLED=false` to disable agent output logging.

### Debug Mode

Enable verbose logging:
```bash
python -m src.main --verbose "Your query"
```

Or set `LOG_LEVEL=DEBUG` in `.env` file.

## Version History

### Version 2.0.0 (Current)

**Major Updates:**
- ✨ **Agent Output Logging**: Added comprehensive agent output logging system with timestamped log files
- 🔄 **Enhanced Model Configuration**: Updated default models to `llama-3.3-70b-versatile` for complex analysis agents (Insight, Report, Export)
- 📊 **Improved PDF Generation**: Enhanced PDF template system with better markdown parsing, SWOT formatting, and table handling
- 🏗️ **Project Structure**: Expanded template utilities, validators, and exception handling
- 📝 **Documentation**: Comprehensive documentation updates with detailed project structure and configuration examples

**Key Features:**
- Automatic agent output logging to timestamped files (`./data/agent_logs/`)
- Tiered model configuration for optimal performance and cost
- Enhanced PDF export with improved formatting and visualizations
- Comprehensive test coverage (80%+)
- Full type safety with type hints throughout

**Configuration Changes:**
- New environment variables: `AGENT_LOG_DIR`, `AGENT_LOG_ENABLED`
- Updated default models for Insight, Report, and Export agents
- Enhanced PDF configuration options

### Version 1.0.0

**Initial Release:**
- Multi-agent architecture with LangGraph
- Planner, Supervisor, Data Collector, Insight, Report, and Export agents
- Validation gates and retry mechanisms
- Basic PDF export functionality
- Web search and scraping capabilities
- Comprehensive error handling

## Contributing

1. Ensure all tests pass: `pytest tests/ -v`
2. Run code quality checks: `make lint`
3. Maintain 80%+ test coverage
4. Follow Google-style docstrings
5. Use type hints for all functions

## License

MIT License

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [LangChain](https://github.com/langchain-ai/langchain) for LLM integration
- Powered by [Groq](https://groq.com/) for fast LLM inference

## Support

For issues, questions, or contributions, please open an issue on the repository.
