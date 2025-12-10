# Competitor Analysis Multi-Agent System

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
- 📄 **PDF Export**: Automatic PDF generation with proper markdown formatting
- 📈 **Visualizations**: SWOT diagrams, trends charts, and opportunities charts
- 🛡️ **Error Handling**: Comprehensive error handling with custom exception hierarchy
- 📝 **Comprehensive Testing**: Unit and integration tests with 80%+ coverage
- 🔧 **Type Safety**: Full type hints throughout the codebase
- 📚 **Well Documented**: Google-style docstrings with usage examples

## Architecture

![](diagrams/system_overview.png)

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
   GROQ_MODEL=llama-3.1-8b-instant
   MAX_RETRIES=3
   LOG_LEVEL=INFO
   DATA_DIR=./data
   TAVILY_API_KEY=your_tavily_api_key_here 
   ```

## Configuration

Configuration is managed through environment variables. Create a `.env` file with the required variables.

### Required Configuration

- `GROQ_API_KEY`: Your Groq API key (required)
- `GROQ_MODEL`: Groq model to use (default: `llama-3.1-8b-instant`)

### Optional Configuration

- `MAX_RETRIES`: Maximum retry attempts (default: 3, range: 1-10)
- `LOG_LEVEL`: Logging level (default: `INFO`, options: DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `DATA_DIR`: Directory for temporary data (default: `./data`)
- `TAVILY_API_KEY`: Tavily API key for enhanced web search (optional)

## Usage

### Command-Line Interface

```bash
python -m src.main "Analyze competitors in the SaaS market"
```

With verbose logging:
```bash
python -m src.main --verbose "Compare pricing strategies of top 5 competitors"
```

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
│   │
│   ├── tools/                   # Stateless tools
│   │   ├── web_search.py       # Web search tool
│   │   ├── scraper.py          # Web scraping tool
│   │   ├── query_generator.py   # Query optimization
│   │   └── text_utils.py       # Text processing utilities
│   │
│   ├── models/                  # Pydantic data models
│   │   ├── plan_model.py       # Execution plan model
│   │   ├── competitor_profile.py # Competitor data model
│   │   ├── insight_model.py    # Business insights model
│   │   └── report_model.py     # Report model
│   │
│   └── exceptions/              # Custom exception hierarchy
│
├── tests/                       # Test suite
│   ├── test_agents.py          # Agent tests
│   ├── test_validators.py      # Validator tests
│   ├── test_tools.py           # Tool tests
│   ├── test_nodes.py           # Node tests
│   ├── test_workflow.py        # Workflow tests
│   ├── test_main.py            # Main entry point tests
│   └── integration/            # Integration tests
│
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
3. **Supervisor Agent** → Validates plan and routes to collector
4. **Data Collector Agent** → Performs web search and scraping
5. **Collector Validator** → Validates collected data quality
6. **Insight Agent** → Generates SWOT analysis and insights
7. **Insight Validator** → Validates insight quality
8. **Report Agent** → Generates formatted report
9. **Report Validator** → Validates report completeness
10. **Export Agent** → Generates PDF and visualizations
11. **Final Report + Exports** → Returned to user

If validation fails at any stage:
- Retry node modifies queries and retries (if retries available)
- Supervisor agent re-evaluates and routes accordingly
- Workflow ends if max retries exceeded

### Supervisor Agent Flow

The Supervisor Agent acts as the quality control and workflow coordinator:

![](diagrams/supervisor_agent_flow.png)

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
- **Solution**: Ensure `.env` file exists with `GROQ_API_KEY` set

**Issue**: `ModuleNotFoundError`
- **Solution**: Install dependencies: `pip install -r requirements.txt`

**Issue**: Workflow fails with validation errors
- **Solution**: Check `validation_errors` in result. May need to adjust query or increase `MAX_RETRIES`

**Issue**: Web search returns no results
- **Solution**: Ensure Tavily API key is set (optional but recommended) or check network connectivity

**Issue**: LLM rate limit errors
- **Solution**: Reduce concurrent requests or upgrade Groq API plan

### Debug Mode

Enable verbose logging:
```bash
python -m src.main --verbose "Your query"
```

Or set `LOG_LEVEL=DEBUG` in `.env` file.

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
