"""Base agent class for all agents in the system.

This module defines the abstract base class that all agents must implement,
following the Agent Pattern with dependency injection.

The Agent Pattern ensures that:
- Agents are self-contained units with clear inputs/outputs
- Agents communicate through state objects, not direct method calls
- Agents are stateless (state passed in, not stored)
- Dependencies (LLM, config) are injected, not created internally

Example:
    ```python
    from src.agents.base_agent import BaseAgent
    from src.graph.state import WorkflowState
    from langchain_groq import ChatGroq
    
    class MyAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "my_agent"
        
        def execute(self, state: WorkflowState) -> WorkflowState:
            # Agent logic here
            return state
    
    llm = ChatGroq(model="llama-3.1-8b-instant")
    config = {"max_retries": 3}
    agent = MyAgent(llm=llm, config=config)
    ```
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.graph.state import WorkflowState


class BaseAgent(ABC):
    """Base class for all agents in the system.
    
    This abstract base class defines the interface that all agents must
    implement. Agents follow the Agent Pattern:
    - Self-contained units with clear inputs/outputs
    - Communicate through state objects, not direct method calls
    - Stateless (state passed in, not stored)
    - Dependencies injected via constructor
    
    All concrete agent implementations must:
    1. Inherit from BaseAgent
    2. Implement the `execute` method
    3. Implement the `name` property
    4. Accept LLM and config via constructor
    
    Attributes:
        llm: Language model instance (injected dependency)
        config: Configuration dictionary (injected dependency)
    
    Example:
        ```python
        from src.agents.base_agent import BaseAgent
        from src.graph.state import WorkflowState
        from langchain_groq import ChatGroq
        
        class PlannerAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "planner_agent"
            
            def execute(self, state: WorkflowState) -> WorkflowState:
                # Generate plan from user request
                user_query = state["messages"][-1].content
                plan = self._generate_plan(user_query)
                state["plan"] = plan
                return state
        
        llm = ChatGroq(model="llama-3.1-8b-instant")
        config = {"temperature": 0}
        agent = PlannerAgent(llm=llm, config=config)
        ```
    """
    
    def __init__(self, llm: BaseChatModel, config: dict[str, Any]) -> None:
        """Initialize agent with dependencies.
        
        This constructor uses dependency injection to provide the agent
        with its required dependencies. This allows for:
        - Easy testing (can inject mocks)
        - Flexible configuration (different LLMs/configs)
        - Loose coupling (agent doesn't create dependencies)
        
        Args:
            llm: Language model instance to use for agent operations.
                Must be a BaseChatModel instance (e.g., ChatGroq, ChatOpenAI).
            config: Configuration dictionary containing agent-specific settings.
                Common keys include:
                - "temperature": float (LLM temperature, default varies by agent)
                - "max_retries": int (retry attempts for operations)
                - Agent-specific configuration keys
        
        Raises:
            TypeError: If llm is not a BaseChatModel instance
            ValueError: If config is not a dictionary
        
        Example:
            ```python
            from langchain_groq import ChatGroq
            
            llm = ChatGroq(model="llama-3.1-8b-instant")
            config = {"temperature": 0, "max_retries": 3}
            agent = PlannerAgent(llm=llm, config=config)
            ```
        """
        if not isinstance(llm, BaseChatModel):
            raise TypeError(
                f"llm must be a BaseChatModel instance, got {type(llm).__name__}"
            )
        if not isinstance(config, dict):
            raise ValueError(
                f"config must be a dictionary, got {type(config).__name__}"
            )
        
        self.llm = llm
        self.config = config
    
    @abstractmethod
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute agent logic.
        
        This method contains the core logic for the agent. It receives
        the current workflow state, performs its operations, and returns
        an updated state. Agents should be stateless - all state is passed
        in and returned, not stored in instance variables.
        
        The agent should:
        1. Extract necessary information from the state
        2. Perform its operations (e.g., generate plan, collect data, analyze)
        3. Update the state with results
        4. Return the updated state
        
        Args:
            state: Current workflow state containing:
                - messages: List of conversation messages
                - plan: Optional execution plan (for collector/insight/report agents)
                - collected_data: Optional collected competitor data (for insight/report agents)
                - insights: Optional business insights (for report agent)
                - report: Optional final report
                - retry_count: Current retry count
                - current_task: Current task being executed
                - validation_errors: List of validation errors
        
        Returns:
            Updated WorkflowState with agent's results added. The agent should
            update the appropriate field (plan, collected_data, insights, or report)
            based on its role.
        
        Raises:
            WorkflowError: If agent execution fails critically and cannot be
                recovered. Most errors should be handled gracefully and added
                to validation_errors in the state.
        
        Example:
            ```python
            def execute(self, state: WorkflowState) -> WorkflowState:
                # Extract user query
                user_message = state["messages"][-1]
                query = user_message.content
                
                # Perform agent operation
                result = self._perform_operation(query)
                
                # Update state
                state["plan"] = result
                state["current_task"] = "Planning completed"
                
                return state
            ```
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return agent name.
        
        This property provides a unique identifier for the agent. It's used
        for logging, error messages, and identification purposes. Each agent
        type should have a distinct name.
        
        Returns:
            String identifier for this agent. Should be lowercase with underscores,
            e.g., "planner_agent", "data_collector_agent", "insight_agent"
        
        Example:
            ```python
            @property
            def name(self) -> str:
                return "planner_agent"
            ```
        """
        pass
