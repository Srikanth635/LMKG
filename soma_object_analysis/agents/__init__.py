"""
Agent modules for SOMA object analysis
"""

from .base_agent import BaseAgent, AnalysisAgent, ValidationAgent, AgentState

# Import agents based on availability
try:
    from .analysis_agent import LangGraphAnalysisAgent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    LangGraphAnalysisAgent = None

from .mock_agent import MockAnalysisAgent
from .validation_agent import SOMAValidationAgent

__all__ = [
    # Base classes
    "BaseAgent",
    "AnalysisAgent",
    "ValidationAgent",
    "AgentState",

    # Concrete agents
    "MockAnalysisAgent",
    "SOMAValidationAgent",
]

# Add LangGraphAnalysisAgent if available
if LANGGRAPH_AVAILABLE:
    __all__.append("LangGraphAnalysisAgent")

# Convenience factory functions
def create_analysis_agent(use_mock: bool = False, **kwargs):
    """
    Factory function to create appropriate analysis agent

    Args:
        use_mock: Whether to use mock agent instead of LLM agent
        **kwargs: Additional arguments for agent initialization

    Returns:
        Analysis agent instance
    """
    if use_mock or not LANGGRAPH_AVAILABLE:
        return MockAnalysisAgent(**kwargs)
    else:
        return LangGraphAnalysisAgent(**kwargs)

def create_validation_agent(**kwargs):
    """
    Factory function to create validation agent

    Args:
        **kwargs: Additional arguments for agent initialization

    Returns:
        Validation agent instance
    """
    return SOMAValidationAgent(**kwargs)

# Module metadata
def get_available_agents():
    """Get list of available agent types"""
    agents = ["MockAnalysisAgent", "SOMAValidationAgent"]
    if LANGGRAPH_AVAILABLE:
        agents.append("LangGraphAnalysisAgent")
    return agents

def check_agent_dependencies():
    """Check agent dependencies and return status"""
    return {
        "mock_agent": True,
        "validation_agent": True,
        "langgraph_agent": LANGGRAPH_AVAILABLE,
        "missing_dependencies": [] if LANGGRAPH_AVAILABLE else ["langgraph", "langchain-openai"]
    }