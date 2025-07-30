"""
Knowledge graph modules for SOMA object analysis
"""

from .base_kg import BaseKnowledgeGraph, QueryableKnowledgeGraph, ExportableKnowledgeGraph, Triple
from .soma_kg import SOMAKnowledgeGraph
from .generator import KnowledgeGraphGenerator

# Import visualizer if dependencies available
try:
    from .visualizer import KnowledgeGraphVisualizer

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    KnowledgeGraphVisualizer = None

__all__ = [
    # Core classes
    "Triple",
    "BaseKnowledgeGraph",
    "QueryableKnowledgeGraph",
    "ExportableKnowledgeGraph",
    "SOMAKnowledgeGraph",
    "KnowledgeGraphGenerator",
]

# Add visualizer if available
if VISUALIZATION_AVAILABLE:
    __all__.append("KnowledgeGraphVisualizer")


# Factory functions
def create_soma_knowledge_graph():
    """Create a new SOMA knowledge graph instance"""
    return SOMAKnowledgeGraph()


def create_knowledge_graph_generator():
    """Create a knowledge graph generator instance"""
    return KnowledgeGraphGenerator()


def create_visualizer(knowledge_graph):
    """
    Create a knowledge graph visualizer

    Args:
        knowledge_graph: SOMAKnowledgeGraph instance

    Returns:
        KnowledgeGraphVisualizer instance

    Raises:
        ImportError: If visualization dependencies not available
    """
    if not VISUALIZATION_AVAILABLE:
        raise ImportError(
            "Visualization dependencies not available. "
            "Install with: pip install matplotlib networkx"
        )

    return KnowledgeGraphVisualizer(knowledge_graph)


# Utility functions
def validate_knowledge_graph(kg):
    """
    Validate a SOMA knowledge graph

    Args:
        kg: SOMAKnowledgeGraph instance

    Returns:
        List of validation errors (empty if valid)
    """
    if hasattr(kg, 'validate_soma_consistency'):
        return kg.validate_soma_consistency()
    else:
        return ["Knowledge graph does not support SOMA validation"]


def get_knowledge_graph_statistics(kg):
    """
    Get statistics about a knowledge graph

    Args:
        kg: Knowledge graph instance

    Returns:
        Dictionary of statistics
    """
    if hasattr(kg, 'get_soma_statistics'):
        return kg.get_soma_statistics()
    elif hasattr(kg, 'get_statistics'):
        return kg.get_statistics()
    else:
        return {"error": "Knowledge graph does not support statistics"}


# Module metadata
def get_capabilities():
    """Get module capabilities"""
    return {
        "base_kg": True,
        "soma_kg": True,
        "generator": True,
        "visualizer": VISUALIZATION_AVAILABLE,
        "missing_dependencies": [] if VISUALIZATION_AVAILABLE else ["matplotlib", "networkx"]
    }


def check_dependencies():
    """Check module dependencies"""
    missing = []

    try:
        import networkx
    except ImportError:
        missing.append("networkx")

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    return {
        "all_available": len(missing) == 0,
        "missing": missing,
        "visualization_available": VISUALIZATION_AVAILABLE
    }