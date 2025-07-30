"""
SOMA Object Analysis System
===========================

A comprehensive system for analyzing objects and generating SOMA-aligned knowledge graphs
using LLMs and structured output.

Main Components:
- ObjectDescription: Pydantic model for structured object analysis
- SOMAObjectAnalysisSystem: Main system orchestrator
- SOMAKnowledgeGraph: Knowledge graph generator
- CLI tools: Command-line interface

Basic Usage:
    >>> from soma_object_analysis import SOMAObjectAnalysisSystem
    >>> system = SOMAObjectAnalysisSystem()
    >>> result = system.analyze_object_sync("red apple on table")
    >>> print(result["object_analysis"]["name"])
    Red Apple

Quick Analysis:
    >>> from soma_object_analysis import analyze_object_sync
    >>> result = analyze_object_sync("kitchen knife", use_mock=True)

Author: AI Assistant
Version: 0.1.0
License: MIT
"""

__version__ = "0.1.0"
__author__ = "AI Assistant"
__license__ = "MIT"

# Core system imports
from .system import (
    SOMAObjectAnalysisSystem,
    create_analysis_system,
    analyze_object,
    analyze_object_sync
)

# Model imports
from .models import (
    ObjectDescription,
    ObjectCategory,
    ShapeType,
    MaterialType,
    ColorDescription,
    GeometricDescription,
    MaterialProperties,
    CapabilityDescription,
    SemanticDescription,
    ObjectState
)

# Knowledge graph imports
from .knowledge_graph import (
    SOMAKnowledgeGraph,
    KnowledgeGraphGenerator
)

# Agent imports
from .agents import (
    LangGraphAnalysisAgent,
    MockAnalysisAgent
)

# Configuration
from .config import config

# Convenience imports for notebook usage
try:
    from .utils.notebook_helpers import NotebookHelpers
    _NOTEBOOK_AVAILABLE = True
except ImportError:
    _NOTEBOOK_AVAILABLE = False

# Version check and warnings
def _check_dependencies():
    """Check for optional dependencies and show warnings"""
    missing_deps = []

    try:
        import langgraph
    except ImportError:
        missing_deps.append("langgraph")

    try:
        import langchain_openai
    except ImportError:
        missing_deps.append("langchain-openai")

    if missing_deps:
        import warnings
        warnings.warn(
            f"Optional dependencies missing: {', '.join(missing_deps)}. "
            "Some features will be limited to mock mode. "
            "Install with: pip install langgraph langchain-openai",
            ImportWarning
        )

# Check dependencies on import
_check_dependencies()

# Public API
__all__ = [
    # Core system
    "SOMAObjectAnalysisSystem",
    "create_analysis_system",
    "analyze_object",
    "analyze_object_sync",

    # Models
    "ObjectDescription",
    "ObjectCategory",
    "ShapeType",
    "MaterialType",
    "ColorDescription",
    "GeometricDescription",
    "MaterialProperties",
    "CapabilityDescription",
    "SemanticDescription",
    "ObjectState",

    # Knowledge graph
    "SOMAKnowledgeGraph",
    "KnowledgeGraphGenerator",

    # Agents
    "LangGraphAnalysisAgent",
    "MockAnalysisAgent",

    # Configuration
    "config",
]

# Add notebook helpers if available
if _NOTEBOOK_AVAILABLE:
    __all__.append("NotebookHelpers")

# Module metadata
def get_version():
    """Get the current version"""
    return __version__

def get_system_info():
    """Get system information"""
    system = create_analysis_system()
    return system.get_system_info()

def validate_installation():
    """Validate the installation and dependencies"""
    system = create_analysis_system()
    return system.validate_setup()

# Convenience function for quick testing
def quick_test():
    """Quick test function to verify installation"""
    print(f"🚀 SOMA Object Analysis System v{__version__}")
    print("=" * 50)

    # Test system creation
    try:
        system = create_analysis_system(use_mock=True)
        print("✅ System initialization: OK")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False

    # Test validation
    validation = system.validate_setup()
    if validation["valid"]:
        print("✅ System validation: OK")
    else:
        print("⚠️  System validation: Issues found")
        for issue in validation["issues"]:
            print(f"   • {issue}")

    # Test mock analysis
    try:
        result = system.analyze_object_sync("test object", export_results=False)
        if result["success"]:
            print("✅ Mock analysis: OK")
            print(f"   Analyzed: {result['object_analysis']['name']}")
        else:
            print(f"❌ Mock analysis failed: {result['error']}")
            return False
    except Exception as e:
        print(f"❌ Mock analysis error: {e}")
        return False

    print("\n🎉 Installation test completed successfully!")
    print("\n📚 Next steps:")
    print("   • Set OPENAI_API_KEY for full LLM features")
    print("   • Try: soma-analyze 'red apple on table'")
    print("   • Check examples/ directory for usage examples")

    return True

# Package initialization message
def _show_welcome_message():
    """Show welcome message on first import"""
    import os
    if os.getenv("SOMA_QUIET") != "1":
        print(f"📊 SOMA Object Analysis System v{__version__} loaded")
        if not config.OPENAI_API_KEY:
            print("⚠️  Set OPENAI_API_KEY for full LLM features")

# Show welcome message unless suppressed
_show_welcome_message()