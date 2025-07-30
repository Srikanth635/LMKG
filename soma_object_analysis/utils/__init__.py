"""
Utility modules for SOMA object analysis
"""

from .logging import setup_logging, get_logger, create_formatter

# Import additional utilities if available
try:
    from .validation import validate_object_description

    VALIDATION_UTILS_AVAILABLE = True
except ImportError:
    VALIDATION_UTILS_AVAILABLE = False
    validate_object_description = None

try:
    from .helpers import (
        clean_filename,
        ensure_directory,
        format_confidence_score,
        format_dimensions
    )

    HELPER_UTILS_AVAILABLE = True
except ImportError:
    HELPER_UTILS_AVAILABLE = False

# Import notebook helpers if available
try:
    from .notebook_helpers import NotebookHelpers

    NOTEBOOK_AVAILABLE = True
except ImportError:
    NOTEBOOK_AVAILABLE = False
    NotebookHelpers = None

__all__ = [
    # Logging utilities
    "setup_logging",
    "get_logger",
    "create_formatter",
]

# Add optional utilities if available
if VALIDATION_UTILS_AVAILABLE:
    __all__.append("validate_object_description")

if HELPER_UTILS_AVAILABLE:
    __all__.extend([
        "clean_filename",
        "ensure_directory",
        "format_confidence_score",
        "format_dimensions"
    ])

if NOTEBOOK_AVAILABLE:
    __all__.append("NotebookHelpers")


# Convenience functions
def get_available_utilities():
    """Get list of available utility modules"""
    available = ["logging"]

    if VALIDATION_UTILS_AVAILABLE:
        available.append("validation")

    if HELPER_UTILS_AVAILABLE:
        available.append("helpers")

    if NOTEBOOK_AVAILABLE:
        available.append("notebook_helpers")

    return available


def check_utility_dependencies():
    """Check utility module dependencies"""
    status = {
        "logging": True,
        "validation": VALIDATION_UTILS_AVAILABLE,
        "helpers": HELPER_UTILS_AVAILABLE,
        "notebook": NOTEBOOK_AVAILABLE
    }

    missing = []
    if not NOTEBOOK_AVAILABLE:
        missing.extend(["ipython", "jupyter", "pandas"])

    status["missing_dependencies"] = missing
    status["all_available"] = len(missing) == 0

    return status