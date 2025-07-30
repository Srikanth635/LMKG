"""
Helper utilities for SOMA object analysis
"""

import re
from pathlib import Path
from typing import Optional, Union
from ..models import Dimensions


def clean_filename(filename: str, max_length: int = 100) -> str:
    """
    Clean a filename to be filesystem-safe

    Args:
        filename: Original filename
        max_length: Maximum length for filename

    Returns:
        Cleaned filename
    """
    # Remove or replace invalid characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Replace multiple spaces/underscores with single underscore
    cleaned = re.sub(r'[\s_]+', '_', cleaned)

    # Remove leading/trailing underscores and whitespace
    cleaned = cleaned.strip('_ ')

    # Truncate if too long
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip('_')

    # Ensure it's not empty
    if not cleaned:
        cleaned = "unnamed"

    return cleaned


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def format_confidence_score(confidence: float, style: str = "percentage") -> str:
    """
    Format confidence score for display

    Args:
        confidence: Confidence value (0-1)
        style: Format style ("percentage", "decimal", "emoji")

    Returns:
        Formatted confidence string
    """
    if style == "percentage":
        return f"{confidence:.1%}"
    elif style == "decimal":
        return f"{confidence:.3f}"
    elif style == "emoji":
        if confidence >= 0.9:
            return f"{confidence:.1%} 🟢"
        elif confidence >= 0.7:
            return f"{confidence:.1%} 🟡"
        else:
            return f"{confidence:.1%} 🔴"
    else:
        return f"{confidence:.2f}"


def format_dimensions(dimensions: Dimensions, unit: str = "m", precision: int = 3) -> str:
    """
    Format dimensions for display

    Args:
        dimensions: Dimensions object
        unit: Unit suffix to add
        precision: Decimal precision

    Returns:
        Formatted dimensions string
    """
    parts = []

    if dimensions.width is not None:
        parts.append(f"W: {dimensions.width:.{precision}f}{unit}")

    if dimensions.height is not None:
        parts.append(f"H: {dimensions.height:.{precision}f}{unit}")

    if dimensions.depth is not None:
        parts.append(f"D: {dimensions.depth:.{precision}f}{unit}")

    if dimensions.length is not None:
        parts.append(f"L: {dimensions.length:.{precision}f}{unit}")

    if dimensions.radius is not None:
        parts.append(f"R: {dimensions.radius:.{precision}f}{unit}")

    if not parts:
        return "No dimensions specified"

    return " × ".join(parts)


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate string to maximum length

    Args:
        text: String to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that handles zero denominator

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero

    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted file size string
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]

    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return f"{s} {size_names[i]}"


def generate_safe_id(name: str, prefix: str = "", max_length: int = 50) -> str:
    """
    Generate a safe identifier from a name

    Args:
        name: Original name
        prefix: Optional prefix
        max_length: Maximum length

    Returns:
        Safe identifier string
    """
    # Convert to lowercase and replace spaces/special chars with underscores
    safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())

    # Remove multiple underscores
    safe_id = re.sub(r'_+', '_', safe_id)

    # Remove leading/trailing underscores
    safe_id = safe_id.strip('_')

    # Add prefix if provided
    if prefix:
        safe_id = f"{prefix}_{safe_id}"

    # Truncate if necessary
    if len(safe_id) > max_length:
        safe_id = safe_id[:max_length].rstrip('_')

    # Ensure it starts with letter or underscore (valid identifier)
    if safe_id and safe_id[0].isdigit():
        safe_id = f"_{safe_id}"

    return safe_id or "unnamed"