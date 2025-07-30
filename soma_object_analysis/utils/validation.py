"""
Validation utilities for SOMA object analysis
"""

from typing import List, Dict, Any, Tuple
from ..models import ObjectDescription
from ..config import config


def validate_object_description(obj_desc: ObjectDescription) -> Tuple[bool, List[str]]:
    """
    Quick validation of ObjectDescription instance

    Args:
        obj_desc: ObjectDescription to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Basic validation
    if not obj_desc.name or not obj_desc.name.strip():
        errors.append("Object name is required")

    if not obj_desc.description or not obj_desc.description.strip():
        errors.append("Object description is required")

    if obj_desc.confidence_score < 0 or obj_desc.confidence_score > 1:
        errors.append(f"Confidence score must be between 0 and 1, got {obj_desc.confidence_score}")

    # Visual validation
    if not obj_desc.visual.colors.primary_color:
        errors.append("Primary color is required")

    # Geometric validation
    dims = obj_desc.geometric.shape.dimensions
    if not any([dims.width, dims.height, dims.depth, dims.radius, dims.length]):
        errors.append("At least one dimension must be specified")

    # Material validation
    if not obj_desc.material.primary_material:
        errors.append("Primary material is required")

    return len(errors) == 0, errors


def validate_confidence_threshold(confidence: float, threshold: float = None) -> bool:
    """
    Check if confidence meets threshold

    Args:
        confidence: Confidence score to check
        threshold: Minimum threshold (uses config default if None)

    Returns:
        True if confidence meets threshold
    """
    threshold = threshold or config.DEFAULT_CONFIDENCE_THRESHOLD
    return confidence >= threshold


def validate_dimension_ranges(dimensions: Dict[str, float]) -> List[str]:
    """
    Validate dimension values are reasonable

    Args:
        dimensions: Dictionary of dimension name -> value

    Returns:
        List of validation errors
    """
    errors = []

    for dim_name, value in dimensions.items():
        if value is not None:
            if value <= 0:
                errors.append(f"{dim_name} must be positive, got {value}")
            elif value > 100:  # 100 meters seems unreasonable for most objects
                errors.append(f"{dim_name} {value}m seems unreasonably large")
            elif value < 0.001:  # 1mm seems like minimum reasonable size
                errors.append(f"{dim_name} {value}m seems unreasonably small")

    return errors


def validate_mass_density_consistency(mass: float, volume: float, material: str) -> List[str]:
    """
    Check if mass and volume are consistent with material

    Args:
        mass: Object mass in kg
        volume: Object volume in m³
        material: Material type

    Returns:
        List of validation errors
    """
    errors = []

    if mass <= 0 or volume <= 0:
        return errors  # Can't validate if invalid inputs

    density = mass / volume

    # Typical density ranges (kg/m³)
    density_ranges = {
        "metal": (1000, 20000),
        "plastic": (800, 2000),
        "wood": (300, 1200),
        "glass": (2200, 2800),
        "ceramic": (2000, 6000),
        "fabric": (200, 1500),
        "rubber": (900, 2000),
        "organic": (200, 1500)
    }

    if material.lower() in density_ranges:
        min_density, max_density = density_ranges[material.lower()]
        if density < min_density * 0.5:  # Allow 50% variance
            errors.append(f"{material} object unusually light: density {density:.0f} kg/m³")
        elif density > max_density * 2:  # Allow 100% variance
            errors.append(f"{material} object unusually heavy: density {density:.0f} kg/m³")

    return errors