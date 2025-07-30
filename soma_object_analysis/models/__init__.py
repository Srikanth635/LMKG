"""
Pydantic models for SOMA-aligned object descriptions
"""

from .base import ObjectCategory, ShapeType, MaterialType, TextureType, DeviceState, CleanlinessState
from .visual import RGBColor, HSVColor, ColorDescription, SurfaceProperties, VisualAppearance
from .geometric import Position3D, Pose6D, Dimensions, GeometricShape, SpatialRelations, GeometricDescription
from .material import MaterialProperties, MechanicalProperties
from .capabilities import BasicCapability, FunctionalAffordances, TaskAffordances, CapabilityDescription
from .semantic import RoleDescription, ContextualInfo, SemanticDescription
from .object_description import ObjectState, ObjectDescription

__all__ = [
    # Base enums
    "ObjectCategory",
    "ShapeType",
    "MaterialType",
    "TextureType",
    "DeviceState",
    "CleanlinessState",

    # Visual models
    "RGBColor",
    "HSVColor",
    "ColorDescription",
    "SurfaceProperties",
    "VisualAppearance",

    # Geometric models
    "Position3D",
    "Pose6D",
    "Dimensions",
    "GeometricShape",
    "SpatialRelations",
    "GeometricDescription",

    # Material models
    "MaterialProperties",
    "MechanicalProperties",

    # Capability models
    "BasicCapability",
    "FunctionalAffordances",
    "TaskAffordances",
    "CapabilityDescription",

    # Semantic models
    "RoleDescription",
    "ContextualInfo",
    "SemanticDescription",

    # Main model
    "ObjectState",
    "ObjectDescription"
]