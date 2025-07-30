"""
Base enums and types for SOMA object analysis
"""

from enum import Enum

class ObjectCategory(str, Enum):
    """SOMA object categories"""
    TOOL = "Tool"
    CONTAINER = "Container"
    APPLIANCE = "Appliance"
    FURNITURE = "DesignedFurniture"
    COMPONENT = "DesignedComponent"
    ITEM = "Item"

class ShapeType(str, Enum):
    """SOMA shape types"""
    BOX = "BoxShape"
    SPHERE = "SphereShape"
    CYLINDER = "CylinderShape"
    CIRCULAR_CYLINDER = "CircularCylinder"
    MESH = "MeshShape"

class MaterialType(str, Enum):
    """Material types"""
    METAL = "metal"
    PLASTIC = "plastic"
    WOOD = "wood"
    GLASS = "glass"
    CERAMIC = "ceramic"
    FABRIC = "fabric"
    RUBBER = "rubber"
    COMPOSITE = "composite"
    ORGANIC = "organic"

class TextureType(str, Enum):
    """Surface texture types"""
    SMOOTH = "smooth"
    ROUGH = "rough"
    GLOSSY = "glossy"
    MATTE = "matte"
    TEXTURED = "textured"
    PATTERNED = "patterned"

class DeviceState(str, Enum):
    """Device operational states"""
    ON = "DeviceTurnedOn"
    OFF = "DeviceTurnedOff"
    UNKNOWN = "unknown"

class CleanlinessState(str, Enum):
    """Object cleanliness states"""
    CLEAN = "Clean"
    DIRTY = "Dirty"
    UNKNOWN = "unknown"