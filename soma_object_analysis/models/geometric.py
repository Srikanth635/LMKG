"""
Geometric property models for object description
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from .base import ShapeType


class Position3D(BaseModel):
    """3D position representation"""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    z: float = Field(..., description="Z coordinate")
    reference_frame: str = Field(default="world", description="Reference coordinate frame")


class Pose6D(BaseModel):
    """6D pose representation (position + orientation)"""
    position: Position3D
    orientation: Dict[str, float] = Field(..., description="Quaternion or Euler angles")


class Dimensions(BaseModel):
    """Physical dimensions"""
    width: Optional[float] = Field(None, gt=0, description="Width in meters")
    height: Optional[float] = Field(None, gt=0, description="Height in meters")
    depth: Optional[float] = Field(None, gt=0, description="Depth in meters")
    length: Optional[float] = Field(None, gt=0, description="Length in meters")
    radius: Optional[float] = Field(None, gt=0, description="Radius in meters")

    def volume(self) -> Optional[float]:
        """Calculate approximate volume if dimensions available"""
        if self.width and self.height and self.depth:
            return self.width * self.height * self.depth
        elif self.radius and self.height:
            # Cylinder volume
            return 3.14159 * self.radius ** 2 * self.height
        elif self.radius:
            # Sphere volume
            return (4 / 3) * 3.14159 * self.radius ** 3
        return None


class GeometricShape(BaseModel):
    """Geometric shape description"""
    primary_shape: ShapeType = Field(..., description="Primary geometric shape")
    dimensions: Dimensions
    shape_parameters: Dict[str, float] = Field(default={}, description="Additional shape parameters")
    complex_geometry: Optional[str] = Field(None, description="Description of complex geometric features")


class SpatialRelations(BaseModel):
    """Spatial relationships with other objects"""
    supported_by: List[str] = Field(default=[], description="Objects supporting this object")
    supports: List[str] = Field(default=[], description="Objects supported by this object")
    contained_in: List[str] = Field(default=[], description="Containers this object is in")
    contains: List[str] = Field(default=[], description="Objects contained in this object")
    adjacent_to: List[str] = Field(default=[], description="Adjacent objects")


class GeometricDescription(BaseModel):
    """Complete geometric description"""
    shape: GeometricShape
    pose: Optional[Pose6D] = Field(None, description="Current pose in space")
    spatial_relations: SpatialRelations = Field(default_factory=SpatialRelations)
    bounding_box: Optional[Dimensions] = Field(None, description="Axis-aligned bounding box")