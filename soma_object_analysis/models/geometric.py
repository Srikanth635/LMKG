"""
Geometric property models for object description
"""

from pydantic import confloat
from pydantic import BaseModel, Field, ConfigDict, confloat
from typing import List, Optional, Dict, Annotated
from .base import ShapeType


class Position3D(BaseModel):
    """3D position representation"""
    model_config = ConfigDict(validate_assignment=True)

    x: float = Field(
        ...,
        description="X coordinate in meters",
        examples=[0.0, 1.5, -2.3]
    )
    y: float = Field(
        ...,
        description="Y coordinate in meters",
        examples=[0.0, 2.1, -1.5]
    )
    z: float = Field(
        ...,
        description="Z coordinate in meters",
        examples=[0.0, 0.8, 1.2]
    )
    reference_frame: str = Field(
        default="world",
        description="Coordinate reference frame",
        examples=["world", "robot_base", "table_top", "camera"]
    )


class Pose6D(BaseModel):
    """6D pose representation (position + orientation)"""
    model_config = ConfigDict(validate_assignment=True)

    position: Position3D = Field(
        ...,
        description="3D position in space"
    )
    orientation: Dict[str, float] = Field(
        ...,
        description="Orientation as quaternion (x,y,z,w) or Euler angles (roll,pitch,yaw)",
        examples=[
            {"x": 0, "y": 0, "z": 0, "w": 1},
            {"roll": 0, "pitch": 0, "yaw": 1.57}
        ]
    )


class Dimensions(BaseModel):
    """Physical dimensions in meters"""
    model_config = ConfigDict(validate_assignment=True)

    width: Optional[Annotated[float, confloat(gt=0)]] = Field(
        None,
        description="Width in meters (left-right extent)",
        examples=[0.05, 0.3, 1.2]
    )
    height: Optional[Annotated[float, confloat(gt=0)]] = Field(
        None,
        description="Height in meters (vertical extent)",
        examples=[0.1, 0.5, 2.0]
    )
    depth: Optional[Annotated[float, confloat(gt=0)]] = Field(
        None,
        description="Depth in meters (front-back extent)",
        examples=[0.05, 0.3, 0.8]
    )
    length: Optional[Annotated[float, confloat(gt=0)]] = Field(
        None,
        description="Length in meters (longest dimension)",
        examples=[0.2, 1.0, 3.0]
    )
    radius: Optional[Annotated[float, confloat(gt=0)]] = Field(
        None,
        description="Radius in meters (for circular/spherical objects)",
        examples=[0.02, 0.1, 0.5]
    )

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
    model_config = ConfigDict(validate_assignment=True)

    primary_shape: ShapeType = Field(
        ...,
        description="Primary geometric primitive that best describes the object",
        examples=[ShapeType.BOX, ShapeType.CYLINDER, ShapeType.SPHERE]
    )
    dimensions: Dimensions = Field(
        ...,
        description="Physical dimensions of the object"
    )
    shape_parameters: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional shape-specific parameters",
        examples=[
            {"corner_radius": 0.01, "taper": 0.9},
            {"eccentricity": 0.5},
            {}
        ]
    )
    complex_geometry: Optional[str] = Field(
        None,
        description="Description of complex geometric features not captured by primary shape",
        examples=[
            "handle attached to cylindrical body",
            "irregular organic shape with multiple protrusions",
            "compound shape with nested cavities"
        ]
    )


class SpatialRelations(BaseModel):
    """Spatial relationships with other objects"""
    model_config = ConfigDict(validate_assignment=True)

    supported_by: List[str] = Field(
        default_factory=list,
        description="IDs/names of objects supporting this object",
        examples=[["table_1", "floor"], ["shelf_2"], []]
    )
    supports: List[str] = Field(
        default_factory=list,
        description="IDs/names of objects this object supports",
        examples=[["cup_3", "plate_1"], [], ["book_5"]]
    )
    contained_in: List[str] = Field(
        default_factory=list,
        description="IDs/names of containers this object is inside",
        examples=[["drawer_1"], ["box_2", "room_1"], []]
    )
    contains: List[str] = Field(
        default_factory=list,
        description="IDs/names of objects contained in this object",
        examples=[["pen_1", "paper_3"], [], ["water"]]
    )
    adjacent_to: List[str] = Field(
        default_factory=list,
        description="IDs/names of nearby/touching objects",
        examples=[["wall", "chair_2"], ["lamp_1"], []]
    )


class GeometricDescription(BaseModel):
    """Complete geometric description"""
    model_config = ConfigDict(validate_assignment=True)

    shape: GeometricShape = Field(
        ...,
        description="Shape and dimensions information"
    )
    pose: Optional[Pose6D] = Field(
        None,
        description="Current position and orientation in 3D space"
    )
    spatial_relations: SpatialRelations = Field(
        default_factory=SpatialRelations,
        description="Relationships with other objects in the environment"
    )
    bounding_box: Optional[Dimensions] = Field(
        None,
        description="Axis-aligned bounding box dimensions for collision detection"
    )