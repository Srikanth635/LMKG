"""
Main ObjectDescription model that combines all aspects
"""

from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional

from .base import CleanlinessState, DeviceState
from .visual import VisualAppearance, ColorDescription, SurfaceProperties, TextureType
from .geometric import GeometricDescription, GeometricShape, Dimensions, ShapeType
from .material import MaterialProperties, MechanicalProperties, MaterialType
from .capabilities import CapabilityDescription, FunctionalAffordances, TaskAffordances
from .semantic import SemanticDescription, ObjectCategory

class ObjectState(BaseModel):
    """Current state of the object"""
    device_state: Optional[DeviceState] = Field(None, description="Device operational state")
    cleanliness: CleanlinessState = Field(default=CleanlinessState.UNKNOWN, description="Cleanliness state")
    integrity: float = Field(default=1.0, ge=0, le=1, description="Physical integrity (0=broken, 1=perfect)")
    functional_state: Optional[str] = Field(None, description="Functional state description")
    temporal_properties: dict = Field(default_factory=dict, description="Time-related properties")

class ObjectDescription(BaseModel):
    """Complete SOMA-aligned object description model"""

    # Basic identification
    name: str = Field(..., description="Object name/identifier")
    description: str = Field(..., description="General description")

    # Core property categories
    visual: VisualAppearance
    geometric: GeometricDescription
    material: MaterialProperties
    mechanical: MechanicalProperties = Field(default_factory=MechanicalProperties)
    capabilities: CapabilityDescription = Field(default_factory=CapabilityDescription)
    semantic: SemanticDescription
    state: ObjectState = Field(default_factory=ObjectState)

    # Metadata
    confidence_score: float = Field(default=0.8, ge=0, le=1, description="Overall confidence in description")
    source: str = Field(default="llm_generated", description="Source of this description")
    timestamp: Optional[str] = Field(None, description="When this description was generated")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "name": "Red Apple",
                "description": "A fresh red apple on a table",
                "visual": {
                    "colors": {
                        "primary_color": "red",
                        "rgb": {"r": 220, "g": 20, "b": 60},
                        "secondary_colors": ["green"]
                    },
                    "surface": {
                        "texture": "smooth",
                        "finish": "natural_waxy"
                    }
                },
                "geometric": {
                    "shape": {
                        "primary_shape": "SphereShape",
                        "dimensions": {"radius": 0.04, "height": 0.08}
                    }
                },
                "material": {
                    "primary_material": "organic",
                    "mass": 0.18
                },
                "semantic": {
                    "category": "Item",
                    "typical_locations": ["kitchen", "table"]
                }
            }
        }

    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()

    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.now().isoformat()

    def get_summary(self) -> str:
        """Generate a brief summary of the object"""
        return f"{self.name}: {self.semantic.category.value} - {self.description[:100]}..."

    def to_simple_dict(self) -> dict:
        """Convert to simplified dictionary for display"""
        return {
            "name": self.name,
            "category": self.semantic.category.value,
            "primary_color": self.visual.colors.primary_color,
            "shape": self.geometric.shape.primary_shape.value,
            "material": self.material.primary_material.value,
            "mass": self.material.mass,
            "graspability": self.capabilities.functional_affordances.graspability,
            "confidence": self.confidence_score
        }

# Factory functions for easy creation
def create_minimal_object(name: str, category: ObjectCategory) -> ObjectDescription:
    """Create a minimal object description with required fields only"""
    return ObjectDescription(
        name=name,
        description=f"A {category.value.lower()}",
        visual=VisualAppearance(
            colors=ColorDescription(primary_color="unknown"),
            surface=SurfaceProperties(texture=TextureType.SMOOTH)
        ),
        geometric=GeometricDescription(
            shape=GeometricShape(
                primary_shape=ShapeType.BOX,
                dimensions=Dimensions()
            )
        ),
        material=MaterialProperties(primary_material=MaterialType.PLASTIC),
        semantic=SemanticDescription(category=category)
    )