"""
Main ObjectDescription model that combines all aspects
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from datetime import datetime
from typing import Optional, Annotated, Dict, Any

from .base import CleanlinessState, DeviceState
from .visual import VisualAppearance, ColorDescription, SurfaceProperties, TextureType
from .geometric import GeometricDescription, GeometricShape, Dimensions, ShapeType
from .material import MaterialProperties, MechanicalProperties, MaterialType
from .capabilities import CapabilityDescription, FunctionalAffordances, TaskAffordances
from .semantic import SemanticDescription, ObjectCategory


class ObjectState(BaseModel):
    """Current state of the object"""
    model_config = ConfigDict(validate_assignment=True)

    device_state: Optional[DeviceState] = Field(
        None,
        description="Operational state if object is a device",
        examples=[DeviceState.ON, DeviceState.OFF, None]
    )
    cleanliness: CleanlinessState = Field(
        default=CleanlinessState.UNKNOWN,
        description="Current cleanliness level",
        examples=[CleanlinessState.CLEAN, CleanlinessState.DIRTY]
    )
    integrity: Annotated[float, Field(ge=0, le=1)] = Field(
        default=1.0,
        description="Physical condition (0=completely broken, 1=perfect condition)",
        examples=[1.0, 0.8, 0.5, 0.2]
    )
    functional_state: Optional[str] = Field(
        None,
        description="Description of current functional state",
        examples=[
            "fully_operational",
            "partially_working",
            "needs_repair",
            "lid_open",
            "half_full"
        ]
    )
    temporal_properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Time-related properties",
        examples=[
            {"last_used": "2024-01-15T10:30:00", "age_days": 365},
            {"expiry_date": "2024-12-31"},
            {}
        ]
    )


class ObjectDescription(BaseModel):
    """Complete SOMA-aligned object description model"""
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        json_schema_extra={
            "example": {
                "name": "Coffee Mug",
                "description": "A white ceramic coffee mug with company logo",
                "visual": {
                    "colors": {
                        "primary_color": "white",
                        "secondary_colors": ["blue"]
                    },
                    "surface": {
                        "texture": "smooth",
                        "finish": "glazed"
                    }
                },
                "geometric": {
                    "shape": {
                        "primary_shape": "CylinderShape",
                        "dimensions": {"radius": 0.04, "height": 0.1}
                    }
                },
                "material": {
                    "primary_material": "ceramic",
                    "mass": 0.3
                },
                "semantic": {
                    "category": "Container",
                    "subcategories": ["drinkware", "mug"]
                }
            }
        }
    )

    # Basic identification
    name: str = Field(
        ...,
        description="Object name or identifier",
        examples=["Coffee Mug", "Red Apple", "Laptop Computer", "Kitchen Knife"]
    )
    description: str = Field(
        ...,
        description="Natural language description of the object",
        examples=[
            "A red ceramic coffee mug with a handle",
            "Fresh green apple with slight bruising",
            "Silver laptop with 15-inch screen"
        ]
    )

    # Core property categories
    visual: VisualAppearance = Field(
        ...,
        description="Visual appearance properties including color and texture"
    )
    geometric: GeometricDescription = Field(
        ...,
        description="Geometric properties including shape, size, and spatial relations"
    )
    material: MaterialProperties = Field(
        ...,
        description="Material composition and physical properties"
    )
    mechanical: MechanicalProperties = Field(
        default_factory=MechanicalProperties,
        description="Mechanical properties like friction and elasticity"
    )
    capabilities: CapabilityDescription = Field(
        default_factory=CapabilityDescription,
        description="Functional capabilities and affordances"
    )
    semantic: SemanticDescription = Field(
        ...,
        description="Semantic category and contextual information"
    )
    state: ObjectState = Field(
        default_factory=ObjectState,
        description="Current state of the object"
    )

    # Metadata
    confidence_score: Annotated[float, Field(ge=0, le=1)] = Field(
        default=0.8,
        description="Overall confidence in this description (0=uncertain, 1=certain)",
        examples=[0.5, 0.8, 0.95, 1.0]
    )
    source: str = Field(
        default="llm_generated",
        description="Source of this description",
        examples=["llm_generated", "human_annotated", "sensor_detected", "database"]
    )
    timestamp: Optional[str] = Field(
        None,
        description="ISO format timestamp when description was generated",
        examples=["2024-01-15T10:30:00Z", "2024-12-31T23:59:59Z"]
    )

    @field_validator('name')
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """Ensure name is not empty"""
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()

    @field_validator('description')
    @classmethod
    def description_must_be_meaningful(cls, v: str) -> str:
        """Ensure description is meaningful"""
        if len(v.strip()) < 10:
            raise ValueError('Description must be at least 10 characters long')
        return v.strip()

    @model_validator(mode='before')
    @classmethod
    def set_timestamp(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set timestamp if not provided"""
        if 'timestamp' not in values or values['timestamp'] is None:
            values['timestamp'] = datetime.now().isoformat()
        return values

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
        description=f"A {category.value.lower()} object named {name}",
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