"""
Material property models for object description
"""

from pydantic import BaseModel, Field, ConfigDict, confloat
from typing import List, Optional, Annotated
from .base import MaterialType


class MaterialProperties(BaseModel):
    """Material and physical properties"""
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "primary_material": "metal",
                "secondary_materials": ["plastic"],
                "mass": 0.15,
                "density": 7800.0,
                "hardness": 5.5,
                "temperature": 20.0
            }
        }
    )

    primary_material: MaterialType = Field(
        ...,
        description="Main material the object is made of",
        examples=[MaterialType.METAL, MaterialType.PLASTIC, MaterialType.WOOD]
    )
    secondary_materials: List[MaterialType] = Field(
        default_factory=list,
        description="Additional materials present in the object",
        examples=[[MaterialType.RUBBER, MaterialType.GLASS], [], [MaterialType.FABRIC]]
    )
    mass: Optional[Annotated[float, confloat(gt=0)]] = Field(
        None,
        description="Mass in kilograms",
        examples=[0.001, 0.5, 2.5, 10.0]
    )
    density: Optional[Annotated[float, confloat(gt=0)]] = Field(
        None,
        description="Material density in kg/m³",
        examples=[1000.0, 7800.0, 500.0, 2700.0]
    )
    hardness: Optional[Annotated[float, Field(ge=1, le=10)]] = Field(
        None,
        description="Hardness on Mohs scale (1=talc, 10=diamond)",
        examples=[2.5, 5.5, 7.0, 9.0]
    )
    temperature: Optional[float] = Field(
        None,
        description="Current temperature in Celsius",
        examples=[20.0, 100.0, -5.0, 37.0]
    )


class MechanicalProperties(BaseModel):
    """Mechanical and force properties"""
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "static_friction": 0.6,
                "kinetic_friction": 0.4,
                "max_force": 150.0,
                "elasticity": 0.1,
                "fragility": 0.3
            }
        }
    )

    static_friction: Optional[Annotated[float, Field(ge=0)]] = Field(
        None,
        description="Static friction coefficient (0=frictionless, >1=very sticky)",
        examples=[0.1, 0.6, 1.2]
    )
    kinetic_friction: Optional[Annotated[float, Field(ge=0)]] = Field(
        None,
        description="Kinetic/sliding friction coefficient",
        examples=[0.05, 0.4, 0.8]
    )
    max_force: Optional[Annotated[float, confloat(gt=0)]] = Field(
        None,
        description="Maximum force before breaking in Newtons",
        examples=[10.0, 150.0, 1000.0]
    )
    elasticity: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        None,
        description="Elasticity/bounciness (0=rigid, 1=perfectly elastic)",
        examples=[0.0, 0.3, 0.8, 1.0]
    )
    fragility: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        None,
        description="How easily broken (0=indestructible, 1=extremely fragile)",
        examples=[0.1, 0.5, 0.9]
    )