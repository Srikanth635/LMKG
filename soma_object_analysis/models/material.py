"""
Material property models for object description
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from .base import MaterialType

class MaterialProperties(BaseModel):
    """Material and physical properties"""
    primary_material: MaterialType = Field(..., description="Primary material")
    secondary_materials: List[MaterialType] = Field(default=[], description="Additional materials")
    mass: Optional[float] = Field(None, gt=0, description="Mass in kilograms")
    density: Optional[float] = Field(None, gt=0, description="Density in kg/m³")
    hardness: Optional[float] = Field(None, ge=1, le=10, description="Hardness on Mohs scale")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")

    class Config:
        json_schema_extra = {
            "example": {
                "primary_material": "metal",
                "secondary_materials": ["plastic"],
                "mass": 0.15,
                "density": 7800.0,
                "hardness": 5.5,
                "temperature": 20.0
            }
        }

class MechanicalProperties(BaseModel):
    """Mechanical and force properties"""
    static_friction: Optional[float] = Field(None, ge=0, description="Static friction coefficient")
    kinetic_friction: Optional[float] = Field(None, ge=0, description="Kinetic friction coefficient")
    max_force: Optional[float] = Field(None, gt=0, description="Maximum force it can withstand (N)")
    elasticity: Optional[float] = Field(None, ge=0, le=1, description="Elasticity coefficient")
    fragility: Optional[float] = Field(None, ge=0, le=1, description="Fragility level")

    class Config:
        json_schema_extra = {
            "example": {
                "static_friction": 0.6,
                "kinetic_friction": 0.4,
                "max_force": 150.0,
                "elasticity": 0.1,
                "fragility": 0.3
            }
        }