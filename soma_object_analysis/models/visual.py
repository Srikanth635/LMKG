"""
Visual property models for object description
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from .base import TextureType

class RGBColor(BaseModel):
    """RGB color representation"""
    r: int = Field(..., ge=0, le=255, description="Red component")
    g: int = Field(..., ge=0, le=255, description="Green component")
    b: int = Field(..., ge=0, le=255, description="Blue component")

    def to_hex(self) -> str:
        """Convert to hex color string"""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

class HSVColor(BaseModel):
    """HSV color representation"""
    h: float = Field(..., ge=0, le=360, description="Hue in degrees")
    s: float = Field(..., ge=0, le=100, description="Saturation percentage")
    v: float = Field(..., ge=0, le=100, description="Value/brightness percentage")

class ColorDescription(BaseModel):
    """Complete color description"""
    primary_color: str = Field(..., description="Primary color name")
    rgb: Optional[RGBColor] = Field(None, description="RGB values")
    hsv: Optional[HSVColor] = Field(None, description="HSV values")
    secondary_colors: List[str] = Field(default=[], description="Additional colors present")
    color_pattern: Optional[str] = Field(None, description="Color pattern if any")

    class Config:
        json_schema_extra = {
            "example": {
                "primary_color": "red",
                "rgb": {"r": 220, "g": 20, "b": 60},
                "secondary_colors": ["green", "yellow"],
                "color_pattern": "gradient"
            }
        }

class SurfaceProperties(BaseModel):
    """Surface and texture properties"""
    texture: TextureType = Field(..., description="Surface texture type")
    finish: Optional[str] = Field(None, description="Surface finish")
    transparency: Optional[float] = Field(None, ge=0, le=1, description="Transparency level")
    reflectivity: Optional[float] = Field(None, ge=0, le=1, description="Surface reflectivity")
    pattern: Optional[str] = Field(None, description="Surface pattern description")

class VisualAppearance(BaseModel):
    """Complete visual description"""
    colors: ColorDescription
    surface: SurfaceProperties
    aesthetic_design: Optional[str] = Field(None, description="Overall aesthetic design style")
    visual_features: List[str] = Field(default=[], description="Notable visual features")