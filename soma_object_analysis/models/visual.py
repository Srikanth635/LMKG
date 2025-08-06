"""
Visual property models for object description
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Annotated
from pydantic import conint
from .base import TextureType


class RGBColor(BaseModel):
    """RGB color representation"""
    model_config = ConfigDict(validate_assignment=True)

    r: Annotated[int, conint(ge=0, le=255)] = Field(
        ...,
        description="Red component (0-255)",
        examples=[255, 128, 0]
    )
    g: Annotated[int, conint(ge=0, le=255)] = Field(
        ...,
        description="Green component (0-255)",
        examples=[255, 128, 0]
    )
    b: Annotated[int, conint(ge=0, le=255)] = Field(
        ...,
        description="Blue component (0-255)",
        examples=[255, 128, 0]
    )

    def to_hex(self) -> str:
        """Convert to hex color string"""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"


class HSVColor(BaseModel):
    """HSV color representation"""
    model_config = ConfigDict(validate_assignment=True)

    h: Annotated[float, Field(ge=0, le=360)] = Field(
        ...,
        description="Hue in degrees (0-360)",
        examples=[0, 120, 240]
    )
    s: Annotated[float, Field(ge=0, le=100)] = Field(
        ...,
        description="Saturation percentage (0-100)",
        examples=[0, 50, 100]
    )
    v: Annotated[float, Field(ge=0, le=100)] = Field(
        ...,
        description="Value/brightness percentage (0-100)",
        examples=[0, 50, 100]
    )


class ColorDescription(BaseModel):
    """Complete color description"""
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "primary_color": "red",
                "rgb": {"r": 220, "g": 20, "b": 60},
                "secondary_colors": ["green", "yellow"],
                "color_pattern": "gradient"
            }
        }
    )

    primary_color: str = Field(
        ...,
        description="Primary color name in common English",
        examples=["red", "blue", "green", "yellow", "silver", "black", "white"]
    )
    rgb: Optional[RGBColor] = Field(
        None,
        description="Exact RGB color values"
    )
    hsv: Optional[HSVColor] = Field(
        None,
        description="HSV color values for better color manipulation"
    )
    secondary_colors: List[str] = Field(
        default_factory=list,
        description="Additional colors present on the object",
        examples=[["green", "yellow"], ["white", "black"], []]
    )
    color_pattern: Optional[str] = Field(
        None,
        description="Pattern of color distribution",
        examples=["solid", "gradient", "striped", "spotted", "marbled"]
    )


class SurfaceProperties(BaseModel):
    """Surface and texture properties"""
    model_config = ConfigDict(validate_assignment=True)

    texture: TextureType = Field(
        ...,
        description="Primary surface texture",
        examples=[TextureType.SMOOTH, TextureType.ROUGH, TextureType.GLOSSY]
    )
    finish: Optional[str] = Field(
        None,
        description="Surface finish or coating",
        examples=["polished", "brushed", "anodized", "painted", "natural", "lacquered"]
    )
    transparency: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        None,
        description="Transparency level (0=opaque, 1=fully transparent)",
        examples=[0.0, 0.5, 0.8, 1.0]
    )
    reflectivity: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        None,
        description="Surface reflectivity (0=matte, 1=mirror-like)",
        examples=[0.1, 0.5, 0.9]
    )
    pattern: Optional[str] = Field(
        None,
        description="Visual pattern on surface",
        examples=["none", "wood grain", "checkered", "dots", "lines", "abstract"]
    )


class VisualAppearance(BaseModel):
    """Complete visual description"""
    model_config = ConfigDict(validate_assignment=True)

    colors: ColorDescription = Field(
        ...,
        description="Color information of the object"
    )
    surface: SurfaceProperties = Field(
        ...,
        description="Surface texture and finish properties"
    )
    aesthetic_design: Optional[str] = Field(
        None,
        description="Overall design style or aesthetic",
        examples=["modern", "minimalist", "industrial", "vintage", "organic", "futuristic"]
    )
    visual_features: List[str] = Field(
        default_factory=list,
        description="Notable visual features or markings",
        examples=[["logo", "label"], ["scratches", "dents"], ["buttons", "display"]]
    )