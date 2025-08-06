"""
Semantic and contextual models for object description
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from .base import ObjectCategory


class RoleDescription(BaseModel):
    """Roles this object can play in actions"""
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "agent_roles": [],
                "patient_roles": ["transported_object", "cleaned_object"],
                "instrument_roles": ["container", "tool"],
                "location_roles": ["storage_location"]
            }
        }
    )

    agent_roles: List[str] = Field(
        default_factory=list,
        description="Roles when object acts as agent/performer",
        examples=[["robot", "actuator"], [], ["heater", "cooler"]]
    )
    patient_roles: List[str] = Field(
        default_factory=list,
        description="Roles when object is acted upon",
        examples=[
            ["cleaned_object", "moved_object"],
            ["filled_container", "heated_object"],
            []
        ]
    )
    instrument_roles: List[str] = Field(
        default_factory=list,
        description="Roles when used as tool/instrument",
        examples=[
            ["cutting_tool", "measuring_device"],
            ["container", "support"],
            []
        ]
    )
    location_roles: List[str] = Field(
        default_factory=list,
        description="Roles when serving as location/place",
        examples=[
            ["storage_location", "work_surface"],
            ["hiding_place"],
            []
        ]
    )


class ContextualInfo(BaseModel):
    """Contextual and semantic information"""
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "typical_locations": ["kitchen", "dining_table"],
                "associated_activities": ["eating", "cooking"],
                "cultural_significance": "Common household item",
                "safety_considerations": ["sharp_edges", "breakable"],
                "maintenance_requirements": ["wash_after_use", "dry_thoroughly"]
            }
        }
    )

    typical_locations: List[str] = Field(
        default_factory=list,
        description="Where this object is typically found",
        examples=[
            ["kitchen", "dining_room"],
            ["office", "desk"],
            ["garage", "workshop"]
        ]
    )
    associated_activities: List[str] = Field(
        default_factory=list,
        description="Activities commonly involving this object",
        examples=[
            ["cooking", "eating", "food_preparation"],
            ["writing", "drawing", "documentation"],
            ["construction", "repair"]
        ]
    )
    cultural_significance: Optional[str] = Field(
        None,
        description="Cultural, social, or symbolic meaning",
        examples=[
            "Symbol of hospitality in many cultures",
            "Traditional tool used in ceremonies",
            "Modern convenience item"
        ]
    )
    safety_considerations: List[str] = Field(
        default_factory=list,
        description="Safety warnings or precautions",
        examples=[
            ["sharp_edges", "hot_surface", "electrical_hazard"],
            ["choking_hazard", "toxic_if_ingested"],
            []
        ]
    )
    maintenance_requirements: List[str] = Field(
        default_factory=list,
        description="Care and maintenance needs",
        examples=[
            ["wash_with_soap", "dry_immediately", "oil_regularly"],
            ["charge_battery", "replace_filter"],
            []
        ]
    )


class SemanticDescription(BaseModel):
    """Complete semantic description"""
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "category": "Container",
                "subcategories": ["drinkware", "cup", "ceramic"],
                "context": {
                    "typical_locations": ["kitchen", "cabinet"],
                    "associated_activities": ["drinking", "serving"]
                },
                "synonyms": ["mug", "teacup"]
            }
        }
    )

    category: ObjectCategory = Field(
        ...,
        description="Primary SOMA object category",
        examples=[ObjectCategory.TOOL, ObjectCategory.CONTAINER, ObjectCategory.ITEM]
    )
    subcategories: List[str] = Field(
        default_factory=list,
        description="More specific classifications",
        examples=[
            ["kitchenware", "utensil", "cutting_tool"],
            ["furniture", "seating", "chair"],
            ["electronics", "computer", "laptop"]
        ]
    )
    roles: RoleDescription = Field(
        default_factory=RoleDescription,
        description="Semantic roles in actions"
    )
    context: ContextualInfo = Field(
        default_factory=ContextualInfo,
        description="Contextual information"
    )
    synonyms: List[str] = Field(
        default_factory=list,
        description="Alternative names or terms",
        examples=[
            ["cup", "mug", "glass"],
            ["knife", "blade", "cutter"],
            []
        ]
    )

    # Convenience properties for backward compatibility
    @property
    def typical_locations(self) -> List[str]:
        """Get typical locations from context"""
        return self.context.typical_locations

    @property
    def associated_activities(self) -> List[str]:
        """Get associated activities from context"""
        return self.context.associated_activities