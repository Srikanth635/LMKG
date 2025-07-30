"""
Semantic and contextual models for object description
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from .base import ObjectCategory

class RoleDescription(BaseModel):
    """Roles this object can play"""
    agent_roles: List[str] = Field(default=[], description="When object acts as agent")
    patient_roles: List[str] = Field(default=[], description="When object is acted upon")
    instrument_roles: List[str] = Field(default=[], description="When object is used as instrument")
    location_roles: List[str] = Field(default=[], description="When object serves as location")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_roles": [],
                "patient_roles": ["consumed_object", "picked_object"],
                "instrument_roles": ["projectile", "teaching_tool"],
                "location_roles": []
            }
        }

class ContextualInfo(BaseModel):
    """Contextual and semantic information"""
    typical_locations: List[str] = Field(default=[], description="Where this object is typically found")
    associated_activities: List[str] = Field(default=[], description="Activities associated with this object")
    cultural_significance: Optional[str] = Field(None, description="Cultural or social significance")
    safety_considerations: List[str] = Field(default=[], description="Safety considerations")
    maintenance_requirements: List[str] = Field(default=[], description="Maintenance needs")

    class Config:
        json_schema_extra = {
            "example": {
                "typical_locations": ["kitchen", "dining_table", "fruit_bowl"],
                "associated_activities": ["eating", "cooking", "snacking"],
                "cultural_significance": "Symbol of health and knowledge",
                "safety_considerations": ["wash_before_eating"],
                "maintenance_requirements": ["store_in_cool_place", "consume_within_week"]
            }
        }

class SemanticDescription(BaseModel):
    """Complete semantic description"""
    category: ObjectCategory = Field(..., description="Primary object category")
    subcategories: List[str] = Field(default=[], description="More specific classifications")
    roles: RoleDescription = Field(default_factory=RoleDescription)
    context: ContextualInfo = Field(default_factory=ContextualInfo)
    synonyms: List[str] = Field(default=[], description="Alternative names")

    # Convenience properties for backward compatibility
    @property
    def typical_locations(self) -> List[str]:
        """Get typical locations from context"""
        return self.context.typical_locations

    @property
    def associated_activities(self) -> List[str]:
        """Get associated activities from context"""
        return self.context.associated_activities

    class Config:
        json_schema_extra = {
            "example": {
                "category": "Item",
                "subcategories": ["food", "fruit", "produce"],
                "context": {
                    "typical_locations": ["kitchen", "table"],
                    "associated_activities": ["eating", "cooking"]
                },
                "synonyms": ["fruit", "pome"]
            }
        }