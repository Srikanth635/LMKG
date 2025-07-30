"""
Capability and affordance models for object description
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union

class BasicCapability(BaseModel):
    """Individual capability description"""
    name: str = Field(..., description="Capability name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in this capability")
    conditions: List[str] = Field(default=[], description="Conditions required for this capability")
    parameters: Dict[str, Union[str, float, bool]] = Field(default={}, description="Capability parameters")

class FunctionalAffordances(BaseModel):
    """What the object affords functionally"""
    can_cut: Optional[bool] = Field(None, description="Can be used for cutting")
    can_contain: Optional[bool] = Field(None, description="Can contain other objects")
    can_support: Optional[bool] = Field(None, description="Can support other objects")
    can_pour: Optional[bool] = Field(None, description="Can pour liquids")
    can_grasp: Optional[bool] = Field(None, description="Can be grasped")
    can_operate: Optional[bool] = Field(None, description="Can be operated/used")
    pourable: Optional[bool] = Field(None, description="Is pourable itself")
    graspability: Optional[float] = Field(None, ge=0, le=1, description="How easily graspable (0-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "can_cut": False,
                "can_contain": False,
                "can_grasp": True,
                "graspability": 0.9,
                "can_support": False
            }
        }

class TaskAffordances(BaseModel):
    """Tasks this object can be used for"""
    primary_tasks: List[str] = Field(default=[], description="Primary intended tasks")
    secondary_tasks: List[str] = Field(default=[], description="Secondary possible tasks")
    requires_tools: List[str] = Field(default=[], description="Tools needed to use this object")
    enables_tasks: List[str] = Field(default=[], description="Tasks this object enables others to do")

    class Config:
        json_schema_extra = {
            "example": {
                "primary_tasks": ["eating", "nutrition"],
                "secondary_tasks": ["throwing", "teaching"],
                "requires_tools": [],
                "enables_tasks": ["hand_feeding"]
            }
        }

class CapabilityDescription(BaseModel):
    """Complete capability description"""
    functional_affordances: FunctionalAffordances = Field(default_factory=FunctionalAffordances)
    task_affordances: TaskAffordances = Field(default_factory=TaskAffordances)
    capabilities: List[BasicCapability] = Field(default=[], description="Detailed capabilities")
    limitations: List[str] = Field(default=[], description="Known limitations")

    class Config:
        json_schema_extra = {
            "example": {
                "functional_affordances": {
                    "can_grasp": True,
                    "graspability": 0.9
                },
                "task_affordances": {
                    "primary_tasks": ["eating", "nutrition"]
                },
                "limitations": ["perishable", "fragile"]
            }
        }