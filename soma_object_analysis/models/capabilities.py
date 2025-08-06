"""
Capability and affordance models for object description
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Union, Annotated


class BasicCapability(BaseModel):
    """Individual capability description"""
    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(
        ...,
        description="Name of the capability",
        examples=["graspable", "pourable", "cuttable", "stackable"]
    )
    confidence: Annotated[float, Field(ge=0, le=1)] = Field(
        ...,
        description="Confidence in this capability (0=unlikely, 1=certain)",
        examples=[0.9, 0.5, 1.0]
    )
    conditions: List[str] = Field(
        default_factory=list,
        description="Conditions required for this capability",
        examples=[
            ["empty", "upright"],
            ["powered_on", "unlocked"],
            []
        ]
    )
    parameters: Dict[str, Union[str, float, bool]] = Field(
        default_factory=dict,
        description="Capability-specific parameters",
        examples=[
            {"max_weight": 5.0, "min_size": 0.01},
            {"requires_tool": True, "tool_type": "knife"},
            {}
        ]
    )


class FunctionalAffordances(BaseModel):
    """What the object affords functionally"""
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "can_cut": False,
                "can_contain": True,
                "can_grasp": True,
                "graspability": 0.9,
                "can_support": True
            }
        }
    )

    can_cut: Optional[bool] = Field(
        None,
        description="Can be used to cut other objects",
        examples=[True, False]
    )
    can_contain: Optional[bool] = Field(
        None,
        description="Can hold/contain other objects inside",
        examples=[True, False]
    )
    can_support: Optional[bool] = Field(
        None,
        description="Can support other objects on top",
        examples=[True, False]
    )
    can_pour: Optional[bool] = Field(
        None,
        description="Can pour out liquids/granular materials",
        examples=[True, False]
    )
    can_grasp: Optional[bool] = Field(
        None,
        description="Can be grasped by a robot/human hand",
        examples=[True, False]
    )
    can_operate: Optional[bool] = Field(
        None,
        description="Has controls that can be operated",
        examples=[True, False]
    )
    pourable: Optional[bool] = Field(
        None,
        description="Contents can be poured out",
        examples=[True, False]
    )
    graspability: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        None,
        description="How easily graspable (0=impossible, 1=very easy)",
        examples=[0.1, 0.5, 0.9, 1.0]
    )


class TaskAffordances(BaseModel):
    """Tasks this object can be used for"""
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "primary_tasks": ["drinking", "storing_liquid"],
                "secondary_tasks": ["measuring", "mixing"],
                "requires_tools": [],
                "enables_tasks": ["serving_beverages", "hydration"]
            }
        }
    )

    primary_tasks: List[str] = Field(
        default_factory=list,
        description="Main intended uses/tasks",
        examples=[
            ["cutting", "slicing"],
            ["storing", "organizing"],
            ["sitting", "resting"]
        ]
    )
    secondary_tasks: List[str] = Field(
        default_factory=list,
        description="Alternative or secondary uses",
        examples=[
            ["hammering", "prying"],
            ["decoration", "paperweight"],
            []
        ]
    )
    requires_tools: List[str] = Field(
        default_factory=list,
        description="Tools needed to use this object effectively",
        examples=[
            ["screwdriver", "wrench"],
            [],
            ["power_source", "batteries"]
        ]
    )
    enables_tasks: List[str] = Field(
        default_factory=list,
        description="Tasks that this object makes possible",
        examples=[
            ["cooking", "food_preparation"],
            ["communication", "documentation"],
            []
        ]
    )


class CapabilityDescription(BaseModel):
    """Complete capability description"""
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "functional_affordances": {
                    "can_grasp": True,
                    "graspability": 0.9,
                    "can_contain": True
                },
                "task_affordances": {
                    "primary_tasks": ["drinking", "storing"],
                    "enables_tasks": ["hydration"]
                },
                "limitations": ["fragile", "not_dishwasher_safe"]
            }
        }
    )

    functional_affordances: FunctionalAffordances = Field(
        default_factory=FunctionalAffordances,
        description="Physical affordances and capabilities"
    )
    task_affordances: TaskAffordances = Field(
        default_factory=TaskAffordances,
        description="Task-oriented capabilities"
    )
    capabilities: List[BasicCapability] = Field(
        default_factory=list,
        description="Detailed list of specific capabilities"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Known limitations or constraints",
        examples=[
            ["fragile", "heavy", "requires_power"],
            ["not_waterproof", "single_use"],
            []
        ]
    )