"""
Mock agent for testing and development without LLM dependencies
"""

from datetime import datetime
from .base_agent import AnalysisAgent, AgentState
from ..models import (
    ObjectDescription, ObjectCategory, ShapeType, MaterialType,
    ColorDescription, RGBColor, VisualAppearance, SurfaceProperties, TextureType,
    GeometricDescription, GeometricShape, Dimensions,
    MaterialProperties, CapabilityDescription, FunctionalAffordances, TaskAffordances,
    SemanticDescription
)


class MockAnalysisAgent(AnalysisAgent):
    """Mock agent that generates realistic object descriptions without LLM calls"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.object_templates = self._create_object_templates()

    def _create_object_templates(self) -> dict:
        """Create templates for common objects"""
        return {
            "apple": {
                "category": ObjectCategory.ITEM,
                "shape": ShapeType.SPHERE,
                "material": MaterialType.ORGANIC,
                "color": "red",
                "rgb": {"r": 220, "g": 20, "b": 60},
                "mass": 0.18,
                "dimensions": {"radius": 0.04, "height": 0.08},
                "can_grasp": True,
                "graspability": 0.9,
                "tasks": ["eating", "nutrition", "snacking"],
                "locations": ["kitchen", "table", "fruit_bowl"]
            },
            "knife": {
                "category": ObjectCategory.TOOL,
                "shape": ShapeType.BOX,
                "material": MaterialType.METAL,
                "color": "silver",
                "rgb": {"r": 192, "g": 192, "b": 192},
                "mass": 0.15,
                "dimensions": {"width": 0.03, "height": 0.02, "length": 0.25},
                "can_cut": True,
                "can_grasp": True,
                "graspability": 0.8,
                "tasks": ["cutting", "food_preparation", "slicing"],
                "locations": ["kitchen", "drawer", "counter"]
            },
            "mug": {
                "category": ObjectCategory.CONTAINER,
                "shape": ShapeType.CYLINDER,
                "material": MaterialType.CERAMIC,
                "color": "blue",
                "rgb": {"r": 0, "g": 100, "b": 200},
                "mass": 0.25,
                "dimensions": {"radius": 0.04, "height": 0.1},
                "can_contain": True,
                "can_grasp": True,
                "graspability": 0.85,
                "tasks": ["drinking", "holding_liquids", "coffee"],
                "locations": ["kitchen", "office", "table"]
            },
            "smartphone": {
                "category": ObjectCategory.APPLIANCE,
                "shape": ShapeType.BOX,
                "material": MaterialType.GLASS,
                "color": "black",
                "rgb": {"r": 20, "g": 20, "b": 20},
                "mass": 0.18,
                "dimensions": {"width": 0.07, "height": 0.15, "depth": 0.008},
                "can_grasp": True,
                "graspability": 0.95,
                "tasks": ["communication", "computing", "photography"],
                "locations": ["pocket", "desk", "hand"]
            }
        }

    def _identify_object_type(self, description: str) -> str:
        """Identify object type from description"""
        description_lower = description.lower()

        for obj_type, template in self.object_templates.items():
            if obj_type in description_lower:
                return obj_type

        # Check for synonyms
        synonyms = {
            "phone": "smartphone",
            "mobile": "smartphone",
            "cup": "mug",
            "blade": "knife",
            "fruit": "apple"
        }

        for synonym, obj_type in synonyms.items():
            if synonym in description_lower:
                return obj_type

        return "generic"

    def _create_generic_object(self, description: str) -> dict:
        """Create a generic object template"""
        return {
            "category": ObjectCategory.ITEM,
            "shape": ShapeType.BOX,
            "material": MaterialType.PLASTIC,
            "color": "gray",
            "rgb": {"r": 128, "g": 128, "b": 128},
            "mass": 0.1,
            "dimensions": {"width": 0.1, "height": 0.1, "depth": 0.1},
            "can_grasp": True,
            "graspability": 0.7,
            "tasks": ["general_use"],
            "locations": ["table", "shelf"]
        }

    def _extract_color_from_description(self, description: str) -> tuple[str, dict]:
        """Extract color information from description"""
        color_mapping = {
            "red": {"r": 220, "g": 20, "b": 60},
            "green": {"r": 34, "g": 139, "b": 34},
            "blue": {"r": 0, "g": 100, "b": 200},
            "yellow": {"r": 255, "g": 255, "b": 0},
            "black": {"r": 20, "g": 20, "b": 20},
            "white": {"r": 255, "g": 255, "b": 255},
            "silver": {"r": 192, "g": 192, "b": 192},
            "brown": {"r": 139, "g": 69, "b": 19}
        }

        description_lower = description.lower()
        for color, rgb in color_mapping.items():
            if color in description_lower:
                return color, rgb

        return "unknown", {"r": 128, "g": 128, "b": 128}

    def _create_object_description(self, description: str, template: dict) -> ObjectDescription:
        """Create ObjectDescription from template"""

        # Extract color from description if mentioned
        extracted_color, extracted_rgb = self._extract_color_from_description(description)
        if extracted_color != "unknown":
            color = extracted_color
            rgb = extracted_rgb
        else:
            color = template["color"]
            rgb = template["rgb"]

        # Create the object description
        obj_desc = ObjectDescription(
            name=self._generate_object_name(description, template),
            description=description,

            visual=VisualAppearance(
                colors=ColorDescription(
                    primary_color=color,
                    rgb=RGBColor(**rgb),
                    secondary_colors=[]
                ),
                surface=SurfaceProperties(
                    texture=TextureType.SMOOTH,
                    finish="natural"
                )
            ),

            geometric=GeometricDescription(
                shape=GeometricShape(
                    primary_shape=template["shape"],
                    dimensions=Dimensions(**template["dimensions"])
                )
            ),

            material=MaterialProperties(
                primary_material=template["material"],
                mass=template["mass"]
            ),

            capabilities=CapabilityDescription(
                functional_affordances=FunctionalAffordances(
                    can_cut=template.get("can_cut", False),
                    can_contain=template.get("can_contain", False),
                    can_grasp=template.get("can_grasp", False),
                    graspability=template.get("graspability", 0.5)
                ),
                task_affordances=TaskAffordances(
                    primary_tasks=template.get("tasks", [])
                )
            ),

            semantic=SemanticDescription(
                category=template["category"],
                typical_locations=template.get("locations", []),
                associated_activities=template.get("tasks", [])
            ),

            confidence_score=0.85,  # Mock confidence
            source="mock_agent"
        )

        return obj_desc

    def _generate_object_name(self, description: str, template: dict) -> str:
        """Generate appropriate object name"""
        obj_type = None

        # Find the object type in the description
        for obj_name in self.object_templates.keys():
            if obj_name in description.lower():
                obj_type = obj_name
                break

        if not obj_type:
            obj_type = "object"

        # Add color if mentioned
        color, _ = self._extract_color_from_description(description)
        if color != "unknown":
            return f"{color.capitalize()} {obj_type.capitalize()}"
        else:
            return obj_type.capitalize()

    async def process(self, state: AgentState) -> AgentState:
        """Process the agent state using mock analysis"""
        self.logger.info(f"Mock analyzing: {state.input_description}")

        try:
            # Identify object type
            obj_type = self._identify_object_type(state.input_description)

            # Get template
            if obj_type in self.object_templates:
                template = self.object_templates[obj_type]
            else:
                template = self._create_generic_object(state.input_description)

            # Create object description
            obj_desc = self._create_object_description(state.input_description, template)

            state.object_analysis = obj_desc
            state.metadata = {
                "analysis_timestamp": datetime.now().isoformat(),
                "mode": "mock",
                "template_used": obj_type,
                "confidence_note": "Mock analysis - not from real LLM"
            }

            self.logger.info(f"Mock analysis completed: {obj_desc.name}")

        except Exception as e:
            error_msg = f"Mock analysis failed: {str(e)}"
            state.errors.append(error_msg)
            self.logger.error(error_msg)

        return state

    async def analyze(self, input_description: str) -> dict:
        """Main analysis method"""
        initial_state = AgentState(input_description=input_description)
        final_state = await self.process(initial_state)

        return {
            "success": len(final_state.errors) == 0,
            "object_analysis": final_state.object_analysis.dict() if final_state.object_analysis else None,
            "errors": final_state.errors,
            "metadata": final_state.metadata
        }