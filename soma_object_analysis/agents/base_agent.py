"""
Base agent classes and interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel

from ..models import ObjectDescription
from ..config import config

class AgentState(BaseModel):
    """State for LangGraph agents"""
    input_description: str
    object_analysis: Optional[ObjectDescription] = None
    knowledge_graph: Optional[str] = None
    export_paths: list = []
    errors: list = []
    metadata: dict = {}

class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logging for the agent"""
        import logging
        logger = logging.getLogger(f"soma.agents.{self.name}")
        logger.setLevel(getattr(logging, config.LOG_LEVEL))
        return logger

    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """Process the agent state"""
        pass

    def validate_input(self, state: AgentState) -> bool:
        """Validate input state"""
        return bool(state.input_description.strip())

class AnalysisAgent(BaseAgent):
    """Base class for object analysis agents"""

    def __init__(self, model_name: str = None, temperature: float = None, **kwargs):
        super().__init__("analysis", **kwargs)
        self.model_name = model_name or config.DEFAULT_MODEL
        self.temperature = temperature or config.DEFAULT_TEMPERATURE
        self.llm = None
        self.structured_llm = None

    def _initialize_llm(self):
        """Initialize the language model - to be implemented by subclasses"""
        pass

    def get_system_prompt(self) -> str:
        """Get the system prompt for analysis"""
        return """You are an expert object analysis system that provides structured descriptions aligned with the SOMA ontology.

Your task is to analyze objects and return a complete ObjectDescription that includes:

1. BASIC INFO: name, description
2. VISUAL: colors (primary_color, rgb values, secondary_colors), surface properties
3. GEOMETRIC: shape type, dimensions (width/height/depth/radius), spatial relations
4. MATERIAL: primary_material, secondary_materials, mass, temperature
5. CAPABILITIES: functional_affordances (can_cut, can_contain, can_grasp, graspability)
6. SEMANTIC: category, subcategories, typical_locations, associated_activities
7. STATE: cleanliness, integrity, functional_state

Guidelines:
- Be precise and specific in your descriptions
- Use the provided enums for categorical values
- Estimate realistic physical properties (mass, dimensions)
- Consider the object's typical use context
- Assign appropriate confidence scores
- Think about what tasks the object affords

Return only the structured ObjectDescription - no additional text."""

class ValidationAgent(BaseAgent):
    """Base class for validation agents"""

    def __init__(self, **kwargs):
        super().__init__("validation", **kwargs)
        self.confidence_threshold = config.DEFAULT_CONFIDENCE_THRESHOLD

    def validate_analysis(self, analysis: ObjectDescription) -> tuple[bool, list[str]]:
        """Validate an object analysis"""
        errors = []

        # Check required fields
        if not analysis.name or not analysis.description:
            errors.append("Missing required fields: name or description")

        # Check confidence score
        if analysis.confidence_score < self.confidence_threshold:
            errors.append(f"Low confidence score: {analysis.confidence_score}")

        # Validate dimensions
        if analysis.geometric.shape.dimensions:
            dims = analysis.geometric.shape.dimensions
            if all(v is None for v in [dims.width, dims.height, dims.depth, dims.radius]):
                errors.append("No geometric dimensions specified")

        # Validate color information
        if not analysis.visual.colors.primary_color or analysis.visual.colors.primary_color == "unknown":
            errors.append("Primary color not specified")

        # Validate material properties
        if analysis.material.primary_material == "unknown":
            errors.append("Primary material not specified")

        return len(errors) == 0, errors