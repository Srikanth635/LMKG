"""
Validation agent for object analysis results
"""

from typing import Tuple, List
from .base_agent import ValidationAgent, AgentState
from ..models import ObjectDescription
from ..config import config


class SOMAValidationAgent(ValidationAgent):
    """SOMA-specific validation agent"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validation_rules = self._setup_validation_rules()

    def _setup_validation_rules(self) -> dict:
        """Setup validation rules for SOMA objects"""
        return {
            "required_fields": ["name", "description"],
            "required_properties": ["visual", "geometric", "material", "semantic"],
            "confidence_threshold": self.confidence_threshold,
            "dimension_rules": {
                "min_dimension": 0.001,  # 1mm minimum
                "max_dimension": 100.0,  # 100m maximum
                "logical_ratios": True  # Check dimension ratios make sense
            },
            "mass_rules": {
                "min_mass": 0.001,  # 1g minimum
                "max_mass": 10000.0,  # 10kg maximum for typical objects
                "density_check": True  # Check mass vs volume makes sense
            }
        }

    async def process(self, state: AgentState) -> AgentState:
        """Process validation of the analysis"""
        if not state.object_analysis:
            state.errors.append("No analysis to validate")
            return state

        self.logger.info("Starting SOMA validation")

        try:
            is_valid, validation_errors = self.validate_analysis(state.object_analysis)

            if not is_valid:
                state.errors.extend(validation_errors)
                self.logger.warning(f"Validation failed with {len(validation_errors)} errors")
            else:
                self.logger.info("Validation passed successfully")

            state.metadata["validation_passed"] = is_valid
            state.metadata["validation_errors"] = len(validation_errors)
            state.metadata["validation_details"] = validation_errors

        except Exception as e:
            error_msg = f"Validation process failed: {str(e)}"
            state.errors.append(error_msg)
            self.logger.error(error_msg)

        return state

    def validate_analysis(self, analysis: ObjectDescription) -> Tuple[bool, List[str]]:
        """Comprehensive validation of object analysis"""
        errors = []

        # Basic field validation
        errors.extend(self._validate_basic_fields(analysis))

        # Visual validation
        errors.extend(self._validate_visual_properties(analysis))

        # Geometric validation
        errors.extend(self._validate_geometric_properties(analysis))

        # Material validation
        errors.extend(self._validate_material_properties(analysis))

        # Capability validation
        errors.extend(self._validate_capabilities(analysis))

        # Semantic validation
        errors.extend(self._validate_semantic_properties(analysis))

        # Cross-property validation
        errors.extend(self._validate_cross_properties(analysis))

        return len(errors) == 0, errors

    def _validate_basic_fields(self, analysis: ObjectDescription) -> List[str]:
        """Validate basic required fields"""
        errors = []

        if not analysis.name or not analysis.name.strip():
            errors.append("Object name is required and cannot be empty")

        if not analysis.description or not analysis.description.strip():
            errors.append("Object description is required and cannot be empty")

        if analysis.confidence_score < self.confidence_threshold:
            errors.append(
                f"Confidence score {analysis.confidence_score:.2f} below threshold {self.confidence_threshold}")

        if analysis.confidence_score < 0 or analysis.confidence_score > 1:
            errors.append(f"Confidence score {analysis.confidence_score} must be between 0 and 1")

        return errors

    def _validate_visual_properties(self, analysis: ObjectDescription) -> List[str]:
        """Validate visual properties"""
        errors = []

        colors = analysis.visual.colors

        if not colors.primary_color or colors.primary_color.lower() in ["unknown", "none", ""]:
            errors.append("Primary color must be specified")

        if colors.rgb:
            if not (0 <= colors.rgb.r <= 255 and 0 <= colors.rgb.g <= 255 and 0 <= colors.rgb.b <= 255):
                errors.append("RGB values must be between 0 and 255")

        if colors.hsv:
            if not (0 <= colors.hsv.h <= 360 and 0 <= colors.hsv.s <= 100 and 0 <= colors.hsv.v <= 100):
                errors.append("HSV values out of valid range")

        return errors

    def _validate_geometric_properties(self, analysis: ObjectDescription) -> List[str]:
        """Validate geometric properties"""
        errors = []

        dims = analysis.geometric.shape.dimensions
        rules = self.validation_rules["dimension_rules"]

        # Check if at least one dimension is specified
        has_dimensions = any([
            dims.width, dims.height, dims.depth, dims.radius, dims.length
        ])

        if not has_dimensions:
            errors.append("At least one geometric dimension must be specified")

        # Validate dimension ranges
        for dim_name, dim_value in [
            ("width", dims.width), ("height", dims.height), ("depth", dims.depth),
            ("radius", dims.radius), ("length", dims.length)
        ]:
            if dim_value is not None:
                if dim_value < rules["min_dimension"]:
                    errors.append(f"{dim_name} {dim_value}m too small (min: {rules['min_dimension']}m)")
                if dim_value > rules["max_dimension"]:
                    errors.append(f"{dim_name} {dim_value}m too large (max: {rules['max_dimension']}m)")

        # Logical dimension checks
        if rules["logical_ratios"] and dims.width and dims.height:
            aspect_ratio = max(dims.width, dims.height) / min(dims.width, dims.height)
            if aspect_ratio > 1000:  # Extremely thin/wide objects
                errors.append(f"Extreme aspect ratio {aspect_ratio:.1f} may be unrealistic")

        return errors

    def _validate_material_properties(self, analysis: ObjectDescription) -> List[str]:
        """Validate material properties"""
        errors = []

        material = analysis.material
        rules = self.validation_rules["mass_rules"]

        # Validate mass
        if material.mass is not None:
            if material.mass < rules["min_mass"]:
                errors.append(f"Mass {material.mass}kg too small (min: {rules['min_mass']}kg)")
            if material.mass > rules["max_mass"]:
                errors.append(f"Mass {material.mass}kg too large for typical object (max: {rules['max_mass']}kg)")

        # Validate density vs volume
        if rules["density_check"] and material.mass and material.density:
            dims = analysis.geometric.shape.dimensions
            volume = dims.volume()
            if volume:
                calculated_mass = material.density * volume
                mass_ratio = abs(calculated_mass - material.mass) / material.mass
                if mass_ratio > 0.5:  # 50% difference threshold
                    errors.append(
                        f"Mass {material.mass}kg inconsistent with density {material.density}kg/m³ and volume {volume:.6f}m³")

        # Temperature validation
        if material.temperature is not None:
            if material.temperature < -273.15:  # Absolute zero
                errors.append("Temperature cannot be below absolute zero")
            if material.temperature > 1000:  # Very hot for typical objects
                errors.append(f"Temperature {material.temperature}°C unusually high for typical objects")

        return errors

    def _validate_capabilities(self, analysis: ObjectDescription) -> List[str]:
        """Validate capability properties"""
        errors = []

        caps = analysis.capabilities.functional_affordances

        # Validate graspability
        if caps.graspability is not None:
            if not (0 <= caps.graspability <= 1):
                errors.append("Graspability must be between 0 and 1")

        # Logical capability checks
        if caps.can_cut and analysis.material.primary_material.value in ["fabric", "rubber"]:
            errors.append("Soft materials typically cannot cut effectively")

        if caps.can_contain and analysis.geometric.shape.primary_shape.value == "SphereShape":
            # Spheres can't typically contain things unless hollow
            if not any("hollow" in task.lower() for task in analysis.capabilities.task_affordances.primary_tasks):
                errors.append("Solid spheres typically cannot contain objects")

        return errors

    def _validate_semantic_properties(self, analysis: ObjectDescription) -> List[str]:
        """Validate semantic properties"""
        errors = []

        semantic = analysis.semantic

        # Validate category consistency
        category = semantic.category.value
        primary_tasks = analysis.capabilities.task_affordances.primary_tasks

        # Category-task consistency checks
        if category == "Tool" and not any(task in ["cutting", "building", "repair", "work"] for task in primary_tasks):
            errors.append("Tool category should have tool-related primary tasks")

        if category == "Container" and not analysis.capabilities.functional_affordances.can_contain:
            errors.append("Container category should have can_contain=True")

        # Location-category consistency
        locations = semantic.typical_locations
        if category == "Appliance" and not any(loc in ["kitchen", "home", "office"] for loc in locations):
            errors.append("Appliances typically found in kitchen, home, or office")

        return errors

    def _validate_cross_properties(self, analysis: ObjectDescription) -> List[str]:
        """Validate consistency across different property groups"""
        errors = []

        # Material-color consistency
        material = analysis.material.primary_material.value
        color = analysis.visual.colors.primary_color.lower()

        inconsistent_combinations = [
            (["metal", "steel"], ["green", "blue", "purple"]),  # Metals rarely these colors naturally
            (["wood"], ["silver", "chrome", "metallic"]),  # Wood not metallic
            (["glass"], ["opaque"]),  # Glass typically transparent
        ]

        for materials, colors in inconsistent_combinations:
            if any(mat in material.lower() for mat in materials) and color in colors:
                errors.append(f"Unusual combination: {material} material with {color} color")

        # Size-mass consistency
        dims = analysis.geometric.shape.dimensions
        mass = analysis.material.mass

        if dims and mass and dims.volume():
            volume = dims.volume()
            implied_density = mass / volume

            # Density sanity checks
            if material == "metal" and implied_density < 1000:  # Metals are dense
                errors.append(f"Metal object unusually light: density {implied_density:.1f} kg/m³")
            elif material in ["wood", "plastic"] and implied_density > 2000:  # Light materials
                errors.append(f"{material} object unusually heavy: density {implied_density:.1f} kg/m³")

        return errors

    def get_validation_summary(self, analysis: ObjectDescription) -> dict:
        """Get a summary of validation results"""
        is_valid, errors = self.validate_analysis(analysis)

        return {
            "valid": is_valid,
            "error_count": len(errors),
            "errors": errors,
            "confidence_score": analysis.confidence_score,
            "passes_confidence_threshold": analysis.confidence_score >= self.confidence_threshold,
            "validation_timestamp": analysis.timestamp,
            "validator_version": "1.0"
        }