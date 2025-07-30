"""
Knowledge graph generator for SOMA object descriptions
"""

import uuid
from datetime import datetime
from typing import Dict, Any

from .soma_kg import SOMAKnowledgeGraph
from ..models import ObjectDescription
from ..config import config


class KnowledgeGraphGenerator:
    """Generates SOMA-aligned knowledge graphs from ObjectDescription instances"""

    def __init__(self):
        self.logger = self._setup_logger()

    def _setup_logger(self):
        import logging
        logger = logging.getLogger(f"soma.kg.generator")
        logger.setLevel(getattr(logging, config.LOG_LEVEL))
        return logger

    def create_from_object_description(self, obj_desc: ObjectDescription) -> SOMAKnowledgeGraph:
        """Create knowledge graph from ObjectDescription"""
        self.logger.info(f"Generating KG for: {obj_desc.name}")

        kg = SOMAKnowledgeGraph()

        # Generate unique instance ID
        obj_id = self._generate_instance_id(obj_desc.name)

        # Add main object instance
        self._add_main_object(kg, obj_id, obj_desc)

        # Add visual properties
        self._add_visual_properties(kg, obj_id, obj_desc)

        # Add geometric properties
        self._add_geometric_properties(kg, obj_id, obj_desc)

        # Add material properties
        self._add_material_properties(kg, obj_id, obj_desc)

        # Add capabilities
        self._add_capabilities(kg, obj_id, obj_desc)

        # Add semantic information
        self._add_semantic_properties(kg, obj_id, obj_desc)

        # Add state information
        self._add_state_properties(kg, obj_id, obj_desc)

        self.logger.info(f"Generated KG with {len(kg.triples)} triples")
        return kg

    def _generate_instance_id(self, name: str) -> str:
        """Generate safe instance ID from name"""
        import re
        # Clean the name
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower().strip())
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{safe_name}_{timestamp}_{uuid.uuid4().hex[:6]}"

    def _add_main_object(self, kg: SOMAKnowledgeGraph, obj_id: str, obj_desc: ObjectDescription):
        """Add main object instance"""
        # Add instance with SOMA category
        kg.add_instance(obj_id, obj_desc.semantic.category.value)

        # Add basic properties
        kg.add_property(obj_id, "hasName", obj_desc.name)
        kg.add_property(obj_id, "hasDescription", obj_desc.description)
        kg.add_property(obj_id, "hasConfidenceScore", obj_desc.confidence_score)

        if obj_desc.timestamp:
            kg.add_property(obj_id, "hasTimestamp", obj_desc.timestamp)

    def _add_visual_properties(self, kg: SOMAKnowledgeGraph, obj_id: str, obj_desc: ObjectDescription):
        """Add visual properties to knowledge graph"""
        colors = obj_desc.visual.colors

        # Add color information
        if colors.primary_color and colors.primary_color != "unknown":
            color_id = f"{obj_id}_color"

            # Use SOMA color classes if available
            soma_colors = ["RedColor", "GreenColor", "BlueColor"]
            color_class = f"{colors.primary_color.capitalize()}Color"

            if color_class in soma_colors:
                kg.add_instance(color_id, color_class)
            else:
                kg.add_instance(color_id, "Color")
                kg.add_property(color_id, "hasColorValue", colors.primary_color)

            # Add RGB values if available
            if colors.rgb:
                rgb_value = f"{colors.rgb.r},{colors.rgb.g},{colors.rgb.b}"
                kg.add_property(color_id, "hasRGBValue", rgb_value)

            # Add HSV values if available
            if colors.hsv:
                hsv_value = f"{colors.hsv.h},{colors.hsv.s},{colors.hsv.v}"
                kg.add_property(color_id, "hasHSVValue", hsv_value)

            kg.add_relation(obj_id, "hasColor", color_id)

        # Add secondary colors
        for i, sec_color in enumerate(colors.secondary_colors):
            sec_color_id = f"{obj_id}_secondary_color_{i}"
            kg.add_instance(sec_color_id, "Color")
            kg.add_property(sec_color_id, "hasColorValue", sec_color)
            kg.add_relation(obj_id, "hasSecondaryColor", sec_color_id)

    def _add_geometric_properties(self, kg: SOMAKnowledgeGraph, obj_id: str, obj_desc: ObjectDescription):
        """Add geometric properties to knowledge graph"""
        geometry = obj_desc.geometric

        # Add shape information
        shape_id = f"{obj_id}_shape"
        kg.add_instance(shape_id, geometry.shape.primary_shape.value)

        # Add dimensions
        dims = geometry.shape.dimensions
        if dims.width is not None:
            kg.add_property(shape_id, "hasWidth", dims.width)
        if dims.height is not None:
            kg.add_property(shape_id, "hasHeight", dims.height)
        if dims.depth is not None:
            kg.add_property(shape_id, "hasDepth", dims.depth)
        if dims.radius is not None:
            kg.add_property(shape_id, "hasRadius", dims.radius)
        if dims.length is not None:
            kg.add_property(shape_id, "hasLength", dims.length)

        # Add volume if calculable
        volume = dims.volume()
        if volume:
            kg.add_property(shape_id, "hasVolume", volume)

        kg.add_relation(obj_id, "hasShape", shape_id)

        # Add spatial relations
        spatial = geometry.spatial_relations

        for supporter in spatial.supported_by:
            supporter_id = self._generate_instance_id(supporter)
            kg.add_relation(obj_id, "isSupportedBy", supporter_id)

        for supported in spatial.supports:
            supported_id = self._generate_instance_id(supported)
            kg.add_relation(obj_id, "supports", supported_id)

        for container in spatial.contained_in:
            container_id = self._generate_instance_id(container)
            kg.add_relation(obj_id, "isContainedIn", container_id)

        for contained in spatial.contains:
            contained_id = self._generate_instance_id(contained)
            kg.add_relation(obj_id, "contains", contained_id)

    def _add_material_properties(self, kg: SOMAKnowledgeGraph, obj_id: str, obj_desc: ObjectDescription):
        """Add material properties to knowledge graph"""
        material = obj_desc.material

        # Add primary material
        material_id = f"{obj_id}_material"
        kg.add_instance(material_id, "Material")
        kg.add_property(material_id, "materialType", material.primary_material.value)
        kg.add_relation(obj_id, "hasMaterial", material_id)

        # Add secondary materials
        for i, sec_material in enumerate(material.secondary_materials):
            sec_mat_id = f"{obj_id}_secondary_material_{i}"
            kg.add_instance(sec_mat_id, "Material")
            kg.add_property(sec_mat_id, "materialType", sec_material.value)
            kg.add_relation(obj_id, "hasSecondaryMaterial", sec_mat_id)

        # Add mass attribute
        if material.mass is not None:
            mass_id = f"{obj_id}_mass"
            kg.add_instance(mass_id, "MassAttribute")
            kg.add_property(mass_id, "hasMassValue", material.mass)
            kg.add_relation(obj_id, "hasMassAttribute", mass_id)

        # Add temperature
        if material.temperature is not None:
            temp_id = f"{obj_id}_temperature"
            kg.add_instance(temp_id, "Temperature")
            kg.add_property(temp_id, "hasTemperatureValue", material.temperature)
            kg.add_relation(obj_id, "hasTemperature", temp_id)

        # Add mechanical properties
        mechanical = obj_desc.mechanical

        if mechanical.static_friction is not None:
            friction_id = f"{obj_id}_static_friction"
            kg.add_instance(friction_id, "StaticFrictionAttribute")
            kg.add_property(friction_id, "hasFrictionValue", mechanical.static_friction)
            kg.add_relation(obj_id, "hasFrictionAttribute", friction_id)

        if mechanical.kinetic_friction is not None:
            friction_id = f"{obj_id}_kinetic_friction"
            kg.add_instance(friction_id, "KineticFrictionAttribute")
            kg.add_property(friction_id, "hasFrictionValue", mechanical.kinetic_friction)
            kg.add_relation(obj_id, "hasFrictionAttribute", friction_id)

    def _add_capabilities(self, kg: SOMAKnowledgeGraph, obj_id: str, obj_desc: ObjectDescription):
        """Add capability information to knowledge graph"""
        caps = obj_desc.capabilities.functional_affordances

        # Add graspability
        if caps.graspability is not None:
            grasp_id = f"{obj_id}_graspability"
            kg.add_instance(grasp_id, "Graspability")
            kg.add_property(grasp_id, "hasGraspabilityValue", caps.graspability)
            kg.add_relation(obj_id, "hasCapability", grasp_id)

        # Add specific capabilities
        if caps.can_cut:
            cut_id = f"{obj_id}_can_cut"
            kg.add_instance(cut_id, "CanCut")
            kg.add_relation(obj_id, "hasCapability", cut_id)

        if caps.can_contain:
            contain_id = f"{obj_id}_containment"
            kg.add_instance(contain_id, "Containment")
            kg.add_relation(obj_id, "hasCapability", contain_id)

        if caps.can_grasp:
            # This is implicit in graspability but can be explicit
            kg.add_property(obj_id, "canBeGrasped", True)

        # Add task affordances
        tasks = obj_desc.capabilities.task_affordances

        for i, task in enumerate(tasks.primary_tasks):
            task_id = f"{obj_id}_primary_task_{i}"
            kg.add_instance(task_id, "Task")
            kg.add_property(task_id, "taskName", task)
            kg.add_relation(obj_id, "affordsTask", task_id)

        for i, task in enumerate(tasks.secondary_tasks):
            task_id = f"{obj_id}_secondary_task_{i}"
            kg.add_instance(task_id, "Task")
            kg.add_property(task_id, "taskName", task)
            kg.add_relation(obj_id, "affordsSecondaryTask", task_id)

    def _add_semantic_properties(self, kg: SOMAKnowledgeGraph, obj_id: str, obj_desc: ObjectDescription):
        """Add semantic properties to knowledge graph"""
        semantic = obj_desc.semantic

        # Add subcategories
        for i, subcat in enumerate(semantic.subcategories):
            kg.add_property(obj_id, f"hasSubcategory_{i}", subcat)

        # Add typical locations
        for i, location in enumerate(semantic.typical_locations):
            location_id = f"{obj_id}_location_{i}"
            kg.add_instance(location_id, "Location")
            kg.add_property(location_id, "locationName", location)
            kg.add_relation(obj_id, "typicallyFoundAt", location_id)

        # Add associated activities
        for i, activity in enumerate(semantic.associated_activities):
            activity_id = f"{obj_id}_activity_{i}"
            kg.add_instance(activity_id, "Activity")
            kg.add_property(activity_id, "activityName", activity)
            kg.add_relation(obj_id, "associatedWithActivity", activity_id)

    def _add_state_properties(self, kg: SOMAKnowledgeGraph, obj_id: str, obj_desc: ObjectDescription):
        """Add state properties to knowledge graph"""
        state = obj_desc.state

        # Add cleanliness state
        if state.cleanliness.value != "unknown":
            clean_id = f"{obj_id}_cleanliness"
            kg.add_instance(clean_id, state.cleanliness.value)
            kg.add_relation(obj_id, "hasQuality", clean_id)

        # Add integrity
        if state.integrity != 1.0:
            kg.add_property(obj_id, "hasIntegrityValue", state.integrity)

        # Add functional state
        if state.functional_state:
            kg.add_property(obj_id, "hasFunctionalState", state.functional_state)

        # Add device state if applicable
        if state.device_state:
            device_state_id = f"{obj_id}_device_state"
            kg.add_instance(device_state_id, state.device_state.value)
            kg.add_relation(obj_id, "hasDeviceState", device_state_id)