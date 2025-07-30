"""
SOMA-specific knowledge graph implementation
"""

from typing import Dict, Any, Optional, List
from .base_kg import ExportableKnowledgeGraph, Triple
from ..config import config


class SOMAKnowledgeGraph(ExportableKnowledgeGraph):
    """Knowledge graph specifically aligned with SOMA ontology"""

    def __init__(self):
        super().__init__()
        self._setup_soma_namespaces()
        self.soma_instances = {}

    def _setup_soma_namespaces(self):
        """Setup standard SOMA namespaces"""
        self.namespaces.update(config.get_namespaces())

    def add_instance(self, instance_id: str, class_type: str, properties: Dict[str, Any] = None):
        """Add a SOMA instance"""
        # Create full URIs
        instance_uri = self.uri('instance', instance_id)
        class_uri = self.uri('soma', class_type)

        # Add type assertion
        self.add_triple(instance_uri, self.uri('rdf', 'type'), class_uri)

        # Store instance info
        self.instances[instance_id] = {
            'type': class_type,
            'properties': properties or {},
            'uri': instance_uri
        }

        self.soma_instances[instance_id] = {
            'class_type': class_type,
            'properties': properties or {}
        }

        # Add properties if provided
        if properties:
            for prop, value in properties.items():
                self.add_property(instance_id, prop, value)

    def add_property(self, instance_id: str, property_name: str, value: Any):
        """Add a property to a SOMA instance"""
        subject = self.uri('instance', instance_id)

        # Format the value
        if isinstance(value, str) and not value.startswith('http'):
            # String literal
            obj = f'"{value}"'
        elif isinstance(value, (int, float)):
            # Numeric literal
            datatype = "double" if isinstance(value, float) else "integer"
            obj = f'"{value}"^^{self.uri("xsd", datatype)}'
        elif isinstance(value, bool):
            # Boolean literal
            obj = f'"{str(value).lower()}"^^{self.uri("xsd", "boolean")}'
        elif isinstance(value, list):
            # Handle lists by creating multiple triples
            for item in value:
                self.add_property(instance_id, property_name, item)
            return
        else:
            # Assume it's a reference to another instance
            obj = self.uri('instance', str(value))

        # Determine predicate URI
        if property_name.startswith('soma:'):
            predicate = self.uri('soma', property_name[5:])
        elif property_name.startswith('has'):
            predicate = self.uri('soma', property_name)
        else:
            predicate = self.uri('soma', f'has{property_name.capitalize()}')

        self.add_triple(subject, predicate, obj)

    def add_relation(self, subject_id: str, relation: str, object_id: str):
        """Add a relation between two SOMA instances"""
        subject_uri = self.uri('instance', subject_id)
        object_uri = self.uri('instance', object_id)
        relation_uri = self.uri('soma', relation)

        self.add_triple(subject_uri, relation_uri, object_uri)

    def get_soma_instances_by_type(self, soma_class: str) -> List[str]:
        """Get all instances of a specific SOMA class"""
        class_uri = self.uri('soma', soma_class)
        return self.get_instances_of_type(class_uri)

    def get_soma_instance_info(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a SOMA instance"""
        return self.soma_instances.get(instance_id)

    def add_affordance(self, affordance_id: str, bearer_id: str, task_type: str,
                       properties: Dict[str, Any] = None):
        """Add a SOMA affordance relation"""
        # Create affordance instance
        self.add_instance(affordance_id, "Affordance", properties)

        # Connect to bearer
        self.add_relation(affordance_id, "affordsBearer", bearer_id)

        # Connect to task type
        if task_type:
            self.add_property(affordance_id, "affordsTask", task_type)

    def add_quality_region(self, object_id: str, quality_type: str,
                           region_id: str, properties: Dict[str, Any] = None):
        """Add a quality region (e.g., ColorRegion, ShapeRegion)"""
        region_class = f"{quality_type}Region"
        self.add_instance(region_id, region_class, properties)

        # Connect to object
        relation_name = f"has{quality_type}Region"
        self.add_relation(object_id, relation_name, region_id)

    def add_spatial_relation(self, subject_id: str, object_id: str, relation_type: str):
        """Add spatial relations between objects"""
        spatial_relations = {
            "supports": "isSupportedBy",
            "contains": "isContainedIn",
            "adjacent_to": "isAdjacentTo",
            "on_top_of": "isOnTopOf"
        }

        # Add forward relation
        self.add_relation(subject_id, relation_type, object_id)

        # Add inverse relation if defined
        if relation_type in spatial_relations:
            inverse_relation = spatial_relations[relation_type]
            self.add_relation(object_id, inverse_relation, subject_id)

    def get_object_properties(self, instance_id: str) -> Dict[str, List[str]]:
        """Get all properties of a SOMA object instance"""
        instance_uri = self.uri('instance', instance_id)
        properties = self.get_instance_properties(instance_uri)

        # Clean up property names and values
        cleaned_properties = {}
        for prop_uri, values in properties.items():
            # Extract property name from URI
            if '#' in prop_uri:
                prop_name = prop_uri.split('#')[-1]
            else:
                prop_name = prop_uri.split('/')[-1]

            # Clean up values
            cleaned_values = []
            for value in values:
                if value.startswith('"') and value.endswith('"'):
                    # Remove quotes from literals
                    cleaned_value = value[1:-1]
                    if '^^' in cleaned_value:
                        cleaned_value = cleaned_value.split('^^')[0]
                    cleaned_values.append(cleaned_value)
                elif '#' in value:
                    # Extract instance name from URI
                    cleaned_values.append(value.split('#')[-1])
                else:
                    cleaned_values.append(value)

            cleaned_properties[prop_name] = cleaned_values

        return cleaned_properties

    def get_related_objects(self, instance_id: str, relation_type: str = None) -> List[Dict[str, str]]:
        """Get objects related to the given instance"""
        instance_uri = self.uri('instance', instance_id)

        if relation_type:
            relation_uri = self.uri('soma', relation_type)
            triples = self.query_pattern(subject=instance_uri, predicate=relation_uri)
        else:
            triples = self.get_triples_by_subject(instance_uri)

        related_objects = []
        for triple in triples:
            if triple.object.startswith(self.namespaces['instance']):
                object_id = triple.object.split('#')[-1]
                relation_name = triple.predicate.split('#')[-1] if '#' in triple.predicate else triple.predicate

                related_objects.append({
                    'object_id': object_id,
                    'relation': relation_name,
                    'object_uri': triple.object
                })

        return related_objects

    def validate_soma_consistency(self) -> List[str]:
        """Validate consistency with SOMA ontology rules"""
        errors = []

        # Check that all instances have proper types
        for instance_id, info in self.soma_instances.items():
            instance_uri = self.uri('instance', instance_id)
            type_triples = self.query_pattern(
                subject=instance_uri,
                predicate=self.uri('rdf', 'type')
            )

            if not type_triples:
                errors.append(f"Instance {instance_id} has no type declaration")

            # Check for SOMA-specific class types
            soma_types = [t for t in type_triples if t.object.startswith(self.namespaces['soma'])]
            if not soma_types:
                errors.append(f"Instance {instance_id} has no SOMA type")

        # Check affordance consistency
        affordance_instances = self.get_soma_instances_by_type("Affordance")
        for affordance_uri in affordance_instances:
            affordance_id = affordance_uri.split('#')[-1]

            # Affordances should have bearers
            bearer_triples = self.query_pattern(
                subject=affordance_uri,
                predicate=self.uri('soma', 'affordsBearer')
            )
            if not bearer_triples:
                errors.append(f"Affordance {affordance_id} has no bearer")

        return errors

    def get_soma_statistics(self) -> Dict[str, Any]:
        """Get statistics specific to SOMA knowledge graph"""
        base_stats = self.get_statistics()

        # Count instances by SOMA class
        soma_class_counts = {}
        for instance_info in self.soma_instances.values():
            class_type = instance_info['class_type']
            soma_class_counts[class_type] = soma_class_counts.get(class_type, 0) + 1

        # Count SOMA-specific relations
        soma_relations = {}
        for triple in self.triples:
            if triple.predicate.startswith(self.namespaces['soma']):
                relation_name = triple.predicate.split('#')[-1]
                soma_relations[relation_name] = soma_relations.get(relation_name, 0) + 1

        return {
            **base_stats,
            "soma_class_counts": soma_class_counts,
            "soma_relation_counts": soma_relations,
            "affordance_count": soma_class_counts.get("Affordance", 0),
            "quality_region_count": sum(1 for k in soma_class_counts.keys() if k.endswith("Region"))
        }