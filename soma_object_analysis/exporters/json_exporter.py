"""
JSON exporter for object analysis results
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path

from .base_exporter import FileExporter
from ..models import ObjectDescription


class JSONExporter(FileExporter):
    """Exports analysis results as JSON"""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, "json")

    async def export(self,
                     object_description: ObjectDescription,
                     knowledge_graph_turtle: Optional[str] = None,
                     metadata: Dict[str, Any] = None) -> str:
        """Export object description as JSON"""

        # Prepare the data structure
        export_data = {
            "object_analysis": object_description.dict(),
            "knowledge_graph": {
                "turtle": knowledge_graph_turtle,
                "available": knowledge_graph_turtle is not None
            },
            "metadata": metadata or {},
            "export_info": {
                "format": "json",
                "version": "1.0",
                "exporter": "JSONExporter"
            }
        }

        # Generate filename
        safe_name = "".join(c for c in object_description.name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_').lower()
        filename = self._generate_filename(f"object_description_{safe_name}", "json", timestamp=False)

        # Convert to JSON with proper formatting
        json_content = json.dumps(export_data, indent=2, ensure_ascii=False, default=self._json_serializer)

        # Write to file
        return self._write_file(json_content, filename)

    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects"""
        if hasattr(obj, 'isoformat'):
            # Handle datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            # Handle custom objects by converting to dict
            return obj.__dict__
        elif isinstance(obj, set):
            # Handle sets
            return list(obj)
        else:
            # Fallback to string representation
            return str(obj)


class CompactJSONExporter(JSONExporter):
    """Exports analysis results as compact JSON (no formatting)"""

    async def export(self,
                     object_description: ObjectDescription,
                     knowledge_graph_turtle: Optional[str] = None,
                     metadata: Dict[str, Any] = None) -> str:
        """Export object description as compact JSON"""

        # Prepare the data structure (same as parent)
        export_data = {
            "object_analysis": object_description.dict(),
            "knowledge_graph": {
                "turtle": knowledge_graph_turtle,
                "available": knowledge_graph_turtle is not None
            },
            "metadata": metadata or {},
            "export_info": {
                "format": "json_compact",
                "version": "1.0",
                "exporter": "CompactJSONExporter"
            }
        }

        # Generate filename
        safe_name = "".join(c for c in object_description.name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_').lower()
        filename = self._generate_filename(f"object_description_compact_{safe_name}", "json", timestamp=False)

        # Convert to compact JSON
        json_content = json.dumps(export_data, separators=(',', ':'), ensure_ascii=False, default=self._json_serializer)

        # Write to file
        return self._write_file(json_content, filename)


class JSONLDExporter(JSONExporter):
    """Exports analysis results as JSON-LD with semantic context"""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.file_extension = "jsonld"

    async def export(self,
                     object_description: ObjectDescription,
                     knowledge_graph_turtle: Optional[str] = None,
                     metadata: Dict[str, Any] = None) -> str:
        """Export object description as JSON-LD"""

        # Create JSON-LD context
        context = {
            "@vocab": "http://www.ease-crc.org/ont/SOMA.owl#",
            "soma": "http://www.ease-crc.org/ont/SOMA.owl#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "name": "rdfs:label",
            "description": "rdfs:comment",
            "hasColor": "soma:hasColor",
            "hasShape": "soma:hasShape",
            "hasMaterial": "soma:hasMaterial",
            "hasCapability": "soma:hasCapability"
        }

        # Create the main object description in JSON-LD format
        jsonld_data = {
            "@context": context,
            "@type": object_description.semantic.category.value,
            "@id": f"_:object_{object_description.name.replace(' ', '_').lower()}",
            "name": object_description.name,
            "description": object_description.description,
            "confidenceScore": object_description.confidence_score,

            # Visual properties
            "hasColor": {
                "@type": "Color",
                "primaryColor": object_description.visual.colors.primary_color,
                "rgbValue": f"{object_description.visual.colors.rgb.r},{object_description.visual.colors.rgb.g},{object_description.visual.colors.rgb.b}" if object_description.visual.colors.rgb else None
            },

            # Geometric properties
            "hasShape": {
                "@type": object_description.geometric.shape.primary_shape.value,
                "dimensions": object_description.geometric.shape.dimensions.dict()
            },

            # Material properties
            "hasMaterial": {
                "@type": "Material",
                "materialType": object_description.material.primary_material.value,
                "mass": object_description.material.mass
            },

            # Capabilities
            "hasCapability": []
        }

        # Add capabilities
        caps = object_description.capabilities.functional_affordances
        if caps.graspability is not None:
            jsonld_data["hasCapability"].append({
                "@type": "Graspability",
                "graspabilityValue": caps.graspability
            })

        if caps.can_cut:
            jsonld_data["hasCapability"].append({"@type": "CanCut"})

        if caps.can_contain:
            jsonld_data["hasCapability"].append({"@type": "Containment"})

        # Add metadata
        export_data = {
            "@context": context,
            "@graph": [jsonld_data],
            "metadata": {
                "export_info": {
                    "format": "json-ld",
                    "version": "1.0",
                    "exporter": "JSONLDExporter"
                },
                "analysis_metadata": metadata or {}
            }
        }

        # Generate filename
        safe_name = "".join(c for c in object_description.name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_').lower()
        filename = self._generate_filename(f"object_description_{safe_name}", "jsonld", timestamp=False)

        # Convert to JSON-LD
        jsonld_content = json.dumps(export_data, indent=2, ensure_ascii=False, default=self._json_serializer)

        # Write to file
        return self._write_file(jsonld_content, filename)