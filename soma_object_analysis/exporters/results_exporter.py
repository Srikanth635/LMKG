"""
Main results exporter that coordinates all export formats
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base_exporter import MultiFormatExporter
from .json_exporter import JSONExporter
from .html_exporter import HTMLExporter
from .markdown_exporter import MarkdownExporter
from .kg_exporter import KnowledgeGraphExporter
from ..models import ObjectDescription


class ResultsExporter(MultiFormatExporter):
    """Main results exporter that handles all output formats"""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self._setup_format_exporters()

    def _setup_format_exporters(self):
        """Initialize all format exporters"""
        # JSON exporter
        self.register_format_exporter("json", JSONExporter(self.output_dir))

        # HTML exporter
        self.register_format_exporter("html", HTMLExporter(self.output_dir))

        # Markdown exporter
        self.register_format_exporter("markdown", MarkdownExporter(self.output_dir))

        # Knowledge Graph exporter
        self.register_format_exporter("knowledge_graph", KnowledgeGraphExporter(self.output_dir))

    async def export_all(self,
                         object_description: ObjectDescription,
                         knowledge_graph_turtle: Optional[str] = None,
                         metadata: Dict[str, Any] = None) -> List[str]:
        """
        Export analysis results in all supported formats

        Returns:
            List of paths to exported files
        """

        self.logger.info(f"Starting export for: {object_description.name}")

        # Create timestamped session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"analysis_{object_description.name.lower().replace(' ', '_')}_{timestamp}"
        session_dir = self.output_dir / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        export_paths = []

        # Update all exporters to use session directory
        for exporter in self.format_exporters.values():
            exporter.output_dir = session_dir

        try:
            # Export in all formats
            format_paths = await self.export_all_formats(
                object_description, knowledge_graph_turtle, metadata
            )
            export_paths.extend(format_paths)

            # Create session metadata
            session_metadata = {
                "session_id": session_name,
                "timestamp": timestamp,
                "object_name": object_description.name,
                "export_formats": list(self.format_exporters.keys()),
                "exported_files": [Path(p).name for p in format_paths],
                "metadata": metadata or {},
                "system_info": {
                    "version": "0.1.0",
                    "export_time": datetime.now().isoformat()
                }
            }

            # Save session metadata
            metadata_path = session_dir / "session_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(session_metadata, f, indent=2, default=str)
            export_paths.append(str(metadata_path))

            # Create session README
            readme_path = await self._create_session_readme(
                session_dir, object_description, session_metadata
            )
            export_paths.append(readme_path)

            self.logger.info(f"Export completed: {len(export_paths)} files in {session_dir}")

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise

        return export_paths

    async def _create_session_readme(self,
                                     session_dir: Path,
                                     object_description: ObjectDescription,
                                     session_metadata: Dict[str, Any]) -> str:
        """Create a README file for the export session"""

        readme_content = f"""# Object Analysis Session: {object_description.name}

**Generated:** {session_metadata['timestamp']}  
**Session ID:** {session_metadata['session_id']}

## Object Summary

- **Name:** {object_description.name}
- **Category:** {object_description.semantic.category.value}
- **Primary Color:** {object_description.visual.colors.primary_color}
- **Shape:** {object_description.geometric.shape.primary_shape.value}
- **Material:** {object_description.material.primary_material.value}
- **Confidence:** {object_description.confidence_score:.2%}

## Files in This Session

### 📊 Analysis Results
- `object_description.json` - Complete structured analysis data
- `analysis_report.html` - Interactive web report
- `summary.md` - Human-readable markdown summary

### 🧠 Knowledge Graph
- `knowledge_graph.ttl` - SOMA knowledge graph (Turtle format)
- `knowledge_graph_visualization.html` - Interactive graph visualization

### 📋 Metadata
- `session_metadata.json` - Session and system metadata
- `README.md` - This file

## How to Use These Files

### View Interactive Report
Open `analysis_report.html` in any web browser for a complete interactive analysis report.

### Explore Knowledge Graph
Open `knowledge_graph_visualization.html` in a web browser to interactively explore the SOMA knowledge graph.

### Import Data
- Use `object_description.json` to import the structured analysis into other systems
- Use `knowledge_graph.ttl` to import into RDF/SPARQL databases or reasoning systems

### Read Summary
Open `summary.md` for a quick human-readable overview of the analysis.

## System Information

- **SOMA Analysis System Version:** {session_metadata['system_info']['version']}
- **Export Formats:** {', '.join(session_metadata['export_formats'])}
- **Total Files:** {len(session_metadata['exported_files'])}

## Next Steps

1. **Review the Analysis:** Start with `analysis_report.html` for a comprehensive overview
2. **Explore Relationships:** Use `knowledge_graph_visualization.html` to understand object relationships
3. **Integrate Data:** Import `object_description.json` into your applications
4. **Validate Results:** Check the confidence scores and validation results

---
*Generated by SOMA Object Analysis System*
"""

        readme_path = session_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        return str(readme_path)

    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats"""
        return list(self.format_exporters.keys())

    async def export_single_format(self,
                                   format_name: str,
                                   object_description: ObjectDescription,
                                   knowledge_graph_turtle: Optional[str] = None,
                                   metadata: Dict[str, Any] = None) -> str:
        """Export in a single specific format"""

        if format_name not in self.format_exporters:
            supported = ', '.join(self.get_supported_formats())
            raise ValueError(f"Format '{format_name}' not supported. Supported formats: {supported}")

        self.logger.info(f"Exporting {format_name} format for: {object_description.name}")

        return await self.export_format(
            format_name, object_description, knowledge_graph_turtle, metadata
        )

    def get_export_statistics(self) -> Dict[str, Any]:
        """Get statistics about export capabilities"""
        return {
            "supported_formats": self.get_supported_formats(),
            "format_count": len(self.format_exporters),
            "output_directory": str(self.output_dir),
            "exporters": {
                name: type(exporter).__name__
                for name, exporter in self.format_exporters.items()
            }
        }