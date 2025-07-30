"""
Base exporter classes and interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from ..models import ObjectDescription


class BaseExporter(ABC):
    """Abstract base class for all exporters"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logging for the exporter"""
        import logging
        logger = logging.getLogger(f"soma.exporters.{self.__class__.__name__}")
        return logger

    @abstractmethod
    async def export(self,
                     object_description: ObjectDescription,
                     knowledge_graph_turtle: Optional[str] = None,
                     metadata: Dict[str, Any] = None) -> str:
        """
        Export the analysis results

        Args:
            object_description: The analyzed object
            knowledge_graph_turtle: Knowledge graph in turtle format
            metadata: Additional metadata

        Returns:
            Path to the exported file
        """
        pass

    def _generate_filename(self,
                           base_name: str,
                           extension: str,
                           timestamp: bool = True) -> Path:
        """Generate a filename with optional timestamp"""
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{ts}.{extension}"
        else:
            filename = f"{base_name}.{extension}"

        return self.output_dir / filename

    def _ensure_output_dir(self, subdir: Optional[str] = None) -> Path:
        """Ensure output directory exists, optionally with subdirectory"""
        if subdir:
            output_path = self.output_dir / subdir
        else:
            output_path = self.output_dir

        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


class FileExporter(BaseExporter):
    """Base class for file-based exporters"""

    def __init__(self, output_dir: Path, file_extension: str):
        super().__init__(output_dir)
        self.file_extension = file_extension.lstrip('.')

    def _write_file(self, content: str, filename: Path, encoding: str = 'utf-8') -> str:
        """Write content to file"""
        try:
            with open(filename, 'w', encoding=encoding) as f:
                f.write(content)

            self.logger.info(f"Exported to: {filename}")
            return str(filename)

        except Exception as e:
            self.logger.error(f"Failed to write file {filename}: {e}")
            raise

    def _write_binary_file(self, content: bytes, filename: Path) -> str:
        """Write binary content to file"""
        try:
            with open(filename, 'wb') as f:
                f.write(content)

            self.logger.info(f"Exported binary to: {filename}")
            return str(filename)

        except Exception as e:
            self.logger.error(f"Failed to write binary file {filename}: {e}")
            raise


class TemplateExporter(FileExporter):
    """Base class for template-based exporters"""

    def __init__(self, output_dir: Path, file_extension: str):
        super().__init__(output_dir, file_extension)
        self.templates = {}

    def _load_template(self, template_name: str) -> str:
        """Load a template (to be overridden by subclasses)"""
        if template_name in self.templates:
            return self.templates[template_name]
        else:
            raise ValueError(f"Template {template_name} not found")

    def _render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a template with context"""
        template = self._load_template(template_name)

        # Simple string formatting (can be enhanced with Jinja2 if needed)
        try:
            return template.format(**context)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            raise

    def _prepare_context(self,
                         object_description: ObjectDescription,
                         knowledge_graph_turtle: Optional[str] = None,
                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare context for template rendering"""
        return {
            'object_description': object_description,
            'knowledge_graph': knowledge_graph_turtle,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


class MultiFormatExporter(BaseExporter):
    """Exporter that supports multiple output formats"""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.format_exporters = {}

    def register_format_exporter(self, format_name: str, exporter: BaseExporter):
        """Register an exporter for a specific format"""
        self.format_exporters[format_name] = exporter

    async def export_format(self,
                            format_name: str,
                            object_description: ObjectDescription,
                            knowledge_graph_turtle: Optional[str] = None,
                            metadata: Dict[str, Any] = None) -> str:
        """Export in a specific format"""
        if format_name not in self.format_exporters:
            raise ValueError(f"Format {format_name} not supported")

        exporter = self.format_exporters[format_name]
        return await exporter.export(object_description, knowledge_graph_turtle, metadata)

    async def export_all_formats(self,
                                 object_description: ObjectDescription,
                                 knowledge_graph_turtle: Optional[str] = None,
                                 metadata: Dict[str, Any] = None) -> List[str]:
        """Export in all registered formats"""
        export_paths = []

        for format_name, exporter in self.format_exporters.items():
            try:
                path = await exporter.export(object_description, knowledge_graph_turtle, metadata)
                export_paths.append(path)
                self.logger.info(f"Successfully exported {format_name} format")
            except Exception as e:
                self.logger.error(f"Failed to export {format_name} format: {e}")

        return export_paths

    async def export(self,
                     object_description: ObjectDescription,
                     knowledge_graph_turtle: Optional[str] = None,
                     metadata: Dict[str, Any] = None) -> str:
        """Export using all formats (returns summary)"""
        export_paths = await self.export_all_formats(
            object_description, knowledge_graph_turtle, metadata
        )

        # Create a summary file
        summary_path = self._generate_filename("export_summary", "txt")
        summary_content = f"""Export Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Object: {object_description.name}

Exported Files ({len(export_paths)}):
"""
        for path in export_paths:
            summary_content += f"  • {Path(path).name}\n"

        with open(summary_path, 'w') as f:
            f.write(summary_content)

        return str(summary_path)