"""
Export modules for SOMA object analysis results
"""

from .base_exporter import BaseExporter, FileExporter, TemplateExporter, MultiFormatExporter
from .results_exporter import ResultsExporter
from .json_exporter import JSONExporter, CompactJSONExporter, JSONLDExporter
from .html_exporter import HTMLExporter
from .markdown_exporter import MarkdownExporter, DetailedMarkdownExporter
from .kg_exporter import KnowledgeGraphExporter, RDFXMLExporter, JSONLDKGExporter

__all__ = [
    # Base classes
    "BaseExporter",
    "FileExporter",
    "TemplateExporter",
    "MultiFormatExporter",

    # Main exporter
    "ResultsExporter",

    # Format-specific exporters
    "JSONExporter",
    "CompactJSONExporter",
    "JSONLDExporter",
    "HTMLExporter",
    "MarkdownExporter",
    "DetailedMarkdownExporter",
    "KnowledgeGraphExporter",
    "RDFXMLExporter",
    "JSONLDKGExporter",
]


# Convenience functions
def get_available_exporters():
    """Get list of available exporter classes"""
    return {
        "json": JSONExporter,
        "json_compact": CompactJSONExporter,
        "json_ld": JSONLDExporter,
        "html": HTMLExporter,
        "markdown": MarkdownExporter,
        "markdown_detailed": DetailedMarkdownExporter,
        "knowledge_graph": KnowledgeGraphExporter,
        "rdf_xml": RDFXMLExporter,
        "jsonld_kg": JSONLDKGExporter,
    }


def create_exporter(format_name: str, output_dir, **kwargs):
    """
    Factory function to create specific exporter

    Args:
        format_name: Name of the export format
        output_dir: Output directory path
        **kwargs: Additional arguments for exporter

    Returns:
        Exporter instance

    Raises:
        ValueError: If format_name is not supported
    """
    exporters = get_available_exporters()

    if format_name not in exporters:
        available = ", ".join(exporters.keys())
        raise ValueError(f"Unsupported export format '{format_name}'. Available: {available}")

    exporter_class = exporters[format_name]
    return exporter_class(output_dir, **kwargs)


def get_supported_formats():
    """Get list of supported export format names"""
    return list(get_available_exporters().keys())