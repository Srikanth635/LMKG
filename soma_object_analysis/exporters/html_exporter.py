"""
HTML exporter for interactive analysis reports
"""

from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .base_exporter import TemplateExporter
from ..models import ObjectDescription

class HTMLExporter(TemplateExporter):
    """Exports analysis results as interactive HTML reports"""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, "html")
        self._setup_templates()

    def _setup_templates(self):
        """Setup HTML templates"""
        self.templates["main_report"] = self._get_main_report_template()
        self.templates["css_styles"] = self._get_css_styles()
        self.templates["javascript"] = self._get_javascript()

    async def export(self,
                    object_description: ObjectDescription,
                    knowledge_graph_turtle: Optional[str] = None,
                    metadata: Dict[str, Any] = None) -> str:
        """Export object description as interactive HTML report"""

        # Prepare context for template
        context = self._prepare_context(object_description, knowledge_graph_turtle, metadata)
        context.update({
            'object': object_description,
            'has_kg': knowledge_graph_turtle is not None,
            'kg_turtle': knowledge_graph_turtle,
            'css_styles': self.templates["css_styles"],
            'javascript': self.templates["javascript"]
        })

        # Render HTML content
        html_content = self._render_template("main_report", context)

        # Generate filename
        safe_name = "".join(c for c in object_description.name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_').lower()
        filename = self._generate_filename(f"analysis_report_{safe_name}", "html", timestamp=False)

        # Write to file
        return self._write_file(html_content, filename)

    def _get_main_report_template(self) -> str:
        """Main HTML report template"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SOMA Analysis Report: {object.name}</title>
    <style>
        {css_styles}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <h1>🔍 SOMA Object Analysis Report</h1>
            <div class="object-title">
                <h2>{object.name}</h2>
                <div class="confidence-badge confidence-{confidence_class}">
                    {object.confidence_score:.1%} Confidence
                </div>
            </div>
            <p class="description">{object.description}</p>
        </header>

        <!-- Quick Stats -->
        <section class="quick-stats">
            <div class="stat-card">
                <div class="stat-icon">🏷️</div>
                <div class="stat-content">
                    <div class="stat-label">Category</div>
                    <div class="stat-value">{object.semantic.category.value}</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🎨</div>
                <div class="stat-content">
                    <div class="stat-label">Color</div>
                    <div class="stat-value">{object.visual.colors.primary_color}</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📐</div>
                <div class="stat-content">
                    <div class="stat-label">Shape</div>
                    <div class="stat-value">{object.geometric.shape.primary_shape.value}</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🧱</div>
                <div class="stat-content">
                    <div class="stat-label">Material</div>
                    <div class="stat-value">{object.material.primary_material.value}</div>
                </div>
            </div>
        </section>

        <!-- Detailed Analysis Sections -->
        <div class="analysis-grid">
            <!-- Visual Properties -->
            <section class="analysis-section">
                <h3><span class="section-icon">🎨</span> Visual Properties</h3>
                <div class="property-grid">
                    <div class="property-item">
                        <span class="property-label">Primary Color:</span>
                        <span class="property-value">{object.visual.colors.primary_color}</span>
                    </div>
                    {rgb_section}
                    {secondary_colors_section}
                    <div class="property-item">
                        <span class="property-label">Surface Texture:</span>
                        <span class="property-value">{object.visual.surface.texture.value}</span>
                    </div>
                    {surface_finish_section}
                </div>
            </section>

            <!-- Geometric Properties -->
            <section class="analysis-section">
                <h3><span class="section-icon">📐</span> Geometric Properties</h3>
                <div class="property-grid">
                    <div class="property-item">
                        <span class="property-label">Primary Shape:</span>
                        <span class="property-value">{object.geometric.shape.primary_shape.value}</span>
                    </div>
                    {dimensions_section}
                    {volume_section}
                </div>
            </section>

            <!-- Material Properties -->
            <section class="analysis-section">
                <h3><span class="section-icon">🧱</span> Material Properties</h3>
                <div class="property-grid">
                    <div class="property-item">
                        <span class="property-label">Primary Material:</span>
                        <span class="property-value">{object.material.primary_material.value}</span>
                    </div>
                    {mass_section}
                    {temperature_section}
                    {secondary_materials_section}
                </div>
            </section>

            <!-- Capabilities -->
            <section class="analysis-section">
                <h3><span class="section-icon">⚡</span> Capabilities & Affordances</h3>
                <div class="capabilities-grid">
                    {capabilities_section}
                </div>
                <div class="tasks-section">
                    <h4>Primary Tasks</h4>
                    <div class="task-tags">
                        {primary_tasks_section}
                    </div>
                </div>
            </section>
        </div>

        <!-- Semantic Information -->
        <section class="analysis-section full-width">
            <h3><span class="section-icon">🏷️</span> Semantic Information</h3>
            <div class="semantic-grid">
                <div class="semantic-item">
                    <h4>Typical Locations</h4>
                    <div class="tag-list">
                        {locations_section}
                    </div>
                </div>
                <div class="semantic-item">
                    <h4>Associated Activities</h4>
                    <div class="tag-list">
                        {activities_section}
                    </div>
                </div>
            </div>
        </section>

        <!-- Knowledge Graph Section -->
        {knowledge_graph_section}

        <!-- Metadata -->
        <section class="analysis-section full-width metadata-section">
            <h3><span class="section-icon">🔧</span> Analysis Metadata</h3>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <span class="metadata-label">Generated:</span>
                    <span class="metadata-value">{export_date}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">System Version:</span>
                    <span class="metadata-value">SOMA Analysis v0.1.0</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Source:</span>
                    <span class="metadata-value">{object.source}</span>
                </div>
                {metadata_items_section}
            </div>
        </section>
    </div>

    <script>
        {javascript}
    </script>
</body>
</html>"""

    def _get_css_styles(self) -> str:
        """CSS styles for the HTML report"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
        }

        .header h1 {
            color: #4a5568;
            margin-bottom: 10px;
            font-size: 2.5em;
        }

        .object-title {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }

        .object-title h2 {
            color: #2d3748;
            font-size: 2em;
        }

        .confidence-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }

        .confidence-high { background: #c6f6d5; color: #22543d; }
        .confidence-medium { background: #fefcbf; color: #744210; }
        .confidence-low { background: #fed7d7; color: #742a2a; }

        .description {
            font-size: 1.2em;
            color: #666;
            max-width: 800px;
            margin: 0 auto;
        }

        .quick-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            display: flex;
            align-items: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.2s;
        }

        .stat-card:hover {
            transform: translateY(-2px);
        }

        .stat-icon {
            font-size: 2em;
            margin-right: 15px;
        }

        .stat-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }

        .stat-value {
            font-weight: bold;
            font-size: 1.1em;
            color: #2d3748;
        }

        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .analysis-section {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }

        .analysis-section h3 {
            color: #2d3748;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            font-size: 1.3em;
        }

        .section-icon {
            margin-right: 10px;
            font-size: 1.2em;
        }

        .full-width {
            grid-column: 1 / -1;
        }

        .property-grid {
            display: grid;
            gap: 12px;
        }

        .property-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #e2e8f0;
        }

        .property-item:last-child {
            border-bottom: none;
        }

        .property-label {
            font-weight: 500;
            color: #4a5568;
        }

        .property-value {
            font-weight: bold;
            color: #2d3748;
        }

        .capabilities-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .capability-item {
            display: flex;
            align-items: center;
            padding: 10px;
            background: #f7fafc;
            border-radius: 8px;
        }

        .capability-item.active {
            background: #c6f6d5;
        }

        .capability-item.inactive {
            background: #fed7d7;
        }

        .task-tags, .tag-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .task-tag, .tag-item {
            background: #e2e8f0;
            color: #4a5568;
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: 500;
        }

        .semantic-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
        }

        .semantic-item h4 {
            color: #4a5568;
            margin-bottom: 15px;
        }

        .metadata-section {
            background: #f8f9fa;
            border: 2px dashed #dee2e6;
        }

        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }

        .metadata-item {
            display: flex;
            justify-content: space-between;
        }

        .metadata-label {
            font-weight: 500;
            color: #6c757d;
        }

        .metadata-value {
            font-weight: bold;
            color: #495057;
        }

        .kg-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .kg-section h3 {
            color: white;
        }

        .kg-preview {
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 15px;
            font-family: monospace;
            font-size: 0.9em;
            overflow-x: auto;
            white-space: pre-wrap;
        }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .analysis-grid {
                grid-template-columns: 1fr;
            }
            
            .quick-stats {
                grid-template-columns: 1fr;
            }
            
            .object-title {
                flex-direction: column;
                gap: 10px;
            }
        }
        """

    def _get_javascript(self) -> str:
        """JavaScript for interactive features"""
        return """
        // Interactive features for the report
        document.addEventListener('DOMContentLoaded', function() {
            // Add hover effects to sections
            const sections = document.querySelectorAll('.analysis-section');
            sections.forEach(section => {
                section.addEventListener('mouseenter', function() {
                    this.style.transform = 'translateY(-2px)';
                    this.style.boxShadow = '0 10px 25px rgba(0,0,0,0.15)';
                });
                
                section.addEventListener('mouseleave', function() {
                    this.style.transform = 'translateY(0)';
                    this.style.boxShadow = '0 5px 15px rgba(0,0,0,0.08)';
                });
            });
            
            // Add click-to-expand for metadata
            const metadataSection = document.querySelector('.metadata-section');
            if (metadataSection) {
                metadataSection.style.cursor = 'pointer';
                metadataSection.addEventListener('click', function() {
                    const grid = this.querySelector('.metadata-grid');
                    if (grid.style.display === 'none') {
                        grid.style.display = 'grid';
                        this.querySelector('h3').innerHTML += ' (Click to collapse)';
                    } else {
                        grid.style.display = 'none';
                        this.querySelector('h3').innerHTML = this.querySelector('h3').innerHTML.replace(' (Click to collapse)', '');
                    }
                });
            }
            
            console.log('SOMA Analysis Report loaded successfully');
        });
        """

    def _render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Enhanced template rendering with dynamic sections"""
        # Get base template
        template = self._load_template(template_name)

        # Generate dynamic sections
        context.update(self._generate_dynamic_sections(context['object']))

        # Render template
        return template.format(**context)

    def _generate_dynamic_sections(self, obj: ObjectDescription) -> Dict[str, str]:
        """Generate dynamic HTML sections based on object properties"""
        sections = {}

        # RGB section
        if obj.visual.colors.rgb:
            rgb = obj.visual.colors.rgb
            sections['rgb_section'] = f"""
            <div class="property-item">
                <span class="property-label">RGB Values:</span>
                <span class="property-value" style="background: rgb({rgb.r},{rgb.g},{rgb.b}); padding: 4px 8px; border-radius: 4px; color: white;">
                    {rgb.r}, {rgb.g}, {rgb.b}
                </span>
            </div>"""
        else:
            sections['rgb_section'] = ""

        # Secondary colors
        if obj.visual.colors.secondary_colors:
            colors_html = " ".join([f'<span class="tag-item">{color}</span>' for color in obj.visual.colors.secondary_colors])
            sections['secondary_colors_section'] = f"""
            <div class="property-item">
                <span class="property-label">Secondary Colors:</span>
                <div class="tag-list">{colors_html}</div>
            </div>"""
        else:
            sections['secondary_colors_section'] = ""

        # Surface finish
        if obj.visual.surface.finish:
            sections['surface_finish_section'] = f"""
            <div class="property-item">
                <span class="property-label">Surface Finish:</span>
                <span class="property-value">{obj.visual.surface.finish}</span>
            </div>"""
        else:
            sections['surface_finish_section'] = ""

        # Dimensions
        dims = obj.geometric.shape.dimensions
        dim_parts = []
        if dims.width: dim_parts.append(f"W: {dims.width}m")
        if dims.height: dim_parts.append(f"H: {dims.height}m")
        if dims.depth: dim_parts.append(f"D: {dims.depth}m")
        if dims.radius: dim_parts.append(f"R: {dims.radius}m")
        if dims.length: dim_parts.append(f"L: {dims.length}m")

        if dim_parts:
            sections['dimensions_section'] = f"""
            <div class="property-item">
                <span class="property-label">Dimensions:</span>
                <span class="property-value">{", ".join(dim_parts)}</span>
            </div>"""
        else:
            sections['dimensions_section'] = ""

        # Volume
        volume = dims.volume()
        if volume:
            sections['volume_section'] = f"""
            <div class="property-item">
                <span class="property-label">Estimated Volume:</span>
                <span class="property-value">{volume:.6f} m³</span>
            </div>"""
        else:
            sections['volume_section'] = ""

        # Mass
        if obj.material.mass:
            sections['mass_section'] = f"""
            <div class="property-item">
                <span class="property-label">Mass:</span>
                <span class="property-value">{obj.material.mass} kg</span>
            </div>"""
        else:
            sections['mass_section'] = ""

        # Temperature
        if obj.material.temperature is not None:
            sections['temperature_section'] = f"""
            <div class="property-item">
                <span class="property-label">Temperature:</span>
                <span class="property-value">{obj.material.temperature}°C</span>
            </div>"""
        else:
            sections['temperature_section'] = ""

        # Secondary materials
        if obj.material.secondary_materials:
            materials_html = " ".join([f'<span class="tag-item">{mat.value}</span>' for mat in obj.material.secondary_materials])
            sections['secondary_materials_section'] = f"""
            <div class="property-item">
                <span class="property-label">Secondary Materials:</span>
                <div class="tag-list">{materials_html}</div>
            </div>"""
        else:
            sections['secondary_materials_section'] = ""

        # Capabilities
        caps = obj.capabilities.functional_affordances
        cap_items = []

        capabilities_map = [
            ("can_cut", "Can Cut", "✂️"),
            ("can_contain", "Can Contain", "📦"),
            ("can_grasp", "Can Grasp", "🤏"),
            ("can_support", "Can Support", "🏗️"),
            ("can_pour", "Can Pour", "🌊")
        ]

        for attr, label, icon in capabilities_map:
            value = getattr(caps, attr)
            if value is not None:
                css_class = "active" if value else "inactive"
                status = "✓" if value else "✗"
                cap_items.append(f'<div class="capability-item {css_class}">{icon} {label}: {status}</div>')

        if caps.graspability is not None:
            cap_items.append(f'<div class="capability-item active">🤏 Graspability: {caps.graspability:.2f}</div>')

        sections['capabilities_section'] = "".join(cap_items)

        # Primary tasks
        tasks = obj.capabilities.task_affordances.primary_tasks
        if tasks:
            tasks_html = " ".join([f'<span class="task-tag">{task}</span>' for task in tasks])
            sections['primary_tasks_section'] = tasks_html
        else:
            sections['primary_tasks_section'] = '<span class="task-tag">No specific tasks identified</span>'

        # Locations
        locations = obj.semantic.typical_locations
        if locations:
            sections['locations_section'] = " ".join([f'<span class="tag-item">{loc}</span>' for loc in locations])
        else:
            sections['locations_section'] = '<span class="tag-item">Unknown</span>'

        # Activities
        activities = obj.semantic.associated_activities
        if activities:
            sections['activities_section'] = " ".join([f'<span class="tag-item">{act}</span>' for act in activities])
        else:
            sections['activities_section'] = '<span class="tag-item">None specified</span>'

        # Confidence class for styling
        if obj.confidence_score >= 0.8:
            sections['confidence_class'] = "high"
        elif obj.confidence_score >= 0.6:
            sections['confidence_class'] = "medium"
        else:
            sections['confidence_class'] = "low"

        # Knowledge graph section
        sections['knowledge_graph_section'] = """
        <section class="analysis-section full-width kg-section">
            <h3><span class="section-icon">🧠</span> Knowledge Graph</h3>
            <p>SOMA-aligned knowledge graph generated for this object.</p>
            <div class="kg-preview">
                Knowledge graph data available in Turtle format.
                See knowledge_graph.ttl file for complete graph data.
            </div>
        </section>""" if sections.get('has_kg') else ""

        # Metadata items
        sections['metadata_items_section'] = ""

        return sections