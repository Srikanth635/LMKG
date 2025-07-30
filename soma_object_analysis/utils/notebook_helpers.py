"""
Jupyter notebook helpers for SOMA object analysis
"""

from typing import Dict, Any, Optional, List
from ..models import ObjectDescription


class NotebookHelpers:
    """Helper functions for Jupyter notebook usage"""

    @staticmethod
    def display_analysis_results(result: Dict[str, Any]):
        """Display analysis results in notebook with rich formatting"""
        try:
            from IPython.display import display, HTML, JSON
            import pandas as pd

            if not result.get("success"):
                display(HTML(f"""
                <div style='color: red; padding: 10px; border: 1px solid red; border-radius: 5px;'>
                    ❌ <strong>Analysis failed:</strong> {result.get('error', 'Unknown error')}
                </div>
                """))
                return

            obj_analysis = result["object_analysis"]

            # Create header
            confidence_color = NotebookHelpers._get_confidence_color(obj_analysis['confidence_score'])
            display(HTML(f"""
            <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='margin: 0; color: white;'>🔍 {obj_analysis['name']}</h2>
                <p style='margin: 10px 0 0 0; opacity: 0.9;'>{obj_analysis['description']}</p>
                <div style='margin-top: 15px;'>
                    <span style='background: {confidence_color}; padding: 5px 15px; 
                                 border-radius: 15px; color: white; font-weight: bold;'>
                        {obj_analysis['confidence_score']:.1%} Confidence
                    </span>
                </div>
            </div>
            """))

            # Create summary table
            summary_data = {
                "Property": ["Category", "Primary Color", "Shape", "Material", "Mass", "Graspability"],
                "Value": [
                    obj_analysis["semantic"]["category"],
                    obj_analysis["visual"]["colors"]["primary_color"],
                    obj_analysis["geometry"]["shape"]["primary_shape"],
                    obj_analysis["material"]["primary_material"],
                    f"{obj_analysis['material'].get('mass', 'Unknown')} kg" if obj_analysis['material'].get(
                        'mass') else 'Unknown',
                    f"{obj_analysis['capabilities']['functional_affordances'].get('graspability', 'Unknown'):.2f}" if
                    obj_analysis['capabilities']['functional_affordances'].get('graspability') else 'Unknown'
                ]
            }

            df = pd.DataFrame(summary_data)
            display(HTML("<h3 style='color: #333; margin-top: 20px;'>📊 Quick Summary</h3>"))
            display(HTML(df.to_html(index=False, escape=False, classes='table table-striped')))

            # Display capabilities
            caps = obj_analysis['capabilities']['functional_affordances']
            cap_items = []

            for cap_name, cap_value in caps.items():
                if cap_value is not None and isinstance(cap_value, bool):
                    icon = "✅" if cap_value else "❌"
                    cap_items.append(f"{icon} {cap_name.replace('_', ' ').title()}")
                elif cap_value is not None and isinstance(cap_value, (int, float)):
                    cap_items.append(f"📊 {cap_name.replace('_', ' ').title()}: {cap_value:.2f}")

            if cap_items:
                display(HTML(f"""
                <h3 style='color: #333; margin-top: 20px;'>⚡ Capabilities</h3>
                <div style='display: flex; flex-wrap: wrap; gap: 10px;'>
                    {''.join([f"<span style='background: #f0f8ff; padding: 8px 12px; border-radius: 15px; border: 1px solid #cce7ff;'>{item}</span>" for item in cap_items])}
                </div>
                """))

            # Display tasks if available
            tasks = obj_analysis['capabilities']['task_affordances']['primary_tasks']
            if tasks:
                display(HTML(f"""
                <h3 style='color: #333; margin-top: 20px;'>🎯 Primary Tasks</h3>
                <div style='display: flex; flex-wrap: wrap; gap: 8px;'>
                    {''.join([f"<span style='background: #e8f5e8; padding: 6px 12px; border-radius: 12px; border: 1px solid #c3e6c3;'>{task}</span>" for task in tasks])}
                </div>
                """))

            # Show knowledge graph info if available
            kg_info = result.get("knowledge_graph", {})
            if kg_info.get("triple_count", 0) > 0:
                display(HTML(f"""
                <h3 style='color: #333; margin-top: 20px;'>🧠 Knowledge Graph</h3>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007acc;'>
                    📊 Generated <strong>{kg_info['triple_count']} triples</strong> and <strong>{kg_info['instance_count']} instances</strong>
                </div>
                """))

            # Show export info
            export_paths = result.get("export_paths", [])
            if export_paths:
                display(HTML(f"""
                <h3 style='color: #333; margin-top: 20px;'>📁 Generated Files</h3>
                <div style='background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;'>
                    📄 Exported <strong>{len(export_paths)} files</strong>:<br>
                    {'<br>'.join([f"• {path}" for path in export_paths[:5]])}
                    {f'<br>• ... and {len(export_paths) - 5} more files' if len(export_paths) > 5 else ''}
                </div>
                """))

            # Expandable full JSON view
            display(HTML(f"""
            <details style='margin-top: 20px;'>
                <summary style='cursor: pointer; color: #007acc; font-weight: bold;'>
                    🔍 View Complete Analysis JSON
                </summary>
                <div style='margin-top: 10px;'>
            """))
            display(JSON(obj_analysis))
            display(HTML("</div></details>"))

        except ImportError:
            print("⚠️  IPython not available - using simple text display")
            NotebookHelpers.print_analysis_results(result)

    @staticmethod
    def print_analysis_results(result: Dict[str, Any]):
        """Print analysis results (fallback for non-notebook environments)"""
        if not result.get("success"):
            print(f"❌ Analysis failed: {result.get('error')}")
            return

        obj_analysis = result["object_analysis"]

        print("🔍 SOMA ANALYSIS RESULTS")
        print("=" * 50)
        print(f"📋 Name: {obj_analysis['name']}")
        print(f"📝 Description: {obj_analysis['description']}")
        print(f"🎯 Confidence: {obj_analysis['confidence_score']:.2%}")
        print()

        print("📊 PROPERTIES")
        print("-" * 20)
        print(f"🏷️  Category: {obj_analysis['semantic']['category']}")
        print(f"🎨 Primary Color: {obj_analysis['visual']['colors']['primary_color']}")
        print(f"📐 Shape: {obj_analysis['geometry']['shape']['primary_shape']}")
        print(f"🧱 Material: {obj_analysis['material']['primary_material']}")

        if obj_analysis['material'].get('mass'):
            print(f"⚖️  Mass: {obj_analysis['material']['mass']} kg")

        print()

        # Capabilities
        caps = obj_analysis['capabilities']['functional_affordances']
        print("⚡ CAPABILITIES")
        print("-" * 15)

        if caps.get('can_grasp'): print("• ✅ Can be grasped")
        if caps.get('can_cut'): print("• ✂️  Can cut")
        if caps.get('can_contain'): print("• 📦 Can contain")
        if caps.get('can_support'): print("• 🏗️  Can support")
        if caps.get('can_pour'): print("• 🌊 Can pour")

        if caps.get('graspability') is not None:
            print(f"• 🤏 Graspability: {caps['graspability']:.2f}/1.0")

        print()

        # Tasks
        tasks = obj_analysis['capabilities']['task_affordances']['primary_tasks']
        if tasks:
            print("🎯 PRIMARY TASKS")
            print("-" * 15)
            for task in tasks:
                print(f"• {task}")
            print()

        # Knowledge graph info
        kg_info = result.get("knowledge_graph", {})
        if kg_info.get("triple_count", 0) > 0:
            print(f"🧠 Knowledge Graph: {kg_info['triple_count']} triples, {kg_info['instance_count']} instances")

        # Export info
        export_paths = result.get("export_paths", [])
        if export_paths:
            print(f"📁 Exported {len(export_paths)} files")

    @staticmethod
    def create_comparison_table(results: List[Dict[str, Any]]):
        """Create comparison table for multiple analysis results"""
        try:
            from IPython.display import display, HTML
            import pandas as pd

            if not results or not any(r.get("success") for r in results):
                display(HTML("<div style='color: red;'>❌ No successful analyses to compare</div>"))
                return

            # Extract data for comparison
            comparison_data = []
            for i, result in enumerate(results):
                if result.get("success"):
                    obj = result["object_analysis"]
                    comparison_data.append({
                        "Name": obj["name"],
                        "Category": obj["semantic"]["category"],
                        "Color": obj["visual"]["colors"]["primary_color"],
                        "Shape": obj["geometry"]["shape"]["primary_shape"],
                        "Material": obj["material"]["primary_material"],
                        "Mass (kg)": obj["material"].get("mass", "Unknown"),
                        "Graspability": f"{obj['capabilities']['functional_affordances'].get('graspability', 0):.2f}" if
                        obj['capabilities']['functional_affordances'].get('graspability') else "Unknown",
                        "Confidence": f"{obj['confidence_score']:.1%}"
                    })

            if comparison_data:
                df = pd.DataFrame(comparison_data)
                display(HTML("<h3>📊 Object Comparison</h3>"))
                display(HTML(df.to_html(index=False, escape=False, classes='table table-striped')))

        except ImportError:
            print("⚠️  IPython/pandas not available for comparison table")
            # Fallback to simple text comparison
            NotebookHelpers._print_comparison_text(results)

    @staticmethod
    def _print_comparison_text(results: List[Dict[str, Any]]):
        """Text-based comparison fallback"""
        successful = [r for r in results if r.get("success")]

        print(f"📊 OBJECT COMPARISON ({len(successful)} objects)")
        print("=" * 60)

        for i, result in enumerate(successful, 1):
            obj = result["object_analysis"]
            print(f"{i}. {obj['name']}")
            print(f"   Category: {obj['semantic']['category']}")
            print(f"   Color: {obj['visual']['colors']['primary_color']}")
            print(f"   Material: {obj['material']['primary_material']}")
            print(f"   Confidence: {obj['confidence_score']:.1%}")
            print()

    @staticmethod
    def create_knowledge_graph_summary(kg_turtle: str):
        """Create a summary visualization of the knowledge graph"""
        try:
            from IPython.display import display, HTML

            if not kg_turtle:
                display(HTML("<div style='color: orange;'>⚠️  No knowledge graph available</div>"))
                return

            # Parse basic statistics from turtle
            lines = kg_turtle.split('\n')
            prefixes = [line for line in lines if line.startswith('@prefix')]
            triples = [line for line in lines if
                       line.strip() and not line.startswith('@prefix') and not line.startswith('#')]

            display(HTML(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px; border-radius: 10px;'>
                <h3 style='margin: 0; color: white;'>🧠 Knowledge Graph Summary</h3>
                <div style='margin-top: 15px; display: flex; gap: 30px;'>
                    <div>
                        <div style='font-size: 2em; font-weight: bold;'>{len(triples)}</div>
                        <div style='opacity: 0.9;'>Triples</div>
                    </div>
                    <div>
                        <div style='font-size: 2em; font-weight: bold;'>{len(prefixes)}</div>
                        <div style='opacity: 0.9;'>Namespaces</div>
                    </div>
                </div>
            </div>
            """))

            # Show sample triples
            if triples:
                sample_triples = triples[:5]
                display(HTML(f"""
                <h4 style='margin-top: 20px; color: #333;'>🔍 Sample Triples</h4>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; font-family: monospace;'>
                    {'<br>'.join(sample_triples)}
                    {f'<br><em>... and {len(triples) - 5} more triples</em>' if len(triples) > 5 else ''}
                </div>
                """))

        except ImportError:
            print("⚠️  IPython not available for KG summary")
            # Fallback
            lines = kg_turtle.split('\n')
            triples = [line for line in lines if
                       line.strip() and not line.startswith('@prefix') and not line.startswith('#')]
            print(f"🧠 Knowledge Graph: {len(triples)} triples")

    @staticmethod
    def _get_confidence_color(confidence: float) -> str:
        """Get color based on confidence score"""
        if confidence >= 0.8:
            return "#28a745"  # Green
        elif confidence >= 0.6:
            return "#ffc107"  # Yellow
        else:
            return "#dc3545"  # Red

    @staticmethod
    def create_interactive_analysis(object_description: str):
        """Create interactive analysis interface"""
        try:
            from IPython.display import display, HTML
            import ipywidgets as widgets
            from IPython.display import clear_output

            # Create input widget
            text_input = widgets.Text(
                value=object_description,
                placeholder='Enter object description...',
                description='Object:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='500px')
            )

            # Create analyze button
            analyze_button = widgets.Button(
                description='🔍 Analyze Object',
                button_style='primary',
                layout=widgets.Layout(width='150px')
            )

            # Create output area
            output = widgets.Output()

            def on_analyze_click(b):
                with output:
                    clear_output()
                    print(f"🔄 Analyzing: {text_input.value}")

                    # Here you would call the actual analysis
                    # For demo purposes, showing placeholder
                    display(HTML("""
                    <div style='background: #fff3cd; padding: 15px; border-radius: 8px; margin-top: 10px;'>
                        ⚠️  Interactive analysis requires the main SOMA system to be imported.<br>
                        Use: <code>from soma_object_analysis import analyze_object_sync</code>
                    </div>
                    """))

            analyze_button.on_click(on_analyze_click)

            # Display interface
            display(HTML("<h3>🔬 Interactive Object Analysis</h3>"))
            display(widgets.HBox([text_input, analyze_button]))
            display(output)

        except ImportError:
            print("⚠️  ipywidgets not available for interactive interface")
            print(f"Analysis request: {object_description}")

    @staticmethod
    def export_to_dataframe(results: List[Dict[str, Any]]):
        """Export analysis results to pandas DataFrame"""
        try:
            import pandas as pd

            data = []
            for result in results:
                if result.get("success"):
                    obj = result["object_analysis"]
                    data.append({
                        'name': obj['name'],
                        'description': obj['description'],
                        'category': obj['semantic']['category'],
                        'primary_color': obj['visual']['colors']['primary_color'],
                        'shape': obj['geometry']['shape']['primary_shape'],
                        'material': obj['material']['primary_material'],
                        'mass': obj['material'].get('mass'),
                        'graspability': obj['capabilities']['functional_affordances'].get('graspability'),
                        'confidence_score': obj['confidence_score'],
                        'primary_tasks': ', '.join(obj['capabilities']['task_affordances']['primary_tasks']),
                        'typical_locations': ', '.join(obj['semantic']['typical_locations'])
                    })

            return pd.DataFrame(data)

        except ImportError:
            print("⚠️  pandas not available - cannot create DataFrame")
            return None