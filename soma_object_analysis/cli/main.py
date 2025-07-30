"""
Command Line Interface for SOMA Object Analysis
"""

import asyncio
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..system import SOMAObjectAnalysisSystem
from ..config import config

console = Console()

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """SOMA Object Analysis System - Analyze objects with SOMA ontology alignment"""
    pass

@cli.command()
@click.argument('description')
@click.option('--model', default=None, help=f'LLM model to use (default: {config.DEFAULT_MODEL})')
@click.option('--output-dir', default=None, help='Output directory for results')
@click.option('--mock', is_flag=True, help='Use mock mode (no LLM calls)')
@click.option('--no-export', is_flag=True, help='Skip exporting results to files')
@click.option('--no-kg', is_flag=True, help='Skip knowledge graph generation')
@click.option('--verbose', is_flag=True, help='Verbose output')
def analyze(description, model, output_dir, mock, no_export, no_kg, verbose):
    """Analyze a single object description"""

    console.print(Panel.fit("🔍 SOMA Object Analysis", style="bold blue"))

    # Initialize system
    system = SOMAObjectAnalysisSystem(
        model_name=model,
        use_mock=mock,
        output_dir=output_dir
    )

    # Validate setup
    validation = system.validate_setup()
    if not validation["valid"]:
        console.print("❌ System validation failed:", style="red")
        for issue in validation["issues"]:
            console.print(f"  • {issue}", style="red")
        return

    if validation["warnings"] and verbose:
        console.print("⚠️  Warnings:", style="yellow")
        for warning in validation["warnings"]:
            console.print(f"  • {warning}", style="yellow")

    # Run analysis
    async def run_analysis():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing object...", total=None)

            result = await system.analyze_object(
                description=description,
                export_results=not no_export,
                generate_kg=not no_kg
            )

            progress.update(task, completed=100, description="Analysis complete!")

        return result

    # Execute analysis
    result = asyncio.run(run_analysis())

    # Display results
    display_analysis_result(result, verbose)

@cli.command()
@click.argument('descriptions', nargs=-1, required=True)
@click.option('--model', default=None, help='LLM model to use')
@click.option('--output-dir', default=None, help='Output directory for results')
@click.option('--mock', is_flag=True, help='Use mock mode')
@click.option('--max-concurrent', default=3, help='Maximum concurrent analyses')
@click.option('--verbose', is_flag=True, help='Verbose output')
def batch(descriptions, model, output_dir, mock, max_concurrent, verbose):
    """Analyze multiple object descriptions"""

    console.print(Panel.fit(f"🔍 SOMA Batch Analysis - {len(descriptions)} objects", style="bold blue"))

    # Initialize system
    system = SOMAObjectAnalysisSystem(
        model_name=model,
        use_mock=mock,
        output_dir=output_dir
    )

    # Run batch analysis
    async def run_batch():
        with Progress(console=console) as progress:
            task = progress.add_task("Batch analysis...", total=len(descriptions))

            results = await system.analyze_batch(
                descriptions=list(descriptions),
                max_concurrent=max_concurrent
            )

            progress.update(task, completed=len(descriptions))

        return results

    results = asyncio.run(run_batch())

    # Display batch results
    display_batch_results(results, verbose)

@cli.command()
@click.option('--output-dir', default=None, help='Output directory to check')
def status():
    """Show system status and configuration"""

    system = SOMAObjectAnalysisSystem(output_dir=output_dir)
    validation = system.validate_setup()
    system_info = system.get_system_info()

    console.print(Panel.fit("🔧 SOMA System Status", style="bold green"))

    # System info table
    info_table = Table(title="System Information")
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("Version", system_info["version"])
    info_table.add_row("Agent Type", system_info["components"]["agent_type"])
    info_table.add_row("Model", system_info["configuration"]["model_name"])
    info_table.add_row("Mock Mode", str(system_info["configuration"]["use_mock"]))
    info_table.add_row("Output Directory", system_info["configuration"]["output_dir"])
    info_table.add_row("LangGraph Available", str(system_info["capabilities"]["langgraph_available"]))
    info_table.add_row("OpenAI Configured", str(system_info["capabilities"]["openai_configured"]))

    console.print(info_table)

    # Validation status
    if validation["valid"]:
        console.print("✅ System validation: PASSED", style="green")
    else:
        console.print("❌ System validation: FAILED", style="red")
        for issue in validation["issues"]:
            console.print(f"  • {issue}", style="red")

    if validation["warnings"]:
        console.print("\n⚠️  Warnings:", style="yellow")
        for warning in validation["warnings"]:
            console.print(f"  • {warning}", style="yellow")

@cli.command()
@click.argument('description')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'yaml', 'table']),
              default='table',
              help='Output format')
def quick(description, output_format):
    """Quick analysis without file export"""

    system = SOMAObjectAnalysisSystem(use_mock=True)  # Use mock for speed

    result = asyncio.run(system.analyze_object(
        description=description,
        export_results=False,
        generate_kg=False
    ))

    if not result.get("success"):
        console.print(f"❌ Analysis failed: {result.get('error')}", style="red")
        return

    obj_analysis = result["object_analysis"]

    if output_format == 'table':
        display_quick_table(obj_analysis)
    elif output_format == 'json':
        import json
        console.print_json(json.dumps(obj_analysis, indent=2))
    elif output_format == 'yaml':
        import yaml
        console.print(yaml.dump(obj_analysis, default_flow_style=False))

def display_analysis_result(result: dict, verbose: bool = False):
    """Display analysis result in a formatted way"""

    if not result.get("success"):
        console.print(f"❌ Analysis failed: {result.get('error')}", style="red")
        return

    obj_analysis = result["object_analysis"]

    # Main result panel
    console.print(Panel.fit(
        f"✅ Analysis Complete: {obj_analysis['name']}\n"
        f"Category: {obj_analysis['semantic']['category']}\n"
        f"Confidence: {obj_analysis['confidence_score']:.2%}",
        style="green"
    ))

    # Properties table
    props_table = Table(title="Object Properties")
    props_table.add_column("Property", style="cyan")
    props_table.add_column("Value", style="white")

    props_table.add_row("Primary Color", obj_analysis['visual']['colors']['primary_color'])
    props_table.add_row("Shape", obj_analysis['geometric']['shape']['primary_shape'])
    props_table.add_row("Material", obj_analysis['material']['primary_material'])

    if obj_analysis['material']['mass']:
        props_table.add_row("Mass", f"{obj_analysis['material']['mass']} kg")

    graspability = obj_analysis['capabilities']['functional_affordances'].get('graspability')
    if graspability is not None:
        props_table.add_row("Graspability", f"{graspability:.2f}")

    console.print(props_table)

    # Knowledge graph info
    kg_info = result.get("knowledge_graph", {})
    if kg_info.get("triple_count", 0) > 0:
        console.print(f"📊 Knowledge Graph: {kg_info['triple_count']} triples, {kg_info['instance_count']} instances")

    # Export info
    export_paths = result.get("export_paths", [])
    if export_paths:
        console.print(f"📁 Exported {len(export_paths)} files:")
        for path in export_paths[:3]:  # Show first 3
            console.print(f"  • {Path(path).name}")
        if len(export_paths) > 3:
            console.print(f"  • ... and {len(export_paths) - 3} more")

    if verbose and result.get("metadata"):
        console.print("\n🔧 Metadata:", style="dim")
        for key, value in result["metadata"].items():
            console.print(f"  {key}: {value}", style="dim")

def display_batch_results(results: list, verbose: bool = False):
    """Display batch analysis results"""

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    console.print(Panel.fit(
        f"📊 Batch Analysis Complete\n"
        f"✅ Successful: {len(successful)}\n"
        f"❌ Failed: {len(failed)}\n"
        f"📈 Success Rate: {len(successful)/len(results):.1%}",
        style="blue"
    ))

    if successful:
        # Results table
        results_table = Table(title="Successful Analyses")
        results_table.add_column("Name", style="cyan")
        results_table.add_column("Category", style="white")
        results_table.add_column("Color", style="white")
        results_table.add_column("Material", style="white")
        results_table.add_column("Confidence", style="white")

        for result in successful[:10]:  # Show first 10
            obj = result["object_analysis"]
            results_table.add_row(
                obj["name"],
                obj["semantic"]["category"],
                obj["visual"]["colors"]["primary_color"],
                obj["material"]["primary_material"],
                f"{obj['confidence_score']:.1%}"
            )

        console.print(results_table)

        if len(successful) > 10:
            console.print(f"... and {len(successful) - 10} more successful analyses")

    if failed and verbose:
        console.print("\n❌ Failed Analyses:", style="red")
        for result in failed[:5]:  # Show first 5 failures
            console.print(f"  • {result.get('error', 'Unknown error')}", style="red")

def display_quick_table(obj_analysis: dict):
    """Display quick analysis in table format"""

    table = Table(title=f"Quick Analysis: {obj_analysis['name']}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Description", obj_analysis['description'])
    table.add_row("Category", obj_analysis['semantic']['category'])
    table.add_row("Primary Color", obj_analysis['visual']['colors']['primary_color'])
    table.add_row("Shape", obj_analysis['geometric']['shape']['primary_shape'])
    table.add_row("Material", obj_analysis['material']['primary_material'])
    table.add_row("Confidence", f"{obj_analysis['confidence_score']:.2%}")

    # Add primary tasks if available
    tasks = obj_analysis['capabilities']['task_affordances']['primary_tasks']
    if tasks:
        table.add_row("Primary Tasks", ", ".join(tasks[:3]))

    console.print(table)

def main():
    """Entry point for console script"""
    cli()

if __name__ == "__main__":
    main()