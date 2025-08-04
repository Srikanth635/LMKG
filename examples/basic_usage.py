"""
Basic usage examples for SOMA Object Analysis System
"""

import asyncio
from soma_object_analysis import SOMAObjectAnalysisSystem, analyze_object_sync


def example_1_basic_analysis():
    """Example 1: Basic object analysis"""
    print("=" * 60)
    print("Example 1: Basic Object Analysis")
    print("=" * 60)

    # Create system instance
    system = SOMAObjectAnalysisSystem(use_mock=False)  # Using mock for this example

    # Analyze an object
    description = "pick up the blue mug from the cupboard /nothink"
    result = system.analyze_object_sync(description)

    if result["success"]:
        obj_analysis = result["object_analysis"]
        print(f"✅ Successfully analyzed: {obj_analysis['name']}")
        print(f"📋 Category: {obj_analysis['semantic']['category']}")
        print(f"🎨 Color: {obj_analysis['visual']['colors']['primary_color']}")
        print(f"📐 Shape: {obj_analysis['geometric']['shape']['primary_shape']}")
        print(f"🧱 Material: {obj_analysis['material']['primary_material']}")
        print(f"⚡ Graspability: {obj_analysis['capabilities']['functional_affordances']['graspability']}")
        print(f"🎯 Confidence: {obj_analysis['confidence_score']:.2%}")

        # Show knowledge graph info
        kg_info = result["knowledge_graph"]
        print(f"📊 Knowledge Graph: {kg_info['triple_count']} triples")

        # Show export paths
        if result["export_paths"]:
            print(f"📁 Results exported to {len(result['export_paths'])} files")
    else:
        print(f"❌ Analysis failed: {result['error']}")


def example_2_multiple_objects():
    """Example 2: Analyze multiple objects"""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Object Analysis")
    print("=" * 60)

    system = SOMAObjectAnalysisSystem(use_mock=True)

    objects = [
        "a sharp stainless steel kitchen knife with black handle",
        "a blue ceramic coffee mug with handle",
        "a rectangular smartphone with cracked screen"
    ]

    print(f"Analyzing {len(objects)} objects...")

    for i, description in enumerate(objects, 1):
        print(f"\n🔍 Object {i}: {description}")
        result = system.analyze_object_sync(description, export_results=False)

        if result["success"]:
            obj = result["object_analysis"]
            print(f"  ✅ {obj['name']} ({obj['semantic']['category']})")
            print(f"     Color: {obj['visual']['colors']['primary_color']}")
            print(f"     Material: {obj['material']['primary_material']}")
        else:
            print(f"  ❌ Failed: {result['error']}")


async def example_3_async_batch():
    """Example 3: Asynchronous batch processing"""
    print("\n" + "=" * 60)
    print("Example 3: Async Batch Processing")
    print("=" * 60)

    system = SOMAObjectAnalysisSystem(use_mock=True)

    descriptions = [
        "wooden pencil with yellow paint",
        "glass water bottle with blue cap",
        "leather wallet with multiple compartments",
        "plastic computer mouse with scroll wheel",
        "metal spoon with wooden handle"
    ]

    print(f"Processing {len(descriptions)} objects concurrently...")

    # Use batch processing
    results = await system.analyze_batch(descriptions, max_concurrent=3)

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print(f"\n📊 Batch Results:")
    print(f"  ✅ Successful: {len(successful)}")
    print(f"  ❌ Failed: {len(failed)}")
    print(f"  📈 Success Rate: {len(successful) / len(results):.1%}")

    print(f"\n🎯 Successful Analyses:")
    for result in successful:
        obj = result["object_analysis"]
        print(f"  • {obj['name']} - {obj['semantic']['category']} - {obj['confidence_score']:.1%}")


def example_4_knowledge_graph_focus():
    """Example 4: Focus on knowledge graph generation"""
    print("\n" + "=" * 60)
    print("Example 4: Knowledge Graph Focus")
    print("=" * 60)

    system = SOMAObjectAnalysisSystem(use_mock=True)

    # Analyze an object with focus on KG
    description = "red apple on wooden table"
    result = system.analyze_object_sync(description)

    if result["success"]:
        print(f"✅ Analyzed: {result['object_analysis']['name']}")

        # Show detailed KG information
        kg_info = result["knowledge_graph"]
        print(f"\n📊 Knowledge Graph Details:")
        print(f"  • Triples: {kg_info['triple_count']}")
        print(f"  • Instances: {kg_info['instance_count']}")

        # Show sample of turtle format
        if kg_info["turtle"]:
            turtle_lines = kg_info["turtle"].split('\n')
            print(f"\n🐢 Sample Turtle Format:")
            for line in turtle_lines[:10]:  # Show first 10 lines
                if line.strip():
                    print(f"  {line}")
            if len(turtle_lines) > 10:
                print(f"  ... and {len(turtle_lines) - 10} more lines")


def example_5_system_configuration():
    """Example 5: Different system configurations"""
    print("\n" + "=" * 60)
    print("Example 5: System Configuration Options")
    print("=" * 60)

    # Configuration 1: Mock mode
    print("🔧 Configuration 1: Mock Mode")
    system1 = SOMAObjectAnalysisSystem(use_mock=True)
    status1 = system1.validate_setup()
    print(f"  Valid: {status1['valid']}")
    print(f"  Agent: {system1.get_system_info()['components']['agent_type']}")

    # Configuration 2: Real LLM (if available)
    print("\n🔧 Configuration 2: Real LLM Mode")
    system2 = SOMAObjectAnalysisSystem(use_mock=False, model_name="gpt-3.5-turbo")
    status2 = system2.validate_setup()
    print(f"  Valid: {status2['valid']}")
    print(f"  Agent: {system2.get_system_info()['components']['agent_type']}")
    if status2['warnings']:
        print(f"  Warnings: {', '.join(status2['warnings'])}")

    # Configuration 3: Custom output directory
    print("\n🔧 Configuration 3: Custom Output")
    system3 = SOMAObjectAnalysisSystem(
        use_mock=True,
        output_dir="./my_custom_results"
    )
    print(f"  Output Dir: {system3.get_system_info()['configuration']['output_dir']}")


def example_6_working_with_results():
    """Example 6: Working with analysis results"""
    print("\n" + "=" * 60)
    print("Example 6: Working with Results")
    print("=" * 60)

    system = SOMAObjectAnalysisSystem(use_mock=True)

    # Analyze object
    result = system.analyze_object_sync("blue ceramic mug")

    if result["success"]:
        # Access the ObjectDescription model
        from soma_object_analysis.models import ObjectDescription
        obj_desc = ObjectDescription(**result["object_analysis"])

        print(f"✅ Object: {obj_desc.name}")

        # Use model methods
        print(f"📄 Summary: {obj_desc.get_summary()}")

        # Access nested properties
        print(f"🎨 RGB Color: {obj_desc.visual.colors.rgb}")
        print(f"📐 Dimensions: {obj_desc.geometric.shape.dimensions}")
        print(f"⚡ Can be grasped: {obj_desc.capabilities.functional_affordances.can_grasp}")

        # Get simplified dictionary
        simple_dict = obj_desc.to_simple_dict()
        print(f"\n📋 Simplified view:")
        for key, value in simple_dict.items():
            print(f"  {key}: {value}")

        # Calculate volume if possible
        volume = obj_desc.geometric.shape.dimensions.volume()
        if volume:
            print(f"📏 Estimated volume: {volume:.6f} m³")


def example_7_convenience_functions():
    """Example 7: Using convenience functions"""
    print("\n" + "=" * 60)
    print("Example 7: Convenience Functions")
    print("=" * 60)

    # Quick analysis using convenience function
    result = analyze_object_sync("smartphone with cracked screen", use_mock=True)

    if result["success"]:
        obj = result["object_analysis"]
        print(f"🚀 Quick analysis: {obj['name']}")
        print(f"   State: {obj['state']['functional_state'] or 'Normal'}")
        print(f"   Integrity: {obj['state']['integrity']:.1%}")


def main():
    """Run all examples"""
    print("🚀 SOMA Object Analysis System - Examples")
    print("=" * 60)

    # Run synchronous examples
    example_1_basic_analysis()
    # example_2_multiple_objects()
    # example_4_knowledge_graph_focus()
    # example_5_system_configuration()
    # example_6_working_with_results()
    # example_7_convenience_functions()
    #
    # # Run async example
    # print("\n" + "=" * 60)
    # print("Running async example...")
    # asyncio.run(example_3_async_batch())
    #
    # print("\n🎉 All examples completed!")
    # print("\n📚 Next steps:")
    # print("  • Try with real LLM by setting OPENAI_API_KEY")
    # print("  • Explore the CLI: soma-analyze 'your object description'")
    # print("  • Check the examples/jupyter_demo.ipynb notebook")
    # print("  • Look at generated files in the output directory")


if __name__ == "__main__":
    main()