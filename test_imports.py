#!/usr/bin/env python3
"""
Quick import test to verify all modules are working
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test all major imports to identify issues"""

    print("🔍 Testing SOMA Object Analysis imports...")
    print("=" * 50)

    # Test base models
    try:
        from soma_object_analysis.models.base import (
            ObjectCategory, ShapeType, MaterialType, TextureType,
            DeviceState, CleanlinessState
        )
        print("✅ Base models imported successfully")
    except ImportError as e:
        print(f"❌ Base models import failed: {e}")
        return False

    # Test visual models
    try:
        from soma_object_analysis.models.visual import (
            RGBColor, HSVColor, ColorDescription, SurfaceProperties, VisualAppearance
        )
        print("✅ Visual models imported successfully")
    except ImportError as e:
        print(f"❌ Visual models import failed: {e}")
        return False

    # Test all models via init
    try:
        from soma_object_analysis.models import (
            ObjectDescription, ObjectCategory, ShapeType, MaterialType, TextureType
        )
        print("✅ Models package imported successfully")
    except ImportError as e:
        print(f"❌ Models package import failed: {e}")
        return False

    # Test agents
    try:
        from soma_object_analysis.agents import MockAnalysisAgent, create_analysis_agent
        print("✅ Agents imported successfully")
    except ImportError as e:
        print(f"❌ Agents import failed: {e}")
        return False

    # Test knowledge graph
    try:
        from soma_object_analysis.knowledge_graph import SOMAKnowledgeGraph, KnowledgeGraphGenerator
        print("✅ Knowledge graph imported successfully")
    except ImportError as e:
        print(f"❌ Knowledge graph import failed: {e}")
        return False

    # Test exporters
    try:
        from soma_object_analysis.exporters import ResultsExporter
        print("✅ Exporters imported successfully")
    except ImportError as e:
        print(f"❌ Exporters import failed: {e}")
        return False

    # Test utils
    try:
        from soma_object_analysis.utils import setup_logging, get_logger
        print("✅ Utils imported successfully")
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False

    # Test main system
    try:
        from soma_object_analysis import SOMAObjectAnalysisSystem, analyze_object_sync
        print("✅ Main system imported successfully")
    except ImportError as e:
        print(f"❌ Main system import failed: {e}")
        return False

    print("\n🎉 All imports successful!")
    return True


def test_basic_functionality():
    """Test basic functionality"""
    print("\n🔧 Testing basic functionality...")
    print("-" * 30)

    try:
        # Test model creation
        from soma_object_analysis.models import ObjectDescription, ObjectCategory, create_minimal_object

        minimal_obj = create_minimal_object("Test Object", ObjectCategory.ITEM)
        print(f"✅ Created minimal object: {minimal_obj.name}")

        # Test mock agent
        from soma_object_analysis.agents import MockAnalysisAgent
        import asyncio

        agent = MockAnalysisAgent()
        result = asyncio.run(agent.analyze("test red apple"))

        if result.get("success"):
            print(f"✅ Mock analysis successful: {result['object_analysis']['name']}")
        else:
            print(f"❌ Mock analysis failed: {result.get('error')}")
            return False

        # Test knowledge graph generation
        from soma_object_analysis.knowledge_graph import KnowledgeGraphGenerator

        kg_gen = KnowledgeGraphGenerator()
        kg = kg_gen.create_from_object_description(minimal_obj)
        stats = kg.get_soma_statistics()

        print(f"✅ Knowledge graph generated: {stats['triple_count']} triples")

        print("\n🎉 Basic functionality test passed!")
        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🚀 SOMA Object Analysis - Import & Functionality Test")
    print("=" * 60)

    # Test imports
    imports_ok = test_imports()

    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()

        if functionality_ok:
            print("\n✅ All tests passed! System is ready to use.")
            print("\nNext steps:")
            print("  • Try: python examples/basic_usage.py")
            print("  • Or: python test_main.py --quick")
            return True
        else:
            print("\n❌ Functionality tests failed.")
            return False
    else:
        print("\n❌ Import tests failed. Please fix import issues first.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)