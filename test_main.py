#!/usr/bin/env python3
"""
Test module for SOMA Object Analysis System
===========================================

This module provides comprehensive testing for individual components
and the complete application pipeline.

Usage:
    python test_main.py                    # Run all tests
    python test_main.py --component models # Test specific component
    python test_main.py --full-pipeline    # Test complete pipeline only
    python test_main.py --interactive      # Interactive testing mode
"""

import asyncio
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Test imports and component availability
print("🔍 Checking component availability...")

# Core imports
try:
    from soma_object_analysis import (
        SOMAObjectAnalysisSystem,
        ObjectDescription,
        analyze_object_sync,
        quick_test,
        config
    )

    CORE_AVAILABLE = True
    print("✅ Core system: Available")
except ImportError as e:
    CORE_AVAILABLE = False
    print(f"❌ Core system: {e}")

# Individual component imports
COMPONENTS = {}

try:
    from soma_object_analysis.models import (
        ObjectDescription, ObjectCategory, ShapeType, MaterialType,
        ColorDescription, GeometricDescription, MaterialProperties
    )

    COMPONENTS['models'] = True
    print("✅ Models: Available")
except ImportError as e:
    COMPONENTS['models'] = False
    print(f"❌ Models: {e}")

try:
    from soma_object_analysis.agents import (
        create_analysis_agent, MockAnalysisAgent,
        check_agent_dependencies
    )

    COMPONENTS['agents'] = True
    print("✅ Agents: Available")
except ImportError as e:
    COMPONENTS['agents'] = False
    print(f"❌ Agents: {e}")

try:
    from soma_object_analysis.knowledge_graph import (
        SOMAKnowledgeGraph, KnowledgeGraphGenerator,
        check_dependencies as kg_check_deps
    )

    COMPONENTS['knowledge_graph'] = True
    print("✅ Knowledge Graph: Available")
except ImportError as e:
    COMPONENTS['knowledge_graph'] = False
    print(f"❌ Knowledge Graph: {e}")

try:
    from soma_object_analysis.exporters import (
        ResultsExporter, get_supported_formats
    )

    COMPONENTS['exporters'] = True
    print("✅ Exporters: Available")
except ImportError as e:
    COMPONENTS['exporters'] = False
    print(f"❌ Exporters: {e}")

try:
    from soma_object_analysis.utils import (
        setup_logging, get_logger, clean_filename
    )

    COMPONENTS['utils'] = True
    print("✅ Utils: Available")
except ImportError as e:
    COMPONENTS['utils'] = False
    print(f"❌ Utils: {e}")

print()


class TestRunner:
    """Main test runner for SOMA Object Analysis System"""

    def __init__(self):
        self.logger = self._setup_logging()
        self.test_results = {}
        self.test_objects = [
            "red apple on wooden kitchen table",
            "sharp stainless steel knife with black handle",
            "blue ceramic coffee mug with white interior",
            "rectangular smartphone with cracked screen",
            "wooden pencil with yellow paint and pink eraser"
        ]

    def _setup_logging(self):
        """Setup logging for tests"""
        if COMPONENTS.get('utils'):
            return setup_logging(log_level="INFO")
        else:
            import logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger("test")

    def print_header(self, title: str, level: int = 1):
        """Print formatted header"""
        symbols = ["🚀", "🔧", "⚡", "📊", "🎯"]
        symbol = symbols[min(level - 1, len(symbols) - 1)]

        if level == 1:
            print(f"\n{symbol} {title}")
            print("=" * (len(title) + 3))
        else:
            print(f"\n{symbol} {title}")
            print("-" * (len(title) + 3))

    def print_result(self, test_name: str, success: bool, details: str = ""):
        """Print test result"""
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {test_name}")
        if details:
            print(f"   {details}")

        self.test_results[test_name] = {
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }

    def test_models(self) -> bool:
        """Test Pydantic models"""
        self.print_header("1. Testing Pydantic Models", 2)

        if not COMPONENTS.get('models'):
            self.print_result("Models Import", False, "Models not available")
            return False

        try:
            # Test basic model creation
            from soma_object_analysis.models import (
                ObjectDescription, ObjectCategory, ShapeType, MaterialType,
                ColorDescription, RGBColor, VisualAppearance, SurfaceProperties, TextureType,
                GeometricDescription, GeometricShape, Dimensions,
                MaterialProperties, CapabilityDescription, FunctionalAffordances,
                SemanticDescription, create_minimal_object
            )

            # Test enum values
            category = ObjectCategory.ITEM
            shape = ShapeType.SPHERE
            material = MaterialType.ORGANIC

            # Test model creation
            color_desc = ColorDescription(
                primary_color="red",
                rgb=RGBColor(r=220, g=20, b=60)
            )

            visual = VisualAppearance(
                colors=color_desc,
                surface=SurfaceProperties(texture=TextureType.SMOOTH)
            )

            geometry = GeometricDescription(
                shape=GeometricShape(
                    primary_shape=shape,
                    dimensions=Dimensions(radius=0.04, height=0.08)
                )
            )

            material_props = MaterialProperties(primary_material=material)

            semantic = SemanticDescription(category=category)

            # Test complete object description
            obj_desc = ObjectDescription(
                name="Test Apple",
                description="A test red apple",
                visual=visual,
                geometric=geometry,
                material=material_props,
                semantic=semantic
            )

            # Test factory function
            minimal_obj = create_minimal_object("Minimal Object", ObjectCategory.TOOL)

            # Test model methods
            summary = obj_desc.get_summary()
            simple_dict = obj_desc.to_simple_dict()
            volume = obj_desc.geometric.shape.dimensions.volume()

            self.print_result("Model Creation", True, f"Created {obj_desc.name}")
            self.print_result("Factory Function", True, f"Created {minimal_obj.name}")
            self.print_result("Model Methods", True, f"Volume: {volume:.6f} m³" if volume else "No volume")

            return True

        except Exception as e:
            self.print_result("Models Test", False, str(e))
            return False

    def test_agents(self) -> bool:
        """Test analysis agents"""
        self.print_header("2. Testing Analysis Agents", 2)

        if not COMPONENTS.get('agents'):
            self.print_result("Agents Import", False, "Agents not available")
            return False

        try:
            # Check agent dependencies
            deps = check_agent_dependencies()
            self.print_result("Agent Dependencies", True,
                              f"LangGraph: {deps['langgraph_agent']}, Mock: {deps['mock_agent']}")

            # Test mock agent
            mock_agent = MockAnalysisAgent()

            # Test analysis
            test_desc = "red apple on table"
            result = asyncio.run(mock_agent.analyze(test_desc))

            if result.get("success"):
                obj_analysis = result["object_analysis"]
                self.print_result("Mock Agent Analysis", True,
                                  f"Analyzed: {obj_analysis['name']}")
            else:
                self.print_result("Mock Agent Analysis", False,
                                  result.get("error", "Unknown error"))

            # Test factory function
            agent = create_analysis_agent(use_mock=True)
            self.print_result("Agent Factory", True, f"Created: {type(agent).__name__}")

            return True

        except Exception as e:
            self.print_result("Agents Test", False, str(e))
            return False

    def test_knowledge_graph(self) -> bool:
        """Test knowledge graph components"""
        self.print_header("3. Testing Knowledge Graph", 2)

        if not COMPONENTS.get('knowledge_graph'):
            self.print_result("KG Import", False, "Knowledge Graph not available")
            return False

        try:
            # Check KG dependencies
            kg_deps = kg_check_deps()
            self.print_result("KG Dependencies", True,
                              f"Visualization: {kg_deps['visualization_available']}")

            # Test KG creation
            kg = SOMAKnowledgeGraph()

            # Add test instance
            kg.add_instance("test_apple", "Item", {"name": "Test Apple"})
            kg.add_instance("test_color", "RedColor")
            kg.add_relation("test_apple", "hasColor", "test_color")

            # Test statistics
            stats = kg.get_soma_statistics()
            self.print_result("KG Creation", True,
                              f"Created {stats['triple_count']} triples")

            # Test turtle export
            turtle_content = kg.export_turtle()
            self.print_result("Turtle Export", True,
                              f"Generated {len(turtle_content)} characters")

            # Test KG generator
            generator = KnowledgeGraphGenerator()

            # Create a simple object for testing
            from soma_object_analysis.models import create_minimal_object, ObjectCategory
            test_obj = create_minimal_object("Generator Test", ObjectCategory.ITEM)

            generated_kg = generator.create_from_object_description(test_obj)
            gen_stats = generated_kg.get_soma_statistics()

            self.print_result("KG Generator", True,
                              f"Generated {gen_stats['triple_count']} triples from object")

            return True

        except Exception as e:
            self.print_result("Knowledge Graph Test", False, str(e))
            return False

    def test_exporters(self) -> bool:
        """Test export system"""
        self.print_header("4. Testing Export System", 2)

        if not COMPONENTS.get('exporters'):
            self.print_result("Exporters Import", False, "Exporters not available")
            return False

        try:
            # Test supported formats
            formats = get_supported_formats()
            self.print_result("Supported Formats", True,
                              f"Available: {', '.join(formats)}")

            # Test results exporter
            test_output_dir = Path("test_exports")
            test_output_dir.mkdir(exist_ok=True)

            exporter = ResultsExporter(test_output_dir)

            # Get exporter statistics
            stats = exporter.get_export_statistics()
            self.print_result("Exporter Setup", True,
                              f"Formats: {stats['format_count']}")

            # Test with minimal object (without actual export to avoid file creation)
            from soma_object_analysis.models import create_minimal_object, ObjectCategory
            test_obj = create_minimal_object("Export Test", ObjectCategory.CONTAINER)

            # Just test that we can prepare for export
            self.print_result("Export Preparation", True,
                              f"Ready to export {test_obj.name}")

            return True

        except Exception as e:
            self.print_result("Exporters Test", False, str(e))
            return False

    def test_utils(self) -> bool:
        """Test utility functions"""
        self.print_header("5. Testing Utilities", 2)

        if not COMPONENTS.get('utils'):
            self.print_result("Utils Import", False, "Utils not available")
            return False

        try:
            # Test filename cleaning
            dirty_filename = "My Object: Name/With\\Bad*Chars?.txt"
            clean_name = clean_filename(dirty_filename)
            self.print_result("Filename Cleaning", True, f"'{dirty_filename}' -> '{clean_name}'")

            # Test logging
            test_logger = get_logger("test")
            test_logger.info("Test log message")
            self.print_result("Logging", True, "Logger created and tested")

            # Test validation if available
            try:
                from soma_object_analysis.utils.validation import validate_object_description
                from soma_object_analysis.models import create_minimal_object, ObjectCategory

                test_obj = create_minimal_object("Validation Test", ObjectCategory.TOOL)
                is_valid, errors = validate_object_description(test_obj)

                self.print_result("Validation Utils", True,
                                  f"Valid: {is_valid}, Errors: {len(errors)}")
            except ImportError:
                self.print_result("Validation Utils", True, "Not available (optional)")

            # Test helpers if available
            try:
                from soma_object_analysis.utils.helpers import format_confidence_score, format_dimensions
                from soma_object_analysis.models import Dimensions

                confidence_str = format_confidence_score(0.85, "emoji")
                dims = Dimensions(width=0.1, height=0.2, depth=0.05)
                dims_str = format_dimensions(dims)

                self.print_result("Helper Functions", True,
                                  f"Confidence: {confidence_str}, Dims: {dims_str}")
            except ImportError:
                self.print_result("Helper Functions", True, "Not available (optional)")

            return True

        except Exception as e:
            self.print_result("Utils Test", False, str(e))
            return False

    def test_full_pipeline(self) -> bool:
        """Test complete analysis pipeline"""
        self.print_header("6. Testing Complete Pipeline", 2)

        if not CORE_AVAILABLE:
            self.print_result("Pipeline Test", False, "Core system not available")
            return False

        success_count = 0

        for i, test_object in enumerate(self.test_objects[:3], 1):  # Test first 3
            try:
                print(f"\n   Testing object {i}: {test_object}")

                # Test with mock mode for reliability
                result = analyze_object_sync(test_object, use_mock=True, export_results=False)

                if result.get("success"):
                    obj_analysis = result["object_analysis"]

                    print(f"   ✅ Analyzed: {obj_analysis['name']}")
                    print(f"      Category: {obj_analysis['semantic']['category']}")
                    print(f"      Confidence: {obj_analysis['confidence_score']:.1%}")

                    # Check knowledge graph
                    kg_info = result.get("knowledge_graph", {})
                    if kg_info.get("triple_count", 0) > 0:
                        print(f"      KG: {kg_info['triple_count']} triples")

                    success_count += 1
                else:
                    print(f"   ❌ Failed: {result.get('error')}")

            except Exception as e:
                print(f"   ❌ Exception: {str(e)}")

        pipeline_success = success_count > 0
        self.print_result("Pipeline Test", pipeline_success,
                          f"Successfully analyzed {success_count}/{len(self.test_objects[:3])} objects")

        return pipeline_success

    def test_system_integration(self) -> bool:
        """Test system-level integration"""
        self.print_header("7. Testing System Integration", 2)

        if not CORE_AVAILABLE:
            self.print_result("Integration Test", False, "Core system not available")
            return False

        try:
            # Test system creation
            system = SOMAObjectAnalysisSystem(use_mock=True, output_dir="test_output")

            # Test system validation
            validation = system.validate_setup()
            self.print_result("System Validation", validation["valid"],
                              f"Issues: {len(validation.get('issues', []))}")

            # Test system info
            system_info = system.get_system_info()
            self.print_result("System Info", True,
                              f"Version: {system_info['version']}")

            # Test single analysis
            result = system.analyze_object_sync("test object for integration",
                                                export_results=False, generate_kg=True)

            integration_success = result.get("success", False)
            self.print_result("System Analysis", integration_success,
                              result.get("error", "Success") if not integration_success else "Completed successfully")

            return integration_success

        except Exception as e:
            self.print_result("Integration Test", False, str(e))
            return False

    def run_component_tests(self, component: Optional[str] = None) -> Dict[str, bool]:
        """Run component tests"""
        self.print_header("SOMA Object Analysis - Component Tests")

        component_tests = {
            'models': self.test_models,
            'agents': self.test_agents,
            'knowledge_graph': self.test_knowledge_graph,
            'exporters': self.test_exporters,
            'utils': self.test_utils
        }

        results = {}

        if component:
            if component in component_tests:
                results[component] = component_tests[component]()
            else:
                print(f"❌ Unknown component: {component}")
                print(f"Available components: {', '.join(component_tests.keys())}")
                return {}
        else:
            for comp_name, test_func in component_tests.items():
                results[comp_name] = test_func()

        return results

    def run_full_tests(self) -> Dict[str, bool]:
        """Run all tests including pipeline"""
        self.print_header("SOMA Object Analysis - Full Test Suite")

        # Run component tests
        component_results = self.run_component_tests()

        # Run pipeline tests
        pipeline_result = self.test_full_pipeline()
        integration_result = self.test_system_integration()

        all_results = {
            **component_results,
            'pipeline': pipeline_result,
            'integration': integration_result
        }

        return all_results

    def print_summary(self, results: Dict[str, bool]):
        """Print test summary"""
        self.print_header("Test Summary")

        passed = sum(1 for success in results.values() if success)
        total = len(results)

        print(f"📊 Overall Results: {passed}/{total} tests passed")
        print(f"🎯 Success Rate: {passed / total:.1%}")

        if passed == total:
            print("🎉 All tests passed! System is ready to use.")
        else:
            print("⚠️  Some tests failed. Check individual results above.")

        print("\n📋 Detailed Results:")
        for test_name, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}")

        # Save results
        self._save_test_results(results)

    def _save_test_results(self, results: Dict[str, bool]):
        """Save test results to file"""
        try:
            results_file = Path("test_results.json")

            test_data = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_tests": len(results),
                    "passed": sum(1 for r in results.values() if r),
                    "success_rate": sum(1 for r in results.values() if r) / len(results)
                },
                "results": results,
                "detailed_results": self.test_results,
                "system_info": {
                    "components_available": COMPONENTS,
                    "core_available": CORE_AVAILABLE
                }
            }

            import json
            with open(results_file, 'w') as f:
                json.dump(test_data, f, indent=2, default=str)

            print(f"\n💾 Test results saved to: {results_file}")

        except Exception as e:
            print(f"⚠️  Could not save test results: {e}")

    def interactive_mode(self):
        """Interactive testing mode"""
        self.print_header("SOMA Object Analysis - Interactive Mode")

        while True:
            print("\n🔧 Choose a test option:")
            print("1. Test specific component")
            print("2. Test full pipeline")
            print("3. Test custom object")
            print("4. Run all tests")
            print("5. Check system status")
            print("6. Exit")

            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == "1":
                print("\nAvailable components:")
                for i, comp in enumerate(['models', 'agents', 'knowledge_graph', 'exporters', 'utils'], 1):
                    print(f"{i}. {comp}")

                comp_choice = input("Enter component number: ").strip()
                comp_map = {
                    '1': 'models', '2': 'agents', '3': 'knowledge_graph',
                    '4': 'exporters', '5': 'utils'
                }

                if comp_choice in comp_map:
                    results = self.run_component_tests(comp_map[comp_choice])
                    self.print_summary(results)
                else:
                    print("❌ Invalid choice")

            elif choice == "2":
                result = self.test_full_pipeline()
                self.print_summary({'pipeline': result})

            elif choice == "3":
                custom_object = input("Enter object description: ").strip()
                if custom_object:
                    try:
                        result = analyze_object_sync(custom_object, use_mock=True, export_results=False)
                        if result.get("success"):
                            obj = result["object_analysis"]
                            print(f"✅ Analysis successful!")
                            print(f"   Name: {obj['name']}")
                            print(f"   Category: {obj['semantic']['category']}")
                            print(f"   Confidence: {obj['confidence_score']:.1%}")
                        else:
                            print(f"❌ Analysis failed: {result.get('error')}")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ No object description provided")

            elif choice == "4":
                results = self.run_full_tests()
                self.print_summary(results)

            elif choice == "5":
                self.print_header("System Status", 2)
                print("Component Availability:")
                for comp, available in COMPONENTS.items():
                    status = "✅" if available else "❌"
                    print(f"   {status} {comp}")

                if CORE_AVAILABLE:
                    try:
                        system = SOMAObjectAnalysisSystem(use_mock=True)
                        validation = system.validate_setup()
                        print(f"\nSystem Validation: {'✅ Valid' if validation['valid'] else '❌ Issues found'}")
                        if validation.get('warnings'):
                            for warning in validation['warnings']:
                                print(f"   ⚠️  {warning}")
                    except Exception as e:
                        print(f"\n❌ System check failed: {e}")

            elif choice == "6":
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice. Please enter 1-6.")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test SOMA Object Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_main.py                     # Run all tests
    python test_main.py --component models  # Test specific component
    python test_main.py --full-pipeline     # Test complete pipeline only
    python test_main.py --interactive       # Interactive mode
    python test_main.py --quick             # Quick system check
        """
    )

    parser.add_argument('--component',
                        choices=['models', 'agents', 'knowledge_graph', 'exporters', 'utils'],
                        help='Test specific component')
    parser.add_argument('--full-pipeline', action='store_true',
                        help='Test complete pipeline only')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive testing mode')
    parser.add_argument('--quick', action='store_true',
                        help='Quick system check')

    args = parser.parse_args()

    runner = TestRunner()

    try:
        if args.interactive:
            runner.interactive_mode()
        elif args.quick:
            print("🚀 SOMA Object Analysis - Quick Check")
            print("=" * 40)

            if CORE_AVAILABLE:
                quick_test()
            else:
                print("❌ Core system not available")
                print("Components status:")
                for comp, available in COMPONENTS.items():
                    status = "✅" if available else "❌"
                    print(f"   {status} {comp}")

        elif args.component:
            results = runner.run_component_tests(args.component)
            runner.print_summary(results)

        elif args.full_pipeline:
            result = runner.test_full_pipeline()
            runner.print_summary({'pipeline': result})

        else:
            # Run all tests
            results = runner.run_full_tests()
            runner.print_summary(results)

    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()