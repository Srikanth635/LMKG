"""
Main system orchestrator that ties all components together
"""

import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from .config import config
from .agents.mock_agent import MockAnalysisAgent
from .knowledge_graph.generator import KnowledgeGraphGenerator
from .exporters.results_exporter import ResultsExporter

# Import LangGraph agent with error handling
try:
    from .agents.analysis_agent import LangGraphAnalysisAgent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    LangGraphAnalysisAgent = None
from .models import ObjectDescription
from .utils.logging import setup_logging

class SOMAObjectAnalysisSystem:
    """Main system class that orchestrates the entire analysis pipeline"""

    def __init__(self,
                 model_name: str = None,
                 use_mock: bool = False,
                 output_dir: str = None,
                 enable_validation: bool = None):
        """
        Initialize the SOMA Object Analysis System

        Args:
            model_name: LLM model to use (default from config)
            use_mock: Whether to use mock agent instead of real LLM
            output_dir: Output directory for results
            enable_validation: Whether to enable validation steps
        """

        # Setup logging
        self.logger = setup_logging()

        # Configuration
        self.model_name = model_name or config.DEFAULT_MODEL
        self.use_mock = use_mock or not config.OPENAI_API_KEY
        self.output_dir = Path(output_dir) if output_dir else config.get_output_dir()
        self.enable_validation = enable_validation if enable_validation is not None else config.ENABLE_VALIDATION

        # Initialize components
        self.agent = self._initialize_agent()
        self.kg_generator = KnowledgeGraphGenerator()
        self.exporter = ResultsExporter(self.output_dir)

        self.logger.info(f"Initialized SOMA Analysis System (mock={self.use_mock})")

    def _initialize_agent(self):
        """Initialize the appropriate analysis agent"""
        if self.use_mock or not LANGGRAPH_AVAILABLE:
            self.logger.info("Using mock analysis agent")
            return MockAnalysisAgent()
        else:
            try:
                agent = LangGraphAnalysisAgent(model_name=self.model_name)
                if hasattr(agent, 'graph') and agent.graph is None:
                    self.logger.warning("LangGraph initialization failed, falling back to mock")
                    return MockAnalysisAgent()
                return agent
            except Exception as e:
                self.logger.error(f"Failed to initialize LangGraph agent: {e}")
                self.logger.info("Falling back to mock agent")
                return MockAnalysisAgent()

    async def analyze_object(self, description: str,
                           export_results: bool = True,
                           generate_kg: bool = True) -> Dict[str, Any]:
        """
        Main analysis pipeline

        Args:
            description: Text description of the object to analyze
            export_results: Whether to export results to files
            generate_kg: Whether to generate knowledge graph

        Returns:
            Dictionary containing analysis results, KG, and export paths
        """

        self.logger.info(f"Starting analysis: '{description[:50]}...'")

        try:
            # Step 1: Object Analysis
            analysis_result = await self.agent.analyze(description)

            # if not analysis_result.get("success"):
            #     self.logger.error("Object analysis failed")
            #     return {
            #         "success": False,
            #         "error": "Object analysis failed",
            #         "details": analysis_result
            #     }

            object_analysis = ObjectDescription(**analysis_result["object_analysis"])
            self.logger.info(f"Analysis completed: {object_analysis.name}")

            # Step 2: Knowledge Graph Generation
            knowledge_graph = None
            kg_turtle = None

            if generate_kg:
                try:
                    knowledge_graph = self.kg_generator.create_from_object_description(object_analysis)
                    kg_turtle = knowledge_graph.export_turtle()
                    self.logger.info(f"Knowledge graph generated with {len(knowledge_graph.triples)} triples")
                except Exception as e:
                    self.logger.error(f"Knowledge graph generation failed: {e}")

            # Step 3: Export Results
            export_paths = []
            if export_results:
                try:
                    export_paths = await self.exporter.export_all(
                        object_description=object_analysis,
                        knowledge_graph_turtle=kg_turtle,
                        metadata=analysis_result.get("metadata", {})
                    )
                    self.logger.info(f"Results exported to {len(export_paths)} files")
                except Exception as e:
                    self.logger.error(f"Export failed: {e}")

            # Compile final results
            result = {
                "success": True,
                "object_analysis": object_analysis.dict(),
                "knowledge_graph": {
                    "turtle": kg_turtle,
                    "instance_count": len(knowledge_graph.instances) if knowledge_graph else 0,
                    "triple_count": len(knowledge_graph.triples) if knowledge_graph else 0
                },
                "export_paths": export_paths,
                "metadata": {
                    **analysis_result.get("metadata", {}),
                    "system_version": "0.1.0",
                    "agent_type": "mock" if self.use_mock else "langgraph",
                    "model_used": self.model_name if not self.use_mock else "mock"
                }
            }

            self.logger.info("Analysis pipeline completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Analysis pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {"error_type": type(e).__name__}
            }

    def analyze_object_sync(self, description: str, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for analyze_object"""
        return asyncio.run(self.analyze_object(description, **kwargs))

    async def analyze_batch(self, descriptions: list[str],
                          max_concurrent: int = 3) -> list[Dict[str, Any]]:
        """
        Analyze multiple objects concurrently

        Args:
            descriptions: List of object descriptions to analyze
            max_concurrent: Maximum number of concurrent analyses

        Returns:
            List of analysis results
        """

        self.logger.info(f"Starting batch analysis of {len(descriptions)} objects")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(desc):
            async with semaphore:
                return await self.analyze_object(desc)

        results = await asyncio.gather(
            *[analyze_with_semaphore(desc) for desc in descriptions],
            return_exceptions=True
        )

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "input_description": descriptions[i]
                })
            else:
                processed_results.append(result)

        successful = sum(1 for r in processed_results if r.get("success"))
        self.logger.info(f"Batch analysis completed: {successful}/{len(descriptions)} successful")

        return processed_results

    def get_system_info(self) -> Dict[str, Any]:
        """Get system configuration and status information"""
        return {
            "version": "0.1.0",
            "configuration": {
                "model_name": self.model_name,
                "use_mock": self.use_mock,
                "output_dir": str(self.output_dir),
                "enable_validation": self.enable_validation
            },
            "components": {
                "agent_type": type(self.agent).__name__,
                "kg_generator": type(self.kg_generator).__name__,
                "exporter": type(self.exporter).__name__
            },
            "capabilities": {
                "langgraph_available": not self.use_mock,
                "openai_configured": bool(config.OPENAI_API_KEY),
                "export_formats": ["json", "html", "markdown", "turtle"]
            }
        }

    def validate_setup(self) -> Dict[str, Any]:
        """Validate system setup and dependencies"""
        issues = []
        warnings = []

        # Check API key
        if not config.OPENAI_API_KEY:
            warnings.append("OpenAI API key not configured - using mock mode")

        # Check output directory
        if not self.output_dir.exists():
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create output directory: {e}")

        # Check agent initialization
        if hasattr(self.agent, 'graph') and self.agent.graph is None:
            warnings.append("LangGraph agent not properly initialized")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "system_info": self.get_system_info()
        }

# Factory function for easy instantiation
def create_analysis_system(**kwargs) -> SOMAObjectAnalysisSystem:
    """Factory function to create a configured analysis system"""
    return SOMAObjectAnalysisSystem(**kwargs)

# Convenience functions
async def analyze_object(description: str, **kwargs) -> Dict[str, Any]:
    """Quick analysis function"""
    system = create_analysis_system(**kwargs)
    return await system.analyze_object(description)

def analyze_object_sync(description: str, **kwargs) -> Dict[str, Any]:
    """Quick synchronous analysis function"""
    system = create_analysis_system(**kwargs)
    return system.analyze_object_sync(description)