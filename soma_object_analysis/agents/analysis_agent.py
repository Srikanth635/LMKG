"""
LangGraph-based object analysis agent (synchronous version)
"""

from datetime import datetime
from typing import Optional
import os
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()


try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from .base_agent import AnalysisAgent, AgentState
from ..models import ObjectDescription
from ..config import config
from ..gcp_ollama import GcpChatOllama
from ..trails_gcp_ollama import ChatGCPOllama
from ..trails_openrouter_ollama import OpenRouterChatModel

class LangGraphAnalysisAgent(AnalysisAgent):
    """LangGraph agent for structured object analysis"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph = None

        if LANGGRAPH_AVAILABLE and config.OPENAI_API_KEY:
            self._initialize_llm()
            self._build_graph()
        else:
            self.logger.warning("LangGraph or OpenAI API key not available")

    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            # self.llm = ChatOpenAI(
            #     model=self.model_name,
            #     # temperature=self.temperature,
            #     # max_tokens=config.MAX_TOKENS
            # )
            # self.structured_llm = self.llm.with_structured_output(ObjectDescription, method="function_calling")

            ollama_turbo_api_key = os.getenv('OLLAMA_TURBO_API_KEY', "")
            client_kw_args = {
                    # "host": "https://ollama.com",
                    "headers": {'Authorization': ollama_turbo_api_key}
                }
            self.llm = ChatOllama(model='gpt-oss:120b',client_kwargs=client_kw_args,base_url="https://ollama.com",extract_reasoning=True)
            self.structured_llm = self.llm.with_structured_output(ObjectDescription,method="json_schema")

            # self.llm = ChatOllama(model="gpt-oss:20b", temperature=0.5, extract_reasoning=True)
            # self.structured_llm = self.llm.with_structured_output(ObjectDescription,method="json_mode")

            # self.llm = GcpChatOllama(model="qwen3:14b", temperature=0.4)
            # self.structured_llm = self.llm.with_structured_output(ObjectDescription,method="json_schema")

            # self.llm = ChatGCPOllama(model="qwen3:14b", temperature=0.4)
            # self.structured_llm = self.llm.with_structured_output(ObjectDescription,method="json_schema")

            # self.llm = OpenRouterChatModel(model="openai/gpt-oss-20b")
            # self.structured_llm = self.llm.with_structured_output(ObjectDescription,method="function_calling")

            self.logger.info(f"Initialized LLM: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
            self.structured_llm = None

    def _build_graph(self):
        """Build the LangGraph workflow"""
        if not self.structured_llm:
            return

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_object", self._analyze_object_node)
        workflow.add_node("validate_analysis", self._validate_analysis_node)
        workflow.add_node("enrich_analysis", self._enrich_analysis_node)

        # Add edges
        workflow.set_entry_point("analyze_object")
        workflow.add_edge("analyze_object", "validate_analysis")
        workflow.add_edge("validate_analysis", "enrich_analysis")
        workflow.add_edge("enrich_analysis", END)

        self.graph = workflow.compile()
        self.logger.info("LangGraph workflow built successfully")

    def _analyze_object_node(self, state: AgentState) -> AgentState:
        """Node: Analyze object using structured output"""
        self.logger.info(f"Analyzing object: {state.input_description[:50]}...")

        try:
            system_prompt = self.get_system_prompt()

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Analyze this object: {state.input_description}")
            ]

            # message = system_prompt + "\n\n" + f"Analyze this object: {state.input_description}"

            # Get structured output (synchronous)
            result = self.structured_llm.invoke(messages)
            print("Object Analysis: ", type(result), result)
            state.object_analysis = result
            state.metadata["analysis_timestamp"] = datetime.now().isoformat()
            state.metadata["model_used"] = self.model_name

            self.logger.info(f"Successfully analyzed: {result.name}")
            return state

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            state.errors.append(error_msg)
            self.logger.error(error_msg)
            return state

    def _validate_analysis_node(self, state: AgentState) -> AgentState:
        """Node: Validate the analysis results"""
        if not state.object_analysis:
            error_msg = "No analysis to validate"
            state.errors.append(error_msg)
            self.logger.error(error_msg)
            return state

        self.logger.info("Validating analysis results...")

        try:
            from .validation_agent import SOMAValidationAgent
            validator = SOMAValidationAgent()

            is_valid, validation_errors = validator.validate_analysis(state.object_analysis)
            print("All validation_errors: ", validation_errors)
            if not is_valid:
                state.errors.extend(validation_errors)
                self.logger.warning(f"Validation failed: {len(validation_errors)} errors")
            else:
                self.logger.info("Analysis validation passed")

            state.metadata["validation_passed"] = is_valid
            state.metadata["validation_errors"] = len(validation_errors)

        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            state.errors.append(error_msg)

            self.logger.error(error_msg)

        return state

    def _enrich_analysis_node(self, state: AgentState) -> AgentState:
        """Node: Enrich analysis with additional information"""
        if not state.object_analysis:
            return state

        self.logger.info("Enriching analysis...")

        try:
            # Add source information
            state.object_analysis.source = f"langgraph_{self.model_name}"

            # Calculate additional metrics
            if state.object_analysis.geometric.shape.dimensions:
                volume = state.object_analysis.geometric.shape.dimensions.volume()
                if volume:
                    state.metadata["estimated_volume"] = volume

            # Add capability scores
            caps = state.object_analysis.capabilities.functional_affordances
            capability_count = sum(1 for attr in [caps.can_cut, caps.can_contain, caps.can_grasp] if attr)
            state.metadata["capability_count"] = capability_count

            self.logger.info("Analysis enrichment completed")

        except Exception as e:
            error_msg = f"Enrichment failed: {str(e)}"
            state.errors.append(error_msg)
            self.logger.error(error_msg)

        return state

    def process(self, state: AgentState) -> AgentState:
        """Process the agent state using LangGraph (synchronous)"""
        if not self.graph:
            state.errors.append("LangGraph not available")
            return state

        if not self.validate_input(state):
            state.errors.append("Invalid input description")
            return state

        try:
            final_state = self.graph.invoke(state)
            self.logger.info("LangGraph processing completed")
            return final_state

        except Exception as e:
            error_msg = f"Graph processing failed: {str(e)}"
            state.errors.append(error_msg)
            self.logger.error(error_msg)
            return state

    def analyze(self, input_description: str) -> dict:
        """Main analysis method (synchronous)"""
        initial_state = AgentState(input_description=input_description)
        final_state = self.process(initial_state)
        print("FINAL STATE : ",type(final_state),final_state)
        if isinstance(final_state, dict):
            out = {
                        "success": len(final_state["errors"]) == 0,
                        "object_analysis": final_state["object_analysis"].model_dump() if final_state["object_analysis"] else None,
                        "errors": final_state["errors"],
                        "metadata": final_state["metadata"]
                    }

            return out

        else:
            out = {
                "success": len(final_state.errors) == 0,
                "object_analysis": final_state.object_analysis.dict() if final_state.object_analysis else None,
                "errors": final_state.errors,
                "metadata": final_state.metadata
            }
            return out