"""
LangGraph-based object analysis agent
"""

from datetime import datetime
from typing import Optional

try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langchain_ollama import ChatOllama

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from ..models import ObjectDescription
from ..config import config
from .base_agent import AnalysisAgent,AgentState

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
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=config.MAX_TOKENS
            )
            # self.llm = ChatOllama(model="qwen3:14b")
            # self.structured_llm = self.llm.with_structured_output(ObjectDescription, method="json_schema")
            # self.logger.info(f"Initialized LLM: OLLAMA")

            self.structured_llm = self.llm.with_structured_output(ObjectDescription, method="json_schema")
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

    async def _analyze_object_node(self, state: dict) -> dict:
        """Node: Analyze object using structured output"""
        input_desc = state.get('input_description', '')
        self.logger.info(f"Analyzing object: {input_desc[:50]}...")

        try:
            system_prompt = self.get_system_prompt()

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Analyze this object: {input_desc}")
            ]

            # Get structured output
            result = await self.structured_llm.ainvoke(messages)

            # Update state dictionary
            state['object_analysis'] = result

            # Initialize metadata if it doesn't exist
            if 'metadata' not in state:
                state['metadata'] = {}

            state['metadata']["analysis_timestamp"] = datetime.now().isoformat()
            state['metadata']["model_used"] = self.model_name

            self.logger.info(f"Successfully analyzed: {result.name}")
            return state

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"

            # Initialize errors list if it doesn't exist
            if 'errors' not in state:
                state['errors'] = []

            state['errors'].append(error_msg)
            self.logger.error(error_msg)
            return state

    async def _validate_analysis_node(self, state: dict) -> dict:
        """Node: Validate the analysis results"""
        object_analysis = state.get('object_analysis')

        if not object_analysis:
            error_msg = "No analysis to validate"

            # Initialize errors list if it doesn't exist
            if 'errors' not in state:
                state['errors'] = []

            state['errors'].append(error_msg)
            self.logger.error(error_msg)
            return state

        self.logger.info("Validating analysis results...")

        try:
            from .validation_agent import SOMAValidationAgent
            validator = SOMAValidationAgent()

            is_valid, validation_errors = validator.validate_analysis(object_analysis)

            if not is_valid:
                # Initialize errors list if it doesn't exist
                if 'errors' not in state:
                    state['errors'] = []

                state['errors'].extend(validation_errors)
                self.logger.warning(f"Validation failed: {len(validation_errors)} errors")
            else:
                self.logger.info("Analysis validation passed")

            # Initialize metadata if it doesn't exist
            if 'metadata' not in state:
                state['metadata'] = {}

            state['metadata']["validation_passed"] = is_valid
            state['metadata']["validation_errors"] = len(validation_errors)

        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"

            # Initialize errors list if it doesn't exist
            if 'errors' not in state:
                state['errors'] = []

            state['errors'].append(error_msg)
            self.logger.error(error_msg)

        return state

    async def _enrich_analysis_node(self, state: dict) -> dict:
        """Node: Enrich analysis with additional information"""
        object_analysis = state.get('object_analysis')

        if not object_analysis:
            return state

        self.logger.info("Enriching analysis...")

        try:
            # Add source information
            object_analysis.source = f"langgraph_{self.model_name}"

            # Calculate additional metrics
            if hasattr(object_analysis, 'geometric') and \
                    hasattr(object_analysis.geometric, 'shape') and \
                    hasattr(object_analysis.geometric.shape, 'dimensions'):

                dimensions = object_analysis.geometric.shape.dimensions
                if hasattr(dimensions, 'volume'):
                    volume = dimensions.volume()
                    if volume:
                        # Initialize metadata if it doesn't exist
                        if 'metadata' not in state:
                            state['metadata'] = {}
                        state['metadata']["estimated_volume"] = volume

            # Add capability scores
            if hasattr(object_analysis, 'capabilities') and \
                    hasattr(object_analysis.capabilities, 'functional_affordances'):

                caps = object_analysis.capabilities.functional_affordances
                capability_count = 0

                # Safely check for capability attributes
                if hasattr(caps, 'can_cut') and caps.can_cut:
                    capability_count += 1
                if hasattr(caps, 'can_contain') and caps.can_contain:
                    capability_count += 1
                if hasattr(caps, 'can_grasp') and caps.can_grasp:
                    capability_count += 1

                # Initialize metadata if it doesn't exist
                if 'metadata' not in state:
                    state['metadata'] = {}
                state['metadata']["capability_count"] = capability_count

            self.logger.info("Analysis enrichment completed")

        except Exception as e:
            error_msg = f"Enrichment failed: {str(e)}"

            # Initialize errors list if it doesn't exist
            if 'errors' not in state:
                state['errors'] = []

            state['errors'].append(error_msg)
            self.logger.error(error_msg)

        return state

    # async def _analyze_object_node(self, state: AgentState) -> AgentState:
    #     """Node: Analyze object using structured output"""
    #     self.logger.info(f"Analyzing object: {state.input_description[:50]}...")
    #
    #     try:
    #         system_prompt = self.get_system_prompt()
    #
    #         messages = [
    #             SystemMessage(content=system_prompt),
    #             HumanMessage(content=f"Analyze this object: {state.input_description}")
    #         ]
    #
    #         # Get structured output
    #         result = await self.structured_llm.ainvoke(messages)
    #
    #         state.object_analysis = result
    #         state.metadata["analysis_timestamp"] = datetime.now().isoformat()
    #         state.metadata["model_used"] = self.model_name
    #
    #         self.logger.info(f"Successfully analyzed: {result.name}")
    #         return state
    #
    #     except Exception as e:
    #         error_msg = f"Analysis failed: {str(e)}"
    #         state.errors.append(error_msg)
    #         self.logger.error(error_msg)
    #         return state
    #
    # async def _validate_analysis_node(self, state: AgentState) -> AgentState:
    #     """Node: Validate the analysis results"""
    #     if not state.object_analysis:
    #         error_msg = "No analysis to validate"
    #         state.errors.append(error_msg)
    #         self.logger.error(error_msg)
    #         return state
    #
    #     self.logger.info("Validating analysis results...")
    #
    #     try:
    #         from .validation_agent import SOMAValidationAgent
    #         validator = SOMAValidationAgent()
    #
    #         is_valid, validation_errors = validator.validate_analysis(state.object_analysis)
    #
    #         if not is_valid:
    #             state.errors.extend(validation_errors)
    #             self.logger.warning(f"Validation failed: {len(validation_errors)} errors")
    #         else:
    #             self.logger.info("Analysis validation passed")
    #
    #         state.metadata["validation_passed"] = is_valid
    #         state.metadata["validation_errors"] = len(validation_errors)
    #
    #     except Exception as e:
    #         error_msg = f"Validation failed: {str(e)}"
    #         state.errors.append(error_msg)
    #         self.logger.error(error_msg)
    #
    #     return state
    #
    # async def _enrich_analysis_node(self, state: AgentState) -> AgentState:
    #     """Node: Enrich analysis with additional information"""
    #     if not state.object_analysis:
    #         return state
    #
    #     self.logger.info("Enriching analysis...")
    #
    #     try:
    #         # Add source information
    #         state.object_analysis.source = f"langgraph_{self.model_name}"
    #
    #         # Calculate additional metrics
    #         if state.object_analysis.geometric.shape.dimensions:
    #             volume = state.object_analysis.geometric.shape.dimensions.volume()
    #             if volume:
    #                 state.metadata["estimated_volume"] = volume
    #
    #         # Add capability scores
    #         caps = state.object_analysis.capabilities.functional_affordances
    #         capability_count = sum(1 for attr in [caps.can_cut, caps.can_contain, caps.can_grasp] if attr)
    #         state.metadata["capability_count"] = capability_count
    #
    #         self.logger.info("Analysis enrichment completed")
    #
    #     except Exception as e:
    #         error_msg = f"Enrichment failed: {str(e)}"
    #         state.errors.append(error_msg)
    #         self.logger.error(error_msg)
    #
    #     return state

    async def process(self, state) -> AgentState:
        """Process the agent state using LangGraph"""
        self.logger.info(f"Starting process method, state type: {type(state)}")

        # Convert dict to AgentState if needed
        if isinstance(state, dict):
            self.logger.info("Converting dict to AgentState")
            agent_state = AgentState(
                input_description=state.get('input_description', ''),
                errors=state.get('errors', [])
            )
        else:
            agent_state = state

        if not self.graph:
            error_msg = "LangGraph not available"
            self.logger.error(error_msg)
            agent_state.errors.append(error_msg)
            return agent_state

        if not self.validate_input(state):  # Pass original state to validate_input
            error_msg = "Invalid input description"
            self.logger.error(error_msg)
            agent_state.errors.append(error_msg)
            return agent_state

        self.logger.info("About to invoke graph")
        try:
            # LangGraph might expect a dict, so convert back

            state_dict = {
                'input_description': agent_state["input_description"],
                'errors': agent_state["errors"]
            }

            final_state_dict = await self.graph.ainvoke(state_dict)

            # # Convert result back to AgentState
            # final_state = AgentState(
            #     input_description=final_state_dict.get('input_description', ''),
            #     errors=final_state_dict.get('errors', [])
            # )

            self.logger.info("LangGraph processing completed successfully")
            return final_state_dict

        except Exception as e:
            error_msg = f"Graph processing failed: {str(e)}"
            agent_state.errors.append(error_msg)
            self.logger.error(error_msg)
            self.logger.error(f"Exception type: {type(e)}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return agent_state

    # async def process(self, state: AgentState) -> AgentState:
    #     """Process the agent state using LangGraph"""
    #     self.logger.info("Starting process method")
    #
    #     if not self.graph:
    #         error_msg = "LangGraph not available"
    #         self.logger.error(error_msg)
    #         state.errors.append(error_msg)
    #         return state
    #
    #     if not self.validate_input(state):
    #         error_msg = "Invalid input description"
    #         self.logger.error(error_msg)
    #         state.errors.append(error_msg)
    #         return state
    #
    #     self.logger.info("About to invoke graph")
    #     try:
    #         final_state = await self.graph.ainvoke(state)
    #         self.logger.info("LangGraph processing completed successfully")
    #         return final_state
    #
    #     except Exception as e:
    #         error_msg = f"Graph processing failed: {str(e)}"
    #         state.errors.append(error_msg)
    #         self.logger.error(error_msg)
    #         self.logger.error(f"Exception type: {type(e)}")
    #         import traceback
    #         self.logger.error(f"Full traceback: {traceback.format_exc()}")
    #         return state

    async def analyze(self, input_description: str) -> dict:
        """Main analysis method"""
        self.logger.info(f"Starting analyze with input: {input_description}")

        initial_state = AgentState(input_description=input_description)
        self.logger.info(f"Created initial state: {initial_state}")

        try:
            final_state = await self.process(initial_state)
            self.logger.info(f"Process completed, final state: {final_state}")
            return final_state
        except Exception as e:
            self.logger.error(f"Analyze method failed: {str(e)}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    # async def process(self, state: AgentState) -> AgentState:
    #     """Process the agent state using LangGraph"""
    #     if not self.graph:
    #         state.errors.append("LangGraph not available")
    #         return state
    #
    #     if not self.validate_input(state):
    #         state.errors.append("Invalid input description")
    #         return state
    #
    #     try:
    #         final_state = await self.graph.ainvoke(state)
    #         self.logger.info("LangGraph processing completed")
    #         return final_state
    #
    #     except Exception as e:
    #         error_msg = f"Graph processing failed: {str(e)}"
    #         state.errors.append(error_msg)
    #         self.logger.error(error_msg)
    #         return state
    #
    # async def analyze(self, input_description: str) -> dict:
    #     """Main analysis method"""
    #     initial_state = AgentState(input_description=input_description)
    #     final_state = await self.process(initial_state)

        return {
            "success": len(final_state.errors) == 0,
            "object_analysis": final_state.object_analysis.dict() if final_state.object_analysis else None,
            "errors": final_state.errors,
            "metadata": final_state.metadata
        }