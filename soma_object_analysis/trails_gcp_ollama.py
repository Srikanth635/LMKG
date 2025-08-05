import asyncio
import re
import json
import logging
import typing
from _operator import itemgetter
from typing import Any, Dict, Iterator, List, Optional, Union, AsyncIterator, Sequence, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
)
import requests
import aiohttp
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.utils import get_from_dict_or_env
# Use only Pydantic v2
from pydantic import Field, model_validator, BaseModel

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)

from langchain_core.language_models.base import (
    LanguageModelInput,
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult
)

if TYPE_CHECKING:
    import uuid

    from langchain_core.output_parsers.base import OutputParserLike
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

def think_remover(res: str):
    if re.search(r"<think>.*?</think>", res, flags=re.DOTALL):
        cleaned_res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
    else:
        cleaned_res = res.strip()
    return cleaned_res

class ChatGCPOllama(BaseChatModel):
    """Custom chat model for GCP-hosted Ollama API that mimics ChatOllama behavior."""

    base_url: str =  Field(default="https://ollama-qwen3-948194141289.us-central1.run.app")
    model: str = Field(default="qwen3:14b")
    temperature: Optional[float] = Field(default=None)
    top_k: int = 40
    top_p: float = 0.9
    num_predict: int = -1
    stop: Optional[List[str]] = None
    timeout: int = Field(default=2000)

    @model_validator(mode='before')
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that base_url is provided."""
        if isinstance(values, dict):
            base_url = get_from_dict_or_env(values, "base_url", "GCP_OLLAMA_BASE_URL", default="")
            if not base_url:
                raise ValueError(
                    "base_url must be provided either directly or via GCP_OLLAMA_BASE_URL environment variable")
            values["base_url"] = base_url.rstrip("/")
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "gcp-ollama"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages to a single prompt string."""
        prompt_parts = []

        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
            else:
                prompt_parts.append(f"{message.__class__.__name__}: {message.content}")

        return "\n\n".join(prompt_parts)

    def _create_payload(self, prompt: str) -> Dict[str, Any]:
        """Create the payload for the API request."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
            }
        }

        if self.num_predict > 0:
            payload["options"]["num_predict"] = self.num_predict

        if self.stop:
            payload["options"]["stop"] = self.stop

        return payload

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response synchronously."""
        prompt = self._convert_messages_to_prompt(messages)

        # Override stop sequences if provided
        if stop:
            self.stop = stop

        # Check if tools are bound (for structured output)
        if hasattr(self, '_bound_tools') and self._bound_tools:
            prompt += f"\n\nYou must respond with a valid JSON object that matches one of these schemas: {self._format_tools_for_api(self._bound_tools)}"

        payload = self._create_payload(prompt)
        url = f"{self.base_url}/api/generate"

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            result = response.json()

            # Extract the response text
            response_text = think_remover(result.get("response", ""))
            if not response_text:
                logger.warning("Empty response from API")
                response_text = ""

            # Create chat generation
            message = AIMessage(content=response_text)
            generation = ChatGeneration(message=message)

            # Extract token usage if available
            llm_output = {}
            if "eval_count" in result:
                llm_output["token_usage"] = {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }

            return ChatResult(generations=[generation], llm_output=llm_output)

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise ValueError(f"Failed to call GCP Ollama API: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            raise ValueError(f"Invalid JSON response from API: {e}")

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response asynchronously."""
        prompt = self._convert_messages_to_prompt(messages)

        # Override stop sequences if provided
        if stop:
            self.stop = stop

        # Check if tools are bound (for structured output)
        if hasattr(self, '_bound_tools') and self._bound_tools:
            prompt += f"\n\nYou must respond with a valid JSON object that matches one of these schemas: {self._format_tools_for_api(self._bound_tools)}"

        payload = self._create_payload(prompt)
        url = f"{self.base_url}/api/generate"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        headers={"Content-Type": "application/json"}
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

            # Extract the response text
            response_text = result.get("response", "")
            if not response_text:
                logger.warning("Empty response from API")
                response_text = ""

            # Create chat generation
            message = AIMessage(content=response_text)
            generation = ChatGeneration(message=message)

            # Extract token usage if available
            llm_output = {}
            if "eval_count" in result:
                llm_output["token_usage"] = {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }

            return ChatResult(generations=[generation], llm_output=llm_output)

        except aiohttp.ClientError as e:
            logger.error(f"Async API request failed: {e}")
            raise ValueError(f"Failed to call GCP Ollama API: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            raise ValueError(f"Invalid JSON response from API: {e}")

    def with_structured_output(
            self,
            schema: Union[Dict, type],
            include_raw: bool = False,
            **kwargs: Any,
    ) -> Any:
        """Return a model that can produce structured output.

        Args:
            schema: The output schema. Can be a Pydantic model class or a dictionary.
            include_raw: Whether to include the raw response alongside the parsed output.
            **kwargs: Additional keyword arguments.

        Returns:
            A runnable that produces structured output.
        """
        try:
            from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough

            # Determine if schema is a Pydantic model or dict
            # if hasattr(schema, "__annotations__") and hasattr(schema, "__fields__"):
            if hasattr(schema, "model_fields"):
                # It's a Pydantic model
                parser = PydanticOutputParser(pydantic_object=schema)
                format_instructions = parser.get_format_instructions()
            elif isinstance(schema, type) and issubclass(schema, BaseModel):
                # It's a Pydantic model class
                parser = PydanticOutputParser(pydantic_object=schema)
                format_instructions = parser.get_format_instructions()
            elif isinstance(schema, dict):
                # It's a dictionary schema
                parser = JsonOutputParser()
                format_instructions = f"Please respond with a JSON object that matches this schema: {json.dumps(schema, indent=2)}"
            else:
                raise ValueError(f"Unsupported schema type: {type(schema)}")

            # Create a prompt template that includes format instructions
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Follow the output format instructions carefully."),
                ("human", "{input}\n\nOutput format instructions:\n{format_instructions}"),
            ])

            if include_raw:
                # Return both raw and parsed output
                def parse_with_raw(response):
                    try:
                        parsed = parser.parse(response.content)
                        return {"raw": response, "parsed": parsed}
                    except Exception as e:
                        logger.warning(f"Failed to parse structured output: {e}")
                        return {"raw": response, "parsed": None}

                chain = (
                        {
                            "input": RunnablePassthrough(),
                            "format_instructions": lambda _: format_instructions
                        }
                        | prompt_template
                        | self
                        | parse_with_raw
                )
            else:
                # Return only parsed output
                def parse_response(response):
                    try:
                        return parser.parse(response.content)
                    except Exception as e:
                        logger.warning(f"Failed to parse structured output: {e}")
                        # Try to extract JSON from the response if parsing fails
                        import re
                        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                        if json_match:
                            try:
                                json_data = json.loads(json_match.group())
                                if isinstance(schema, dict):
                                    return json_data
                                elif hasattr(schema, "parse_obj"):
                                    return schema.parse_obj(json_data)
                                elif hasattr(schema, "model_validate"):
                                    return schema.model_validate(json_data)
                            except:
                                pass
                        raise ValueError(f"Could not parse response into structured format: {e}")

                chain = (
                        {
                            "input": RunnablePassthrough(),
                            "format_instructions": lambda _: format_instructions
                        }
                        | prompt_template
                        | self
                        | parse_response
                )

            return chain

        except ImportError as e:
            raise ImportError(f"Required dependencies not available for structured output: {e}")

    def bind_tools(
            self,
            tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
            *,
            tool_choice: Optional[Union[str]] = None,
            **kwargs: Any,
    ) -> "Runnable[LanguageModelInput, BaseMessage]":
        """Bind tools to the model - minimal implementation for structured output."""
        from copy import deepcopy

        # Create a new instance with tools bound
        new_instance = deepcopy(self)
        new_instance._bound_tools = tools
        new_instance._tool_choice = tool_choice
        new_instance._bind_kwargs = kwargs

        return new_instance

    def _format_tools_for_api(self, tools, tool_choice=None):
        """Convert tools to format expected by your API."""
        # This is a simplified version - you may need to adapt based on your API
        formatted_tools = []
        for tool in tools:
            if hasattr(tool, '__name__'):
                # It's a Pydantic class
                formatted_tools.append({
                    "name": tool.__name__,
                    "description": tool.__doc__ or "",
                    "parameters": tool.model_json_schema() if hasattr(tool, 'model_json_schema') else {}
                })
            elif isinstance(tool, dict):
                formatted_tools.append(tool)

        return formatted_tools


# Example usage and structured output classes
class MathResult(BaseModel):
    """A mathematical calculation result."""
    calculation: str
    result: int
    explanation: str


class PersonInfo(BaseModel):
    """Information about a person."""
    name: str
    age: int
    occupation: str


# Usage example:
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, SystemMessage

    # Initialize the model
    llm = ChatGCPOllama(
        base_url="https://ollama-qwen3-948194141289.us-central1.run.app",
        model="qwen3:14b",
        temperature=0.7
    )

    # Basic usage
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is 2+2? /nothink")
    ]

    result = llm.invoke("What is 2+2? /nothink")
    print("Response:", result.content)

    # Now you can remove the custom with_structured_output since it will use the parent class method
    # Structured output examples (using the built-in method)
    try:
        print("\n=== Structured Output Examples (Built-in Method) ===")

        # Example 1: Using Pydantic model with built-in method
        structured_llm = llm.with_structured_output(MathResult)
        structured_result = structured_llm.invoke("Calculate 15 + 27 and explain the process /nothink")
        print("Math Result:", structured_result)
        print(f"Calculation: {structured_result.calculation}")
        print(f"Result: {structured_result.result}")

        # Example 2: With raw output included
        structured_with_raw = llm.with_structured_output(PersonInfo, include_raw=True)
        raw_result = structured_with_raw.invoke("Tell me about Marie Curie /nothink")
        print("\nStructured with raw:")
        print("Parsed:", raw_result["parsed"])
        print("Raw content:", raw_result["raw"].content[:100] + "...")

    except Exception as e:
        print(f"Structured output error: {e}")
        print("The built-in method requires bind_tools to be implemented")

    # Alternative initialization with environment variable
    # Set environment variable: export GCP_OLLAMA_BASE_URL="https://your-api-url.run.app"
    # llm = ChatGCPOllama(model="qwen3:14b")