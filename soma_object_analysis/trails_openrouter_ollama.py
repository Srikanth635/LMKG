import requests
import json
import os
import re
from typing import Any, Dict, List, Optional, Iterator, Union, Sequence, Literal, Type
from dotenv import load_dotenv

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ChatMessage,
    FunctionMessage,
    ToolMessage,
    ToolCall
)
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

# Use Pydantic v2
from pydantic import Field, model_validator, BaseModel

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")

class OpenRouterChatModel(BaseChatModel):
    """Custom LangChain chat model for OpenRouter API with structured output support."""

    model: str = Field(default="openai/gpt-oss-120b", description="Model to use")
    api_key: Optional[str] = Field(default=api_key, description="OpenRouter API key")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1/chat/completions",
        description="OpenRouter API endpoint"
    )
    temperature: Optional[float] = Field(default=None, description="Temperature for sampling")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens to generate")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling parameter")
    frequency_penalty: Optional[float] = Field(default=None, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=None, description="Presence penalty")
    remove_think_tags: bool = Field(default=True, description="Remove <think> tags from response")

    # Tool/function calling related fields
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tools to bind")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Tool choice strategy")

    @model_validator(mode='before')
    @classmethod
    def validate_api_key(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set API key."""
        api_key = values.get("api_key")
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenRouter API key must be provided either as a parameter "
                    "or as OPENROUTER_API_KEY environment variable"
                )
            values["api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "openrouter"

    def _think_remover(self, res: str) -> str:
        """Remove <think> tags from response."""
        if re.search(r"<think>.*?</think>", res, flags=re.DOTALL):
            cleaned_res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
        else:
            cleaned_res = res.strip()
        return cleaned_res

    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        """Convert LangChain message to OpenRouter format."""
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": str(message.content)}
        elif isinstance(message, HumanMessage):
            return {"role": "user", "content": str(message.content)}
        elif isinstance(message, AIMessage):
            msg_dict = {"role": "assistant", "content": str(message.content)}
            # Add tool calls if present
            if hasattr(message, 'tool_calls') and message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(tc.get("args", {})) if isinstance(tc.get("args"), dict) else tc.get(
                                "args", "{}")
                        }
                    } for tc in message.tool_calls
                ]
            elif hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                msg_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            return msg_dict
        elif isinstance(message, ChatMessage):
            return {"role": message.role, "content": str(message.content)}
        elif isinstance(message, FunctionMessage):
            return {"role": "function", "name": message.name, "content": str(message.content)}
        elif isinstance(message, ToolMessage):
            return {"role": "tool", "content": str(message.content), "tool_call_id": message.tool_call_id}
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    def _create_request_payload(
            self,
            messages: List[BaseMessage],
            **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the request payload for OpenRouter API."""
        # Convert messages to OpenRouter format
        formatted_messages = [self._convert_message_to_dict(msg) for msg in messages]

        # Build payload with only the actual values (not Field objects)
        payload = {
            "model": self.model,
            "messages": formatted_messages
        }

        # Add tools if bound
        if self.tools:
            payload["tools"] = self.tools
            if self.tool_choice:
                if isinstance(self.tool_choice, str):
                    if self.tool_choice == "any":
                        # Force the model to use one of the provided tools
                        payload["tool_choice"] = "required"
                    else:
                        payload["tool_choice"] = self.tool_choice
                else:
                    payload["tool_choice"] = self.tool_choice

        # Add optional parameters if they have actual values
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            payload["presence_penalty"] = self.presence_penalty

        # Add stop sequences if provided
        if 'stop' in kwargs and kwargs['stop']:
            payload["stop"] = kwargs['stop']

        # Override with other kwargs (excluding internal parameters)
        excluded_keys = {'stop', 'run_manager', 'stream'}
        for key, value in kwargs.items():
            if key not in excluded_keys and value is not None:
                payload[key] = value

        return payload

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response."""
        # Create request payload
        payload = self._create_request_payload(messages, stop=stop, **kwargs)

        # Make API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                url=self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )
            response.raise_for_status()
            json_res = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenRouter API request failed: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse OpenRouter API response: {e}")

        # Extract the response
        if "choices" not in json_res or len(json_res["choices"]) == 0:
            raise RuntimeError("No choices returned from OpenRouter API")

        choice = json_res["choices"][0]
        message_dict = choice.get("message", {})
        message_content = message_dict.get("content", "")

        # Remove think tags if configured
        if self.remove_think_tags and message_content:
            message_content = self._think_remover(message_content)

        # Create AIMessage with tool calls if present
        additional_kwargs = {}
        tool_calls = []

        if "tool_calls" in message_dict:
            additional_kwargs["tool_calls"] = message_dict["tool_calls"]
            # Convert to LangChain ToolCall format
            for tc in message_dict["tool_calls"]:
                try:
                    args = tc.get("function", {}).get("arguments", "{}")
                    if isinstance(args, str):
                        args = json.loads(args)
                    tool_calls.append(
                        ToolCall(
                            name=tc.get("function", {}).get("name", ""),
                            args=args,
                            id=tc.get("id", "")
                        )
                    )
                except json.JSONDecodeError:
                    # If arguments can't be parsed, keep as string
                    tool_calls.append(
                        ToolCall(
                            name=tc.get("function", {}).get("name", ""),
                            args={},
                            id=tc.get("id", "")
                        )
                    )

        ai_message = AIMessage(
            content=message_content or "",
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls if tool_calls else []
        )

        # Create ChatGeneration
        generation = ChatGeneration(
            message=ai_message,
            generation_info={
                "finish_reason": choice.get("finish_reason"),
                "model": json_res.get("model", self.model),
                "usage": json_res.get("usage", {})
            }
        )

        return ChatResult(generations=[generation])

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat responses (not implemented for this basic version)."""
        raise NotImplementedError("Streaming is not yet implemented for OpenRouterChatModel")

    def bind_tools(
            self,
            tools: Sequence[Union[Dict[str, Any], Type[BaseModel], BaseTool]],
            *,
            tool_choice: Optional[Union[str, Literal["auto", "none", "required", "any"], Dict[str, Any]]] = None,
            **kwargs: Any,
    ) -> "OpenRouterChatModel":
        """Bind tools to the model."""
        formatted_tools = []

        for tool in tools:
            if isinstance(tool, dict):
                # Already in OpenAI format
                formatted_tools.append(tool)
            elif isinstance(tool, type) and issubclass(tool, BaseModel):
                # Convert Pydantic model to OpenAI tool format
                formatted_tools.append(convert_to_openai_tool(tool))
            elif hasattr(tool, 'name') and hasattr(tool, 'description'):
                # BaseTool
                formatted_tools.append(convert_to_openai_tool(tool))
            else:
                # Try to convert whatever it is
                formatted_tools.append(convert_to_openai_tool(tool))

        # Create new instance with bound tools
        return self.__class__(
            **{
                **self.dict(),
                "tools": formatted_tools,
                "tool_choice": tool_choice,
                **kwargs
            }
        )

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }


# Example usage
if __name__ == "__main__":
    # Initialize the model
    chat_model = OpenRouterChatModel(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.7,
        max_tokens=1000
    )

    # Example 1: Simple message
    messages = [
        HumanMessage(content="name the puranas in hinduism. explain in plain text only.")
    ]

    response = chat_model.invoke(messages)
    print("Response:", response.content)


    # Example 2: With structured output using Pydantic

    class AnswerWithJustification(BaseModel):
        """An answer to the user question along with justification for the answer."""
        answer: str = Field(description="The answer.")
        justification: str = Field(description="The justification.")


    # Use with_structured_output
    structured_llm = chat_model.with_structured_output(AnswerWithJustification,method="json_schema")

    result = structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers?")
    print("\nStructured output:")
    print(f"Answer: {result.answer}")
    print(f"Justification: {result.justification}")

    # Example 3: With structured output including raw response
    structured_llm_raw = chat_model.with_structured_output(AnswerWithJustification, method="json_schema",include_raw=True)

    result_raw = structured_llm_raw.invoke("What is the capital of France?")
    print("\nStructured output with raw:")
    print(f"Parsed: {result_raw['parsed']}")
    print(f"Raw message type: {type(result_raw['raw'])}")
    print(f"Parsing error: {result_raw['parsing_error']}")

    # Example 4: Using with LangChain chains
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ])

    chain = prompt | chat_model | StrOutputParser()

    result = chain.invoke({"input": "Tell me a short joke"})
    print("\nChain result:", result)