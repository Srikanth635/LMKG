import re
import asyncio
import aiohttp
import requests
from typing import Optional, List, Any, Dict, Iterator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field
from typing import Union
from models import (ObjectDescription, VisualAppearance, GeometricDescription, MaterialProperties, MechanicalProperties,
                    CapabilityDescription, SemanticDescription, ObjectState)
import json

def think_remover(res: str):
    if re.search(r"<think>.*?</think>", res, flags=re.DOTALL):
        cleaned_res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
    else:
        cleaned_res = res.strip()
    return cleaned_res

# def parse_key_value_string(s: str) -> Dict[str, Any]:
#     """Parses a comma-separated string of key: value pairs into a dictionary."""
#     result = {}
#     if not s:
#         return result
#
#     parts = re.split(r', (?=\w+:)', s)  # split only where the next part looks like "key:"
#     for part in parts:
#         if ':' not in part:
#             continue
#         key, value = part.split(':', 1)
#         key = key.strip()
#         value = value.strip()
#         # Attempt to parse JSON-like values
#         if value.startswith('[') or value.startswith('{'):
#             try:
#                 result[key] = json.loads(value.replace("'", "\""))
#             except Exception:
#                 result[key] = value
#         elif re.match(r'^\(.*\)$', value):  # Tuple-like values
#             nums = re.findall(r'\d+\.?\d*', value)
#             result[key] = [float(n) if '.' in n else int(n) for n in nums]
#         elif value.lower() in ['true', 'false']:
#             result[key] = value.lower() == 'true'
#         elif re.match(r'^\d+\.?\d*$', value):
#             result[key] = float(value) if '.' in value else int(value)
#         else:
#             result[key] = value
#     return result
#
#
# def flatten_to_object_description(flat: Dict[str, Any]) -> ObjectDescription:
#     """Converts a flat dictionary with string fields to a fully nested ObjectDescription."""
#     return ObjectDescription(
#         name=flat.get("name", ""),
#         description=flat.get("description", ""),
#         visual=VisualAppearance(**parse_key_value_string(flat.get("visual", ""))),
#         geometric=GeometricDescription(**parse_key_value_string(flat.get("geometric", ""))),
#         material=MaterialProperties(**parse_key_value_string(flat.get("material", ""))),
#         mechanical=MechanicalProperties(**parse_key_value_string(flat.get("mechanical", ""))),
#         capabilities=CapabilityDescription(**parse_key_value_string(flat.get("capabilities", ""))),
#         semantic=SemanticDescription(**parse_key_value_string(flat.get("semantic", ""))),
#         state=ObjectState(**parse_key_value_string(flat.get("state", ""))),
#         confidence_score=float(flat.get("confidence_score", 0.8)),
#         source=flat.get("source", "llm_generated"),
#         timestamp=flat.get("timestamp")
#     )

class GcpChatOllama(BaseChatModel):
    """Custom ChatOllama that connects to GCP-hosted Ollama server"""

    model: str = Field(default="qwen3:14b")
    gcp_base_url: str = Field(default="https://ollama-qwen3-948194141289.us-central1.run.app")
    temperature: Optional[float] = Field(default=None)
    timeout: int = Field(default=60)

    def __init__(
            self,
            model: str = "qwen3:14b",
            gcp_base_url: str = "https://ollama-qwen3-948194141289.us-central1.run.app",
            temperature: Optional[float] = None,
            timeout: int = 60,
            **kwargs
    ):
        super().__init__(
            model=model,
            gcp_base_url=gcp_base_url,
            temperature=temperature,
            timeout=timeout,
            **kwargs
        )

    @property
    def _llm_type(self) -> str:
        return "gcp-ollama"

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        parts = []
        for message in messages:
            if isinstance(message, SystemMessage):
                parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                parts.append(f"User: {message.content}")
            elif isinstance(message, AIMessage):
                parts.append(f"Assistant: {message.content}")
            else:
                parts.append(f"{message.type.capitalize()}: {message.content}")
        return "\n".join(parts)

    def _generate(
            self,
            messages: Union[List[BaseMessage], str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response using GCP Ollama server"""
        try:
            # Convert messages to Ollama format
            ollama_messages = self._convert_messages_to_ollama_format(messages)

            params = {
                "model": self.model,
                "prompt": self._messages_to_prompt(messages),
                "stream": False
            }

            # Add temperature if specified
            if self.temperature is not None:
                params["options"] = {"temperature": self.temperature}

            # Add stop sequences if provided
            if stop:
                if "options" not in params:
                    params["options"] = {}
                params["options"]["stop"] = stop

            # Add any additional kwargs
            params.update(kwargs)

            # Make request to GCP server
            response = requests.post(
                f"{self.gcp_base_url}/api/generate",
                json=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            response_data = response.json()
            content = response_data.get("response", "")
            print("CONTENT" ,type(think_remover(content)),think_remover(content))


            # Clean up the response
            cleaned_content = think_remover(content)
            # flatten_to_object_description(cleaned_content)
            # Create AIMessage with the response
            message = AIMessage(content=cleaned_content)
            generation = ChatGeneration(message=message)

            return ChatResult(generations=[generation])

        except Exception as e:
            raise Exception(f"Error calling GCP Ollama server: {str(e)}")

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response asynchronously using GCP Ollama server"""
        try:
            # Convert messages to Ollama format
            ollama_messages = self._convert_messages_to_ollama_format(messages)

            params = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": False
            }

            # Add temperature if specified
            if self.temperature is not None:
                params["options"] = {"temperature": self.temperature}

            # Add stop sequences if provided
            if stop:
                if "options" not in params:
                    params["options"] = {}
                params["options"]["stop"] = stop

            # Add any additional kwargs
            params.update(kwargs)

            # Make async request to GCP server
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.gcp_base_url}/api/chat",
                        json=params,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    content = response_data.get("message", {}).get("content", "")

                    # Clean up the response
                    cleaned_content = think_remover(content)

                    # Create AIMessage with the response
                    message = AIMessage(content=cleaned_content)
                    generation = ChatGeneration(message=message)

                    return ChatResult(generations=[generation])

        except Exception as e:
            raise Exception(f"Error calling GCP Ollama server: {str(e)}")

    def _convert_messages_to_ollama_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to Ollama chat format"""
        ollama_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                ollama_messages.append({
                    "role": "user",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                ollama_messages.append({
                    "role": "assistant",
                    "content": message.content
                })
            elif isinstance(message, SystemMessage):
                ollama_messages.append({
                    "role": "system",
                    "content": message.content
                })
            else:
                # Default to user role for unknown message types
                ollama_messages.append({
                    "role": "user",
                    "content": str(message.content)
                })

        return ollama_messages

    def get_structured_output(self, messages: List[BaseMessage], schema_class: type) -> Any:
        """
        Custom method to get structured output with better error handling
        """
        import json
        from pydantic import ValidationError

        # Get schema info
        schema = schema_class.model_json_schema()
        properties = schema.get('properties', {})

        # Create a detailed prompt
        schema_description = []
        for prop_name, prop_info in properties.items():
            prop_type = prop_info.get('type', 'string')
            schema_description.append(f"'{prop_name}': {prop_type}")

        schema_str = "{" + ", ".join(schema_description) + "}"

        # Add JSON formatting instruction to the last message
        if messages and hasattr(messages[-1], 'content'):
            original_content = messages[-1].content
            json_instruction = f"""

Respond ONLY with valid JSON in this exact format:
{schema_str}

Do not include any explanation, markdown formatting, or additional text. Just the JSON object."""

            # Create new message with JSON instruction
            new_messages = messages[:-1] + [type(messages[-1])(content=original_content + json_instruction)]
        else:
            new_messages = messages

        # Get response
        response = self.invoke(new_messages)
        content = response.content.strip()

        # Clean up response
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()

        # Try to parse JSON
        try:
            parsed_data = json.loads(content)
            return schema_class(**parsed_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {content}. Error: {e}")
        except ValidationError as e:
            raise ValueError(f"Response doesn't match expected schema: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error processing structured output: {e}")

    def bind_tools(
            self,
            tools: List[BaseTool],
            **kwargs: Any,
    ) -> Runnable:
        """
        Bind tools to the model. For structured output, we'll implement a simple approach.
        """
        # For now, return self as we handle structured output differently
        return self

    def with_structured_output(
            self,
            schema: type,
            **kwargs: Any,
    ) -> Runnable:
        """
        Override to use our custom structured output method
        """

        class StructuredOutputRunnable(Runnable):
            def __init__(self, llm, schema_class):
                self.llm = llm
                self.schema_class = schema_class

            def invoke(self, input_data, config=None, **kwargs):
                if isinstance(input_data, list):
                    messages = input_data
                else:
                    messages = [HumanMessage(content=str(input_data))]
                return self.llm.get_structured_output(messages, self.schema_class)

            async def ainvoke(self, input_data, config=None, **kwargs):
                if isinstance(input_data, list):
                    messages = input_data
                else:
                    messages = [HumanMessage(content=str(input_data))]
                return await self.llm.aget_structured_output(messages, self.schema_class)

        return StructuredOutputRunnable(self, schema)

    async def aget_structured_output(self, messages: List[BaseMessage], schema_class: type) -> Any:
        """
        Async version of get_structured_output
        """
        import json
        from pydantic import ValidationError

        # Get schema info
        schema = schema_class.model_json_schema()
        properties = schema.get('properties', {})

        # Create a detailed prompt
        schema_description = []
        for prop_name, prop_info in properties.items():
            prop_type = prop_info.get('type', 'string')
            schema_description.append(f"'{prop_name}': {prop_type}")

        schema_str = "{" + ", ".join(schema_description) + "}"

        # Add JSON formatting instruction to the last message
        if messages and hasattr(messages[-1], 'content'):
            original_content = messages[-1].content
            json_instruction = f"""

Respond ONLY with valid JSON in this exact format:
{schema_str}

Do not include any explanation, markdown formatting, or additional text. Just the JSON object."""

            # Create new message with JSON instruction
            new_messages = messages[:-1] + [type(messages[-1])(content=original_content + json_instruction)]
        else:
            new_messages = messages

        # Get response
        response = await self.ainvoke(new_messages)
        content = response.content.strip()

        # Clean up response
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()

        # Try to parse JSON
        try:
            parsed_data = json.loads(content)
            return schema_class(**parsed_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {content}. Error: {e}")
        except ValidationError as e:
            raise ValueError(f"Response doesn't match expected schema: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error processing structured output: {e}")

# class GcpChatOllama(BaseChatModel):
#     """Custom ChatOllama that connects to GCP-hosted Ollama server"""
#
#     model: str = Field(default="qwen3:14b")
#     gcp_base_url: str = Field(default="https://ollama-qwen3-948194141289.us-central1.run.app")
#     temperature: Optional[float] = Field(default=None)
#     timeout: int = Field(default=60)
#
#     def __init__(
#             self,
#             model: str = "qwen3:14b",
#             gcp_base_url: str = "https://ollama-qwen3-948194141289.us-central1.run.app",
#             temperature: Optional[float] = None,
#             timeout: int = 60,
#             **kwargs
#     ):
#         super().__init__(
#             model=model,
#             gcp_base_url=gcp_base_url,
#             temperature=temperature,
#             timeout=timeout,
#             **kwargs
#         )
#
#     @property
#     def _llm_type(self) -> str:
#         return "gcp-ollama"
#
#     def _generate(
#             self,
#             messages: List[BaseMessage],
#             stop: Optional[List[str]] = None,
#             run_manager: Optional[CallbackManagerForLLMRun] = None,
#             **kwargs: Any,
#     ) -> ChatResult:
#         """Generate chat response using GCP Ollama server"""
#         try:
#             # Convert messages to Ollama format
#             # ollama_messages = self._convert_messages_to_ollama_format(messages)
#
#             params = {
#                 "model": self.model,
#                 "prompt": self._messages_to_prompt(messages),
#                 "stream": False
#             }
#
#             # Add temperature if specified
#             if self.temperature is not None:
#                 params["options"] = {"temperature": self.temperature}
#
#             # Add stop sequences if provided
#             if stop:
#                 if "options" not in params:
#                     params["options"] = {}
#                 params["options"]["stop"] = stop
#
#             # Add any additional kwargs
#             params.update(kwargs)
#
#             # import requests
#
#             # url = "https://ollama-gcs-<some-number-here>.us-central1.run.app/api/generate"
#             # payload = {
#             #     "model": "gemma2:2b",
#             #     "prompt": "Why is the sky blue? Give the shortest answer possible",
#             #     "stream": False
#             # }
#
#             # response = requests.post(url, json=payload)
#             # print(response.text)
#
#             # Make request to GCP server
#             response = requests.post(
#                 f"{self.gcp_base_url}/api/generate",
#                 json=params,
#                 timeout=self.timeout
#             )
#             response.raise_for_status()
#
#             response_data = response.json()
#             content = response_data.get("response", "")
#
#             # Clean up the response
#             cleaned_content = think_remover(content)
#
#             # Create AIMessage with the response
#             message = AIMessage(content=cleaned_content)
#             generation = ChatGeneration(message=message)
#
#             return ChatResult(generations=[generation])
#
#         except Exception as e:
#             raise Exception(f"Error calling GCP Ollama server: {str(e)}")
#
#     async def _agenerate(
#             self,
#             messages: List[BaseMessage],
#             stop: Optional[List[str]] = None,
#             run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
#             **kwargs: Any,
#     ) -> ChatResult:
#         """Generate chat response asynchronously using GCP Ollama server"""
#         try:
#             # Convert messages to Ollama format
#             # ollama_messages = self._convert_messages_to_ollama_format(messages)
#
#             params = {
#                 "model": self.model,
#                 "prompt": self._messages_to_prompt(messages),
#                 "stream": False
#             }
#
#             # Add temperature if specified
#             if self.temperature is not None:
#                 params["options"] = {"temperature": self.temperature}
#
#             # Add stop sequences if provided
#             if stop:
#                 if "options" not in params:
#                     params["options"] = {}
#                 params["options"]["stop"] = stop
#
#             # Add any additional kwargs
#             params.update(kwargs)
#
#             # Make async request to GCP server
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(
#                         f"{self.gcp_base_url}/api/generate",
#                         json=params,
#                         timeout=aiohttp.ClientTimeout(total=self.timeout)
#                 ) as response:
#                     response.raise_for_status()
#                     response_data = await response.json()
#                     content = response_data.get("response", "")
#
#                     # Clean up the response
#                     cleaned_content = think_remover(content)
#
#                     # Create AIMessage with the response
#                     message = AIMessage(content=cleaned_content)
#                     generation = ChatGeneration(message=message)
#
#                     return ChatResult(generations=[generation])
#
#         except Exception as e:
#             raise Exception(f"Error calling GCP Ollama server: {str(e)}")
#
#     def _convert_messages_to_ollama_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
#         """Convert LangChain messages to Ollama chat format"""
#         ollama_messages = []
#
#         for message in messages:
#             if isinstance(message, HumanMessage):
#                 ollama_messages.append({
#                     "role": "user",
#                     "content": message.content
#                 })
#             elif isinstance(message, AIMessage):
#                 ollama_messages.append({
#                     "role": "assistant",
#                     "content": message.content
#                 })
#             elif isinstance(message, SystemMessage):
#                 ollama_messages.append({
#                     "role": "system",
#                     "content": message.content
#                 })
#             else:
#                 # Default to user role for unknown message types
#                 ollama_messages.append({
#                     "role": "user",
#                     "content": str(message.content)
#                 })
#
#         return ollama_messages
#
#     def get_structured_output(self, messages: List[BaseMessage], schema_class: type) -> Any:
#         """
#         Custom method to get structured output with better error handling
#         """
#         import json
#         from pydantic import ValidationError
#
#         # Get schema info
#         schema = schema_class.model_json_schema()
#
#         # Create detailed schema description with examples
#         schema_str = self._create_detailed_schema_prompt(schema)
#
#         # Add JSON formatting instruction to the last message
#         if messages and hasattr(messages[-1], 'content'):
#             original_content = messages[-1].content
#             json_instruction = f"""
#
# You must respond with ONLY a valid JSON object that exactly matches this schema:
#
# {schema_str}
#
# CRITICAL REQUIREMENTS:
# - Return ONLY the JSON object, no other text
# - All required fields must be present
# - Use exact field names as shown
# - Follow the data types specified
# - Do not add extra fields not in the schema
# - Ensure proper JSON syntax with quotes around strings"""
#
#             # Create new message with JSON instruction
#             new_messages = messages[:-1] + [type(messages[-1])(content=original_content + json_instruction)]
#         else:
#             new_messages = messages
#
#         # Get response using the sync invoke method
#         response = self.invoke(new_messages)
#         content = response.content.strip()
#
#         # Debug: Print raw response
#         print(f"DEBUG - Raw model response: {content}")
#
#         # Clean up response - handle various formats
#         content = self._clean_json_response(content)
#
#         # Debug: Print cleaned response
#         print(f"DEBUG - Cleaned JSON: {content}")
#
#         # Try to parse JSON
#         try:
#             parsed_data = json.loads(content)
#             print(
#                 f"DEBUG - Parsed data keys: {list(parsed_data.keys()) if isinstance(parsed_data, dict) else 'Not a dict'}")
#
#             # Try to create the schema object
#             try:
#                 result = schema_class(**parsed_data)
#                 print(f"DEBUG - Successfully created {schema_class.__name__} object")
#                 return result
#             except ValidationError as ve:
#                 print(f"DEBUG - Validation errors: {ve}")
#
#                 # Try to fix common issues automatically
#                 fixed_data = self._fix_common_schema_issues(parsed_data, schema)
#                 if fixed_data != parsed_data:
#                     print(f"DEBUG - Attempting with fixed data: {fixed_data}")
#                     return schema_class(**fixed_data)
#                 else:
#                     raise ve
#
#         except json.JSONDecodeError as e:
#             print(f"DEBUG - JSON decode error: {e}")
#             raise ValueError(f"Failed to parse JSON response: {content[:500]}... Error: {e}")
#         except ValidationError as e:
#             print(f"DEBUG - Validation error details: {e}")
#             raise ValueError(f"Response doesn't match expected schema. Raw response: {content[:500]}... Errors: {e}")
#         except Exception as e:
#             print(f"DEBUG - Unexpected error: {e}")
#             raise ValueError(f"Unexpected error processing structured output: {e}")
#
#     async def aget_structured_output(self, messages: List[BaseMessage], schema_class: type) -> Any:
#         """
#         Async version of get_structured_output
#         """
#         import json
#         from pydantic import ValidationError
#
#         # Get schema info
#         schema = schema_class.model_json_schema()
#
#         # Create detailed schema description with examples
#         schema_str = self._create_detailed_schema_prompt(schema)
#
#         # Add JSON formatting instruction to the last message
#         if messages and hasattr(messages[-1], 'content'):
#             original_content = messages[-1].content
#             json_instruction = f"""
#
# You must respond with ONLY a valid JSON object that exactly matches this schema:
#
# {schema_str}
#
# CRITICAL REQUIREMENTS:
# - Return ONLY the JSON object, no other text
# - All required fields must be present
# - Use exact field names as shown
# - Follow the data types specified
# - Do not add extra fields not in the schema
# - Ensure proper JSON syntax with quotes around strings"""
#
#             # Create new message with JSON instruction
#             new_messages = messages[:-1] + [type(messages[-1])(content=original_content + json_instruction)]
#         else:
#             new_messages = messages
#
#         # Get response using the async ainvoke method
#         response = await self.ainvoke(new_messages)
#         content = response.content.strip()
#
#         # Debug: Print raw response
#         print(f"DEBUG - Raw model response: {content}")
#
#         # Clean up response
#         content = self._clean_json_response(content)
#
#         # Debug: Print cleaned response
#         print(f"DEBUG - Cleaned JSON: {content}")
#
#         # Try to parse JSON
#         try:
#             parsed_data = json.loads(content)
#             print(
#                 f"DEBUG - Parsed data keys: {list(parsed_data.keys()) if isinstance(parsed_data, dict) else 'Not a dict'}")
#
#             # Try to create the schema object
#             try:
#                 result = schema_class(**parsed_data)
#                 print(f"DEBUG - Successfully created {schema_class.__name__} object")
#                 return result
#             except ValidationError as ve:
#                 print(f"DEBUG - Validation errors: {ve}")
#
#                 # Try to fix common issues automatically
#                 fixed_data = self._fix_common_schema_issues(parsed_data, schema)
#                 if fixed_data != parsed_data:
#                     print(f"DEBUG - Attempting with fixed data: {fixed_data}")
#                     return schema_class(**fixed_data)
#                 else:
#                     raise ve
#
#         except json.JSONDecodeError as e:
#             print(f"DEBUG - JSON decode error: {e}")
#             raise ValueError(f"Failed to parse JSON response: {content[:500]}... Error: {e}")
#         except ValidationError as e:
#             print(f"DEBUG - Validation error details: {e}")
#             raise ValueError(f"Response doesn't match expected schema. Raw response: {content[:500]}... Errors: {e}")
#         except Exception as e:
#             print(f"DEBUG - Unexpected error: {e}")
#             raise ValueError(f"Unexpected error processing structured output: {e}")
#
#     def _clean_json_response(self, content: str) -> str:
#         """Clean up JSON response from various formats"""
#         content = content.strip()
#
#         # Remove markdown code blocks
#         if content.startswith('```json'):
#             content = content[7:]
#         elif content.startswith('```'):
#             content = content[3:]
#
#         if content.endswith('```'):
#             content = content[:-3]
#
#         content = content.strip()
#
#         # Try to find JSON object if there's extra text
#         start_idx = content.find('{')
#         end_idx = content.rfind('}')
#
#         if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
#             content = content[start_idx:end_idx + 1]
#
#         return content
#
#     def bind_tools(
#             self,
#             tools: List[BaseTool],
#             **kwargs: Any,
#     ) -> Runnable:
#         """
#         Bind tools to the model. For structured output, we'll implement a simple approach.
#         """
#         # For now, return self as we handle structured output differently
#         return self
#
#     def with_structured_output(
#             self,
#             schema: type,
#             method: str = "function_calling",  # Accept method parameter for compatibility
#             **kwargs: Any,
#     ) -> Runnable:
#         """
#         Override to use our custom structured output method
#         """
#
#         class StructuredOutputRunnable(Runnable):
#             def __init__(self, llm, schema_class):
#                 self.llm = llm
#                 self.schema_class = schema_class
#
#             def invoke(self, input_data, config=None, **kwargs):
#                 if isinstance(input_data, list):
#                     messages = input_data
#                 else:
#                     messages = [HumanMessage(content=str(input_data))]
#                 return self.llm.get_structured_output(messages, self.schema_class)
#
#             async def ainvoke(self, input_data, config=None, **kwargs):
#                 if isinstance(input_data, list):
#                     messages = input_data
#                 else:
#                     messages = [HumanMessage(content=str(input_data))]
#                 return await self.llm.aget_structured_output(messages, self.schema_class)
#
#         return StructuredOutputRunnable(self, schema)
#
#     def _create_detailed_schema_prompt(self, schema: dict) -> str:
#         """Create a detailed schema description for better model understanding"""
#
#         def format_nested_object(obj_schema, indent=2):
#             """Recursively format nested object schemas"""
#             if not isinstance(obj_schema, dict):
#                 return str(obj_schema)
#
#             properties = obj_schema.get('properties', {})
#             required = obj_schema.get('required', [])
#
#             if not properties:
#                 return "{}"
#
#             lines = ["{"]
#             prop_items = list(properties.items())
#
#             for i, (prop_name, prop_info) in enumerate(prop_items):
#                 prop_type = prop_info.get('type', 'string')
#                 prop_description = prop_info.get('description', '')
#                 is_required = prop_name in required
#                 is_last = i == len(prop_items) - 1
#
#                 indent_str = " " * (indent + 2)
#
#                 if prop_type == 'object' and 'properties' in prop_info:
#                     # Recursively handle nested objects
#                     nested_obj = format_nested_object(prop_info, indent + 2)
#                     line = f'{indent_str}"{prop_name}": {nested_obj}'
#                 elif prop_type == 'array':
#                     items_info = prop_info.get('items', {})
#                     if items_info.get('type') == 'object' and 'properties' in items_info:
#                         nested_obj = format_nested_object(items_info, indent + 2)
#                         line = f'{indent_str}"{prop_name}": [{nested_obj}]'
#                     else:
#                         items_type = items_info.get('type', 'string')
#                         line = f'{indent_str}"{prop_name}": ["{items_type}"]'
#                 elif prop_type == 'string':
#                     line = f'{indent_str}"{prop_name}": "string"'
#                 elif prop_type == 'number':
#                     line = f'{indent_str}"{prop_name}": 0.0'
#                 elif prop_type == 'integer':
#                     line = f'{indent_str}"{prop_name}": 0'
#                 elif prop_type == 'boolean':
#                     line = f'{indent_str}"{prop_name}": true'
#                 else:
#                     line = f'{indent_str}"{prop_name}": "{prop_type}"'
#
#                 if is_required:
#                     line += f" // REQUIRED"
#                 if prop_description:
#                     line += f" // {prop_description}"
#
#                 if not is_last:
#                     line += ","
#
#                 lines.append(line)
#
#             lines.append(" " * indent + "}")
#             return "\n".join(lines)
#
#         formatted_schema = format_nested_object(schema)
#
#         return f"""EXACT JSON FORMAT REQUIRED:
# {formatted_schema}
#
# CRITICAL:
# - Each nested object field (visual, geometric, material, etc.) MUST be a JSON object, NOT a string
# - Follow the exact structure shown above
# - All nested properties must be included as separate fields within their parent objects
# - Use actual JSON objects {{ }} not descriptive strings"""
#
#     def _fix_common_schema_issues(self, data: dict, schema: dict) -> dict:
#         """Attempt to fix common schema validation issues automatically"""
#         if not isinstance(data, dict):
#             return data
#
#         fixed_data = data.copy()
#         properties = schema.get('properties', {})
#         required = schema.get('required', [])
#
#         # Define enum mappings for common mismatches
#         enum_mappings = {
#             'semantic.category': {
#                 'container': 'Container',
#                 'tool': 'Tool',
#                 'appliance': 'Appliance',
#                 'furniture': 'DesignedFurniture',
#                 'component': 'DesignedComponent',
#                 'item': 'Item'
#             },
#             'state.cleanliness': {
#                 'clean': 'Clean',
#                 'dirty': 'Dirty',
#                 'unknown': 'unknown'
#             },
#             'material.secondary_materials': {
#                 'glaze': 'ceramic',  # glaze is ceramic-based
#                 'coating': 'composite',
#                 'paint': 'composite'
#             }
#         }
#
#         # Value mappings for string-to-number conversions
#         value_mappings = {
#             'high': 0.8,
#             'medium': 0.5,
#             'low': 0.2,
#             'very high': 0.9,
#             'very low': 0.1,
#             'intact': 1.0,
#             'damaged': 0.5,
#             'broken': 0.0,
#             'room temperature': 20.0,
#             'hot': 60.0,
#             'cold': 5.0,
#             'warm': 30.0
#         }
#
#         def fix_nested_object(obj, obj_schema, path=""):
#             """Recursively fix nested objects"""
#             if not isinstance(obj, dict) or not isinstance(obj_schema, dict):
#                 return obj
#
#             obj_properties = obj_schema.get('properties', {})
#             obj_required = obj_schema.get('required', [])
#             fixed_obj = obj.copy()
#
#             # Fix existing fields
#             for field_name, field_value in obj.items():
#                 field_path = f"{path}.{field_name}" if path else field_name
#
#                 if field_name in obj_properties:
#                     field_schema = obj_properties[field_name]
#                     expected_type = field_schema.get('type')
#
#                     # Handle enum fields
#                     if 'enum' in field_schema:
#                         enum_values = field_schema['enum']
#                         if isinstance(field_value, str):
#                             # Try direct mapping first
#                             if field_path in enum_mappings and field_value.lower() in enum_mappings[field_path]:
#                                 fixed_obj[field_name] = enum_mappings[field_path][field_value.lower()]
#                             # Try case-insensitive match
#                             elif any(field_value.lower() == enum_val.lower() for enum_val in enum_values):
#                                 matching_enum = next(
#                                     enum_val for enum_val in enum_values if field_value.lower() == enum_val.lower())
#                                 fixed_obj[field_name] = matching_enum
#                             # Use first enum value as fallback
#                             else:
#                                 fixed_obj[field_name] = enum_values[0]
#
#                     # Handle array fields
#                     elif expected_type == 'array' and isinstance(field_value, list):
#                         items_schema = field_schema.get('items', {})
#                         if 'enum' in items_schema:
#                             enum_values = items_schema['enum']
#                             fixed_array = []
#                             for item in field_value:
#                                 if isinstance(item, str):
#                                     # Try mapping
#                                     if field_path in enum_mappings and item.lower() in enum_mappings[field_path]:
#                                         fixed_array.append(enum_mappings[field_path][item.lower()])
#                                     # Try case-insensitive match
#                                     elif any(item.lower() == enum_val.lower() for enum_val in enum_values):
#                                         matching_enum = next(
#                                             enum_val for enum_val in enum_values if item.lower() == enum_val.lower())
#                                         fixed_array.append(matching_enum)
#                                     # Use first enum value as fallback
#                                     else:
#                                         fixed_array.append(enum_values[0])
#                                 else:
#                                     fixed_array.append(item)
#                             fixed_obj[field_name] = fixed_array
#
#                     # Handle number fields that are strings
#                     elif expected_type in ['number', 'integer'] and isinstance(field_value, str):
#                         if field_value.lower() in value_mappings:
#                             fixed_obj[field_name] = value_mappings[field_value.lower()]
#                         else:
#                             # Try to extract numbers from string
#                             import re
#                             numbers = re.findall(r'\d+\.?\d*', field_value)
#                             if numbers:
#                                 fixed_obj[field_name] = float(numbers[0]) if expected_type == 'number' else int(
#                                     float(numbers[0]))
#                             else:
#                                 fixed_obj[field_name] = 0.0 if expected_type == 'number' else 0
#
#                     # Handle nested objects
#                     elif expected_type == 'object' and isinstance(field_value, dict):
#                         fixed_obj[field_name] = fix_nested_object(field_value, field_schema, field_path)
#
#                     # Handle spatial_relations string -> object conversion
#                     elif field_name == 'spatial_relations' and isinstance(field_value, str):
#                         fixed_obj[field_name] = {
#                             'container_relations': [],
#                             'support_relations': [],
#                             'proximity_relations': [],
#                             'relative_position': field_value
#                         }
#
#             # Add missing required fields
#             for req_field in obj_required:
#                 if req_field not in fixed_obj:
#                     if req_field in obj_properties:
#                         field_schema = obj_properties[req_field]
#                         field_type = field_schema.get('type', 'string')
#
#                         if field_type == 'string':
#                             fixed_obj[req_field] = ""
#                         elif field_type == 'number':
#                             fixed_obj[req_field] = 0.0
#                         elif field_type == 'integer':
#                             fixed_obj[req_field] = 0
#                         elif field_type == 'boolean':
#                             fixed_obj[req_field] = False
#                         elif field_type == 'array':
#                             fixed_obj[req_field] = []
#                         elif field_type == 'object':
#                             # Create minimal nested object
#                             nested_props = field_schema.get('properties', {})
#                             nested_required = field_schema.get('required', [])
#                             nested_obj = {}
#
#                             for nested_field in nested_required:
#                                 if nested_field in nested_props:
#                                     nested_info = nested_props[nested_field]
#                                     nested_type = nested_info.get('type', 'string')
#
#                                     if nested_type == 'string':
#                                         nested_obj[nested_field] = ""
#                                     elif nested_type in ['number', 'integer']:
#                                         nested_obj[nested_field] = 0
#                                     elif nested_type == 'boolean':
#                                         nested_obj[nested_field] = False
#                                     elif nested_type == 'array':
#                                         nested_obj[nested_field] = []
#                                     elif nested_type == 'object':
#                                         nested_obj[nested_field] = {}
#
#                             fixed_obj[req_field] = nested_obj
#
#             # Handle specific field mappings
#             if 'shape_type' in fixed_obj and 'shape' not in fixed_obj:
#                 # Move shape_type to shape object
#                 fixed_obj['shape'] = {
#                     'shape_type': fixed_obj['shape_type'],
#                     'shape_description': f"A {fixed_obj['shape_type']} shape"
#                 }
#                 del fixed_obj['shape_type']
#
#             if 'surface_properties' in fixed_obj and 'surface' not in fixed_obj:
#                 # Move surface_properties to surface object
#                 fixed_obj['surface'] = {
#                     'texture': fixed_obj['surface_properties'],
#                     'material_finish': 'smooth'
#                 }
#                 del fixed_obj['surface_properties']
#
#             return fixed_obj
#
#         # Apply fixes recursively
#         fixed_data = fix_nested_object(fixed_data, schema)
#
#         return fixed_data
#
#     def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
#         parts = []
#         for message in messages:
#             if isinstance(message, SystemMessage):
#                 parts.append(f"System: {message.content}")
#             elif isinstance(message, HumanMessage):
#                 parts.append(f"User: {message.content}")
#             elif isinstance(message, AIMessage):
#                 parts.append(f"Assistant: {message.content}")
#             else:
#                 parts.append(f"{message.type.capitalize()}: {message.content}")
#         return "\n".join(parts)



# Usage example:
if __name__ == "__main__":
    from pydantic import BaseModel, Field
    from langchain_core.messages import HumanMessage, SystemMessage

    class Accuracy(BaseModel):
        confidence: float = Field(..., description="The confidence of the answer")
        correct: bool = Field(..., description="Whether the answer is correct")

    class Answer(BaseModel):
        answer: str = Field(..., description="The answer to the question")
        accuracy: Accuracy = Field(..., description="The accuracy of the answer")


    # Initialize the LLM
    llm = GcpChatOllama(
        model="qwen3:14b",
        temperature=0.4
    )

    # Test basic invoke
    print("Testing basic invoke...")
    try:
        response = llm.invoke("What is the capital of France? /nothink")
        print("Basic response:", response)
    except Exception as e:
        print(f"Error in basic invoke: {e}")


    # try:
    #     structured_llm = llm.with_structured_output(Answer, method="json_schema")
    #     structured_response = structured_llm.invoke("What is the capital of France? /nothink")
    #     print("Structured response:", structured_response)
    # except Exception as e:
    #     print(f"Error in structured output: {e}")
    #     import traceback
    #     traceback.print_exc()

    # Test with system message
    print("\nTesting with system message...")
    try:
        messages = [
            SystemMessage(content="You are a helpful assistant that provides concise answers."),
            HumanMessage(content="What is 2+2? /nothink")
        ]
        response = llm.invoke(messages)
        print("Response with system message:", response.content)
    except Exception as e:
        print(f"Error with system message: {e}")

    # Test structured output
    print("\nTesting structured output...")
    try:
        # First test what the raw response looks like
        # test_response = llm.invoke([
        #     HumanMessage(
        #         content="What is 2+2? Please respond in JSON format with 'answer' and 'confidence' fields. Confidence should be a number between 0 and 1.")
        # ])
        # print("Raw response for structured output:", repr(test_response.content))

        structured_llm = llm.with_structured_output(Answer, method="json_schema")
        structured_response = structured_llm.invoke("What is 2+2? /nothink")
        out = structured_response.model_dump()
        print("Structured response:", type(out), out)
        # if hasattr(structured_response, 'answer'):
        #     print("Answer:", structured_response.answer)
        #     print("Confidence:", structured_response.confidence)
    except Exception as e:
        print(f"Error in structured output: {e}")
        import traceback

        traceback.print_exc()

    # Test async
    print("\nTesting async...")


    async def async_test():
        try:
            response = await llm.ainvoke([HumanMessage(content="What is3*2? /nothink")])
            print("Async response:", response.content)
        except Exception as e:
            print(f"Error in async test: {e}")


    asyncio.run(async_test())



