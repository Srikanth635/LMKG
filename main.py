import os

from langchain_core.output_parsers import PydanticOutputParser
from ollama import Client
import requests
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

ollama_turbo_api_key = os.getenv('OLLAMA_TURBO_API_KEY',"")

turbo_client = Client(
    headers={'Authorization': ollama_turbo_api_key}
)

client_kw_args = {
                    "headers": {'Authorization': ollama_turbo_api_key}
                }

# print(turbo_client.list())

class AnswerWithJustification(BaseModel):
        """An answer to the user question along with justification for the answer."""
        answer: str = Field(description="The answer to the user question.")
        justification: str = Field(description="The justification for the answer.")

chat_model = ChatOllama(base_url="https://ollama.com", client_kwargs=client_kw_args, model='gpt-oss:120b',
                        extract_reasoning=True, format="json")

# First, test without structured output to see raw response
raw_response = chat_model.invoke("What is the capital of France? Respond in JSON format with 'answer' and 'justification' fields.")
print("Raw response:", raw_response.content)

# Then try with structured output
try:
    structured_llm = chat_model.with_structured_output(AnswerWithJustification)
    result = structured_llm.invoke("What is the capital of France?")
    print("Structured result:", result)
except Exception as e:
    print(f"Error: {e}")
    # Try to see what the model actually returned
    print("Model likely returned plain text instead of JSON")