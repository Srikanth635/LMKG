import requests
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.output_parsers.pydantic import PydanticOutputParser

llm = ChatOllama(model="qwen3:14b", temperature=0.1, client_kwargs={"timeout": 600})

class AnswerWithJustification(BaseModel):
    """An answer to the user question along with justification for the answer."""
    answer: str = Field(description="The answer.")
    justification: str = Field(description="The justification.")


# Use with_structured_output
structured_llm = llm.with_structured_output(AnswerWithJustification, method="json_schema")

result = structured_llm.invoke("What is 2+2?")
print("\nStructured output:")
print(f"Answer: {result.answer}")
print(f"Justification: {result.justification}")

# url = "https://ollama-gcp-oss20b-948194141289.us-central1.run.app/api/generate"
# payload = {
#     "model": "gpt-oss:20b",
#     "prompt": "Why is 2+2 /nothink",
#     "stream": False
# }
#
# # url = "https://ollama-qwen3-948194141289.us-central1.run.app/api/generate"
# # payload = {
# #     "model": "qwen3:14b",
# #     "prompt": "Why is 2+2 /nothink",
# #     "stream": False
# # }
#
# response = requests.post(url, json=payload)
# res_json = response.json()
# print(res_json.get("response",""))