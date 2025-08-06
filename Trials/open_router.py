import requests
import json
from dotenv import load_dotenv
load_dotenv()
import os
import re

def think_remover(res: str):
    if re.search(r"<think>.*?</think>", res, flags=re.DOTALL):
        cleaned_res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
    else:
        cleaned_res = res.strip()
    return cleaned_res

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    },
    data=json.dumps({
        # "model": "openai/gpt-oss-20b",
        "model":"openai/gpt-oss-120b",
        "messages": [
            {
                "role": "user",
                "content": "name the puranas in hinduism. explain in plain text only."
            }
        ],

    })
)
json_res = response.json()

print(think_remover(json_res.get("choices")[0].get("message").get("content")))

# with open("openrouter_response.txt",'w') as f:
#     json_res = response.json()
#     answer = think_remover(json_res.get("choices")[0].get("message").get("content"))
#     reasoning = think_remover(json_res.get("choices")[0].get("message").get("reasoning"))
#
#     text_str = f"Answer : \n\n {answer} \n\n and Reasoning : \n\n {reasoning} \n\n"
#
#     f.write(text_str)


# print(response.json())