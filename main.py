import requests

url = "https://ollama-qwen3-948194141289.us-central1.run.app/api/generate"
payload = {
    "model": "qwen3:14b",
    "prompt": "Why is 2+2 /nothink",
    "stream": False
}

response = requests.post(url, json=payload)
res_json = response.json()
print(res_json.get("response",""))