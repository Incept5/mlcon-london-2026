import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3.5:4b",
        "prompt": "Hello",
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": 256,
            "seed": 42,
            "num_ctx": 4096,
        },
    },
)

data = response.json()
print(data["response"])
