import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("FIREWORKS_API_KEY", "")


def generate_response(prompt):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": "accounts/fireworks/models/deepseek-v3p2",
        "max_tokens": 500, "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Accept": "application/json", "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(generate_response("Hello"))
