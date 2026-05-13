import requests

key = open("TOGETHER_KEY.txt").read().strip()

r = requests.post(
    "https://api.together.xyz/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}"},
    json={ "model": "Qwen/Qwen3.5-9B",
        "messages": [{"role": "user", "content": "Hello"}],},
    timeout=180,
)
print(r.json()["choices"][0]["message"]["content"])
