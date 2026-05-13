import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="qwen/qwen3-32b",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
