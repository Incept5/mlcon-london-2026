import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROK_API_KEY", "")


def generate_response(prompt):
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    completion = client.chat.completions.create(
        model="grok-4-latest",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    print(generate_response("Hello"))
