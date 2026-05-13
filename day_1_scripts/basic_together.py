import os
from together import Together
from dotenv import load_dotenv

load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY", ""))


def generate_response(prompt):
    response = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(generate_response("Hello"))
