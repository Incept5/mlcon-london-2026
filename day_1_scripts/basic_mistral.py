import os
from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))


def generate_response(prompt):
    response = client.chat.complete(
        model="mistral-medium-latest",
        messages=[{"content": prompt, "role": "user"}],
        stream=False,
        temperature=0.3,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(generate_response("Hello"))
