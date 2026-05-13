import ollama
import json


def generate_formatted_response(prompt):
    try:
        response = ollama.generate(
            model="qwen3.5:4b",
            prompt=prompt,
            format="json",
            think=False,
            options={
                "num_ctx": 8192,
                "temperature": 0.3
            }
        )
        return response['response']
    except Exception as e:
        print("Error:", e)
        return None


def main():
    prompt = """List the numbers from 1 to 10 and their names in
    English, French, German, Dutch, Chinese, Russian, Arabic, Polish, Hungarian.
    Provide the output in this exact JSON format:
    {
      "numbers": [
        {
          "number": 7,
          "English": "seven",
          "French": "sept",
          "German": "sieben",
          "Dutch": "zeven",
          "Chinese": "七",
          "Russian": "семь",
          "Arabic": "سبعة",
          "Polish": "siedem",
          "Hungarian": "hét"
        },
        ...and so on for numbers 1-10
      ]
    }"""

    response = generate_formatted_response(prompt)

    try:
        if response:
            parsed = json.loads(response)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("Received non-JSON response:")
        print(response)


if __name__ == "__main__":
    main()
