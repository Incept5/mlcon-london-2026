import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def generate_response(prompt, system_prompt=None):
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        client = Groq(api_key=api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=0.95,
            stream=False
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None


def summarise(text):
    system_prompt = "You are a summary assistant. Provide very concise, one-sentence summaries."
    prompt = f"Summarise the following text into a very short sentence:\n\n{text}"
    return generate_response(prompt, system_prompt)


def extract(text, what):
    system_prompt = "You are a concise data extraction assistant. Provide only the extracted information, nothing else."
    prompt = f"Extract {what} from the following text. Give the answer only, nothing else:\n\n{text}"
    return generate_response(prompt, system_prompt)


def main():
    # Source: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
    text = """# Qwen3.6-35B-A3B Model Card

Qwen3.6-35B-A3B is the first open-weight variant of the Qwen3.6 series, built on community feedback to prioritize stability and real-world utility. It offers developers a more intuitive, responsive, and productive coding experience.

Type: Causal Language Model with Vision Encoder
License: Apache 2.0

## Highlights
- Agentic Coding: Enhanced handling of frontend workflows and repository-level reasoning with greater fluency and precision
- Thinking Preservation: New option to retain reasoning context from historical messages

## Specifications
- Parameters: 35B total, 3B activated
- Context Length: 262,144 tokens natively, extensible to 1,010,000 tokens
- Experts: 256 total, 8 Routed + 1 Shared activated

## Recommended Sampling Parameters

### Instruct Mode for General Tasks
- temperature: 0.7
- top_p: 0.8
- top_k: 20
- min_p: 0.0
- presence_penalty: 1.5
- repetition_penalty: 1.0

### Instruct Mode for Reasoning Tasks
- temperature: 1.0
- top_p: 1.0
- top_k: 40
- min_p: 0.0
- presence_penalty: 2.0
- repetition_penalty: 1.0
"""

    print("=" * 70)
    print("Data Extraction Demo using Groq")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("SUMMARIZATION DEMO")
    print("=" * 70)
    summary = summarise(text)
    if summary:
        print(f"Summary: {summary}\n")
    else:
        print("Failed to generate summary\n")

    print("=" * 70)
    print("DATA EXTRACTION DEMO")
    print("=" * 70)
    data = "Sampling parameters for Instruct Mode for General Tasks"
    extracted_info = extract(text, data)
    if extracted_info:
        print(f"{data}: {extracted_info}")
    else:
        print(f"Failed to extract {data}")


if __name__ == "__main__":
    main()
