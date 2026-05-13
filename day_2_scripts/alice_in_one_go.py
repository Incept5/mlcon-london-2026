from typing import List
import requests
import re
from pathlib import Path

LLM_MODEL = "qwen3.5:4b"
HERE = Path(__file__).parent


class OllamaClient:
    def __init__(self):
        self.base_url = "http://localhost:11434/api"

    def generate_response(self, prompt: str) -> str:
        data = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                # 64k context to fit the whole book plus the question
                "num_ctx": 65536,
                "temperature": 0.05,
                "top_p": 0.85
            }
        }

        print(f"Sending request to Ollama... (prompt length: {len(prompt)} chars)")
        try:
            response = requests.post(f"{self.base_url}/chat", json=data, timeout=300)
            response.raise_for_status()
            result = response.json()["message"]["content"]
            print("Response received successfully!")
            return result
        except requests.exceptions.Timeout:
            return "Error: Request timed out - document may be too large for model context"
        except requests.exceptions.RequestException as e:
            return f"Error: Network/API issue - {e}"


class FullContextQA:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.ollama = OllamaClient()
        self.full_text = ""
        self._test_connection()
        self._load_document()

    def _test_connection(self):
        print("Testing Ollama connection...")
        try:
            test_response = self.ollama.generate_response("Say 'Hello' to test the connection.")
            if "Error:" not in test_response:
                print(f"Connection test successful: {test_response[:50]}...")
            else:
                print(f"Connection test failed: {test_response}")
                return False
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
        return True

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        if 'Contents' in text:
            match = re.search(r'CHAPTER I\.\s*\n\s*[\w\s-]+\n\s*\n', text)
            if match:
                text = text[match.end():]

        return text.strip()

    def _load_document(self):
        print(f"Loading document: {self.file_path.name}")

        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        self.full_text = self._clean_text(text)

        print(f"Document loaded: {len(self.full_text)} characters")
        print(f"Estimated tokens: ~{len(self.full_text) // 4}")

    def query(self, question: str) -> str:
        if not question.strip():
            return "Please provide a question."

        prompt = f"""Answer the question using the provided complete text.

Be specific and detailed. Quote relevant text when appropriate using quotation marks.
If the text doesn't contain enough information, say so clearly.
Focus on accuracy - only state what is supported by the text.

Question: {question}

Complete Text:
{self.full_text}

Answer:"""

        try:
            answer = self.ollama.generate_response(prompt)
            return answer.strip()
        except Exception as e:
            return f"Error generating response: {e}"


def main():
    print("Full Context QA Demo System (64k context)")
    print("Make sure Ollama is running on port 11434\n")

    try:
        qa = FullContextQA(str(HERE / "data" / "alice_in_wonderland.txt"))
        print(f"\nFull-context system ready!\n")

        test_questions = [
            "What was Alice doing at the beginning of the story?",
            "What was written on the bottle that made Alice shrink?",
            "What did the White Rabbit say when Alice first saw him?"
        ]

        print("Testing with first 3 questions:")
        print("=" * 70)

        for i, question in enumerate(test_questions, 1):
            print(f"Question {i}/3: {question}")
            print("-" * 50)

            answer = qa.query(question)
            print(f"A: {answer}\n")
            print("=" * 70)

            if "Error:" in answer:
                print("Stopping due to error. Check Ollama configuration.")
                break

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
