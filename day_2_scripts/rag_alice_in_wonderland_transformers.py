from typing import List, Dict, Tuple
import numpy as np
import re
from pathlib import Path
from dataclasses import dataclass
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

LLM_MODEL = "qwen3.5:4b"
HERE = Path(__file__).parent


@dataclass
class Document:
    text: str
    metadata: Dict
    embedding: np.ndarray = None


def mean_pooling(model_output, attention_mask):
    # Average token embeddings weighted by attention mask (ignore padding)
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TransformerEmbedder:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        encoded_input = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize to unit length so cosine similarity = dot product
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings[0].numpy()


class OllamaLLM:
    def __init__(self, model: str = LLM_MODEL):
        self.model = model
        self.base_url = "http://localhost:11434/api"

    def generate_response(self, prompt: str) -> str:
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": 8192,
                "temperature": 0.3
            }
        }

        try:
            response = requests.post(f"{self.base_url}/generate", json=data)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error generating LLM response: {str(e)}"


class TextChunker:
    def __init__(self, chunk_size: int = 250, overlap_percentage: float = 0.2, min_chunk_ratio: float = 0.4, min_break_ratio: float = 0.75):
        self.chunk_size = chunk_size
        self.overlap_percentage = overlap_percentage
        self.min_chunk_size = int(chunk_size * min_chunk_ratio)
        self.min_break_size = int(chunk_size * min_break_ratio)
        self.final_chunk_min_size = int(chunk_size * min_chunk_ratio)
        self.overlap_words = int(chunk_size * overlap_percentage)

    def get_config_info(self) -> str:
        return (f"Chunking: {self.chunk_size} words, {self.overlap_percentage:.0%} overlap")

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences

    def chunk_text(self, text: str) -> List[Document]:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+', ' ', text).strip()

        sentences = self._split_sentences(text)

        chunks = []
        chunk_id = 0
        current_chunk = []
        current_words = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = len(sentence.split())

            if (current_words + sentence_words > self.chunk_size and
                current_words >= self.min_chunk_size):

                best_break = len(current_chunk)
                for j in range(len(current_chunk) - 1, max(0, len(current_chunk) - 5), -1):
                    if ('\n' in current_chunk[j] or
                        current_chunk[j].strip().endswith(('!', '?', '."', '.\'')) and
                        len(' '.join(current_chunk[:j+1]).split()) >= self.min_break_size):
                        best_break = j + 1
                        break

                final_chunk = current_chunk[:best_break]
                chunk_text = ' '.join(final_chunk)
                chunks.append(Document(text=chunk_text, metadata={"chunk_id": chunk_id}))
                chunk_id += 1

                target_overlap_words = self.overlap_words
                overlap_sentences = []
                overlap_words = 0

                for j in range(best_break - 1, -1, -1):
                    sentence_words = len(current_chunk[j].split())
                    if overlap_words + sentence_words <= target_overlap_words:
                        overlap_sentences.insert(0, current_chunk[j])
                        overlap_words += sentence_words
                    else:
                        break

                if not overlap_sentences and best_break > 0:
                    overlap_sentences = [current_chunk[best_break - 1]]

                current_chunk = overlap_sentences
                current_words = overlap_words

            current_chunk.append(sentence)
            current_words += sentence_words
            i += 1

        if current_chunk:
            if current_words >= self.final_chunk_min_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Document(text=chunk_text, metadata={"chunk_id": chunk_id}))
            elif chunks:
                chunks[-1].text += ' ' + ' '.join(current_chunk)
            else:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Document(text=chunk_text, metadata={"chunk_id": chunk_id}))

        return chunks


class SimpleRetriever:
    def __init__(self):
        self.documents = []

    def add_documents(self, documents: List[Document]):
        self.documents = [doc for doc in documents if doc.embedding is not None]

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_relevant_chunks(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Document, float]]:
        similarities = []
        for doc in self.documents:
            similarity = self.cosine_similarity(query_embedding, doc.embedding)
            similarities.append((doc, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class TransformerRAG:
    def __init__(self, file_path: str, chunker: TextChunker = None):
        self.chunker = chunker if chunker else TextChunker()
        self.embedder = TransformerEmbedder()
        self.llm = OllamaLLM()
        self.retriever = SimpleRetriever()

        self.file_path = Path(file_path)
        self.documents = self._load_and_process_text()
        self.retriever.add_documents(self.documents)

    def _load_and_process_text(self) -> List[Document]:
        try:
            if not self.file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")

            print(f"Loading {self.file_path.name} - {self.chunker.get_config_info()}")

            with open(self.file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            text = self._clean_text(text)
            chunks = self.chunker.chunk_text(text)

            for i, chunk in enumerate(chunks):
                try:
                    chunk.embedding = self.embedder.get_embedding(chunk.text)
                    if (i + 1) % 25 == 0 or i == len(chunks) - 1:
                        print(f"  Embedded {i + 1}/{len(chunks)} chunks")
                except Exception as e:
                    print(f"Failed to embed chunk {i}: {e}")
                    chunk.embedding = None

            print(f"Ready! {len([c for c in chunks if c.embedding is not None])} chunks loaded.")
            return chunks

        except Exception as e:
            raise Exception(f"Error processing file {self.file_path}: {str(e)}")

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        if 'Contents' in text:
            match = re.search(r'CHAPTER I\.\s*\n\s*[\w\s-]+\n\s*\n', text)
            if match:
                text = text[match.end():]

        return text.strip()

    def _show_match_details(self, matches: List[Tuple[Document, float]], query: str) -> None:
        if not matches:
            return

        query_embedding = self.embedder.get_embedding(query)
        all_scores = []
        for doc in self.documents:
            if doc.embedding is not None:
                score = self.retriever.cosine_similarity(query_embedding, doc.embedding)
                all_scores.append(score)

        mean_score = sum(all_scores) / len(all_scores)
        variance = sum((score - mean_score) ** 2 for score in all_scores) / len(all_scores)
        std_dev = variance ** 0.5 if variance > 0 else 0.001

        print("\nBest matches (cosine similarity scores & significance):")
        for i, (doc, score) in enumerate(matches, 1):
            significance = (score - mean_score) / std_dev if std_dev > 0 else 0
            preview = doc.text[:80].replace('\n', ' ') + "..."
            print(f"  {i}. Chunk: {doc.metadata['chunk_id']:03d} | Score: {score:.4f} | "
                  f"Significance: {significance:+.2f}σ | \"{preview}\"")
        print()

    def query(self, question: str, show_matches: bool = False) -> str:
        if not question.strip():
            return "Please provide a question."

        try:
            query_embedding = self.embedder.get_embedding(question)
            relevant_docs = self.retriever.get_relevant_chunks(query_embedding)

            if not relevant_docs:
                return "I couldn't find relevant information to answer your question."

            if show_matches:
                self._show_match_details(relevant_docs, question)

            context_parts = []
            for i, (doc, score) in enumerate(relevant_docs, 1):
                context_parts.append(f"Context {i}:\n{doc.text}")

            context = "\n\n".join(context_parts)

            prompt = f"""Answer the question using the provided context passages.

Be specific and detailed. Quote relevant text when appropriate using quotation marks.
If the context doesn't contain enough information, say so clearly.

Question: {question}

Context:
{context}

Answer:"""

            answer = self.llm.generate_response(prompt)
            return answer.strip()

        except Exception as e:
            return f"Error generating response: {e}"


def main():
    print("Transformer RAG Demo System")
    print("Make sure Ollama is running on port 11434\n")

    try:
        rag = TransformerRAG(str(HERE / "data" / "alice_in_wonderland.txt"))
        print()

        questions = [
            "What was Alice doing at the beginning of the story?",
            "What was written on the bottle that made Alice shrink?",
            "Where was the Cheshire Cat when Alice first met him?",
            "What happens when Alice falls down the rabbit hole?",
            "What did the White Rabbit say when Alice first saw him?",
            "Was stand auf der Flasche, die Alice schrumpfen ließ?",
            "Que faisait Alice au début de l'histoire?",
            "¿Qué dijo el Conejo Blanco cuando Alice lo vio por primera vez?"
        ]

        for question in questions:
            print("=" * 70)
            print(f"Q: {question}")
            print("=" * 70)

            answer = rag.query(question, show_matches=True)
            print(f"\nA: {answer}\n")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
