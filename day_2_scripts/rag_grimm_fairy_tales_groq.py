from typing import List, Tuple
import numpy as np
from dataclasses import dataclass
import re
from pathlib import Path
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
HERE = Path(__file__).parent


@dataclass
class Document:
    text: str
    chunk_id: int
    embedding: np.ndarray = None


class EmbeddingClient:
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model ready!")

    def get_embedding(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding


class GroqClient:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=api_key)

    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=0.95,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Groq API error: {e}")


class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap_percentage: float = 0.15, min_chunk_ratio: float = 0.3, min_break_ratio: float = 0.7):
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

        # Grimm text uses quadruple newlines as story separators - chunk per story when present
        stories = re.split(r'\n\n\n\n', text)

        if len(stories) > 3:
            chunks = []
            chunk_id = 0

            for story in stories:
                cleaned = ' '.join(story.strip().split())
                if len(cleaned) < 100:
                    continue

                story_words = len(cleaned.split())

                if story_words <= self.chunk_size * 1.5:
                    chunks.append(Document(text=cleaned, chunk_id=chunk_id))
                    chunk_id += 1
                else:
                    story_chunks = self._chunk_long_text(cleaned, chunk_id)
                    chunks.extend(story_chunks)
                    chunk_id += len(story_chunks)

            return chunks

        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return self._chunk_long_text(text, 0)

    def _chunk_long_text(self, text: str, start_id: int) -> List[Document]:
        sentences = self._split_sentences(text)

        chunks = []
        chunk_id = start_id
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
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))
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
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))
            elif chunks:
                chunks[-1].text += ' ' + ' '.join(current_chunk)
            else:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Document(text=chunk_text, chunk_id=chunk_id))

        return chunks


class GrimmRAG:
    def __init__(self, file_path: str, chunker: TextChunker = None):
        self.file_path = Path(file_path)
        self.chunker = chunker if chunker else TextChunker()
        self.embedding_client = EmbeddingClient()
        self.groq_client = GroqClient()
        self.documents: List[Document] = []
        self._load_document()

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        if 'Project Gutenberg' in text:
            matches = list(re.finditer(r'\*\*\* START OF (THIS|THE) PROJECT GUTENBERG', text))
            if matches:
                text = text[matches[0].end():]

            matches = list(re.finditer(r'\*\*\* END OF (THIS|THE) PROJECT GUTENBERG', text))
            if matches:
                text = text[:matches[0].start()]

        return text.strip()

    def _load_document(self):
        print(f"\nLoading {self.file_path.name} - {self.chunker.get_config_info()}")

        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            with open(self.file_path, 'r', encoding='latin-1') as file:
                text = file.read()

        text = self._clean_text(text)

        chunks = self.chunker.chunk_text(text)
        print(f"  Created {len(chunks)} chunks from document")

        print("  Generating embeddings...")
        for i, chunk in enumerate(chunks):
            try:
                chunk.embedding = self.embedding_client.get_embedding(chunk.text)
                self.documents.append(chunk)
                if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                    print(f"  Embedded {i + 1}/{len(chunks)} chunks")
            except Exception as e:
                print(f"  Failed to embed chunk {i}: {e}")

        print(f"Ready! {len(self.documents)} chunks loaded.\n")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _retrieve_chunks(self, query: str, top_k: int = 6) -> List[Tuple[Document, float]]:
        query_embedding = self.embedding_client.get_embedding(query)

        similarities = []
        for doc in self.documents:
            score = self._cosine_similarity(query_embedding, doc.embedding)
            similarities.append((doc, score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _show_match_details(self, matches: List[Tuple[Document, float]], query: str) -> None:
        if not matches:
            return

        query_embedding = self.embedding_client.get_embedding(query)
        all_scores = []
        for doc in self.documents:
            score = self._cosine_similarity(query_embedding, doc.embedding)
            all_scores.append(score)

        mean_score = sum(all_scores) / len(all_scores)
        variance = sum((score - mean_score) ** 2 for score in all_scores) / len(all_scores)
        std_dev = variance ** 0.5 if variance > 0 else 0.001

        print("\nBest matches (cosine similarity scores & significance):")
        for i, (doc, score) in enumerate(matches, 1):
            significance = (score - mean_score) / std_dev if std_dev > 0 else 0
            preview = doc.text[:100].replace('\n', ' ') + "..."
            print(f"  {i}. Chunk: {doc.chunk_id:03d} | Score: {score:.4f} | "
                  f"Significance: {significance:+.2f}σ | \"{preview}\"")
        print()

    def query(self, question: str, show_matches: bool = False) -> str:
        if not question.strip():
            return "Please provide a question."

        relevant_docs = self._retrieve_chunks(question)

        if not relevant_docs:
            return "I couldn't find relevant information to answer your question."

        if show_matches:
            self._show_match_details(relevant_docs, question)

        context_parts = []
        for i, (doc, score) in enumerate(relevant_docs, 1):
            text = doc.text[:1000] + "..." if len(doc.text) > 1000 else doc.text
            context_parts.append(f"Context {i}:\n{text}")

        context = "\n\n".join(context_parts)

        system_prompt = """You are a helpful assistant that answers questions about Grimm fairy tales.
Use the provided context from the fairy tales to answer the user's question accurately and concisely.
Provide your answer in English, even though the source text may be in German.
Keep your response short and focused on directly answering the question."""

        user_prompt = f"""Based on the following context from Grimm fairy tales, please answer this question: {question}

Context:
{context}

Please provide a short, direct answer in English."""

        try:
            answer = self.groq_client.generate_response(user_prompt, system_prompt)
            return answer.strip()
        except Exception as e:
            return f"Error generating response: {e}"


def main():
    print("=" * 70)
    print("Grimm Fairy Tales RAG Demo System (Groq + SentenceTransformers)")
    print("=" * 70)

    try:
        fairy_tale_file = None
        for filename in ["Kinder-und-Hausmärchen-der-Gebrüder-Grimm.txt", "Grimms-Fairy-Tales.txt"]:
            candidate = HERE / "data" / filename
            if candidate.exists():
                fairy_tale_file = str(candidate)
                break

        if not fairy_tale_file:
            print("Error: Could not find Grimm fairy tales file in data/")
            return

        chunker = TextChunker(chunk_size=500, overlap_percentage=0.15)
        rag = GrimmRAG(fairy_tale_file, chunker)

        questions = [
            "What did the frog king promise the princess in exchange for her golden ball?",
            "What happened to Hansel and Gretel in the forest?",
            "What did Little Red Riding Hood's mother tell her to do?",
            "Who helped Cinderella go to the ball?",
            "What happened to the wolf in the story of the seven little goats?",
            "Was versprach der Froschkönig der Prinzessin für ihre goldene Kugel?",
            "Was geschah mit Hänsel und Gretel im Wald?"
        ]

        for question in questions:
            print("=" * 70)
            print(f"Q: {question}")
            print("=" * 70)

            answer = rag.query(question, show_matches=True)
            print(f"\nA: {answer}\n")

    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
