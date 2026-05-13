from typing import List, Tuple
import numpy as np
import requests
from dataclasses import dataclass
import re
from pathlib import Path
import chromadb
from chromadb.config import Settings
import hashlib

LLM_MODEL = "qwen3.5:4b"
EMBEDDING_MODEL = "embeddinggemma"
HERE = Path(__file__).parent


@dataclass
class Document:
    text: str
    chunk_id: int
    embedding: np.ndarray = None


class OllamaClient:
    def __init__(self):
        self.base_url = "http://localhost:11434/api"

    def get_embedding(self, text: str, task_type: str = "document") -> np.ndarray:
        # EmbeddingGemma uses task-specific prompts for document vs query
        if task_type == "query":
            formatted_text = f"task: search result | query: {text}"
        else:
            formatted_text = f"title: none | text: {text}"

        data = {"model": EMBEDDING_MODEL, "prompt": formatted_text}
        response = requests.post(f"{self.base_url}/embeddings", json=data, timeout=30)
        response.raise_for_status()
        return np.array(response.json()["embedding"])

    def generate_response(self, prompt: str) -> str:
        data = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": 16384,
                "temperature": 0.05,
                "top_p": 0.85
            }
        }
        response = requests.post(f"{self.base_url}/chat", json=data, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]


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


class PersistentChromaRetriever:
    def __init__(self, collection_name: str = "text_chunks", persist_directory: str = "./chroma_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        self.collection_name = collection_name
        self.documents = []

        try:
            self.collection = self.client.get_collection(name=collection_name)
            count = self.collection.count()
            if count > 0:
                print(f"Found existing ChromaDB with {count} chunks")
                self._load_documents_from_chroma()
        except:
            self.collection = self.client.create_collection(name=collection_name)

    def _get_file_hash(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def needs_update(self, file_path: str) -> bool:
        hash_file = self.persist_directory / f"{self.collection_name}_hash.txt"
        current_hash = self._get_file_hash(file_path)

        if not hash_file.exists():
            return True

        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()

        return current_hash != stored_hash

    def _save_file_hash(self, file_path: str):
        hash_file = self.persist_directory / f"{self.collection_name}_hash.txt"
        with open(hash_file, 'w') as f:
            f.write(self._get_file_hash(file_path))

    def _load_documents_from_chroma(self):
        results = self.collection.get(
            include=["documents", "metadatas", "embeddings"]
        )

        for i, (text, metadata, embedding) in enumerate(zip(results['documents'], results['metadatas'], results['embeddings'])):
            doc = Document(
                text=text,
                chunk_id=int(metadata['chunk_id']),
                embedding=np.array(embedding)
            )
            self.documents.append(doc)

    def add_documents(self, documents: List[Document], file_path: str = None):
        valid_docs = [doc for doc in documents if doc.embedding is not None]

        if not valid_docs:
            raise ValueError("No documents with valid embeddings provided")

        if self.collection.count() > 0:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(self.collection_name)

        ids = [str(i) for i in range(len(valid_docs))]
        texts = [doc.text for doc in valid_docs]
        embeddings = [doc.embedding.tolist() for doc in valid_docs]
        metadatas = [{"chunk_id": str(doc.chunk_id)} for doc in valid_docs]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        self.documents = valid_docs

        if file_path:
            self._save_file_hash(file_path)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_relevant_chunks(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Document, float]]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "embeddings"]
        )

        relevant_docs = []
        for i in range(len(results['ids'][0])):
            text = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            embedding = np.array(results['embeddings'][0][i])

            similarity = self._cosine_similarity(query_embedding, embedding)

            doc = Document(
                text=text,
                chunk_id=int(metadata['chunk_id']),
                embedding=embedding
            )
            relevant_docs.append((doc, similarity))

        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        return relevant_docs


class GenericRAG:
    def __init__(self, file_path: str, chunker: TextChunker = None, persist_directory: str = "./chroma_db"):
        self.file_path = Path(file_path)
        self.chunker = chunker if chunker else TextChunker()
        self.ollama = OllamaClient()
        self.retriever = PersistentChromaRetriever(persist_directory=persist_directory)
        self.documents: List[Document] = []

        if self.retriever.needs_update(str(self.file_path)):
            self._load_document()
        else:
            print("Using existing embeddings from ChromaDB")
            self.documents = self.retriever.documents

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
        print(f"Loading {self.file_path.name} - {self.chunker.get_config_info()}")

        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        text = self._clean_text(text)
        chunks = self.chunker.chunk_text(text)

        for i, chunk in enumerate(chunks):
            try:
                chunk.embedding = self.ollama.get_embedding(chunk.text, "document")
                self.documents.append(chunk)
                if (i + 1) % 25 == 0 or i == len(chunks) - 1:
                    print(f"  Embedded {i + 1}/{len(chunks)} chunks")
            except Exception as e:
                print(f"Failed to embed chunk {i}: {e}")

        self.retriever.add_documents(self.documents, str(self.file_path))
        print(f"Ready! {len(self.documents)} chunks loaded and saved to ChromaDB.")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _retrieve_chunks(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        query_embedding = self.ollama.get_embedding(query, "query")
        return self.retriever.get_relevant_chunks(query_embedding, top_k)

    def _show_match_details(self, matches: List[Tuple[Document, float]], query: str) -> None:
        if not matches:
            return

        query_embedding = self.ollama.get_embedding(query, "query")
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
            preview = doc.text[:80].replace('\n', ' ') + "..."
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
            context_parts.append(f"Context {i}:\n{doc.text}")

        context = "\n\n".join(context_parts)

        prompt = f"""Answer the question using the provided context passages.

Be specific and detailed. Quote relevant text when appropriate using quotation marks.
If the context doesn't contain enough information, say so clearly.

Question: {question}

Context:
{context}

Answer:"""

        try:
            answer = self.ollama.generate_response(prompt)
            return answer.strip()
        except Exception as e:
            return f"Error generating response: {e}"


def main():
    print("ChromaDB RAG Demo System")
    print("Make sure Ollama is running on port 11434\n")

    try:
        rag = GenericRAG(
            str(HERE / "data" / "alice_in_wonderland.txt"),
            persist_directory=str(HERE / "chroma_db"),
        )
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
