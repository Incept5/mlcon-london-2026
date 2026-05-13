import numpy as np
import requests


def get_ollama_embedding(text: str) -> list:
    data = {"model": "all-minilm", "prompt": text}
    try:
        response = requests.post("http://localhost:11434/api/embeddings", json=data)
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_most_similar_sentence(question: str, sentences: list) -> tuple:
    question_embedding = get_ollama_embedding(question)
    if question_embedding is None:
        return [], 0, 0

    similarities = []
    similarity_scores = []
    for sentence in sentences:
        sentence_embedding = get_ollama_embedding(sentence)
        if sentence_embedding is not None:
            similarity = cosine_similarity(question_embedding, sentence_embedding)
            similarities.append((sentence, similarity))
            similarity_scores.append(similarity)

    if not similarity_scores:
        return [], 0, 0

    mean_similarity = np.mean(similarity_scores)
    std_similarity = np.std(similarity_scores)

    # z-score: standard deviations from the mean, so the caller can flag outliers
    enhanced_similarities = [
        (s, sim, (sim - mean_similarity) / std_similarity if std_similarity > 0 else 0)
        for s, sim in similarities
    ]
    enhanced_similarities.sort(key=lambda x: x[1], reverse=True)
    return enhanced_similarities, mean_similarity, std_similarity


def main():
    sentences = [
        "The cat sat on the mat.",
        "What is the weather like today?",
        "Python is a popular programming language.",
        "How do I write code?",
        "The quick brown fox jumps over the lazy dog.",
        "What's the best way to learn programming?",
        "Machine learning involves training models on data.",
        "Can you help me debug this code?",
    ]

    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        print("\nFinding similar sentences...")
        results, mean_sim, std_sim = find_most_similar_sentence(question, sentences)

        if results:
            print(f"\nStatistics:")
            print(f"Mean similarity: {mean_sim:.4f}")
            print(f"Standard deviation: {std_sim:.4f}")
            print("\nResults (sorted by similarity):")
            print("=" * 70)
            for sentence, similarity, std_devs in results:
                print(f"Similarity: {similarity:.4f}, {std_devs:+.2f}σ")
                print(f"Sentence: {sentence}")
                print("-" * 70)
        else:
            print("No results found.")


if __name__ == "__main__":
    print("Enhanced Sentence Similarity Matcher")
    print("Make sure Ollama is running locally on port 11434")
    main()
