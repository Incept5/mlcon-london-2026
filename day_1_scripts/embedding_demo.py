import numpy as np
import ollama
from tabulate import tabulate

client = ollama.Client()


def get_embedding(text):
    return client.embeddings(model="all-minilm", prompt=text)["embedding"]


def main():
    words = ["tea", "coffee", "mud", "dirt"]

    embeddings = np.array([get_embedding(w) for w in words])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = np.dot(embeddings, embeddings.T)

    header = [""] + words
    table = [
        [words[i]] + [f"{similarities[i][j]:.3f}" for j in range(len(words))]
        for i in range(len(words))
    ]

    print(tabulate(table, headers=header, tablefmt="grid"))


if __name__ == "__main__":
    main()
