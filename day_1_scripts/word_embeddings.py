from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'notebook'

EMBEDDINGS_FILE = "embeddings.pkl"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def load_dictionary(word_file):
    with open(word_file, 'r') as f:
        return [line.strip() for line in f]


def generate_and_store_embeddings(all_words):
    embeddings = model.encode(all_words)
    result = dict(zip(all_words, embeddings))
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(result, f)
    return result


def find_similar(embeddings, target_words, n_neighbors=20):
    target_embeddings = model.encode(target_words)
    scored = [
        (word, float(np.mean(np.dot(vec, target_embeddings.T))))
        for word, vec in embeddings.items()
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:n_neighbors]


def load_embeddings():
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            data = pickle.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


if __name__ == '__main__':
    word_file = '/usr/share/dict/words'
    all_words = load_dictionary(word_file)

    embeddings = load_embeddings()
    if embeddings is None:
        print("Generating embeddings for the first time (this may take a few minutes)...")
        embeddings = generate_and_store_embeddings(all_words)
        print("Embeddings generated and cached.")

    target_words = ['tea', 'coffee', 'mud', 'dirt', "pretzel", "grape"]
    n = 100
    similar_words = find_similar(embeddings, target_words, n)

    print(f"Top {n} similar words to {', '.join(target_words)}:")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.4f}")

    neighbor_words = [w for w, _ in similar_words]
    combined_words = target_words + neighbor_words
    combined_embeddings = np.array([embeddings[w] for w in combined_words])

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(combined_embeddings)

    fig = go.Figure()
    target_idx = range(len(target_words))
    neighbor_idx = range(len(target_words), len(reduced))

    fig.add_trace(go.Scatter3d(
        x=reduced[target_idx, 0], y=reduced[target_idx, 1], z=reduced[target_idx, 2],
        mode='markers+text',
        marker=dict(color='red', size=10),
        text=target_words,
        name='Target Words',
        textposition='bottom center',
        textfont=dict(size=18, color='black')
    ))

    fig.add_trace(go.Scatter3d(
        x=reduced[neighbor_idx, 0], y=reduced[neighbor_idx, 1], z=reduced[neighbor_idx, 2],
        mode='markers+text',
        marker=dict(color='blue', size=5),
        text=neighbor_words,
        name='Nearest Neighbors'
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(xaxis_title='PCA1', yaxis_title='PCA2', zaxis_title='PCA3')
    )

    file_path = '3d_plot.html'
    with open(file_path, 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    print(f"\nVisualization saved to {file_path}")

    fig.show(renderer="browser")
