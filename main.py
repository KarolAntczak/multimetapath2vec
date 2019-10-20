from tensorflow_core.python.keras.layers.embeddings import Embedding

import node2vec
from graph.load_graph import load_graph_from_csv

if __name__ == '__main__':
    g = load_graph_from_csv("./data/dane.csv")
    node2vec_g = node2vec.Graph(g, is_directed=False, p=1., q=1.)
    node2vec_g.preprocess_transition_probs()
    walks = node2vec_g.simulate_walks(10, 80)

    Embedding
    print(len(walks))