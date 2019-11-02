import pickle

from keras_preprocessing.sequence import skipgrams

import node2vec
from data_encoding import encode_labels
from graph.load_graph import load_graph_from_csv


def create_dataset_skipgrams(graph_filename):
    """
    Create dataset with skipgrams from random walking.

    :param graph_filename: name of file containing graph
    """
    g = load_graph_from_csv(graph_filename)
    node2vec_g = node2vec.Graph(g, is_directed=False, p=1., q=1.)
    node2vec_g.preprocess_transition_probs()
    walks = node2vec_g.simulate_walks(10, 80)

    print("Encoding to integers")
    walks_encoded = encode_labels(g.nodes, walks)

    print("Generating skipgrams")

    all_couples = []
    all_labels = []
    for walk_encoded in walks_encoded:
        couples, labels = skipgrams(sequence=walk_encoded, vocabulary_size=len(g.nodes)+1)
        all_couples += couples
        all_labels += labels

    print(len(all_couples))
    print(len(all_labels))

    print("Saving dataset")

    pickle.dump((all_couples, all_labels), open(file=graph_filename + "_train.pickle", mode='wb'))


if __name__ == '__main__':
    create_dataset_skipgrams("./data/dane_small.csv")