import pickle

from keras_preprocessing.sequence import skipgrams

import metapath2vec
import multimetapath2vec
import node2vec
from data_encoding import to_integers
from graph.load_graph import load_graph_from_csv


def generate_skipgrams(graph_filename, algorithm):
    """
    Create dataset with skipgrams from random walking.

    :param algorithm: algorithm for random walking. Available values: 'node2vec', 'metapath2vec', 'multimetapath2vec'
    :param graph_filename: name of file containing graph
    """
    g = load_graph_from_csv(graph_filename)

    num_walks = 10
    walk_length = 80


    walks = []
    if algorithm == "node2vec":
        ng = node2vec.Graph(g, is_directed=False, p=1., q=1.)
        ng.preprocess_transition_probs()
        walks = ng.simulate_walks(num_walks, walk_length)
    elif algorithm == "metapath2vec":
        walks = metapath2vec.Graph(g).simulate_walks(num_walks, walk_length, metapath=["JCH", "O", "NO", "O", "WO", "O", "JCH"])
    elif algorithm == "multimetapath2vec":
        walks = multimetapath2vec.Graph(g).simulate_walks(num_walks, walk_length, metapaths=
                                                                            [["JCH", "O", "NO", "O", "JCH"],
                                                                             ["JCH", "O", "WO", "O", "JCH"]])

    print("Encoding to integers")
    walks_encoded = to_integers(g.nodes, walks)

    print("Generating skipgrams")

    all_couples = []
    all_labels = []
    for walk_encoded in walks_encoded:
        couples, labels = skipgrams(sequence=walk_encoded, vocabulary_size=len(g.nodes) + 1)
        all_couples += couples
        all_labels += labels

    print(len(all_couples))
    print(len(all_labels))

    print("Saving dataset")

    pickle.dump((all_couples, all_labels), open(file=graph_filename + "_" + algorithm + ".pickle", mode='wb'))


if __name__ == '__main__':
    generate_skipgrams("./data/dane.csv", algorithm="node2vec")
