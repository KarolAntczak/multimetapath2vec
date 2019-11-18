import pickle

import numpy as np
from sklearn.utils import shuffle

from data_encoding import to_integers, to_binary
from graph.load_graph import load_graph_from_csv


def create_dataset_classification(graph_filename, labels_filename):
    """
    Create dataset for node classification task.

    :param graph_filename: name of file containing graph
    :param labels_filename: name of CSV file with pairs <node id, label>
    """
    g = load_graph_from_csv(graph_filename)
    labels_dict = dict(np.loadtxt(labels_filename, skiprows=1, delimiter=";", dtype=np.str))
    nodes = []
    labels = []

    for node in g.nodes:
        if node in labels_dict:
            nodes.append([node])
            labels.append([labels_dict[node]])

    print("Encoding to integers")

    nodes = to_integers(g.nodes, nodes)
    labels = to_integers(list(labels_dict.values()), labels)

    nodes, labels = shuffle(nodes, labels, random_state=42)
    assert len(nodes) == len(labels)

    print("Saving dataset. Size: %d" % len(nodes))

    pickle.dump((nodes, labels), open(file=graph_filename + "_classification.pickle", mode='wb'))


def load_data(filename):
    x_data, y_data = pickle.load(open(filename, 'rb'))
    y_data = to_binary(y_data, np.amax(y_data))
    return x_data, y_data


if __name__ == '__main__':
    create_dataset_classification("../data/dane.csv", "../data/dictionaries/jednostki_chorobowe_icd_grupa.csv")
