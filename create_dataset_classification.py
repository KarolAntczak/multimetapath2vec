import pickle
import numpy as np
from data_encoding import encode_labels
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

    nodes = encode_labels(g.nodes, nodes)
    labels = encode_labels(list(labels_dict.values()), labels)

    print("Saving dataset")

    pickle.dump((nodes, labels), open(file=graph_filename + "_classification.pickle", mode='wb'))


if __name__ == '__main__':
    create_dataset_classification("./data/dane_small.csv", "./data/dictionaries/jednostki_chorobowe_icd_grupa.csv")