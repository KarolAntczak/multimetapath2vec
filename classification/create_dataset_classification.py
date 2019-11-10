import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from data_encoding import to_integers
from graph.load_graph import load_graph_from_csv


def create_dataset_classification(graph_filename, labels_filename, train_size  = 0.8):
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

    nodes_train, nodes_test, labels_train, labels_test = train_test_split(nodes, labels, train_size=train_size, random_state=42)
    assert len(nodes_train) == len(labels_train)
    assert len(nodes_test) == len(labels_test)

    print("Saving dataset. Train size: %d Test size: %d" % (len(nodes_train), len(nodes_test)))

    pickle.dump((nodes_train, labels_train, nodes_test, labels_test), open(file=graph_filename + "_classification.pickle", mode='wb'))


if __name__ == '__main__':
    create_dataset_classification("./data/dane.csv", "./data/dictionaries/jednostki_chorobowe_icd_grupa.csv")