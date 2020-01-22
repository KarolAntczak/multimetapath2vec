import pickle

import numpy as np
from networkx import Graph
from sklearn.utils import shuffle

from data_encoding import to_integers, to_binary
from graph.load_graph import load_graph_from_csv


def create_dataset_diagnosis(graph_filename, number_of_cases):
    """
    Create dataset for disease diagnosis task.

    :param number_of_cases: number of cases per disease
    :param graph_filename: name of file containing graph
    """
    g = load_graph_from_csv(graph_filename)
    all_train_cases = dict()
    all_train_labels = dict()
    all_val_cases = dict()
    all_val_labels = dict()
    all_diseases = [node for node in g.nodes if g.nodes[node]['type'] == "JCH"]

    for missing_percentage in range(90, 100, 1):

        subgraph = get_subgraph(g, missing_percentage)

        all_train_cases[missing_percentage] = []
        all_train_labels[missing_percentage] = []
        all_val_cases[missing_percentage] = []
        all_val_labels[missing_percentage] = []

        for disease_id in all_diseases:

            if disease_id in subgraph.nodes:
                train_cases, train_labels = create_cases(subgraph, disease_id, number_of_cases)
            else:
                train_cases = []
                train_labels = []

            val_cases, val_labels = create_cases(g, disease_id, number_of_cases)

            train_cases = to_integers(g.nodes, train_cases)
            train_labels = to_integers(all_diseases, train_labels)
            val_cases = to_integers(g.nodes, val_cases)
            val_labels = to_integers(all_diseases, val_labels)

            all_train_cases[missing_percentage].extend(train_cases)
            all_train_labels[missing_percentage].extend(train_labels)
            all_val_cases[missing_percentage].extend(val_cases)
            all_val_labels[missing_percentage].extend(val_labels)

        all_train_cases[missing_percentage], all_train_labels[missing_percentage] = shuffle(all_train_cases[missing_percentage], all_train_labels[missing_percentage])
        all_val_cases[missing_percentage], all_val_labels[missing_percentage] = shuffle(all_val_cases[missing_percentage], all_val_labels[missing_percentage])

        print(" %d%%\ttrain: %d val: %d" %(missing_percentage , len(all_train_cases[missing_percentage]), len(all_val_cases[missing_percentage])))

    print("Saving dataset. Cases per disease: %d" % number_of_cases)

    pickle.dump((all_train_cases, all_train_labels, all_val_cases, all_val_labels), open(file=graph_filename + "_diagnosis.pickle", mode='wb'))


def create_cases(g: Graph, disease_id, number_of_cases):
    cases = []
    symptoms = [neighbor for neighbor in g.neighbors(disease_id) if g.nodes[neighbor]['type'] == "O"]
    labels = []

    if not symptoms:
        return cases, labels

    for i in range(0, number_of_cases):
        number_of_symptoms = min(max(1, len(symptoms)), 10)
        case = np.random.choice(symptoms, number_of_symptoms, replace=False)
        cases.append(case)
        labels.append([disease_id])

    return cases, labels

def get_subgraph(g : Graph, missing_percentage) -> Graph :
    subgraph = g.copy()
    all_nodes_count = len(subgraph.nodes)
    to_delete_count = int(all_nodes_count * (missing_percentage/100))
    to_delete_nodes = np.random.choice(subgraph.nodes, to_delete_count)
    subgraph.remove_nodes_from(to_delete_nodes)
    return subgraph




def load_data(filename):
    x_train, y_train, x_val, y_val = pickle.load(open(filename, 'rb'))
    return x_train, y_train, x_val, y_val


if __name__ == '__main__':
    create_dataset_diagnosis("../data/dane.csv", number_of_cases=100)
