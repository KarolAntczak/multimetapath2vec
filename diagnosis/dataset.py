import pickle

import numpy as np
from networkx import Graph
from sklearn.utils import shuffle

from data_encoding import to_integers
from graph.load_graph import load_graph_from_csv


def create_dataset_diagnosis(graph_filename, number_of_cases):
    """
    Create dataset for disease diagnosis task.

    :param number_of_cases: number of cases per disease
    :param graph_filename: name of file containing graph
    """
    g = load_graph_from_csv(graph_filename)
    all_cases = dict()
    all_labels = dict()

    for missing_percentage in np.arange(99.99, 100., 0.1):
        cases = []
        labels = []

        for node_id in g.nodes():
            node = g.nodes[node_id]

            if node["type"] == "JCH":
                cases += create_cases(g, node_id, positive=True, number_of_cases=number_of_cases, missing_percentage=missing_percentage)
                cases += create_cases(g, node_id, positive=False, number_of_cases=number_of_cases, missing_percentage=missing_percentage)
                labels += np.ones(number_of_cases).tolist()
                labels += np.zeros(number_of_cases).tolist()



        cases = to_integers(g.nodes, cases)

        labels = np.asarray(labels)
        cases, labels = shuffle(cases, labels, random_state=42)
        assert len(cases) == len(labels)

        all_cases[missing_percentage] = cases
        all_labels[missing_percentage] = labels

        print(cases.shape)
        print(" %f %d" %(missing_percentage , np.count_nonzero(cases)))

    print("Saving dataset. Cases per disease: %d" % number_of_cases)

    pickle.dump((all_cases, all_labels), open(file=graph_filename + "_3_diagnosis.pickle", mode='wb'))


def create_cases(g: Graph, disease_id, number_of_cases, positive=True, missing_percentage=0):
    cases = []

    max_length = len([node for node in g.nodes if g.nodes[node]['type'] == "O"])
    other_diseases = [node for node in g.nodes if g.nodes[node]['type'] == "JCH"]
    other_diseases.remove(disease_id)

    for i in range(0, number_of_cases):
        if positive:
            case = [disease_id]
        else:
            case = np.random.choice(other_diseases, 1).tolist()
        case += _get_random_associated_symptoms(g, disease_id, missing_percentage).tolist()
        case = np.pad(case, (0, max_length - len(case)), pad_with)

        cases.append(case)
    return cases


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 'empty')
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def _get_random_associated_symptoms(g: Graph, disease_id: int, missing_percentage):
    symptoms = [neighbor for neighbor in g.neighbors(disease_id) if g.nodes[neighbor]['type'] == "O"]
    number_of_symptoms = int(len(symptoms) * ((100 - missing_percentage) / 100))
    return np.random.choice(symptoms, number_of_symptoms)

def load_data(filename):
    x_data, y_data = pickle.load(open(filename, 'rb'))
    return x_data, y_data


if __name__ == '__main__':
    create_dataset_diagnosis("../data/dane.csv", number_of_cases=10)
