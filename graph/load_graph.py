import networkx as nx
import numpy as np


def load_graph_from_csv(filename: str) -> nx.Graph:
    data = np.loadtxt(filename, delimiter=",", skiprows=1, dtype=np.str)

    data_jch = ["jch_%s" % jch for jch in data[:, 0]]
    data_wo = ["wo_%s" % wo for wo in data[:, 1]]
    data_no = ["no_%s" % no for no in data[:, 2]]
    data_o = ["o_%s_%s" % (i, j) for i, j in zip(data[:, 1], data[:, 2])]

    graph = nx.Graph()

    graph.add_nodes_from(data_jch, type="JCH", color='red')
    graph.add_nodes_from(data_wo, type="WO", color='green')
    graph.add_nodes_from(data_no, type="NO", color='yellow')
    graph.add_nodes_from(data_o, type="O", color='blue')

    graph.add_edges_from(zip(data_jch, data_o))
    graph.add_edges_from(zip(data_wo, data_o))
    graph.add_edges_from(zip(data_no, data_o))

    return graph
