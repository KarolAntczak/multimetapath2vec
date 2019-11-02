import networkx as nx
import matplotlib.pyplot as plt
from graph.load_graph import load_graph_from_csv


def draw_graph(graph: nx.Graph):
    nodes = graph.nodes('color')
    labels = dict()
    for node in nodes:
        labels[node[0]] = node[0]

    nodes, colors = zip(*nodes)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_edges(graph, pos, alpha=0.2)
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=colors, node_size=10)
    nx.draw_networkx_labels(graph, pos, labels=labels)
    plt.draw()
    plt.show()


if __name__ == '__main__':
    g = load_graph_from_csv("../data/dane_small.csv")
    draw_graph(g)
