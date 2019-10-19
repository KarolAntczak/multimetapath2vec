import networkx as nx
import matplotlib.pyplot as plt
from graph.load_graph import load_graph_from_csv


def draw_graph(graph: nx.Graph):
    nodes = graph.nodes('color')

    nodes, colors = zip(*nodes)
    pos = nx.fruchterman_reingold_layout(graph)
    nx.draw_networkx_edges(graph, pos, alpha=0.2)
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=colors, with_labels=False, node_size=20)

    plt.draw()
    plt.show()


if __name__ == '__main__':
    g = load_graph_from_csv("../data/dane.csv")
    draw_graph(g)
