import random


class Graph:
    def __init__(self, nx_G):
        self.G = nx_G

    def metapath2vec_walk(self, walk_length, start_node, metapath):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G

        walk = [start_node]

        for i in range(1, walk_length):
            current_node = walk[-1]
            next_type_index = i % (len(metapath)-1)
            next_type = metapath[next_type_index]

            allowed_neighbors = [neighbor for neighbor in G.neighbors(current_node) if
                                 G.nodes[neighbor]['type'] == next_type]
            if not allowed_neighbors:
                break
            next_node = random.choice(allowed_neighbors)
            walk.append(next_node)

        return walk

    def simulate_walks(self, num_walks, walk_length, metapath):
        """
       Repeatedly simulate random walks from each node.
        """
        G = self.G
        walks = []

        start_type = metapath[0]
        start_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == start_type]

        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(start_nodes)
            for start_node in start_nodes:
                walks.append(self.metapath2vec_walk(walk_length=walk_length, start_node=start_node, metapath=metapath))

        good = 0
        for walk in walks:
            print(len(walk))
            if len(walk) == walk_length:
                good += 1
        print("Correct length walks: %d/%d" % (good, len(walks)))

        return walks
