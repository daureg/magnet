#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Implement random spanning tree generation for weighted undirected graph as
described in
Wilson, D. B. (1996). Generating Random Spanning Trees More Quickly Than the
Cover Time. STOCS'96 (pp. 296–303). doi:10.1145/237814.237880
"""
import random
import bisect


# http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python
class WeightedRandomGenerator(object):
    def __init__(self, items, weights):
        self.totals = []
        self.items = items
        running_total = 0

        for w in weights:
            running_total += w
            self.totals.append(running_total)
        self.total = running_total

    def next(self):
        rnd = random.random() * self.total
        return self.items[bisect.bisect_right(self.totals, rnd)]

    def __call__(self):
        return self.next()


def random_tree_with_root(graph_adj, edge_weight, root):
    in_tree = {n: n == root for n in graph_adj}
    parent = {root: None}
    successors = {}
    for u, neighbors in graph_adj.items():
        _neighbors = list(neighbors)
        weights = [edge_weight[(u, v) if u < v else (v, u)]
                   for v in _neighbors]
        successors[u] = WeightedRandomGenerator(_neighbors, weights)
    for i in range(len(graph_adj)):
        u = i
        while not in_tree[u]:
            parent[u] = successors[u]()
            u = parent[u]
        u = i
        while not in_tree[u]:
            in_tree[u] = True
            u = parent[u]
    return parent


def get_tree(graph, edge_weight):
    # For undirected graphs and Eulerian graphs, \tilde{\pi} is just the
    # uniform distribution on vertices.  In the case of undirected graphs,
    # since any vertex $r$ may be used to generate a free spanning tree, it
    # turns out to be more efficient to pick $r$ to be a random endpoint of a
    # random edge, sample \Gamma_r, and then pick a uniformly random vertex to
    # be the root.
    from pred_on_tree import add_edge_to_tree
    root = random.choice(list(graph.keys()))
    parent = random_tree_with_root(graph, edge_weight, root)
    rst_edges, tree_adj = {}, {}
    for u, v in parent.items():
        if v is None:
            continue
        edge = (u, v) if u < v else (v, u)
        rst_edges[edge] = edge_weight[edge]
        add_edge_to_tree(tree_adj, u, v)
    return tree_adj, rst_edges, parent


if __name__ == '__main__':
    # pylint: disable=C0103
    import convert_experiment as cexp
    from timeit import default_timer as clock
    cexp.fast_preferential_attachment(25000, 4)
    edge_weight = {e: random.random()*7 for e in cexp.redensify.EDGES_SIGN}

    timing = []
    for i in range(80):
        start = clock()
        random_tree_with_root(cexp.redensify.G, edge_weight, 34)
        print('done in {:.3f} sec'.format(clock() - start))
        if i > 10:
            timing.append(clock() - start)
    print('avrg run: {:.3f}'.format(sum(timing)/len(timing)))
