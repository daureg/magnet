#! /usr/bin/env python
# vim: set fileencoding=utf-8
import convert_experiment as cexp
import random as r
import numpy as np
import pred_on_tree as pot
import new_galaxy as ng
from graph_tool.generation import lattice
from graph_tool.topology import shortest_distance


def perturbed_bfs(G, root=0):
    tree = []
    current_border, next_border = set(), set()
    discovered = set()
    discovered.add(root)
    for u in G[root]:
        current_border.add(u)
        e = (root, u) if root < u else (u, root)
        tree.append(e)
    empty_border = len(current_border) == 0
    while not empty_border:
        destination = list(current_border)
        r.shuffle(destination)
        for v in destination:
            for w in G[v].difference(discovered.union(next_border)):
                next_border.add(w)
                e = (v, w) if v < w else (w, v)
                tree.append(e)
            discovered.add(v)
        current_border, next_border = next_border, set()
        empty_border = len(current_border) == 0
    return tree


def make_grid(side=10):
    cexp.new_graph()
    n = side
    p = np.zeros((2, n*n))
    for i in range(n*n):
        p[:, i] = (i % n, n-i//n)
        neighbors = []
        if i % n != 0:
            neighbors.append(i-1)
        if (i+1) % n != 0:
            neighbors.append(i+1)
        if i >= n:
            neighbors.append(i-n)
        if i < n*(n-1):
            neighbors.append(i+n)
        for v in neighbors:
            cexp.add_signed_edge(i, v, True)
    cexp.finalize_graph()
    return lattice([side, side])


def compute_stretch(k, edges):
    """Compute the stretch of all edges of `k` but those in the graph spanned
    by `edges`"""
    test_graph = {}
    k.set_vertex_filter(None)
    k.set_edge_filter(None)
    n = k.num_vertices()
    bfsmap = k.new_edge_property('boolean')
    for e in k.edges():
        u, v = int(e.source()), int(e.target())
        if (u, v) in edges:
            bfsmap[e] = True
        else:
            bfsmap[e] = False
            pot.add_edge_to_tree(test_graph, u, v)
    k.set_edge_filter(bfsmap)

    tree_dst = shortest_distance(k, dense=False)
    tree_mat = np.zeros((n, n), dtype=np.uint8)
    for v in k.vertices():
        tree_mat[int(v), :] = tree_dst[v].a.astype(np.uint8)

    edge_paths = {}
    for v in range(n):
        if v in test_graph:
            edge_paths.update({(v, w): tree_mat[v, w]
                               for w in sorted(test_graph[v]) if v < w})
    return edge_paths


if __name__ == '__main__':
    # pylint: disable=C0103
    from timeit import default_timer as clock
    side = 100
    n_rep = 2
    start = clock()
    k = make_grid(side)
    nb_test_edges = len(cexp.redensify.EDGES_SIGN) - (cexp.redensify.N - 1)
    res = np.zeros((2*n_rep, nb_test_edges))
    print(clock() - start); start = clock()
    for i in range(n_rep):
        bfs = perturbed_bfs(cexp.redensify.G, 0)
        res[i, :] =  np.array(list(compute_stretch(k, bfs).values()))
        print(clock() - start); start = clock()
        gtx, _ = ng.galaxy_maker(cexp.redensify.G, 1000, short=True)
        res[n_rep+i, :] = np.array(list(compute_stretch(k, gtx).values()))
        print(clock() - start); start = clock()
    print(res[:n_rep, :].mean(), res[n_rep:, :].mean())
    np.savez_compressed('stgrid_{}'.format(side), res=res)
