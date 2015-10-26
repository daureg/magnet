from itertools import product
from msst_heuristic import tree_path
from math import log
import random
"""Baselines for active binary node classification"""
QUERIED = {}


def short_bfs(G, root, target):
    """Start build a BFS tree from `root` until it reaches `target`"""
    tree = []
    parents = {root: None}
    current_border, next_border = set(), set()
    discovered = set([root])
    for u in G[root]:
        current_border.add(u)
        discovered.add(u)
        parents[u] = root
        e = (root, u) if root < u else (u, root)
        tree.append(e)

    empty_border = len(current_border) == 0
    while not empty_border:
        destination = list(current_border)
        random.shuffle(destination)
        for v in destination:
            for w in G[v]:
                if w in discovered:
                    continue
                next_border.add(w)
                discovered.add(w)
                parents[w] = v
                e = (v, w) if v < w else (w, v)
                tree.append(e)
                if w == target:
                    return tree, parents
        current_border, next_border = next_border, set()
        empty_border = len(current_border) == 0
    return tree, parents


def actual_tree_path(u, v, parents):
    """Return the nodes on a path from `u` to `v`, according to the `parents`
    relationship"""
    length, common = tree_path(u, v, parents)
    path = []
    parent = u
    # if common is None, then it's the root
    while parent != common:
        if parent == v:
            return path + [v]
        path.append(parent)
        parent = parents[parent]
    if common is not None:
        path.append(parent)
    npath = []
    parent = v
    while parent != common:
        npath.append(parent)
        parent = parents[parent]
    return path+npath[::-1]



def get_label(node):
    """Return label of `node`, keeping track of oracle queries"""
    global QUERIED, LABELS
    if node not in QUERIED:
        QUERIED[node] = LABELS[node]
    return QUERIED[node]


def find_cute_edge(graph, a, b):
    """performs dichotomic queries to return either a cut-edge along the 
    shortest path from a to b in the current graph, or None if no such edge exists."""
    if a > b:
        a, b = b, a 
    _, prt = short_bfs(graph, a, b)
    if b not in prt:
        # a and b are disconnected
        return None
    #TODO inverse a and b, factor path part and binary search
    path = list(reversed(actual_tree_path(b, a, prt)))
    # print('{} -> {}: {}'.format(a, b, len(path)))
    a_label = get_label(a)
    lo, hi = 0, len(path)
    local_labels = {a: labels[a]}
    while lo+1 < hi:
        mid = lo + (hi-lo)//2
        mid_label = get_label(path[mid])
        local_labels[path[mid]] = mid_label
        if mid_label == a_label:
            lo = mid
        else:
            hi = mid
    u, v = path[lo], path[hi]
    if u not in local_labels:
        local_labels[u] = get_label(u)
    if v not in local_labels:
        local_labels[v] = get_label(v)
    if local_labels[u] == local_labels[v]:
        return None
    return (u, v) if u < v else (v, u)   


def dichotomic_baseline(G, alpha):
    seeds = random.sample(list(G.keys()), int(log(len(G))//alpha))
    for s in seeds:
        get_label(s)
    pos_nodes, neg_nodes = [], []
    for u in seeds: 
        (pos_nodes if labels[u] == 1 else neg_nodes).append(u)
    nG = G.copy()
    cut_edges = set()
    for u, v in product(neg_nodes, pos_nodes):
        e = find_cute_edge(nG, u, v)
        while e is not None:
            cut_edges.add(e)
            a, b = e
            nG[a].remove(b)
            nG[b].remove(a)
            e = find_cute_edge(nG, u, v)
    return cut_edges, seeds


def predict_labels(G, cut_edges):
    """Predict the label of all the non queried nodes of `G` by
    finding the its connected component according to `cut_edges`"""
    pass


if __name__ == "__main__":
    from collections import Counter
    from galaxy import to_graph_tool_simple
    import draw_utils as du
    import graph_tool.draw
    import grid_stretch as gs
    import numpy as np
    # creating the grid
    n = 20
    G, E = gs.make_grid(n)
    kk = to_graph_tool_simple(G)
    p = np.zeros((2, n*n))
    for i in range(n*n):
        p[:, i] = (i%n, n-i//n)
    pos = kk.new_vertex_property('vector<double>')
    pos.set_2d_array(p)
    LABELS = {}
    for u in G:
        i, j = u//n, u%n
        LABELS[u] = 0
        if (0 <= i <= n//3 and 0 <= j <= n//3 or
            2*n//3 <= i and 2*n//3 <= j):
            LABELS[u] = 1
    comp = kk.new_vertex_property('int')
    for v in kk.vertices():
        comp[v] = LABELS[int(v)]

    alpha = min(Counter(labels.values()).values())/2/len(G)

    cut_edges, seeds = dichotomic_baseline(G, alpha)

    halo = kk.new_vertex_property('boolean')
    halo.a = np.array([u in queried and u not in seeds
                       for u in sorted(G.keys())])
    ecol, esize = du.color_graph(kk, cut_edges)
    graph_tool.draw.graph_draw(kk, pos=pos, output_size=(800,800),
                               vprops={'size': 15, 'fill_color': comp,
                                       'shape': comp, 'halo': halo,
                                       'halo_color': du.good_edge},
                               eprops={'color': ecol, 'pen_width': esize},)
    msg = 'Found {} cut edges after {}+{} queries'
    print(msg.format(len(cut_edges), len(seeds), len(QUERIED)-len(seeds)))
