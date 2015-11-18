import collections
import grid_stretch as gs
from copy import deepcopy
from itertools import product
from msst_heuristic import tree_path
from math import log
import random
"""Baselines for active binary node classification"""
QUERIED = {}
LABELS = None


def short_bfs(G, root, targets, find_all=True):
    """Start build a BFS tree from `root` until it reaches one (or all) node in
    `targets`"""
    tree = []
    parents = {root: None}
    current_border, next_border = set(), set()
    discovered = set([root])
    if not isinstance(targets, collections.Sequence):
        targets = [targets]
    targets = set(targets)
    for u in G[root]:
        current_border.add(u)
        discovered.add(u)
        parents[u] = root
        e = (root, u) if root < u else (u, root)
        tree.append(e)
        if u in targets:
            return tree, parents

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
                if w in targets:
                    targets.remove(w)
                    if not find_all or len(targets) == 0:
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


def find_path(graph, source, destinations):
    """Find the shortest path betwee `source` node and one or more
    `destinations`. Return None is there is no such path."""
    if isinstance(destinations, collections.Sequence):
        _, prt = short_bfs(graph, source, destinations)
        for w in destinations:
            if w in prt:
                return actual_tree_path(w, source, prt)
        # can't reach any destinations
        return None
    else:
        a, b = source, destinations
    if a > b:
        a, b = b, a 
    _, prt = short_bfs(graph, a, b)
    if b not in prt:
        # a and b are disconnected
        return None
    return actual_tree_path(b, a, prt)


def find_cute_edge(graph, a, b):
    """performs dichotomic queries to return either a cut-edge along the 
    shortest path from a to b in the current graph, or None if no such edge exists."""
    path = find_path(graph, a, b)
    if not path:
        return None
    if get_label(path[0]) == 0:
        path = list(reversed(path))
    lo, hi = 0, len(path)-1
    while lo+1 < hi:
        mid = lo + (hi-lo)//2
        mid_label = get_label(path[mid])
        if mid_label == 1:
            lo = mid
        else:
            hi = mid
    u, v = path[lo], path[hi]
    if get_label(u) == get_label(v):
        return None
    return (u, v) if u < v else (v, u)   


def pick_seed_labels(G, alpha):
    """Query the label of |V|/alpha nodes and return the positive and negative
    ones"""
    QUERIED.clear()
    seeds = random.sample(list(G.keys()), int(log(len(G))//alpha))
    pos_nodes, neg_nodes = [], []
    for u in seeds: 
        (pos_nodes if get_label(u) == 1 else neg_nodes).append(u)
    return pos_nodes, neg_nodes


def dichotomic_baseline(G, seeds):
    pos_nodes, neg_nodes = seeds
    nG = deepcopy(G)
    cut_edges = set()
    for u, v in product(neg_nodes, pos_nodes):
        e = find_cute_edge(nG, u, v)
        while e is not None:
            cut_edges.add(e)
            a, b = e
            nG[a].remove(b)
            nG[b].remove(a)
            e = find_cute_edge(nG, u, v)
    return cut_edges


def triplet_strategy(G, seeds):
    pos_nodes, neg_nodes = (set(_) for _ in seeds)
    pairs_of_ones = []
    for u in pos_nodes:
        for v in G[u].intersection(pos_nodes).difference([u]):
            if u < v:
                pairs_of_ones.append((u, v))
    if not pairs_of_ones:
        raise NotImplemented('need to query additional labels, maybe around pos_nodes, or those with a lot of common neighbors')
    nG = deepcopy(G)
    cut_edges = set()
    for pair in pairs_of_ones:
        ancestors, root = build_triplets_spanner(nG, pair)
        while ancestors:
            other, other_label, cedges = find_cut_triplet(ancestors, root, pair)
            while other_label == 0:
                other, other_label, cedges = find_cut_triplet(ancestors, other, pair)
            cut_edges.update(cedges)
            for e in cedges:
                nG[e[0]].remove(e[1])
                nG[e[1]].remove(e[0])
            ancestors, root = build_triplets_spanner(nG, pair)
    return cut_edges


def find_cut_triplet(ancestors, root, pair):
    u, v = find_cute_edge(ancestors, root, pair)
    edges = [(u, v) if u < v else (v, u)]
    if v in ancestors[u]:
        other = next(iter(ancestors[u].difference([v])))
        edges.append((other, u) if other < u else (u, other))
    else:
        other = next(iter(ancestors[v].difference([u])))
        edges.append((other, v) if other < v else (v, other))
    return other, get_label(other), edges


def remove_edge(G, to_delete):
    nG = {}
    for u in G:
        for v in G[u]:
            e = (u, v) if u < v else (v, u)
            if e in to_delete:
                continue
            gs.add_edge(nG, u, v)
    return nG


def predict_labels(G, cut_edges, seeds, triplet_strategy=False):
    """Predict the label of all the non queried nodes of `G` by
    finding their connected component according to `cut_edges`"""
    if triplet_strategy:
        heads, tails = zip(*cut_edges)
        border = set(heads).union(set(tails))
        cut_edges = set()
        for u in border:
            assert u in QUERIED
            if QUERIED[u] == 1:
                continue
            neighbors = [v for v in G[u] if v in QUERIED and QUERIED[v] == 1]
            cut_edges.update((u, v) if u < v else (v, u) for v in neighbors)
    nG = remove_edge(G, cut_edges)
    if triplet_strategy:
        return _predict_triangle(nG, seeds)
    return _predict_baseline(nG, seeds)
    

def _predict_triangle(graph, seeds):
    pos_nodes, neg_nodes = (set(_) for _ in seeds)
    pairs_of_ones = []
    predicted_labels = {}
    for u in pos_nodes:
        for v in graph[u].intersection(pos_nodes).difference([u]):
            if u < v:
                pairs_of_ones.append((u, v))
    for pair in pairs_of_ones:
        if pair[0] in predicted_labels:
            continue
        ancestors, _ = build_triplets_spanner(graph, pair, find_component=True)
        if not ancestors:
            print(pair)
            raise RuntimeError('could not find')
        predicted_labels.update({u: 1 for u in ancestors})
    predicted_labels.update({u: QUERIED[u] for u in pos_nodes.union(neg_nodes)})
    predicted_labels.update({u: 0 for u in graph if u not in predicted_labels})
    return predicted_labels


def _predict_baseline(graph, seeds):
    pos_nodes, neg_nodes = seeds
    predicted_labels = {}
    for label, nodes in zip([1, 0], [pos_nodes, neg_nodes]):
        for root in nodes:
            if root in predicted_labels:
                continue
            _, prt = short_bfs(graph, root, [])
            predicted_labels.update({u: label for u in prt})
    predicted_labels.update({u: QUERIED[u] for u in pos_nodes + neg_nodes})
    # TODO: take majority vote instead (or random?)
    predicted_labels.update({u: 0 for u in graph if u not in predicted_labels})
    return predicted_labels


def build_triplets_spanner(G, root_pair, find_component=False):
    u, v = root_pair
    seen = set(root_pair)
    border, next_border = G[u].intersection(G[v]), set()
    ancestors = {u: set() for u in root_pair}
    for w in border:
        ancestors[w] = root_pair
    while border:
        seen.update(border)
        border = list(border)
        random.shuffle(border)
        for u in border:
            for v in G[u].difference(seen):
                for w in G[v].intersection(seen):
                    if w == u:
                        continue
                    next_border.add(v)
                    ancestors[v] = {u, w}
                    if v in QUERIED and QUERIED[v] == 0:
                        # print('{} is a 0 seed'.format(v))
                        return ancestors, v
        border, next_border = next_border, set()
    if find_component:
        return ancestors, None
    return None, None

if __name__ == "__main__":
    from collections import Counter
    import numpy as np
    # random.seed(4656987)
    draw = False
    # creating the grid
    n = 20
    G, E = gs.make_grid(n)
    for i, j in product(range(n-1), range(n-1)):
        e = (i*n+j, (i+1)*n+j+1)
        E.add(e)
        gs.add_edge(G, *e)
    LABELS = {}
    for u in G:
        i, j = u//n, u%20
        LABELS[u] = 0
        if (0 <= i <= n//3 and 0 <= j <= n//3 or
            2*n//3 <= i and 2*n//3 <= j):
            LABELS[u] = 1
            if ((i in {n//3, 2*n//3} and j % 2 == int(i==(2*n//3))) or
                (j in {n//3, 2*n//3} and i % 2 == int(j==(2*n//3)))):
                LABELS[u] = 0
    gold = [LABELS[u] for u in sorted(G.keys())]
    if draw:
        import draw_utils as du
        import graph_tool.draw
        from galaxy import to_graph_tool_simple
        kk = to_graph_tool_simple(G)
        p = np.zeros((2, n*n))
        for i in range(n*n):
            p[:, i] = (i%n, n-i//n)
        pos = kk.new_vertex_property('vector<double>')
        pos.set_2d_array(p)
        comp = kk.new_vertex_property('int')
        for v in kk.vertices():
            comp[v] = LABELS[int(v)]

    alpha = min(Counter(LABELS.values()).values())/2/len(G)

    seeds = pick_seed_labels(G, alpha)
    # seeds = [[21, 22, 377, 378], [210]]
    rseeds = seeds[0] + seeds[1]
    for u in rseeds:
        get_label(u)
    cut_edges = dichotomic_baseline(G, seeds)
    pred = predict_labels(G, cut_edges, seeds, triplet_strategy=False)
    baseline_pred = [pred[u] for u in sorted(G.keys())]
    accuracy = sum((int(g==p) for g, p in zip(gold, baseline_pred)))/len(G)

    if draw:
        halo = kk.new_vertex_property('boolean')
        halo.a = np.array([u in QUERIED and u not in rseeds
                           for u in sorted(G.keys())])
        ecol, esize = du.color_graph(kk, cut_edges)
        graph_tool.draw.graph_draw(kk, pos=pos, output_size=(800,800),
                                   vprops={'size': 15, 'fill_color': comp,
                                           'shape': comp, 'halo': halo,
                                           'halo_color': du.good_edge},
                                   eprops={'color': ecol, 'pen_width': esize},)
    msg = 'Found {} cut edges after {}+{} queries: {}% accuracy'
    print(msg.format(len(cut_edges), len(rseeds), len(QUERIED)-len(rseeds),
                     100*accuracy))

    QUERIED.clear()
    for u in rseeds:
        get_label(u)
    cut_edges = triplet_strategy(G, seeds)
    pred = predict_labels(G, cut_edges, seeds, triplet_strategy=True)
    triplet_pred = [pred[u] for u in sorted(G.keys())]
    accuracy = sum((int(g==p) for g, p in zip(gold, triplet_pred)))/len(G)
    msg = 'Found {} cut edges after {}+{} queries: {}% accuracy'
    print(msg.format(len(cut_edges), len(rseeds), len(QUERIED)-len(rseeds),
                     100*accuracy))
