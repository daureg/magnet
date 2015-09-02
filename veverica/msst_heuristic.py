#! /usr/bin/python
# vim: set fileencoding=utf-8
from collections import defaultdict
import grid_stretch as gs
from itertools import product
import persistent
"""Various methods to improve a existing spanning tree."""


def profile(func):
    return func


@profile
def augmented_ancestor(tree_adj, X):
    tree_root = max(((node, len(adj)) for node, adj in tree_adj.items()),
                    key=lambda x: x[1])[0]
    prt = gs.ancestor_info(tree_adj, tree_root)
    if len(prt) != len(tree_adj):
        persistent.save_var('bug_parent.my', (prt, tree_adj))
    assert len(prt) == len(tree_adj), set(tree_adj.keys()) - set(prt.keys())
    leaves = {u for u, adj in tree_adj.items() if len(adj) == 1}
    infos = {u: (prt[u], 1, int(u in X)) for u in leaves}
    possible_inclusion = defaultdict(int)
    for _ in infos.values():
        if _[0] is not None:
            possible_inclusion[_[0]] += 1

    def ready(u, vote):
        threshold = len(tree_adj[u]) - 1
        if prt[u] is None:
            threshold += 1
        return vote == threshold
    border = {u for u, vote in possible_inclusion.items() if ready(u, vote)}
    while border:
        for u in border:
            children = {v for v in tree_adj[u] if v in infos}
            subtree_size, num_in_x, parent = 1, int(u in X), prt[u]
            for v in children:
                subtree_size += infos[v][1]
                num_in_x += infos[v][2]
            infos[u] = (parent, subtree_size, num_in_x)
            del possible_inclusion[u]
            if parent is not None:
                possible_inclusion[parent] += 1
        border = {u for u, vote in possible_inclusion.items()
                  if ready(u, vote)}
    return infos, {u: v[0] for u, v in infos.items()}


@profile
def edge_tail(edge, tree_parents):
    u, v = edge
    pu, pv = tree_parents[u], tree_parents[v]
    child = u if (pv is None or pu == v) else v
    parent = v if child == u else u
    return child, parent


@profile
def fast_cost(infos, X, tree):
    prt = {v: val[0] for v, val in infos.items()}
    cost = 0
    V = len(infos)
    x_size = len(X)
    xbar_size = V - x_size
    for edge in tree:
        tail = edge_tail(edge, prt)[0]
        subtree_size, x_nodes = infos[tail][1:]
        other_side = V - subtree_size
        other_x = x_size - x_nodes
        other_xbar = other_side - other_x
        xbar_nodes = subtree_size - x_nodes    
        cost += xbar_nodes*other_x + x_nodes*other_xbar
    return cost/(x_size * xbar_size)


@profile
def tree_cost(X, tree_edges, tree_adj):
    infos = augmented_ancestor(tree_adj, X)[0]
    return fast_cost(infos, X, tree_edges)


def slow_tree_cost(X, tree=None, prt=None, g_variant=False):
    assert tree or prt
    if not prt:
        tree_adj = {}
        for u, v in tree:        
            gs.add_edge(tree_adj, u, v)
        prt = gs.ancestor_info(tree_adj, 0)
    X_bar = set(range(len(prt)))-X
    if g_variant:
        distances = list(min((gs.tree_path(i, j, prt) for i in X)) for j in X_bar)
    distances = list(gs.tree_path(i, j, prt) for i, j in product(X_bar, X))
    return sum(distances)/len(distances)



@profile
def cut_edges(edge, graph, tree_adj, tree_parents):
    """return the set of edges in `graph` between the 2 components of
    `tree_adj` when `edge` is removed."""
    child, parent = edge_tail(edge, tree_parents)
    component, border = set([child]), tree_adj[child].copy()
    border.remove(parent)
    while border:
        component.update(border)
        new_border = {v for u in border for v in tree_adj[u] if v not in component}
        border = new_border
    return {(u, v) if u < v else (v, u) for u in component for v in graph[u] 
            if v not in component and (u, v) != (child, parent)}


@profile
def remove_edge(tree_adj, u, v):
    tree_adj[u].remove(v)
    tree_adj[v].remove(u)


@profile
def edge_improvement(edge, X, graph, tree_adj, tree_parents, tree):
    best_cost, best_edge = 1<<62, None
    u, v = edge
    for eprime in cut_edges(edge, graph, tree_adj, tree_parents):
        new_cost = replacement_cost(edge, eprime, X, tree_adj, tree)
        if new_cost < best_cost:
            best_cost, best_edge = new_cost, eprime
    return best_cost, best_edge


@profile
def replacement_cost(removed_edge, added_edge, X, tree_adj, tree):
    """compute the cost function if switching `removed_edge` in favor of
    `add_edge`"""
    remove_edge(tree_adj, *removed_edge)
    gs.add_edge(tree_adj, *added_edge)
    infos = augmented_ancestor(tree_adj, X)[0]
    ntree = [(u, v) for u in tree_adj for v in tree_adj[u] if u < v]
    new_cost = fast_cost(infos, X, ntree)
    gs.add_edge(tree_adj, *removed_edge)
    remove_edge(tree_adj, *added_edge)
    return new_cost


@profile
def improve_tree(tree_edges, tree_adj, G, X):
    nb_iter = 0
    infos, prt = augmented_ancestor(tree_adj, X)
    current_cost = fast_cost(infos, X, tree_edges)
    while True:
        nb_iter += 1
        # print('begin iter {}: {:.4f}'.format(nb_iter, current_cost))
        improv = {e: edge_improvement(e, X, G, tree_adj, prt, tree_edges)
                  for e in tree_edges}
        candidate_edges = sorted({e: info for e, info in improv.items() 
                                  if info[0] < .9999*current_cost}.items(), 
                                 key=lambda x: x[1][0])
        can_be_improved = len(candidate_edges) > 0
        if not can_be_improved:
            break
        e, (c, ne) = candidate_edges[0]
        remove_edge(tree_adj, *e)
        gs.add_edge(tree_adj, *ne)
        # print('replace {} by {}: {:.3f}'.format(e, ne, current_cost - c))
        tree_edges = [(u, v) for u in tree_adj for v in tree_adj[u] if u < v]
        infos, prt = augmented_ancestor(tree_adj, X)
        current_cost = fast_cost(infos, X, tree_edges)
        # print('end iter {}: {:.4f}'.format(nb_iter, current_cost))
    return current_cost, tree_edges, tree_adj


def tree_path(u, v, parents):
    path1, path2, history = set(), set(), {}
    parent = u
    while parent is not None:
        if parent == v:
            return len(path1), None
        path1.add(parent)
        history[parent] = len(path1)
        parent = parents[parent]
    parent = v
    while parent is not None:
        path2.add(parent)
        if parent in path1:
            break
        parent = parents[parent]
    common = path1.intersection(path2)
    assert len(common) == 1
    common = next(iter(common))
    return len(path2) + history[common] - 2, common


def actual_tree_path(u, v, parents):
    def reorder_edges(nodes):
        return [(u, v) if u < v else (v, u)
                for u, v in zip(nodes, nodes[1:])]
    length, common = tree_path(u, v, parents)
    path = []
    parent = u
    # if common is None, then it's the root
    while parent != common:
        if parent == v:
            return reorder_edges(path+[v])
        path.append(parent)
        parent = parents[parent]
    if common is not None:
        path.append(parent)
    npath = []
    parent = v
    while parent != common:
        npath.append(parent)
        parent = parents[parent]
    return reorder_edges(path+npath[::-1])


def full_cost(tree_adj, tree, gt_graph, X):
    import graph_tool.centrality as gt
    from bidict import bidict
    vbetween, ebetween = gt.betweenness(gt_graph)
    info, prt = augmented_ancestor(tree_adj, X)
    root = None
    for u, parent in prt.items():
        if parent is None:
            root = u
            break
    assert root is not None
    Xbar = set(tree_adj.keys()) - X
    edge_to_vertex = bidict({})
    raw_edge_features = {}
    for edge in tree:
        tail = edge_tail(edge, prt)[0]
        edge_to_vertex[edge] = tail
        raw_edge_features[edge] = [ebetween[gt_graph.edge(*edge)],
                                   vbetween[gt_graph.vertex(tail)],
                                   info[tail][1]/len(tree_adj),
                                   info[tail][2]/len(X), [], []]
    distances = []
    for i in X:
        for j in Xbar:
            path = actual_tree_path(i, j, prt)
            path_len = len(path)
            distances.append(path_len)
            if j != root:
                raw_edge_features[edge_to_vertex[:j]][4].append(path_len)
            for edge in path:
                raw_edge_features[edge][5].append(path_len)
    return raw_edge_features, sum(distances)/len(distances)


def process_features(raw_features):
    import numpy as np
    edges_idx = {}
    features = np.zeros((len(raw_features), 11))
    for i, (edge, feats) in enumerate(raw_features.items()):
        edges_idx[i] = edge
        if len(feats[4]) == 0:
            mmm = [0, 0, 0]
        else:
            mmm = [min(feats[4]), np.median(feats[4]), max(feats[4])]
        plen = np.percentile(feats[5], [25, 50, 75])
        features[i,:] = np.hstack([feats[:4], mmm, [len(feats[5])], plen])
    return edges_idx, features

if __name__ == '__main__':
    # pylint: disable=C0103
    pass
    import convert_experiment as cexp
    import random
    import partiallst as plst
    from copy import deepcopy
    # 6944 fail one assertion (with N,m,b=45,2,.18)
    random.seed(6944)
    N = 45
    cexp.fast_preferential_attachment(N, m=2, bonus_neighbor_prob=.18)
    G, E = cexp.redensify.G, cexp.redensify.EDGES_SIGN
    X = set(random.sample(G.keys(), 3))
    btree = plst.get_one_bfs_tree(G, X)
    btree_adj = {}
    for u, v in btree:        
        gs.add_edge(btree_adj, u, v)                                                                                            
    print(len(E))
    infos, prt = augmented_ancestor(btree_adj, X)
    print(fast_cost(infos, X, btree))
    new_cost, tree_edges, tree_adj = improve_tree(deepcopy(btree), deepcopy(btree_adj), G, X)
    print(new_cost)
