#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Implement heuristics to solve Partial Low Stretch problem."""
import sys
from collections import deque
from grid_stretch import perturbed_bfs
from new_galaxy import galaxy_maker
import random
from heap.heap import heap


def get_mbfs_tree(G, X):
    tree = []
    q = deque()
    label = {i: i if i in X else None for i in G}
    conn = {i: set((j for j in G[i] if j in X)) for i in X}
    for i in X:
        q.append(i)
    while q:
        v = q.popleft()
        label_v = label[v]
        assert label_v is not None
        for w in G[v]:
            label_w = label[w]
            a, b = label_v, label_w
            a, b = (a, b) if b is None or a < b else (b, a)
            if label_w is None:
                q.append(w)
                tree.append((v, w) if v < w else (w, v))
                label[w] = label_v
            elif a == b or b in conn[a]:
                continue
            else:
                tree.append((v, w) if v < w else (w, v))
                update_connectivity(conn, a, b)
    return tree


def update_connectivity(conn, a, b):
    """`a` < `b` got connected inside X, which mean `conn` might get new True
    entries"""
    for v in conn[a]:
        conn[v].add(b)
        conn[b].add(v)
    conn[a].add(b)
    for v in conn[b]:
        conn[v].add(a)
        conn[a].add(v)
    conn[b].add(a)


def get_one_bfs_tree(G, X):
    degrees = {node: len(G[node]) for node in X}
    max_degree = max(degrees.values())
    roots = [node for node, deg in degrees.items() if deg == max_degree]
    root = random.choice(roots)
    return perturbed_bfs(G, root)


def get_short_galaxy_tree(G, X=None):
    return galaxy_maker(G, 10, short=True)[0]


def get_Xaware_galaxy_tree(G, X):
    return galaxy_maker(G, 10, short=True, **{'X': X})[0]


def get_merged_bfs(G, X):
    bfts = [perturbed_bfs(G, x) for x in X]
    edge_weights = {(u, v): 0 for u in G for v in G[u] if u < v}
    for tree in bfts:
        for edge in tree:
            edge_weights[edge] -= 1
    return minimum_spanning_tree(G, edge_weights)


def minimum_spanning_tree(graph, weights, root=0):
    connection_cost = heap({v: 0 if v == root else sys.maxsize for v in graph})
    connection_edge = {v: None for v in graph}
    outside = set(graph.keys())
    tree = []
    while connection_cost:
        v = connection_cost.pop()
        outside.remove(v)
        if connection_edge[v] is not None:
            tree.append(connection_edge[v])
        for w in graph[v]:
            edge = (v, w) if v < w else (w, v)
            if w in outside and weights[edge] < connection_cost[w]:
                connection_cost[w] = weights[edge]
                connection_edge[w] = edge
    return tree

if __name__ == '__main__':
    # pylint: disable=C0103
    pass
