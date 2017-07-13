#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Implement heuristics to solve Partial Low Stretch problem."""
import random
import sys
from collections import defaultdict, deque

from grid_stretch import perturbed_bfs
from heap import heap
from new_galaxy import galaxy_maker
from pred_on_tree import get_dfs_tree


def profile(func): return func


@profile
def get_mbfs_tree(G, X, momentum=False):
    tree = []
    q = deque()
    label = {i: i if i in X else None for i in G}
    conn = defaultdict(set)
    # It's kind of hacky to add edge between nodes of X directly in the tree
    # but they would grow onto each other anyway (except since both nodes are
    # already labeled, the edge wouldn't be added in the while loop)
    for i in X:
        for j in (j for j in G[i] if j in X and i < j):
            if j not in conn[i]:
                tree.append((i, j))
            update_connectivity(conn, i, j)
        q.append(i)

    def xmax(seq): return 0 if not seq else max(seq)
    max_component_size = xmax([len(_) for _ in conn.values()])
    while q:
        v = q.popleft()
        label_v = label[v]
        assert label_v is not None
        if momentum:
            # each component need to add its root to its size
            v_component_size = len(conn[label_v]) + 1
            ratio = v_component_size/float(max_component_size + 1)
            proba = max(0.3, ratio**.5)
            if random.random() > proba:
                q.append(v)
                continue
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
                max_component_size = xmax([len(_) for _ in conn.values()])
    return tree


@profile
def update_connectivity(conn, a, b):
    """`a` < `b` got connected inside X, which mean `conn` might get new True
    entries"""
    # it could be cheapest to compute component max_size here
    component_a = conn[a].union((a, b))
    component_b = conn[b].union((b, a))
    for v in component_a:
        conn[v].update(component_b)
    for v in component_b:
        conn[v].update(component_a)
    conn[a].remove(a)
    conn[b].remove(b)


def get_one_bfs_tree(G, X):
    degrees = {node: len(G[node]) for node in X}
    max_degree = max(degrees.values())
    roots = [node for node, deg in degrees.items() if deg == max_degree]
    root = random.choice(roots)
    return perturbed_bfs(G, root)


def get_one_dfs_tree(G, X):
    degrees = {node: len(G[node]) for node in X}
    max_degree = max(degrees.values())
    roots = [node for node, deg in degrees.items() if deg == max_degree]
    root = random.choice(roots)
    return get_dfs_tree(G, root)


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
    import convert_experiment as cexp
    from timeit import default_timer as clock
    timings = []
    n = 500
    for i in range(30):
        cexp.fast_preferential_attachment(n, m=3, bonus_neighbor_prob=.13)
        G, N = cexp.redensify.G, cexp.redensify.N
        X = set(random.sample(list(G.keys()), int(.1*N)))
        start = clock()
        tree = get_mbfs_tree(G, X)
        timings.append(clock() - start)
        print(timings[-1])
        assert len(tree) == N-1, i
    # print(timings)
    print('--\n{:.4f}'.format(sum(timings)/len(timings)))
