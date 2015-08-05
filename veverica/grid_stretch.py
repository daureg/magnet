#! /usr/bin/env python
# vim: set fileencoding=utf-8
from collections import deque
import random as r
import new_galaxy as ng


def add_edge(tree, u, v):
    """Update adjacency list `tree` with the (u, v) edge"""
    if u in tree:
        tree[u].add(v)
    else:
        tree[u] = set([v])
    if v in tree:
        tree[v].add(u)
    else:
        tree[v] = set([u])


def ancestor_info(G, root):
    q = deque()
    discovered = {u: False for u in G}
    q.append(root)
    discovered[root] = True
    parents = {root: None}
    while q:
        v = q.popleft()
        for w in G[v]:
            if not discovered[w]:
                q.append(w)
                discovered[w] = True
                parents[w] = v
    return parents


def perturbed_bfs(G, root=0):
    tree = []
    current_border, next_border = set(), set()
    discovered = set([root])
    for u in G[root]:
        current_border.add(u)
        discovered.add(u)
        e = (root, u) if root < u else (u, root)
        tree.append(e)
    empty_border = len(current_border) == 0
    while not empty_border:
        destination = list(current_border)
        r.shuffle(destination)
        for v in destination:
            for w in G[v]:
                if w in discovered:
                    continue
                next_border.add(w)
                discovered.add(w)
                e = (v, w) if v < w else (w, v)
                tree.append(e)
        current_border, next_border = next_border, set()
        empty_border = len(current_border) == 0
    return tree


def make_grid(side=10):
    n = side
    graph = {}
    test_edges = set()
    for i in range(n*n):
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
            add_edge(graph, i, v)
            test_edges.add((i, v) if i < v else (v, i))
    return graph, test_edges


def tree_path(u, v, parents):
    path1, path2, history = set(), set(), {}
    parent = u
    while parent is not None:
        if parent == v:
            return len(path1)
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
    common = list(common)[0]
    return len(path2) + history[common] - 2


if __name__ == '__main__':
    # pylint: disable=C0103
    import persistent as p
    import sys
    rside = int(sys.argv[1])
    if 0 <= rside <= 50:
        supp = []
    if 50 < rside <= 200:
        supp = [25]
    if 200 < rside:
        supp = [25, 45]
    if 500 < rside:
        supp = 5*[55, ]
    for side in supp+[rside]:
        # run first on small ones to give Pypy time to compile
        graph, all_test_edges = make_grid(side)
        # gtx, _ = ng.galaxy_maker(graph, 1000, short=True)
        bfs = perturbed_bfs(graph)
        #for name, edges in zip(['sgt', 'bfs'], [gtx, bfs]):
        for name, edges in zip(['bfs'], [bfs]):
            tree_adj = {}
            for u, v in edges:
                add_edge(tree_adj, u, v)
            prt = ancestor_info(tree_adj, 42)
            test_edges = all_test_edges.difference(edges)
            estretch = [tree_path(u, v, prt) for u, v in test_edges]
            if side < 50:
                continue
            p.save_var('ngrid_{}_{}'.format(name, side), estretch)
            #print(name, len(estretch), sum(estretch)/len(estretch))
