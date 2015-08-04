#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Implement heuristics to solve Partial Low Stretch problem."""
from collections import deque


def get_bfs_tree(G, X):
    tree = []
    q = deque()
    label = {i: i if i in X else None for i in G}
    conn = {i: set((j for j in G[i] if j in X)) for i in X}
    print(conn)
    for i in X:
        q.append(i)
    nb_iter = 0
    max_iter = 3*sum((len(_) for _ in G.values()))
    while q and nb_iter < max_iter:
        nb_iter += 1
        v = q.popleft()
        label_v = label[v]
        assert label_v
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

if __name__ == '__main__':
    # pylint: disable=C0103
    pass
