# coding: utf-8
from grid_stretch import add_edge
from pred_on_tree import get_bfs_tree
from collections import defaultdict

def compute_stats(labels, E):
    G = {}
    for u, v in E:
        add_edge(G, u, v)

    return (find_corrupted(G, labels), get_components_size(E, labels),
            within_triangle(G, E))

def find_corrupted(G, labels):
    corrupted = defaultdict(lambda : [set(), set(), set()])
    for u in G:
        num_pos_neigbors = len({v for v in G[u] if labels[v] == 1})
        for k in range(2, 5+1):
            if len(G[u]) < k:
                continue
            corrupted[k][0].add(u)
            if labels[u] == 1 and num_pos_neigbors < k:
                corrupted[k][1].add(u)
                continue
            if labels[u] == 0 and num_pos_neigbors >= k:
                corrupted[k][2].add(u)
    return corrupted

def get_components_size(E, labels):
    nE = {e for e in E if labels[e[0]] == labels[e[1]]}
    nG = {}
    for u, v in nE:
        add_edge(nG, u, v)
    discovered = {u: False for u in nG}
    components_size = {}
    for u in discovered:
        if discovered[u]:
            continue
        tree = get_bfs_tree(nG, u)
        nodes = set((_[0] for _ in tree)).union(set((_[1] for _ in tree)))
        for x in nodes:
            discovered[x] = True
        components_size[u] = len(tree)+1
    assert all(discovered.values())
    assert sum(components_size.values()) == len(nG)
    return components_size

def within_triangle(G, E):
    in_triangle = set()
    for u in G:
        if u in in_triangle:
            continue
        found = False
        for v in G[u]:
            for w in G[v]:
                if w == u:
                    continue
                e = (w, u) if w < u else (u, w)
                if e in E:
                    in_triangle.update([u, v, w])
                    found = True
                    break
            if found:
                break
    return in_triangle
