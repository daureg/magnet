# coding: utf-8
import random
from collections import defaultdict

from grid_stretch import add_edge
from pred_on_tree import get_bfs_tree
from real_world import reindex_nodes


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
        if u in in_triangle or len(G[u]) < 2:
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


def triangle_subgraph(E, in_triangle):
    """Return the subgraph of `E` induces by nodes belonging to at least one
    triangle"""
    tE = {(u, v) for u, v in E if u in in_triangle and v in in_triangle}
    G = {}
    for u, v in tE:
        add_edge(G, u, v)
    triangle_mapping = {u: i for i, u in enumerate(sorted(G))}
    fG, fE = reindex_nodes(G, {e: 0 for e in tE}, triangle_mapping)
    fE = set(fE.keys())
    return fG, fE, triangle_mapping


def synthetic_label(G, E, decay=.98, num_seeds=10):
    """affect a label to each node of G"""
    from multiprocessing import Pool
    from collections import deque
    common_neighbors = {nodes: G[nodes[0]].intersection(G[nodes[1]])
                        for nodes in E}
    # pool = Pool(14)
    # common_neighbors = dict(pool.imap_unordered(lambda nodes: G[nodes[0]].intersection(G[nodes[1]]),
    #                                             E, len(E)//14))
    # pool.close()
    # pool.join()
    seed_edges = random.sample(list(E), num_seeds)
    labels = {u: 0 for u in G}
    first_ones = {u: 1 for edge in seed_edges for u in edge}
    labels.update(first_ones)
    visited = set(first_ones.keys())
    queue = deque()
    for u, v in seed_edges:
        for w in common_neighbors[(u, v)]:
            queue.append((u, w) if u < w else (w, u))
            queue.append((v, w) if v < w else (w, v))
    queue.append(None)
    proba = decay
    nb_iter = 0
    nb_phases = 0
    while len(visited) < 0.98*len(G) and nb_iter < 1.1*len(E):
        nb_iter += 1
        e = queue.popleft()
        if e is None:
            nb_phases += 1
            if not queue:
                break
            print(len(queue))
            proba *= decay
            queue.append(None)
            
            continue
        u, v = e
        for w in common_neighbors[e]:
            if w in visited:
                continue
            visited.add(w)
            if random.random() < proba:
                queue.append((u, w) if u < w else (w, u))
                queue.append((v, w) if v < w else (w, v))
                labels[w] = 1
    print(nb_iter, nb_phases, len(visited))
    return labels


if __name__ == "__main__":
    filename = '../../data/higgs/higgs-social_network.edgelist'
    filename = '../../data/g_plusAnonymized.csv'
    E = set()
    with open(filename) as f:
        for i, line in enumerate(f):
            u, v = [int(_) for _ in line.strip().split(',')]
            E.add((u, v) if u < v else (v, u))
    from grid_stretch import add_edge
    import persistent as p
    G = {}
    for u, v in E:
        add_edge(G, u, v)
    print(len(G), len(E))
    from timeit import default_timer as clock
    start = clock()
    in_t = within_triangle(G, E)
    print('{:.2f}% (in {:.2f}s)'.format(100*len(in_t)/len(G), clock() - start))
    p.save_var('google_triangles.my', in_t)
