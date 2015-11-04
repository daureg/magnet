#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""A more efficient implementation of tBFS."""
from collections import defaultdict, deque
from grid_stretch import add_edge
from itertools import combinations
from multiprocessing import Pool
import naive_tbfs as nt


def opposite_triangle_edges(u, Gr, Ed, common_neighbors):
    "return edges facing u and forming a triangle with it"""
    tedges=set()
    for v in Gr[u]:
        for w in common_neighbors[(u,v) if u<v else (v,u)]:            
            e = (u,w) if u < w else (w,u)
            if e in Ed:
                tedges.add((v,w) if v<w else (w,v))
    return tedges


def triangles_shared_edge(t1, t2):
    t1, t2 = sorted(t1), sorted(t2)
    et1 = {tuple(t1[:2]), tuple(t1[1:]), (t1[0], t1[-1])}
    et2 = {tuple(t2[:2]), tuple(t2[1:]), (t2[0], t2[-1])}
    shared = et1.intersection(et2)
    if len(shared)==1:
        return next(iter(shared))
    return None


def compute_common_neighbors(G, E):
    return {e: G[e[0]].intersection(G[e[1]]) for e in E}
    pool = Pool(14)
    def cn(nodes):
        return G[nodes[0]].intersection(G[nodes[1]])
    common_neighbors = dict(pool.imap_unordered(cn, E, len(E)//14))
    pool.close()
    pool.join()
    return common_neighbors


def get_tbfs(G, E, src, dst=None):
    root=src
    Q = deque()
    tree = defaultdict(dict)
    parents = defaultdict(lambda: defaultdict(set))
    all_closed=set()
    common_neighbors = compute_common_neighbors(G, E)
    for u, v in opposite_triangle_edges(root, G, E, common_neighbors):
        outgoing = (root, u)
        next_edge = (u, v)
        closing = (v, root)
        tree[outgoing][0] = v
        parents[u][0].add(outgoing)
        tree[next_edge][1] = root
        parents[v][1].add(next_edge)
        tree[closing][1] = u
        parents[root][1].add(closing)
        all_closed.add(nt.hash_triangle((root, u, v)))
        # print('{:02}: closed {} [{} -> {} -> {}]'.format(0, (root, u, v), outgoing, next_edge, closing))
        Q.append(next_edge)
        Q.append(closing)
    Q.append(None)

    phase = 1
    while Q and phase < len(G):
        e = Q.popleft()
        if e is None:
            phase += 1
            if Q:
                Q.append(None)
            continue
        base, tip = e
        for u in common_neighbors[(base, tip) if base<tip else (tip, base)]:
            next_edge = (tip, u)
            closing = (u, base)
            thash = nt.hash_triangle((tip, base, u))
            if tree[next_edge] and base in {tree[next_edge][max(tree[next_edge])],
                                            tree[next_edge].get(phase, -1),
                                            tree[next_edge].get(phase-1, -1)}:
                continue
            if thash in all_closed:
                # print('not closing {} [{} -> {} -> {}] because {}=={}?'.format((base, tip, u), e, next_edge, closing,
                #                                                                tree[next_edge], base))
                continue
            parents[u][phase+1].add(next_edge)
            tree[e][phase+1] = u
            tree[next_edge][phase+1] = base
            parents[base][phase+1].add(closing)
            tree[closing][phase+1] = tip
            # print('{:02}: closed {} [{} -> {} -> {}]'.format(phase, (base, tip, u), e, next_edge, closing))
            all_closed.add(thash)
            # print('{:02}: going forward with {} & {}'.format(phase, next_edge, closing))
            Q.append(next_edge)
            Q.append(closing)
            if u == dst:
                return parents
    if dst is None:
        return parents
    return None


def get_path(parents, src, dst, common_neighbors):
    np = {}
    for u, back in parents.items():
        np[u] = {k: v for k, v in back.items()}
    from copy import deepcopy
    parents = deepcopy(np)

    path, parent = [], dst
    level = min(parents[dst])
    res=[]
    while level>=0:
        path.append(parent)
        p = parents[parent]
        if level not in parents[parent]:
            level += 1
        candidates = parents[parent][level]
        assert candidates, (parents[parent], level)
        if len(candidates) == 1 or len(path) == 1:
            e = next(iter(candidates))
        else:
            for e in candidates:
                if path[-2] in common_neighbors[tuple(sorted(e))]:
                    break
            else:
                raise RuntimeError("can't find a shared edge")
        res.append(e)
        level -= 1
        parent = e[0]
    all_t = [tuple(sorted(set(list(e)+list(f)))) for e, f in zip(res, res[1:])]
    present = set()
    nodup = []
    for t in all_t:
        if t not in present:
            present.add(t)
            nodup.append(t)
    shared_edges=[triangles_shared_edge(t1, t2)
                  for t1, t2 in zip(nodup, nodup[1:])]
    return clean_shared_edges(shared_edges, src, dst)


def clean_shared_edges(shared_edges, src, dst):
    if len(shared_edges) == 0:
        e = (src, dst) if src < dst else (dst, src)
        return [e], [e]
    if len(shared_edges) == 1:  # diamond
        p = shared_edges[0][0]
        e1 = (dst, p) if dst < p else (p, dst)
        e2 = (src, p) if src < p else (p, src)
        return [e1, e2], [e1, e2]
    pG = {}
    for u, v in shared_edges:
        add_edge(pG, u, v)
    full_path = [nt.pick_edge(dst, shared_edges[0], pG)]+shared_edges
    add_edge(pG, *full_path[0])
    full_path += [nt.pick_edge(src, shared_edges[-1], pG)]    
    add_edge(pG, *full_path[-1])
    # discard redundant support edges
    res = []
    in_wheel, wheel_size, count = False, -1, len(pG)
    wheel_centers = set()
    for e, nx in zip(full_path, full_path[1:]):
        d0, d1 = len(pG[e[0]]), len(pG[e[1]])
        wheel_center = e[0] if d0 > d1 else e[1]
        if (d0 > 3 or d1 > 3) and not in_wheel and wheel_center not in wheel_centers:
            in_wheel = True            
            wheel_size = len(pG[wheel_center])
            count = 0
            wheel_centers.add(wheel_center)
        if in_wheel and count == wheel_size-3:
            in_wheel, count = False, len(pG)
            u, v = e
            res.append((u, v) if u < v else (v, u))            
            continue
        if in_wheel and wheel_center in nx:
            u = e[0] if d0 < d1 else e[1]
            v = nx[0] if nx[0] != wheel_center else nx[1]
            count += 1
            res.append((u, v) if u < v else (v, u))
            continue
        if src not in e and dst not in e and (d0==1 or d1==1):
            continue
        res.append(e)
    res.append(full_path[-1])
    return res, full_path


if __name__ == '__main__':
    # pylint: disable=C0103
    import convert_experiment as cexp
    import draw_utils as du
    import graph_tool.draw
    import graph_tool.generation
    import numpy as np
    import redensify
    import random
    import sys
    N=60
    nt.N = N
    points = np.random.rand(N, 2)*5
    k, pos = graph_tool.generation.triangulation(points, type='delaunay')
    cexp.to_python_graph(k)
    G, E = redensify.G, set(redensify.EDGES_SIGN.keys())
    common_neighbors = compute_common_neighbors(G, E)
    root = max(G.items(), key=lambda x: len(x[1]))[0]
    tG, tE, node_to_triangles = nt.build_triangle_graph(G, E)

    parents = get_tbfs(G, E, root)
    assert len(parents) == len(G)
    for dst in G:
        if dst==root or dst in G[root]:
            continue
        epath, _ = get_path(parents, root, dst, common_neighbors)
        ttree, tparents, w = nt.perturbed_bfs(tG, root, dst, node_to_triangles)
        tpath = nt.path_to_root(w, tparents)
        shared_edges = [tE[(tr1, tr2) if tr1 < tr2 else (tr2, tr1)]
                        for tr1, tr2 in zip(tpath, tpath[1:])]
        tpath, _ = clean_shared_edges(shared_edges, root, dst)
        if len(epath) > len(tpath)+1:
            k.save('fail.gt')
            print('{}: {} > {}'.format(dst, len(epath), len(tpath)))
            # raise RuntimeError('{}: {} > {}'.format(dst, len(epath), len(tpath)))
