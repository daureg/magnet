import random
from collections import defaultdict
from itertools import combinations

from grid_stretch import add_edge
from tbfs import memoize

N = 50


@memoize
def hash_triangle(points):
    """Give an unique id to each vertices triplet"""
    a, b, c = sorted(points)
    return N*(a*N+b)+c
@memoize
def triangle_nodes(hash_):
    """Return the 3 vertices index making a triangle id"""
    c = hash_ % N
    hash_ = (hash_ - c) // N
    b = hash_ % N
    a = (hash_ - b) // N
    return a, b, c


def build_triangle_graph(G, E):
    common_neighbors = {nodes: G[nodes[0]].intersection(G[nodes[1]]) for nodes in E}
    # find the triangles
    tG = {}
    node_to_triangles = defaultdict(set)
    for (u, v), cn in common_neighbors.items():
        for w in cn:
            tr = hash_triangle((u, v, w))
            tG[tr] = set()
            node_to_triangles[u].add(tr)
            node_to_triangles[v].add(tr)
            node_to_triangles[w].add(tr)
    # link them
    tE={}
    for e, cn in common_neighbors.items():
        u, v = e
        for (w1, w2) in combinations(cn, 2):
            tr1 = hash_triangle((u, v, w1))
            tr2 = hash_triangle((u, v, w2))
            assert tr1 in tG and tr2 in tG, e
            tr1, tr2 = (tr1, tr2) if tr1 < tr2 else (tr2, tr1)
            tE[(tr1, tr2)] = e
            add_edge(tG, tr1, tr2)
    return tG, tE, node_to_triangles


def perturbed_bfs(G, src, dst, node_to_triangles):
    src_expanded = node_to_triangles[src].copy()
    dst_expanded = node_to_triangles[dst].copy()
    tree = []
    current_border, next_border = src_expanded, set()
    discovered = src_expanded
    parents = {s: None for s in src_expanded}    
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
                e = (v, w) if v < w else (w, v)
                tree.append(e)
                parents[w] = v
                if w in dst_expanded:
                    return tree, parents, w
        current_border, next_border = next_border, set()
        empty_border = len(current_border) == 0
    return tree, parents


def path_to_root(u, parents):
    path, parent = [], u
    while parent is not None:                
        path.append(parent)
        parent = parents[parent]
    return path


def pick_edge(u, edge, e_graph):
    e_deg = set([len(e_graph[edge[0]]), len(e_graph[edge[1]])])
    min_d, max_d = sorted(edge, key=lambda x: len(e_graph[x]))
    assert e_deg in [{1,2}, {1,3}], (edge, e_deg)
    v = min_d if e_deg == {1, 3} else max_d
    return (u, v) if u < v else (v, u)


def clean_path(path, tE, src, dst):
    shared_edges = [tE[(tr1, tr2) if tr1 < tr2 else (tr2, tr1)]
                    for tr1, tr2 in zip(path, path[1:])]
    pG = {}
    for u, v in shared_edges:
        add_edge(pG, u, v)
    full_path = [pick_edge(dst, shared_edges[0], pG)]+shared_edges
    full_path += [pick_edge(src, shared_edges[-1], pG)]
    add_edge(pG, *full_path[0])
    add_edge(pG, *full_path[-1])
    # discard redundant support edges
    res = []
    for e in full_path:
        if src not in e and dst not in e and (len(pG[e[0]])==1 or len(pG[e[1]])==1):
            continue
        res.append(e)
    return res, full_path


if __name__ == "__main__":
    import convert_experiment as cexp
    import draw_utils as du
    import graph_tool.draw
    import graph_tool.generation
    import numpy as np
    import redensify

    points = np.random.randn(N, 2)*5
    k, pos = graph_tool.generation.triangulation(points, type='delaunay')
    name = k.new_vertex_property('string')
    for v in k.vertices():
        name[v] = str(int(v))
    cexp.to_python_graph(k)
    G, E = redensify.G, set(redensify.EDGES_SIGN.keys())
    root = max(G.items(), key=lambda x: len(x[1]))[0]
    spos=graph_tool.draw.sfdp_layout(k)
    ecol, esize = 'gray', 2
    vhalo = k.new_vertex_property('boolean')
    vhalo.a = np.arange(len(G)) == root
    graph_tool.draw.graph_draw(k, pos=spos, output_size=(900, 900),
                               output='delauny.pdf',
                               vprops={'size': 25, 'fill_color': [1,1,1,.5],
                                       'text': name, 'halo': vhalo, 'halo_color': du.good_edge},
                               eprops={'color': ecol, 'pen_width': esize})

    tG, tE, node_to_triangles = build_triangle_graph(G, E)

    kk = graph_tool.Graph(directed=False)
    mapping={}
    for i, vt in enumerate(sorted(tG)):
        kk.add_vertex()
        mapping[vt] = i
    for (ut, vt) in tE:
        kk.add_edge(kk.vertex(mapping[ut]), kk.vertex(mapping[vt]))
    inv_mapping = {v: k for k, v in mapping.items()}
    nname = kk.new_vertex_property('string')
    ppos = kk.new_vertex_property('vector<double>')
    for v in kk.vertices():
        tr = triangle_nodes(inv_mapping[int(v)])    
        nname[v] = '{:02}.{:02}.{:02}'.format(*tr)
        ppos[v] = np.vstack([spos[kk.vertex(tr[0])],
                            spos[kk.vertex(tr[1])],
                            spos[kk.vertex(tr[2])]]).mean(0)

    src, dst = 25, 33
    tree, parents, w = perturbed_bfs(tG, src, dst, node_to_triangles)

    ecol, esize = du.color_graph(kk, {(mapping[e[0]], mapping[e[1]]) for e in tree})
    graph_tool.draw.graph_draw(kk, pos=ppos, output_size=(900, 900),
                               output='delauny_tr.pdf',
                               vprops={'size': 10, 'fill_color': [1,1,1,.8], 'font_size': 8,
                                       'text': nname,},
                               eprops={'color': ecol, 'pen_width': esize})

    path = path_to_root(w, parents)
    res, full_path = clean_path(path, tE, src, dst)

    ecol, esize = du.color_graph(k, res)
    def color_edge(k, root, colmap=ecol, col=du.bad_edge):
        colmap[k.edge(k.vertex(root[0]), k.vertex(root[1]))] = col
    for e in full_path:
        if e not in res:
            color_edge(k, e)
    graph_tool.draw.graph_draw(k, pos=spos, output_size=(900, 900),
                               output='delauny_path.pdf',
                               vprops={'size': 25, 'fill_color': [1,1,1,.5],
                                       'text': name, 'halo': vhalo, 'halo_color': du.good_edge},
                               eprops={'color': ecol, 'pen_width': esize})
