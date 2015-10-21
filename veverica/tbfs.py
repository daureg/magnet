#! /usr/bin/python
# vim: set fileencoding=utf-8
"""Compute a BFS where each path is a t-path."""
from collections import deque
from grid_stretch import add_edge

# http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/#c8
def memoize(f):
    class memodict(dict):
        __slots__ = ()
        def __missing__(self, key):
            self[key] = ret = f(key)
            return ret
    return memodict().__getitem__


def tbfs(G, root):
    @memoize
    def get_common_neighbors(nodes):
        return G[nodes[0]].intersection(G[nodes[1]])

    def found_edge(edge, endpoint=None, parent=None):
        tree.add(edge)    
        reached[edge[0]] = True
        reached[edge[1]] = True
        if endpoint is not None:
            add_more(endpoint, edge)
            parents[endpoint] = parent
        else:        
            add_more(edge[0], edge)
            add_more(edge[1], edge)

    def add_more(just_reached, added_edge):
        for v in G[just_reached]:
            if not reached[v]:
                Q.append(((just_reached, v) if just_reached < v else (v, just_reached), added_edge))
    Q = deque()
    tree = set()
    reached = {u: False for u in G}
    parents = {u: root for u in G[root]}
    for v in G[root]:
        found_edge((root, v) if root<v else (v,root), None)

    while Q:
        e, p = Q.popleft()
        u, v = p
        endpoint = e[0] if e[1] == u or e[1] == v else e[1]
        attach = e[1] if endpoint == e[0] else e[0]
        if reached[endpoint]:
            continue
        cn = get_common_neighbors(p)
        if endpoint in cn:
            others_triangle_tips = get_common_neighbors(e)
            others_triangle_tips.discard(u)
            others_triangle_tips.discard(v)
            if others_triangle_tips:
                found_edge(e, endpoint, attach)
            continue
        for w in G[endpoint]:
            if w in cn:
                found_edge(e, endpoint, attach)

    return tree, parents


if __name__ == '__main__':
    # pylint: disable=C0103
    from itertools import product
    from msst_heuristic import actual_tree_path
    import convert_experiment as cexp
    import draw_utils as du
    import graph_tool.draw
    import graph_tool.generation
    import numpy as np
    import redensify
    import random

    points = np.random.rand(50, 2)*5
    k, pos = graph_tool.generation.triangulation(points, type='delaunay')
    name = k.new_vertex_property('string')
    for v in k.vertices():
        name[v] = str(int(v))

    cexp.to_python_graph(k)
    G = redensify.G
    root = max(G.items(), key=lambda x: len(x[1]))[0]

    def get_common_neighbors(nodes):
        return G[nodes[0]].intersection(G[nodes[1]])
    def color_edge(k, root, col=du.bad_edge):
        ecol[k.edge(k.vertex(root[0]), k.vertex(root[1]))] = col
    def find_shared_edge(pair):
        found = False
        e, f = pair
        e_tr = []
        for tip in get_common_neighbors(e):
            _ = sorted([tip, e[0], e[1]])
            e_tr.append({tuple(_[:2]), tuple(_[1:]), (_[0], _[-1])})
        for tip in get_common_neighbors(f):
            _ = sorted([tip, f[0], f[1]])
            tr = {tuple(_[:2]), tuple(_[1:]), (_[0], _[-1])}
            if tr in e_tr:
                continue
            for edge, otr in product(tr, e_tr):
                if edge in otr:
                    return (tr, edge, otr)
        return None

    tree, parents = tbfs(G, root)
    T = {}
    for u, v in tree:
        add_edge(T, u, v)
    leaves = [u for u, adj in T.items() if len(adj) == 1]
    ecol, esize = du.color_graph(k, tree)

    for leaf in leaves:
        path = actual_tree_path(leaf, root, parents)
        assert((find_shared_edge(e,f) is not None for e, f in zip(path, path[1:]))), (leaf, path)

    leaf = random.choice(leaves)
    print(leaf)
    path = actual_tree_path(leaf, root, parents)
    for e, f in zip(path, path[1:]):
        shared_edge = find_shared_edge((e, f))[1]
        color_edge(k, shared_edge,
                   col=[.612, .153, .69, .9] if shared_edge in tree else du.bad_edge)
    vhalo = k.new_vertex_property('boolean')
    vhalo.a = np.arange(len(G)) == root
    graph_tool.draw.graph_draw(k, output='tBFS.pdf', output_size=(1100, 1100),
                               vprops={'size': 25, 'fill_color': [1,1,1,.5],
                                       'text': name, 'halo': vhalo, 'halo_color': du.good_edge},
                               eprops={'color': ecol, 'pen_width': esize})
