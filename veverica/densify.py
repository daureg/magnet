#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Take a general signed graph and complete it randomly (to some extent)."""
import cc_pivot as cc
import random as r
from itertools import combinations

CLOSEABLE_TRIANGLES = set()
N = -1
GRAPH = None


def hash_triangle(a, b, c):
    """Give an unique id to each vertices triplet"""
    a, b, c = sorted([a, b, c])
    return N*(a*N+b)+c


def triangle_nodes(hash_):
    """Return the 3 vertices index making a triangle id"""
    c = hash_ % N
    hash_ = (hash_ - c) / N
    b = hash_ % N
    a = (hash_ - b) / N
    return a, b, c


def triangle_edges(hash_):
    """Return the edges of a triangle"""
    u, v, w = triangle_nodes(hash_)
    return GRAPH.edge(v, w), GRAPH.edge(u, w), GRAPH.edge(u, v)


def triangle_is_closeable(hash_):
    """A triangle is closeable if one edge is missing and at least another
    one is positive"""
    # score each edge as : {+: 1, -: -1, absent: -10}
    edge_score = lambda e: -10 if e is None else (GRAPH.ep['sign'][e]*2 - 1)
    return sum((edge_score(e) for e in triangle_edges(hash_))) in [-10, -8]


def triangle_is_closed(hash_):
    """Tell if a triangle has 3 edges"""
    edge_score = lambda e: -10 if e is None else (GRAPH.ep['sign'][e]*2 - 1)
    return sum((edge_score(e) for e in triangle_edges(hash_))) >= -3


def ego_triangle(v):
    """Return all triangles (as int tuples) involving `v`"""
    neighbors = (_+(v,) for _ in combinations(v.out_neighbours(), 2))
    return [hash_triangle(*map(int, nodes)) for nodes in neighbors]


def how_to_complete_triangle(hash_):
    """Return the endpoints and the boolean sign of the missing edge in the
    triangle `hash_`"""
    u, v, w = triangle_nodes(hash_)
    eu, ev, ew = GRAPH.edge(v, w), GRAPH.edge(u, w), GRAPH.edge(u, v)
    if not eu:
        a, b, first, second = v, w, ev, ew
    if not ev:
        a, b, first, second = u, w, eu, ew
    if not ew:
        a, b, first, second = u, v, eu, ev
    s1, s2 = GRAPH.ep['sign'][first], GRAPH.ep['sign'][second]
    return a, b, s1 and s2


def add_signed_edge(src, dst, positive=False):
    """Add a edge between `src` and `dst`, potentially a positive one"""
    e = GRAPH.add_edge(src, dst)
    GRAPH.ep['fake'][e] = True
    GRAPH.ep['sign'][e] = positive


def complete_pivot(v):
    """Close all triangles related to `v`"""
    candidates = ego_triangle(GRAPH.vertex(v))
    removed = []
    for triangle in candidates:
        if triangle in CLOSEABLE_TRIANGLES:
            a, b, sign = how_to_complete_triangle(triangle)
            add_signed_edge(a, b, sign)
            removed.append((a, b, triangle))
    for done in removed:
        CLOSEABLE_TRIANGLES.remove(done[2])
        update_triangle_status(done[0], done[1])


def update_triangle_status(a, b):
    """Look for all closeable triangles involving edge (`a`,`b`)"""
    Na = set(map(int, GRAPH.vertex(a).out_neighbours()))
    Nb = set(map(int, GRAPH.vertex(b).out_neighbours()))
    common_neighbors = Na.union(Nb).difference((a, b))
    for v in common_neighbors:
        h = hash_triangle(a, b, v)
        if triangle_is_closeable(h):
            CLOSEABLE_TRIANGLES.add(h)
        # FIXME: those triangle should be identified before this point
        if triangle_is_closed(h):
            CLOSEABLE_TRIANGLES.discard(h)


def complete_graph():
    """Close every possible triangles and then add negative edges"""
    N = GRAPH.num_vertices()
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                h = hash_triangle(i, j, k)
                if triangle_is_closeable(h):
                    CLOSEABLE_TRIANGLES.add(h)
    nb_iter = 0
    while CLOSEABLE_TRIANGLES and nb_iter < N*N*N/6:
        # TODO: choose pivot without replacement
        complete_pivot(r.randint(0, N-1))
        nb_iter += 1
    for i in range(N):
        for j in range(i+1, N):
            if not GRAPH.edge(i, j):
                add_signed_edge(i, j)

if __name__ == '__main__':
    # pylint: disable=C0103
    N = 15
    GRAPH = cc.make_signed_graph(cc.gtgeneration.circular_graph(N))
    name = GRAPH.new_vertex_property('string')
    fake = GRAPH.new_edge_property('bool')
    GRAPH.ep['fake'] = fake
    for i, v in enumerate(GRAPH.vertices()):
        name[v] = str(i)
    for i, e in enumerate(GRAPH.edges()):
        GRAPH.ep['sign'][e] = i != 0
    pos = cc.gtdraw.sfdp_layout(GRAPH, cooling_step=0.95, epsilon=5e-2)
    complete_graph()
    cc.cc_pivot(GRAPH)
    print(GRAPH.vp['cluster'].a)
    cc.draw_clustering(GRAPH, filename="completed.pdf", pos=pos,
                       vmore={'text': name})
