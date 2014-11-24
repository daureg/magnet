#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Take a general signed graph and complete it randomly (to some extent)."""
import random as r
from itertools import combinations
import matplotlib.pyplot as plt
import prettyplotlib as ppl

CLOSEABLE_TRIANGLES = None
N = -1
GRAPH = None
EDGES_SIGN = {}
ADJ_NAME = None
ADJ_NUMBER = 0
SIGN = None
import graph_tool.spectral as spectral
import numpy as np
triangle_is_closeable_ = {
    (None, None, None): False, (None, None, False): False,
    (None, None, True): False, (None, False, None): False,
    (None, False, False): False, (None, False, True): True,
    (None, True, None): False, (None, True, False): True,
    (None, True, True): True, (False, None, None): False,
    (False, None, False): False, (False, None, True): True,
    (False, False, None): False, (False, False, False): False,
    (False, False, True): False, (False, True, None): True,
    (False, True, False): False, (False, True, True): False,
    (True, None, None): False, (True, None, False): True,
    (True, None, True): True, (True, False, None): True,
    (True, False, False): False, (True, False, True): False,
    (True, True, None): True, (True, True, False): False,
    (True, True, True): False}
triangle_is_closed_ = {
    (None, None, None): False, (None, None, False): False,
    (None, None, True): False, (None, False, None): False,
    (None, False, False): False, (None, False, True): False,
    (None, True, None): False, (None, True, False): False,
    (None, True, True): False, (False, None, None): False,
    (False, None, False): False, (False, None, True): False,
    (False, False, None): False, (False, False, False): True,
    (False, False, True): True, (False, True, None): False,
    (False, True, False): True, (False, True, True): True,
    (True, None, None): False, (True, None, False): False,
    (True, None, True): False, (True, False, None): False,
    (True, False, False): True, (True, False, True): True,
    (True, True, None): False, (True, True, False): True,
    (True, True, True): True}


def profile(f):
    return f


@profile
def hash_triangle(a, b, c):
    """Give an unique id to each vertices triplet"""
    a, b, c = sorted([a, b, c])
    return N*(a*N+b)+c


@profile
def triangle_nodes(hash_):
    """Return the 3 vertices index making a triangle id"""
    c = hash_ % N
    hash_ = (hash_ - c) / N
    b = hash_ % N
    a = (hash_ - b) / N
    return a, b, c


@profile
def triangle_edges(hash_):
    """Return the edges of a triangle"""
    u, v, w = triangle_nodes(hash_)
    return (EDGES_SIGN.get((v, w), None), EDGES_SIGN.get((u, w), None),
            EDGES_SIGN.get((u, v), None))


def triangle_score(hash_):
    """Return a number characterizing the triangle's edge type"""
    # score each edge as : {+: 1, -: -1, absent: -10}
    edge_score = lambda e: -10 if e is None else (e*2 - 1)
    return sum((edge_score(e) for e in triangle_edges(hash_)))


@profile
def triangle_is_closeable(hash_):
    """A triangle is closeable if one edge is missing and at least another
    one is positive"""
    return triangle_is_closeable_[triangle_edges(hash_)]


@profile
def triangle_is_closed(hash_):
    """Tell if a triangle has 3 edges"""
    return triangle_is_closed_[triangle_edges(hash_)]


@profile
def ego_triangle(v):
    """Return all triangles (as int tuples) involving `v`"""
    neighbors = (_+(v,) for _ in combinations(v.out_neighbours(), 2))
    return [hash_triangle(*list(map(int, nodes))) for nodes in neighbors]


@profile
def how_to_complete_triangle(hash_):
    """Return the endpoints and the boolean sign of the missing edge in the
    triangle `hash_`"""
    u, v, w = triangle_nodes(hash_)
    eu, ev, ew = triangle_edges(hash_)
    if eu is None:
        a, b, first, second = v, w, ev, ew
    if ev is None:
        a, b, first, second = u, w, eu, ew
    if ew is None:
        a, b, first, second = u, v, eu, ev
    return a, b, first and second


@profile
def add_signed_edge(graph, src, dst, positive=False):
    """Add a edge between `src` and `dst`, potentially a positive one"""
    e = graph.add_edge(src, dst)
    graph.ep['fake'][e] = True
    graph.ep['sign'][e] = positive
    src, dst = min(src, dst), max(src, dst)
    EDGES_SIGN[(src, dst)] = positive
    SIGN.a = graph.ep['sign'].a.astype(np.int8)*2-1
    A = np.array(spectral.adjacency(graph, SIGN).todense())
    global ADJ_NUMBER
    plot_adj(A, ADJ_NUMBER)
    ADJ_NUMBER += 1


def plot_adj(A, seq=0):
    nodes = list(map(str, range(1, A.shape[0]+1)))
    f = ppl.pcolormesh((np.flipud(A)), xticklabels=nodes,
                       yticklabels=list(reversed(nodes)))
    f.axes[0].set_aspect('equal')
    f.set_figheight(5)
    f.set_figwidth(5)
    f.tight_layout()
    plt.savefig('a_{}_{:05d}.png'.format(abs(ADJ_NAME), seq))
    f.clear()
    plt.close()


@profile
def complete_pivot(graph, v):
    """Close all triangles related to `v`"""
    candidates = ego_triangle(graph.vertex(v))
    removed = []
    for triangle in candidates:
        if triangle in CLOSEABLE_TRIANGLES:
            a, b, sign = how_to_complete_triangle(triangle)
            add_signed_edge(graph, a, b, sign)
            removed.append((a, b, triangle))
    for done in removed:
        CLOSEABLE_TRIANGLES.remove(done[2])
        update_triangle_status(graph, done[0], done[1])


@profile
def update_triangle_status(graph, a, b):
    """Look for all closeable triangles involving edge (`a`,`b`)"""
    Na = set(map(int, graph.vertex(a).out_neighbours()))
    Nb = set(map(int, graph.vertex(b).out_neighbours()))
    common_neighbors = Na.union(Nb).difference((a, b))
    for v in common_neighbors:
        h = hash_triangle(a, b, v)
        if triangle_is_closeable(h):
            CLOSEABLE_TRIANGLES.add(h)
        # FIXME: those triangle should be identified before this point
        if triangle_is_closed(h):
            CLOSEABLE_TRIANGLES.discard(h)


@profile
def complete_graph(graph):
    """Close every possible triangles and then add negative edges"""
    global CLOSEABLE_TRIANGLES
    global ADJ_NAME
    global SIGN
    SIGN = graph.new_edge_property('int')
    ADJ_NAME = hash(graph)
    N = graph.num_vertices()
    CLOSEABLE_TRIANGLES = set()
    for i, j, k in combinations(range(N), 3):
        h = hash_triangle(i, j, k)
        if triangle_is_closeable(h):
            CLOSEABLE_TRIANGLES.add(h)
    nb_iter = 0
    closed = len(EDGES_SIGN)
    adj = []
    while CLOSEABLE_TRIANGLES and nb_iter < N*N*N/6:
        # TODO: choose pivot without replacement
        complete_pivot(graph, r.randint(0, N-1))
        if len(EDGES_SIGN) != closed:
            closed = len(EDGES_SIGN)
            # sign.a = graph.ep['sign'].a.astype(np.int8)*2-1
            # adj.append(np.array(spectral.adjacency(graph, sign).todense()))
        nb_iter += 1
    random_completion(graph, -1)
    return adj


def random_completion(graph, positive_proba=0.5):
    """Set `graph` absent edges positive with `positive_proba`ility."""
    for i, j in combinations(range(N), 2):
        if (i, j) not in EDGES_SIGN:
            add_signed_edge(graph, i, j, r.random() < positive_proba)

if __name__ == '__main__':
    # pylint: disable=C0103
    N = 15
    import cc_pivot as cc
    GRAPH = cc.make_signed_graph(cc.gtgeneration.circular_graph(N))
    EDGES_SIGN.clear()
    name = GRAPH.new_vertex_property('string')
    fake = GRAPH.new_edge_property('bool')
    GRAPH.ep['fake'] = fake
    for i, v in enumerate(GRAPH.vertices()):
        name[v] = str(i)
    for i, e in enumerate(GRAPH.edges()):
        GRAPH.ep['sign'][e] = i != 0
        src, dst = int(e.source()), int(e.target())
        src, dst = min(src, dst), max(src, dst)
        EDGES_SIGN[(src, dst)] = bool(GRAPH.ep['sign'][e])
    pos = cc.gtdraw.sfdp_layout(GRAPH, cooling_step=0.95, epsilon=5e-2)

    complete_graph(GRAPH)
    cc.cc_pivot(GRAPH)
    print(GRAPH.vp['cluster'].a)
    cc.draw_clustering(GRAPH, filename="completed.pdf", pos=pos,
                       vmore={'text': name})
