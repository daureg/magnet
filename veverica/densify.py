#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Take a general signed graph and complete it randomly (to some extent)."""
import random as r
import bisect
from itertools import combinations, accumulate
import numpy as np

CLOSEABLE_TRIANGLES = None
N = -1
GRAPH = None
EDGES_SIGN = {}
# EDGES_DEPTH = {}
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


class WeightedRandomGenerator(object):
    """Draw integer between 0 and len(weights)-1 according to
    `Weights` probability."""
    def __init__(self, weights):
        self.totals = list(accumulate(weights))

    def __next__(self):
        rnd = r.random() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)


@profile
def hash_triangle(a, b, c):
    """Give an unique id to each vertices triplet"""
    a, b, c = sorted([a, b, c])
    return int(N*(a*N+b)+c)


@profile
def choose_pivot(N, nb_iter, pivots_gen=None, uniform_ending=True):
    """Choose a index from `pivots_gen` or uniformly from [0, N-1]"""
    threshold = 1.5*int(N*np.log(N))
    if not pivots_gen or (nb_iter > threshold and uniform_ending):
        return r.randint(0, N-1)
    return next(pivots_gen)


@profile
def triangle_nodes(hash_):
    """Return the 3 vertices index making a triangle id"""
    c = hash_ % N
    hash_ = (hash_ - c) // N
    b = hash_ % N
    a = (hash_ - b) // N
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
    return [hash_triangle(int(nodes[0]), int(nodes[1]), int(nodes[2]))
            for nodes in neighbors]


@profile
def how_to_complete_triangle(hash_):
    """Return the endpoints and the boolean sign of the missing edge in the
    triangle `hash_`"""
    u, v, w = triangle_nodes(hash_)
    eu, ev, ew = triangle_edges(hash_)
    # du, dv, dw = (EDGES_DEPTH.get((v, w), None), EDGES_DEPTH.get((u, w), None),
    #               EDGES_DEPTH.get((u, v), None))
    du, dv, dw = 0, 0, 0
    if eu is None:
        a, b, first, second, depth = v, w, ev, ew, dv+dw
    if ev is None:
        a, b, first, second, depth = u, w, eu, ew, du+dw
    if ew is None:
        a, b, first, second, depth = u, v, eu, ev, du+dv
    return a, b, first and second, depth


@profile
def add_signed_edge(graph, src, dst, depth, positive=False):
    """Add a edge between `src` and `dst`, potentially a positive one"""
    e = graph.add_edge(src, dst)
    graph.ep['fake'][e] = True
    graph.ep['sign'][e] = positive
    src, dst = min(src, dst), max(src, dst)
    EDGES_SIGN[(src, dst)] = positive
    # EDGES_DEPTH[(src, dst)] = depth


@profile
def complete_pivot(graph, v):
    """Close all triangles related to `v`"""
    candidates = ego_triangle(graph.vertex(v))
    r.shuffle(candidates)
    removed = []
    for triangle in candidates:
        if triangle in CLOSEABLE_TRIANGLES:
            a, b, sign, depth = how_to_complete_triangle(triangle)
            add_signed_edge(graph, a, b, depth, sign)
            removed.append((a, b, triangle))
            break
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
        # FIXME: those triangle should be identified before this point
        if triangle_is_closed(h):
            CLOSEABLE_TRIANGLES.discard(h)
        if triangle_is_closeable(h):
            CLOSEABLE_TRIANGLES.add(h)


def non_shared_vertices(N, shared_edges):
    """Return the vertices not part of `shared_edges` in a `N` nodes graph"""
    src, dst = zip(*shared_edges)
    shared_vertices = set(list(src)+list(dst))
    return list(set(range(N)).difference(shared_vertices))


@profile
def complete_graph(graph, shared_edges=None, close_all=True, by_degree=False):
    """Close every possible triangles and then add negative edges"""
    global CLOSEABLE_TRIANGLES
    N = graph.num_vertices()
    CLOSEABLE_TRIANGLES = set()
    for i, j, k in combinations(range(N), 3):
        h = hash_triangle(i, j, k)
        if triangle_is_closeable(h):
            CLOSEABLE_TRIANGLES.add(h)
    nb_iter = 0
    if shared_edges:
        assert not by_degree, ("can't cheat and use by_degree"
                               "at the same time")
        non_shared = non_shared_vertices(N, shared_edges)
        vertices_gen = iter(lambda: r.choice(non_shared), -1)
    else:
        if by_degree:
            degs = graph.degree_property_map('total').a
            weights = np.exp(-degs)/np.sum(np.exp(-degs))
            vertices_gen = WeightedRandomGenerator(weights)
        else:
            vertices_gen = None
    threshold = int(N*np.log(N))
    while CLOSEABLE_TRIANGLES and nb_iter < N*threshold:
        pivot_index = choose_pivot(N, nb_iter, vertices_gen)
        complete_pivot(graph, pivot_index)
        nb_iter += 1
    print(nb_iter, len(CLOSEABLE_TRIANGLES))
    if close_all:
        random_completion(graph, -1)
    # transfer_depth(graph)


@profile
def transfer_depth(graph):
    """Copy EDGES_DEPTH back to graph property"""
    for e in graph.edges():
        src, dst = int(e.source()), int(e.target())
        src, dst = min(src, dst), max(src, dst)
        EDGES_SIGN[(src, dst)] = bool()
        graph.ep['depth'][e] = EDGES_DEPTH[(src, dst)]


def random_completion(graph, positive_proba=0.5):
    """Set `graph` absent edges positive with `positive_proba`ility."""
    max_depth = int(graph.ep['depth'].a.max())
    larger_depth = int(1.4*max_depth)
    for i, j in combinations(range(N), 2):
        if (i, j) not in EDGES_SIGN:
            add_signed_edge(graph, i, j, larger_depth,
                            r.random() < positive_proba)

if __name__ == '__main__':
    # pylint: disable=C0103
    N = 15
    import cc_pivot as cc
    GRAPH = cc.make_signed_graph(cc.gtgeneration.circular_graph(N))
    EDGES_SIGN.clear()
    EDGES_DEPTH.clear()
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
        EDGES_DEPTH[(src, dst)] = 1
    pos = cc.gtdraw.sfdp_layout(GRAPH, cooling_step=0.95, epsilon=5e-2)

    complete_graph(GRAPH)
    cc.cc_pivot(GRAPH)
    print(GRAPH.vp['cluster'].a)
    cc.draw_clustering(GRAPH, filename="completed.pdf", pos=pos,
                       vmore={'text': name})
