#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Take a general signed graph and complete it randomly (to some extent)."""
import random as r
import bisect
from itertools import combinations, accumulate
import numpy as np
from graph_tool import centrality
from TriangleCache import TriangleStatus
from enum import Enum, unique
import persistent as p


@unique
class PivotStrategy(Enum):
    """How to choose the pivot at each iteration"""
    uniform = 1
    by_degree = 2
    by_betweenness = 3
    weighted = 4
    no_pivot = 5


CLOSEABLE_TRIANGLES = None
TWO_PATHS = None
N = -1
GRAPH = None
EDGES_SIGN = {}
# EDGES_DEPTH = {}
DATA = p.load_var('triangle_cache.my')
TRIANGLE_STATUS_CACHE = lambda x: DATA[x.value]


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
    # TODO is node order significant?
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


@profile
def triangle_is_closeable(hash_):
    """A triangle is closeable if one edge is missing and at least another
    one is positive"""
    edges = triangle_edges(hash_)
    return TRIANGLE_STATUS_CACHE(TriangleStatus.closeable)[edges]


@profile
def triangle_is_closed(hash_):
    """Tell if a triangle has 3 edges"""
    edges = triangle_edges(hash_)
    return TRIANGLE_STATUS_CACHE(TriangleStatus.closed)[edges]


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
    # du, dv, dw = (EDGES_DEPTH.get((v, w), None),
    #               EDGES_DEPTH.get((u, w), None),
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
def update_triangle_status(graph, a, b):
    """Look for all closeable triangles involving edge (`a`,`b`)"""
    Na = set(map(int, graph.vertex(a).out_neighbours()))
    Nb = set(map(int, graph.vertex(b).out_neighbours()))
    common_neighbors = Na.union(Nb).difference((a, b))
    for v in common_neighbors:
        h = hash_triangle(a, b, v)
        if triangle_is_closed(h):
            CLOSEABLE_TRIANGLES.discard(h)
            TWO_PATHS.discard(h)
        if triangle_is_closeable(h):
            CLOSEABLE_TRIANGLES.add(h)
    for v in Na.union(Nb).difference(common_neighbors):
        # v is a exclusive neighbor of either a or b
        h = hash_triangle(a, b, v)
        TWO_PATHS.add(h)


def non_shared_vertices(N, shared_edges):
    """Return the vertices not part of `shared_edges` in a `N` nodes graph"""
    src, dst = zip(*shared_edges)
    shared_vertices = set(list(src)+list(dst))
    return list(set(range(N)).difference(shared_vertices))


def is_a_two_path(hash_):
    edges = triangle_edges(hash_)
    return TRIANGLE_STATUS_CACHE(TriangleStatus.one_edge_missing)[edges]


@profile
def complete_graph(graph, shared_edges=None, close_all=True,
                   pivot_strategy=PivotStrategy.uniform,
                   pivot_gen=None,
                   triangle_strategy=TriangleStatus.closeable):
    """Close every possible triangles and then add negative edges"""
    global CLOSEABLE_TRIANGLES, TWO_PATHS
    N = graph.num_vertices()
    CLOSEABLE_TRIANGLES = set()
    TWO_PATHS = set()
    for i, j, k in combinations(range(N), 3):
        h = hash_triangle(i, j, k)
        if triangle_is_closeable(h):
            CLOSEABLE_TRIANGLES.add(h)
        if is_a_two_path(h):
            TWO_PATHS.add(h)
    nb_iter = 0
    vertices_gen = build_pivot_generator(N, graph, shared_edges,
                                         pivot_strategy, pivot_gen)
    threshold = int(N*np.log(N))
    while CLOSEABLE_TRIANGLES and nb_iter < N*threshold:
        if pivot_strategy is PivotStrategy.no_pivot:
            pivot = None
        else:
            pivot = choose_pivot(N, nb_iter, vertices_gen)
        triangle = pick_triangle(graph, pivot, triangle_strategy)
        complete_triangle(graph, triangle)
        nb_iter += 1
    print(nb_iter, len(CLOSEABLE_TRIANGLES))
    # print('completed {}, {}'.format(hash(graph),
    #                                 int(''.join(map(lambda x: str(int(x)),
    #                                             EDGES_SIGN.values())), 2)))
    if close_all:
        random_completion(graph, -1)
    # transfer_depth(graph)


def pick_triangle(graph, pivot, triangle_strategy):
    """Choose randomly the first triangle in `graph` that match
    `triangle_strategy` involving `pivot`"""
    if pivot is None:
        candidates = TWO_PATHS
    else:
        candidates = ego_triangle(graph.vertex(pivot))
    for idx in randperm(len(candidates)):
        edges = triangle_edges(candidates[idx])
        if TRIANGLE_STATUS_CACHE(triangle_strategy)[edges]:
            return candidates[idx]


def complete_triangle(graph, triangle):
    """Close `triangle` in `graph` and make necessary updates."""
    if triangle in CLOSEABLE_TRIANGLES:
        a, b, sign, depth = how_to_complete_triangle(triangle)
        add_signed_edge(graph, a, b, depth, sign)
        CLOSEABLE_TRIANGLES.remove(triangle)
        update_triangle_status(graph, a, b)


def randperm(seq_len):
    """Yield indices in [0, `seq_len`] at random without replacement"""
    indices = list(range(seq_len))
    r.shuffle(indices)
    for idx in indices:
        yield idx


def build_pivot_generator(N, graph, shared_edges, pivot_strategy, pivot_gen):
    """Return a vertex generator according to the chosen strategy"""
    if shared_edges:
        non_shared = non_shared_vertices(N, shared_edges)
        return iter(lambda: r.choice(non_shared), -1)
    if pivot_strategy is PivotStrategy.weighted:
        assert pivot_gen, "provide your own generator"
        return pivot_gen
    if pivot_strategy in [PivotStrategy.uniform, PivotStrategy.no_pivot]:
        return None
    if pivot_strategy is PivotStrategy.by_degree:
        degs = graph.degree_property_map('total').a
        weights = np.exp(-degs)/np.sum(np.exp(-degs))
        return WeightedRandomGenerator(weights)
    if pivot_strategy is PivotStrategy.by_betweenness:
        vb, _ = centrality.betweenness(graph)
        vb = N*vb.a/5
        weights = np.exp(-vb)/np.sum(np.exp(-vb))
        return WeightedRandomGenerator(weights)
    raise ValueError('check build_pivot_generator call')


@profile
def transfer_depth(graph):
    """Copy EDGES_DEPTH back to graph property"""
    for e in graph.edges():
        src, dst = int(e.source()), int(e.target())
        src, dst = min(src, dst), max(src, dst)
        # graph.ep['depth'][e] = EDGES_DEPTH[(src, dst)]


def random_completion(graph, positive_proba=0.5):
    """Set `graph` absent edges positive with `positive_proba`ility."""
    # max_depth = int(graph.ep['depth'].a.max())
    # larger_depth = int(1.4*max_depth)
    larger_depth = 2
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
    # EDGES_DEPTH.clear()
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
        # EDGES_DEPTH[(src, dst)] = 1
    pos = cc.gtdraw.sfdp_layout(GRAPH, cooling_step=0.95, epsilon=5e-2)

    complete_graph(GRAPH)
    cc.cc_pivot(GRAPH)
    print(GRAPH.vp['cluster'].a)
    cc.draw_clustering(GRAPH, filename="completed.pdf", pos=pos,
                       vmore={'text': name})
