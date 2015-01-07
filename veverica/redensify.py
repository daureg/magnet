#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Complete a graph (again)."""

G = {}
N = None
N_CLOSEABLE = 0
TMP_SET = set()
CLOSEABLE_TRIANGLES = {}
EDGES_SIGN = {}
EDGES_ORIG = {}
EDGES_DEPTH = {}
NODE_DEPTH = {}
NODE_PIVOT_COUNT = []
ALPHA = 1
DEGREES = []
DEGREES_SEQUENCE = []
DEGREE_THRESHOLD_INDEX = 0
DEPTH_METHOD = 'constant'
DEPTH_COMPUTATION = {'constant': lambda a, b: 1,
                     'sum': lambda a, b: 1 + a + b,
                     'max': lambda a, b: 1 + max(a, b)}[DEPTH_METHOD]
VALS = []
import random as r
import gc
import os
from itertools import combinations
from enum import Enum, unique
import warnings


@unique
class PivotSelection(Enum):
    """How to choose the pivot at each iteration"""
    Uniform = 1
    Preferential = 2
    ByDegree = 3
    FromEdge = 4
    NoPivot = 5


PIVOT_SELECTION = PivotSelection.Uniform


def profile(func):
    """no op"""
    return func


def memodict(func):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        __slots__ = ()

        def __missing__(self, key):
            self[key] = ret = func(key)
            return ret
    return memodict().__getitem__


# http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python
@profile
def weighted_choice(objects, weights):
    """Return an element from `objects` with probability defined by the list
    of `weights`."""
    rnd = r.random() * sum(weights)
    for obj, weight in zip(objects, weights):
        rnd -= weight
        if rnd < 0:
            return obj


@memodict
@profile
def hash_triangle(points):
    """Give an unique id to each vertices triplet"""
    # TODO is node order significant?
    a, b, c = sorted(points)
    return N*(a*N+b)+c


@memodict
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
    return (EDGES_SIGN.get((v, w), None),
            EDGES_SIGN.get((u, w), None),
            EDGES_SIGN.get((u, v), None))


@profile
def compute_original_degree():
    """Fill global variables related to degree in the original graph"""
    global DEGREES, DEGREES_SEQUENCE, DEGREE_THRESHOLD_INDEX
    DEGREES = [len(G[n]) for n in range(len(G))]
    DEGREES_SEQUENCE = sorted(set(DEGREES))
    DEGREE_THRESHOLD_INDEX = 0


@profile
def get_potential_pivot(dictionary):
    """Return a list of (nodes, attached closeable triangle) of not too high
    degree"""
    global DEGREE_THRESHOLD_INDEX
    max_degree = DEGREES_SEQUENCE[DEGREE_THRESHOLD_INDEX]
    res = [(n, t) for n, t in dictionary.items()
           if DEGREES[n] <= max_degree]
    if res:
        return res
    assert DEGREE_THRESHOLD_INDEX+1 < len(DEGREES_SEQUENCE)
    DEGREE_THRESHOLD_INDEX += 1
    return get_potential_pivot(dictionary)


@profile
def sample_key(dictionary):
    """Return a key, value tuple at random from dictionary"""
    if PIVOT_SELECTION is PivotSelection.NoPivot:
        raise ValueError
    if PIVOT_SELECTION is PivotSelection.Uniform:
        return r.choice(list(dictionary.items()))
    if PIVOT_SELECTION is PivotSelection.ByDegree:
        return r.choice(get_potential_pivot(dictionary))
    objects = []
    weights = []
    for n, t in dictionary.items():
        objects.append((n, t))
        if PIVOT_SELECTION is PivotSelection.FromEdge:
            weight = NODE_DEPTH[n]
        if PIVOT_SELECTION is PivotSelection.Preferential:
            weight = 1+NODE_PIVOT_COUNT[n]**ALPHA
        weights.append(weight)
    return weighted_choice(objects, weights)


@profile
def sample_set(set_to_sample, one_at_a_time=False):
    """Return an element at random from a set or all of them"""
    if one_at_a_time:
        return [r.choice(list(set_to_sample))]
    return list(set_to_sample)


@profile
def add_signed_edge(src, dst, positive=False, depth_a=0, depth_b=0):
    """Add a edge between `src` and `dst`, potentially a positive one"""
    src, dst = min(src, dst), max(src, dst)
    # print('{} {} {}'.format(src, {True: '+', False: '-'}[positive], dst))
    EDGES_SIGN[(src, dst)] = positive
    depth = DEPTH_COMPUTATION(depth_a, depth_b)
    EDGES_DEPTH[(src, dst)] = depth
    G[src].add(dst)
    G[dst].add(src)
    NODE_DEPTH[src] += depth
    NODE_DEPTH[dst] += depth


@profile
def how_to_complete_triangle(hash_):
    """Return the endpoints and the boolean sign of the missing edge in the
    triangle `hash_`"""
    u, v, w = triangle_nodes(hash_)
    eu, ev, ew = triangle_edges(hash_)
    if eu is None:
        a, b, first, second = v, w, ev, ew
        da, db = EDGES_DEPTH[(u, w)], EDGES_DEPTH[(u, v)]
    if ev is None:
        a, b, first, second = u, w, eu, ew
        da, db = EDGES_DEPTH[(v, w)], EDGES_DEPTH[(u, v)]
    if ew is None:
        a, b, first, second = u, v, eu, ev
        da, db = EDGES_DEPTH[(v, w)], EDGES_DEPTH[(u, w)]
    return a, b, first and second, da, db


@profile
def find_initial_closeable():
    """Find the initial set of closeable triangles"""
    for middle, neighbors in G.items():
        for a, b in combinations(neighbors, 2):
            s1 = EDGES_SIGN[(a, middle) if a < middle else (middle, a)]
            s2 = EDGES_SIGN[(b, middle) if b < middle else (middle, b)]
            if s1 or s2:
                add_triangle(a, middle, b)


@profile
def complete_graph(one_at_a_time=True):
    no_pivot = bool(os.environ.get('NO_PIVOT', False))
    if PIVOT_SELECTION is PivotSelection.ByDegree and one_at_a_time:
        warnings.warn('ByDegree strategy do not work with one_at_a_time')
        one_at_a_time = False
    compute_original_degree()
    from math import log
    # r.seed(800)
    global N_CLOSEABLE, NODE_PIVOT_COUNT
    N_CLOSEABLE = 0
    NODE_PIVOT_COUNT = [0 for node in G.keys()]
    find_initial_closeable()
    threshold = int(N*N*log(N))
    nb_iter = 0
    while N_CLOSEABLE > 0 and nb_iter < threshold:
        # current = set((v for _ in CLOSEABLE_TRIANGLES.values() for v in _))
        # assert N_CLOSEABLE == len(current), '{}\n{}\n{}'.format(TMP_SET,
        #                                                         current,
        #                                                         TMP_SET.symmetric_difference(current))
        if no_pivot:
            closeables = [t for s in CLOSEABLE_TRIANGLES.values() for t in s]
        else:
            pivot, closeables = sample_key(CLOSEABLE_TRIANGLES)
            NODE_PIVOT_COUNT[pivot] += 1
        # if (nb_iter % 1) == 0:
        #     VALS.append([NODE_DEPTH[i] for i in range(N)])
        # if pivot < 2 and nb_iter < threshold//N:
        #     nb_iter += 1
        #     continue
        # print('pivot {}'.format(pivot))
        triangles = sample_set(closeables, one_at_a_time)
        closed = [close_triangle(triangle) for triangle in triangles]
        for a, b, sign in closed:
            if a is not None:
                update_triangle_status(a, b, sign)
        # print(triangle_nodes(triangle))
        # assert triangle
        nb_iter += 1
        if ((nb_iter + 1) % 5000) == 0:
            gc.collect()
    print(nb_iter, threshold, len(CLOSEABLE_TRIANGLES), N_CLOSEABLE)
    random_completion(-1)


@profile
def close_triangle(triangle):
    """Close triangle and return the added edge and its sign"""
    try:
        a, b, sign, da, db = how_to_complete_triangle(triangle)
    except UnboundLocalError:
        delete_triangle(*triangle_nodes(triangle))
        return None, None, None
    add_signed_edge(a, b, sign, da, db)
    return a, b, sign


@profile
def delete_triangle(a, p, b):
    """Remove (a, p, b) from all the lists of closeable triangles"""
    global N_CLOSEABLE
    h = hash_triangle((a, p, b))
    if p in CLOSEABLE_TRIANGLES:
        if len(CLOSEABLE_TRIANGLES[p]) == 1 and \
           h in CLOSEABLE_TRIANGLES[p]:
            del CLOSEABLE_TRIANGLES[p]
            N_CLOSEABLE -= 1
            # print('delete {}'.format((a, p, b)))
            # TMP_SET.remove(h)
        elif h in CLOSEABLE_TRIANGLES[p]:
            CLOSEABLE_TRIANGLES[p].remove(h)
            N_CLOSEABLE -= 1
            # print('delete {}'.format((a, p, b)))
            # TMP_SET.remove(h)
        else:
            # assert not triangle_is_closeable(h)
            pass


@profile
def add_triangle(a, p, b):
    """Add (a, p, b) from all the lists of closeable triangles"""
    global N_CLOSEABLE
    h = hash_triangle((a, p, b))
    ta, tb = (a, b) if a < b else (b, a)
    if (ta, tb) in EDGES_SIGN:
        return
    # assert triangle_is_closeable(h), ((a,p,b), triangle_edges(h))
    # if not triangle_is_closeable(h):
    #     return
    # print('add {}'.format((a, p, b)))
    # TMP_SET.add(h)
    N_CLOSEABLE += 1
    if p in CLOSEABLE_TRIANGLES:
        CLOSEABLE_TRIANGLES[p].add(h)
    else:
        CLOSEABLE_TRIANGLES[p] = set([h])


@profile
def update_triangle_status(a, b, sign):
    Na = G[a].difference([b])
    Nb = G[b].difference([a])
    common = Na.intersection(Nb)
    Na.difference_update(Nb)
    Nb.difference_update(Na)
    # print(common, a, Na, G[a], b, Nb, G[b])
    for v in common:
        delete_triangle(a, v, b)
    for v in Na:
        second_sign = EDGES_SIGN[(v, a) if v < a else (a, v)]
        if sign or second_sign:
            add_triangle(v, a, b)
        # else:
        #     assert not triangle_is_closeable(hash_triangle((a,b,v)))
    for v in Nb:
        second_sign = EDGES_SIGN[(v, b) if v < b else (b, v)]
        if sign or second_sign:
            add_triangle(v, b, a)
        # else:
        #     assert not triangle_is_closeable(hash_triangle((a,b,v)))


@profile
def random_completion(graph, positive_proba=0.5):
    """Set `graph` absent edges positive with `positive_proba`ility."""
    how_many_closed = 0
    for i, j in combinations(range(N), 2):
        if (i, j) not in EDGES_SIGN:
            how_many_closed += 1
            add_signed_edge(i, j, r.random() < positive_proba)
    return how_many_closed
