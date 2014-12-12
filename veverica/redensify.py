#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Complete a graph (again)."""

G = {}
N = None
N_CLOSEABLE = 0
TMP_SET = set()
CLOSEABLE_TRIANGLES = {}
EDGES_SIGN = {}
EDGES_ORIG = None
import random as r
from itertools import combinations
import persistent as p
from TriangleCache import TriangleStatus
DATA = p.load_var('triangle_cache.my')
TRIANGLE_IS_CLOSED_ = DATA[TriangleStatus.closed.value]
TRIANGLE_IS_CLOSEABLE_ = DATA[TriangleStatus.closeable.value]


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
def triangle_is_closeable(hash_):
    """A triangle is closeable if one edge is missing and at least another
    one is positive"""
    return TRIANGLE_IS_CLOSEABLE_[triangle_edges(hash_)]


@profile
def triangle_is_closed(hash_):
    """Tell if a triangle has 3 edges"""
    return TRIANGLE_IS_CLOSED_[triangle_edges(hash_)]


@profile
def sample_key(dictionary):
    """Return a key, value tuple at random from dictionary"""
    n = len(dictionary)
    if n == 0:
        return None, None
    idx = r.randint(0, n-1)
    for i, (k, v) in enumerate(dictionary.items()):
        if i == idx:
            return k, v
    raise ValueError("Can't reach that point")


@profile
def sample_set(sett):
    """Return an element at random from a set"""
    n = len(sett)
    if n == 0:
        return None
    idx = r.randint(0, n-1)
    for i, el in enumerate(sett):
        if i == idx:
            return el
    raise ValueError("Can't reach that point")


@profile
def add_signed_edge(src, dst, positive=False):
    """Add a edge between `src` and `dst`, potentially a positive one"""
    src, dst = min(src, dst), max(src, dst)
    # print('{} {} {}'.format(src, {True: '+', False: '-'}[positive], dst))
    EDGES_SIGN[(src, dst)] = positive
    G[src].add(dst)
    G[dst].add(src)


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
def find_initial_closeable():
    """Find the initial set of closeable triangles"""
    for middle, neighbors in G.items():
        for a, b in combinations(neighbors, 2):
            s1, s2 = EDGES_SIGN[(a, middle) if a < middle else (middle, a)], EDGES_SIGN[(b, middle) if b < middle else (middle, b)],
            if s1 or s2:
                add_triangle(a, middle, b)


@profile
def complete_graph():
    from math import log
    # r.seed(800)
    global N_CLOSEABLE
    N_CLOSEABLE = 0
    find_initial_closeable()
    threshold = int(N*N*log(N))
    nb_iter = 0
    while N_CLOSEABLE > 0 and nb_iter < threshold:
        # current = set((v for _ in CLOSEABLE_TRIANGLES.values() for v in _))
        # assert N_CLOSEABLE == len(current), '{}\n{}\n{}'.format(TMP_SET,
        #                                                         current,
        #                                                         TMP_SET.symmetric_difference(current))
        pivot, closeables = sample_key(CLOSEABLE_TRIANGLES)
        # print('pivot {}'.format(pivot))
        triangle = sample_set(closeables)
        # print(triangle_nodes(triangle))
        # assert triangle
        a, b, sign = how_to_complete_triangle(triangle)
        add_signed_edge(a, b, sign)
        update_triangle_status(a, b, sign)
        nb_iter += 1
    print(nb_iter, N*threshold, CLOSEABLE_TRIANGLES, N_CLOSEABLE)
    print(random_completion(-1))


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
