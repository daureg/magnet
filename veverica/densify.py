#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Take a general signed graph and complete it randomly (to some extent)."""
import random as r
from itertools import combinations
# import matplotlib.pyplot as plt
# import prettyplotlib as ppl

CLOSEABLE_TRIANGLES = None
N = -1
GRAPH = None
EDGES_SIGN = {}
EDGES_DEPTH = {}
# ADJ_NAME = None
# ADJ_NUMBER = 0
# SIGN = None
# import graph_tool.spectral as spectral
# import numpy as np
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
    return int(N*(a*N+b)+c)


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
    du, dv, dw = (EDGES_DEPTH.get((v, w), None), EDGES_DEPTH.get((u, w), None),
                  EDGES_DEPTH.get((u, v), None))
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
    # graph.ep['depth'][e] = depth
    src, dst = min(src, dst), max(src, dst)
    EDGES_SIGN[(src, dst)] = positive
    EDGES_DEPTH[(src, dst)] = depth
    # SIGN.a = graph.ep['sign'].a.astype(np.int8)*2-1
    # A = np.array(spectral.adjacency(graph, SIGN).todense())
    # global ADJ_NUMBER
    # plot_adj(A, ADJ_NUMBER)
    # ADJ_NUMBER += 1


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
            a, b, sign, depth = how_to_complete_triangle(triangle)
            add_signed_edge(graph, a, b, depth, sign)
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
        # FIXME: those triangle should be identified before this point
        if triangle_is_closed(h):
            CLOSEABLE_TRIANGLES.discard(h)
        # if a < 2 or b < 2 or v < 2:
        #     continue
        if triangle_is_closeable(h):
            CLOSEABLE_TRIANGLES.add(h)


@profile
def complete_graph(graph, cheating=False, close_all=True):
    """Close every possible triangles and then add negative edges"""
    global CLOSEABLE_TRIANGLES
    # global ADJ_NAME
    # global SIGN
    # SIGN = graph.new_edge_property('int')
    # ADJ_NAME = hash(graph)
    N = graph.num_vertices()
    CLOSEABLE_TRIANGLES = set()
    for i, j, k in combinations(range(N), 3):
        # if i < 2 or j < 2 or k < 2:
        #     continue
        h = hash_triangle(i, j, k)
        if triangle_is_closeable(h):
            CLOSEABLE_TRIANGLES.add(h)
    nb_iter = 0
    # closed = len(EDGES_SIGN)
    adj = []
    threshold = 300 if cheating else 0
    non_shared = [_ for _ in range(N) if _ % 32 >= 2]
    while CLOSEABLE_TRIANGLES and (nb_iter < (800 if cheating else N*N*N/6)):
        # print(len(CLOSEABLE_TRIANGLES))
        if cheating and nb_iter < threshold:
            pivot_index = r.choice(non_shared)
        else:
            pivot_index = r.randint((0 if nb_iter >= threshold else 2), N-1)
        complete_pivot(graph, pivot_index)
        # if len(EDGES_SIGN) != closed:
        #     closed = len(EDGES_SIGN)
            # sign.a = graph.ep['sign'].a.astype(np.int8)*2-1
            # adj.append(np.array(spectral.adjacency(graph, sign).todense()))
        nb_iter += 1
    # print(list(map(triangle_nodes, CLOSEABLE_TRIANGLES)))
    print(nb_iter, len(CLOSEABLE_TRIANGLES))
    if close_all:
        random_completion(graph, -1)
    transfer_depth(graph)
    return adj


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
