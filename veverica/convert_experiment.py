#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Convert graph_tool graph to dict of sets and back within the redensify
module."""
import redensify
import random as r
from collections import Counter
from timeit import default_timer
from itertools import repeat
from operator import itemgetter
import persistent as p
import time
NUM_THREADS = 1


def to_python_graph(graph):
    """populate redensify global variable with a representation of `graph`"""
    redensify.N = graph.num_vertices()
    redensify.G.clear()
    redensify.CLOSEABLE_TRIANGLES.clear()
    redensify.TMP_SET.clear()
    for node in graph.vertices():
        redensify.G[int(node)] = set(map(int, node.out_neighbours()))
    redensify.EDGES_SIGN.clear()
    for edge in graph.edges():
        src, dst = (min(map(int, edge)), max(map(int, edge)))
        redensify.EDGES_SIGN[(src, dst)] = bool(graph.ep['sign'][edge])
    redensify.EDGES_ORIG = list(redensify.EDGES_SIGN.keys())


def make_rings(size, nb_rings, shared_sign=True, rigged=False):
    """Create a graph with around `size` nodes splits into `nb_rings`. They
    have one common edge with `shared_sign`. If `rigged` is True, the negative
    edge of each ring is at the middle, or at a position specified by a
    list."""
    if nb_rings == 1:
        return make_circle(size)
    new_graph()

    redensify.G[0] = set([1])
    redensify.G[1] = set([0])
    redensify.EDGES_SIGN[(0, 1)] = shared_sign
    ring_id = 0

    def add_cycle(length, ring_id):
        start = 1
        # if the shared edge is not positive, then we don't need to add any
        # other negative edges in the rings
        negative_index = r.randint(0, length) if shared_sign else -1
        if negative_index >= 0 and rigged:
            negative_index = int(length/2)
        if negative_index >= 0 and isinstance(rigged, list):
            negative_index = rigged[ring_id] - 1
            assert 0 <= negative_index <= length
        for i in range(length-1):
            end = len(redensify.G)
            redensify.EDGES_SIGN[(start, end)] = i != negative_index
            redensify.G[end] = set([start])
            redensify.G[start].add(end)
            start = end
        end = len(redensify.G)
        redensify.EDGES_SIGN[(end-1, end)] = (length - 1) != negative_index
        redensify.G[end-1].add(end)
        redensify.G[end] = set([end-1])
        redensify.EDGES_SIGN[(0, end)] = length != negative_index
        redensify.G[0].add(end)
        redensify.G[end].add(0)

    for _ in range(nb_rings):
        add_cycle(size//nb_rings, ring_id)
        ring_id += 1
    redensify.EDGES_ORIG = list(redensify.EDGES_SIGN.keys())
    redensify.N = len(redensify.G)


def new_graph():
    redensify.G.clear()
    redensify.CLOSEABLE_TRIANGLES.clear()
    redensify.TMP_SET.clear()
    redensify.EDGES_SIGN.clear()


def make_circle(n):
    new_graph()
    redensify.N = n
    for i in range(n):
        a, p, b = (i-1) % n, i, (i+1) % n
        redensify.G[i] = set([a, b])
        redensify.EDGES_SIGN[(p, b) if p < b else (b, p)] = i != 0
    redensify.EDGES_ORIG = list(redensify.EDGES_SIGN.keys())


def to_graph_tool():
    import graph_tool as gt
    graph = gt.Graph(directed=False)
    graph.ep['fake'] = graph.new_edge_property('bool')
    graph.ep['sign'] = graph.new_edge_property('bool')
    graph.vp['cluster'] = graph.new_vertex_property('int')
    graph.add_vertex(redensify.N)
    for edge, sign in redensify.EDGES_SIGN.items():
        e = graph.add_edge(edge[0], edge[1])
        graph.ep['sign'][e] = sign
        graph.ep['fake'][e] = edge not in redensify.EDGES_ORIG
    # from operator import itemgetter
    # all_vert = list(range(redensify.N))
    # graph.vp['cluster'].a = itemgetter(*all_vert)(cc_pivot())
    return graph


def cc_pivot():
    """Fill g's cluster_index according to Ailon algorithm"""
    N = redensify.N
    clustered = set()
    unclustered = set(range(N))
    cluster = {}
    current_cluster_index = 0

    def add_to_current_cluster(node):
        cluster[node] = current_cluster_index
        clustered.add(node)
        unclustered.remove(node)

    while unclustered:
        pivot = r.choice(list(unclustered))
        add_to_current_cluster(pivot)
        for n in redensify.G[pivot]:
            edge = (n, pivot) if n < pivot else (pivot, n)
            positive_neighbor = redensify.EDGES_SIGN[edge]
            if positive_neighbor and n in unclustered:
                add_to_current_cluster(n)
        current_cluster_index += 1
    return cluster


def count_disagreements(cluster):
    """Return a boolean edge map of disagreement with current clustering"""

    def disagree(edge, positive):
        return (cluster[edge[0]] == cluster[edge[1]] and not positive) or \
               (cluster[edge[0]] != cluster[edge[1]] and positive)

    return [disagree(edge, redensify.EDGES_SIGN[edge])
            for edge in redensify.EDGES_ORIG]


def process_rings(kwargs):
    make_rings(kwargs['size'], kwargs['nb_rings'], kwargs['shared_sign'],
               kwargs['rigged'])
    return run_one_experiment(100, kwargs['one_at_a_time'])


def run_rings_experiment(size, nb_rings, shared_sign, rigged, one_at_a_time,
                         n_rep=100, pool=None):
    args = repeat({"size": size, "nb_rings": nb_rings, "rigged": rigged,
                   "shared_sign": shared_sign,
                   "one_at_a_time": one_at_a_time}, n_rep)

    if pool:
        runs = list(pool.imap_unordered(process_rings, args,
                                        chunksize=n_rep//NUM_THREADS))
    else:
        runs = [process_rings(_) for _ in args]
    res = {'time': list(map(itemgetter(0), runs)),
           'nb_error': list(map(itemgetter(2), runs))}
    p.save_var('rings_new_{:04d}_{:03d}_{}.my'.format(size, nb_rings,
                                                      int(time.time())),
               res)


def process_circle(kwargs):
    make_circle(kwargs['circle_size'])
    return run_one_experiment(100, kwargs['one_at_a_time'])


def run_circle_experiment(size, one_at_a_time, n_rep=100, pool=None):
    args = repeat({"circle_size": size,
                   "one_at_a_time": one_at_a_time}, n_rep)

    if pool:
        runs = list(pool.imap_unordered(process_circle, args,
                                        chunksize=n_rep//NUM_THREADS))
    else:
        runs = list(map(process_circle, args))
    res = {'time': list(map(itemgetter(0), runs)),
           'nb_error': list(map(itemgetter(2), runs))}
    p.save_var('circle_new_{:04d}_{}.my'.format(size, int(time.time())),
               res)


def run_one_experiment(cc_run, one_at_a_time):
    start = default_timer()
    redensify.complete_graph(one_at_a_time=one_at_a_time)
    elapsed = default_timer() - start
    res = []
    for _ in range(cc_run):
        clusters = cc_pivot()
        disagreements = count_disagreements(clusters)
        res.append(sum(disagreements))
    nb_cluster = len(Counter(clusters).keys())
    mean = sum(res)/len(res)
    return elapsed, nb_cluster, mean
