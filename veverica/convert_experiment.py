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
NUM_THREADS = 14


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


def make_circle(n):
    redensify.N = n
    redensify.G.clear()
    redensify.CLOSEABLE_TRIANGLES.clear()
    redensify.TMP_SET.clear()
    redensify.EDGES_SIGN.clear()
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
    from operator import itemgetter
    all_vert = list(range(redensify.N))
    graph.vp['cluster'].a = itemgetter(*all_vert)(cc_pivot())
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


def process_circle(kwargs):
    make_circle(kwargs['circle_size'])
    return run_one_experiment(100, kwargs['one_at_a_time'])


def run_circle_experiment(size, one_at_a_time=True, n_rep=100, pool=None):
    args = repeat({"circle_size": size,
                   "one_at_a_time": one_at_a_time}, n_rep)

    if pool:
        runs = list(pool.imap_unordered(process_circle, args,
                                        chunksize=n_rep//NUM_THREADS))
    else:
        runs = list(map(process_circle, args))
    res = {'time': list(map(itemgetter(0), runs)),
           'nb_error': list(map(itemgetter(2), runs))}
    p.save_var('circle_new_{:04d}_{}.my'.format(size,
                                               int(time.time())),
               res)


def run_one_experiment(cc_run=100, one_at_a_time=True):
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
    print(mean)
    return elapsed, nb_cluster, mean
