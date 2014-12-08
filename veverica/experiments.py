#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""."""
from timeit import default_timer
import time
import numpy as np
import densify
import cc_pivot as cc
import random as r
import graph_tool as gt
from operator import itemgetter
from itertools import product, combinations, repeat
from TriangleCache import TriangleStatus
import persistent as p
import os
NUM_THREADS = os.environ.get('NUM_THREADS', 14)


def negative_pattern(n, quantity=None, distance=None):
    """create position for `quantity` negative edges or two of them separated
    by `distance` vertices."""
    assert quantity or distance, "give a argument"
    vertices = list(range(n))
    if quantity:
        starts = sorted(r.sample(vertices, int(quantity)))
        return [(_, (_+1) % n) for _ in starts]
    assert distance < n
    return [(0, 1), (distance, (distance+1) % n)]


def make_circle(n, rigged=False):
    circle = cc.gtgeneration.circular_graph(n)
    graph = cc.make_signed_graph(circle)
    densify.N = n
    densify.EDGES_SIGN.clear()
    # densify.EDGES_DEPTH.clear()
    fake = graph.new_edge_property('bool')
    graph.ep['fake'] = fake
    graph.ep['depth'] = graph.new_edge_property('long')
    graph.ep['depth'].a = 1
    for i, e in enumerate(graph.edges()):
        src, dst = int(e.source()), int(e.target())
        src, dst = min(src, dst), max(src, dst)
        if not rigged:
            graph.ep['sign'][e] = i != 0
        else:
            graph.ep['sign'][e] = (src, dst) not in rigged
        densify.EDGES_SIGN[(src, dst)] = bool(graph.ep['sign'][e])
        # densify.EDGES_DEPTH[(src, dst)] = 1
    return graph


def empty_graph():
    graph = gt.Graph(directed=False)
    graph.ep['fake'] = graph.new_edge_property('bool')
    edge_is_positive = graph.new_edge_property("bool")
    graph.ep['sign'] = edge_is_positive
    graph.ep['depth'] = graph.new_edge_property('short')
    graph.ep['depth'].a = 1
    graph.vp['cluster'] = graph.new_vertex_property("int")
    return graph, edge_is_positive


def finalize_graph(graph):
    edge_tuple = lambda e: (min(map(int, e)), max(map(int, e)))
    densify.N = graph.num_vertices()
    densify.EDGES_SIGN = {edge_tuple(e): bool(graph.ep['sign'][e])
                          for e in graph.edges()}
    # densify.EDGES_DEPTH = {edge_tuple(e): int(graph.ep['depth'][e])
    #                        for e in graph.edges()}
    # print('finalize {}'.format(hash(graph)))


def make_rings(size, nb_rings, ring_size_ratio=1, shared_sign=True,
               rigged=False):
    """Create a graph with around `size` nodes splits into `nb_rings`. Half of
    them are `ring_size_ratio` smaller than the others. They have one common
    edge with `shared_sign`. If `rigged` is True, the negative edge of each
    ring is at the middle, or at a position specified by a list."""
    if nb_rings == 1:
        return make_circle(size)
    graph, edge_is_positive = empty_graph()

    v1, v2 = graph.add_vertex(), graph.add_vertex()
    shared = graph.add_edge(v1, v2)
    edge_is_positive[shared] = shared_sign
    graph.ep['depth'][shared] = 1
    ring_id = 0

    def add_cycle(length, ring_id):
        start = graph.vertex(1)
        # if the shared edge is not positive, then we don't need to add any
        # other negative edges in the rings
        negative_index = r.randint(0, length-1) if shared_sign else -1
        if negative_index >= 0 and rigged:
            negative_index = int(length/2)
        if negative_index >= 0 and isinstance(rigged, list):
            negative_index = rigged[ring_id] - 1
            assert 0 <= negative_index <= length
        for i in range(length-1):
            end = graph.add_vertex()
            e = graph.add_edge(start, end)
            graph.ep['depth'][e] = 1
            edge_is_positive[e] = i != negative_index
            start = end
        e = graph.add_edge(end, 0)
        graph.ep['depth'][e] = 1
        edge_is_positive[e] = (length - 1) != negative_index

    nb_small_rings = int(nb_rings / 2)
    nb_large_rings = nb_rings - nb_small_rings
    large_length = int((size + nb_rings)/(nb_small_rings * ring_size_ratio +
                                          nb_large_rings))
    for _ in range(nb_large_rings):
        add_cycle(large_length, ring_id)
        ring_id += 1
    for _ in range(nb_small_rings):
        add_cycle(max(int(large_length*ring_size_ratio), 2), ring_id)
        ring_id += 1

    finalize_graph(graph)
    return graph


def planted_clusters(ball_size=12, nb_balls=5):
    graph, __ = empty_graph()
    graph.vp['true_cluster'] = graph.new_vertex_property('int')
    balls = [make_ball(graph, ball_size) for _ in range(nb_balls)]
    pos = cc.gtdraw.sfdp_layout(graph)
    for b1, b2 in combinations(balls, 2):
        link_balls(graph, b1, b2)
    flip_random_edges(graph)
    finalize_graph(graph)
    return graph, pos


def make_ball(graph, n):
    gsize = graph.num_vertices()
    if gsize == 0:
        cluster_index = 0
    else:
        cluster_index = graph.vp['true_cluster'][graph.vertex(gsize-1)] + 1
    size = r.randint(int(0.7*n), int(1.3*n))
    index = set()
    for _ in range(size):
        v = graph.add_vertex()
        graph.vp['true_cluster'][v] = cluster_index
        index.add(v)
    edges = r.sample(list(combinations(index, 2)), int(1.5*size))
    endpoints = set()
    for u, v in edges:
        e = graph.add_edge(u, v)
        endpoints.add(u)
        endpoints.add(v)
        graph.ep['sign'][e] = True
    # make sure the ball forms a connected component
    alone = index.difference(endpoints)
    endpoints = list(endpoints)
    for u in alone:
        e = graph.add_edge(u, r.choice(endpoints))
        graph.ep['sign'][e] = True
    return index


def link_balls(graph, b1, b2):
    edges = r.sample(list(product(b1, b2)), int(1.0*(len(b1)+len(b2))/2))
    for u, v in edges:
        e = graph.add_edge(u, v)
        graph.ep['sign'][e] = False


def flip_random_edges(graph, fraction=0.1):
    """Change the sign of `fraction` of `graph` edges"""
    E = list(graph.edges())
    for e in r.sample(E, int(fraction*len(E))):
        graph.ep['sign'][e] = not graph.ep['sign'][e]


def run_one_experiment(graph, cc_run=500, shared_edges=None,
                       pivot_strategy=densify.PivotStrategy.uniform,
                       triangle_strategy=TriangleStatus.closeable,
                       one_at_a_time=False):
    start = default_timer()
    densify.complete_graph(graph, shared_edges=shared_edges,
                           pivot_strategy=pivot_strategy,
                           triangle_strategy=triangle_strategy,
                           one_at_a_time=one_at_a_time)
    elapsed = default_timer() - start
    res = []
    for _ in range(cc_run):
        tmp_graph = graph.copy()
        cc.cc_pivot(tmp_graph)
        disagreements = cc.count_disagreements(tmp_graph)
        res.append(disagreements.a.sum().ravel()[0])
    nb_cluster = np.unique(graph.vp['cluster'].a).size
    return elapsed, nb_cluster, np.mean(res)


def process_rings(kwargs):
    g = make_rings(kwargs['size'], kwargs['nb_rings'],
                   kwargs['ring_size_ratio'], kwargs['shared_sign'],
                   kwargs['rigged'])
    # print(hash(g))
    # print(id(g))
    # print(g)
    return run_one_experiment(g, 150, kwargs['shared_edges'],
                              kwargs['pivot_strategy'],
                              kwargs['triangle_strategy'],
                              kwargs['one_at_a_time'])


def run_ring_experiment(size, nb_rings, ring_size_ratio=1.0, shared_sign=True,
                        rigged=False, n_rep=100, shared_edges=None,
                        pivot_strategy=densify.PivotStrategy.uniform,
                        triangle_strategy=TriangleStatus.closeable,
                        one_at_a_time=True,
                        pool=None):
    args = repeat({"size": size, "nb_rings": nb_rings, "ring_size_ratio":
                   ring_size_ratio, "shared_sign": shared_sign, "rigged":
                   rigged, "shared_edges": shared_edges,
                   "pivot_strategy": pivot_strategy, "triangle_strategy":
                   triangle_strategy, "one_at_a_time": one_at_a_time}, n_rep)
    if pool:
        runs = list(pool.imap_unordered(process_rings, args,
                                        chunksize=n_rep//NUM_THREADS))
    else:
        runs = list(map(process_rings, args))
    res = {'time': list(map(itemgetter(0), runs)),
           'nb_error': list(map(itemgetter(2), runs))}
    suffix = 'pos' if shared_sign else 'neg'
    suffix += '_rigged' if rigged else ''
    suffix += '_' + str(n_rep)
    heuristic = strategy_to_str(pivot_strategy, triangle_strategy,
                                one_at_a_time)
    exp_name = 'square_{:04d}_{:02d}_{:.1f}_{}_{}_{}.my'
    p.save_var(exp_name.format(size, nb_rings, ring_size_ratio, suffix,
                               heuristic, int(time.time())), res)


def process_planted(kwargs):
    g, _ = planted_clusters(kwargs['ball_size'], kwargs['nb_balls'])
    delta = cc.count_disagreements(g, alt_index='true_cluster')
    delta = delta.a.sum().ravel()[0]
    times, _, errors = run_one_experiment(g, 150, kwargs['shared_edges'],
                                          kwargs['pivot_strategy'],
                                          kwargs['triangle_strategy'],
                                          kwargs['one_at_a_time'])
    return [times, delta, errors]


def run_planted_experiment(ball_size, nb_balls,
                           pivot_strategy=densify.PivotStrategy.uniform,
                           triangle_strategy=TriangleStatus.closeable,
                           one_at_a_time=True,
                           n_rep=100, pool=None):
    args = repeat({"ball_size": ball_size, "nb_balls": nb_balls,
                   "shared_edges": False,
                   "pivot_strategy": pivot_strategy, "triangle_strategy":
                   triangle_strategy, "one_at_a_time": one_at_a_time}, n_rep)

    if pool:
        runs = list(pool.imap_unordered(process_planted, args,
                                        chunksize=n_rep//NUM_THREADS))
    else:
        runs = list(map(process_planted, args))
    res = {'time': list(map(itemgetter(0), runs)),
           'delta': list(map(itemgetter(1), runs)),
           'nb_error': list(map(itemgetter(2), runs))}
    heuristic = strategy_to_str(pivot_strategy, triangle_strategy,
                                one_at_a_time)
    p.save_var('planted_{:04d}_{:02d}_{}_{}.my'.format(ball_size, nb_balls,
                                                       heuristic,
                                                       int(time.time())),
               res)


def strategy_to_str(pivot_strategy, triangle_strategy, one_at_a_time):
    return '{}_{}_{}'.format(pivot_strategy.name, triangle_strategy.name,
                             'ONE' if one_at_a_time else 'ALL')


def process_circle(kwargs):
    g = make_circle(kwargs['circle_size'], kwargs['rigged'])
    return run_one_experiment(g, 100, kwargs['shared_edges'],
                              kwargs['pivot_strategy'],
                              kwargs['triangle_strategy'],
                              kwargs['one_at_a_time'])


def run_circle_experiment(size, rigged=False,
                          pivot_strategy=densify.PivotStrategy.uniform,
                          triangle_strategy=TriangleStatus.closeable,
                          one_at_a_time=True,
                          n_rep=100, pool=None):
    args = repeat({"circle_size": size, "rigged": rigged,
                   "shared_edges": False,
                   "pivot_strategy": pivot_strategy, "triangle_strategy":
                   triangle_strategy, "one_at_a_time": one_at_a_time}, n_rep)

    if pool:
        runs = list(pool.imap_unordered(process_circle, args,
                                        chunksize=n_rep//NUM_THREADS))
    else:
        runs = list(map(process_planted, args))
    res = {'time': list(map(itemgetter(0), runs)),
           'nb_error': list(map(itemgetter(2), runs))}
    heuristic = strategy_to_str(pivot_strategy, triangle_strategy,
                                one_at_a_time)
    p.save_var('circle_{:04d}_{}_{}.my'.format(size, heuristic,
                                               int(time.time())),
               res)


def delta_fas_circle(n, p, k=100):
    orig = make_circle(n)
    densify.random_completion(orig, p)
    res, best_g, best_d, worst_g, worst_d = [], None, n, None, 0
    for _ in range(k):
        graph = orig.copy()
        cc.cc_pivot(graph)
        disagreements = cc.count_disagreements(graph)
        d = disagreements.a.sum().ravel()[0]
        if d < best_d:
            best_g, best_d = graph.copy(), d
        if d > worst_d:
            worst_g, worst_d = graph.copy(), d
        res.append(d)
    return res, best_d, best_g, worst_d, worst_g
