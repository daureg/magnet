#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""."""
from timeit import default_timer
import numpy as np
import densify
import cc_pivot as cc
import random as r
import graph_tool as gt
from itertools import product, combinations


def make_circle(n):
    circle = cc.gtgeneration.circular_graph(n)
    graph = cc.make_signed_graph(circle)
    densify.N = n
    densify.EDGES_SIGN.clear()
    fake = graph.new_edge_property('bool')
    graph.ep['fake'] = fake
    for i, e in enumerate(graph.edges()):
        graph.ep['sign'][e] = i != 0
        src, dst = int(e.source()), int(e.target())
        src, dst = min(src, dst), max(src, dst)
        densify.EDGES_SIGN[(src, dst)] = bool(graph.ep['sign'][e])
    return graph


def empty_graph():
    graph = gt.Graph(directed=False)
    graph.ep['fake'] = graph.new_edge_property('bool')
    edge_is_positive = graph.new_edge_property("bool")
    graph.ep['sign'] = edge_is_positive
    graph.vp['cluster'] = graph.new_vertex_property("int")
    return graph, edge_is_positive


def finalize_graph(graph):
    edge_tuple = lambda e: (min(map(int, e)), max(map(int, e)))
    densify.N = graph.num_vertices()
    densify.EDGES_SIGN = {edge_tuple(e): bool(graph.ep['sign'][e])
                          for e in graph.edges()}


def make_rings(size, nb_rings, ring_size_ratio=1, shared_sign=True):
    graph, edge_is_positive = empty_graph()

    v1, v2 = graph.add_vertex(), graph.add_vertex()
    shared = graph.add_edge(v1, v2)
    edge_is_positive[shared] = shared_sign

    def add_cycle(length):
        start = graph.vertex(1)
        # if the shared edge is not positive, then we don't need to add any
        # other negative edges in the rings
        negative_index = r.randint(0, length-1) if shared_sign else -1
        for i in range(length-1):
            end = graph.add_vertex()
            e = graph.add_edge(start, end)
            edge_is_positive[e] = i != negative_index
            start = end
        e = graph.add_edge(end, 0)
        edge_is_positive[e] = (length - 1) != negative_index

    nb_small_rings = nb_rings / 2
    nb_large_rings = nb_rings - nb_small_rings
    large_length = int((size + nb_rings)/(nb_small_rings * ring_size_ratio +
                                          nb_large_rings))
    for _ in range(nb_large_rings):
        add_cycle(large_length)
    for _ in range(nb_small_rings):
        add_cycle(max(int(large_length*ring_size_ratio), 2))

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


def run_one_experiment(graph):
    start = default_timer()
    densify.complete_graph(graph)
    elapsed = default_timer() - start
    cc.cc_pivot(graph)
    nb_cluster = np.unique(graph.vp['cluster'].a).size
    disagreements = cc.count_disagreements(graph)
    return elapsed, nb_cluster, disagreements.a.sum().ravel()[0]


def delta_fas_circle(n, p, k=100):
    orig = make_circle(n)
    densify.random_completion(orig, p)
    res, best_g, best_d, worst_g, worst_d = [], None, N, None, 0
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

if __name__ == '__main__':
    # pylint: disable=C0103
    # ring = make_rings(35, 5)
    # name = ring.new_vertex_property('string')
    # for i, v in enumerate(ring.vertices()):
    #     name[v] = str(i)
    # pos = cc.gtdraw.sfdp_layout(ring, cooling_step=0.95, epsilon=5e-2)
    # print(run_one_experiment(ring))
    # cc.draw_clustering(ring, filename="ring.pdf", pos=pos,
    #                    vmore={'text': name})
    import persistent as p
    N, proba = 20, 2
    res, _, best_g, _, worst_g = delta_fas_circle(N, proba, 1000)
    p.save_var('test_fas_pos.my', res)
    best_g.save('fas_best_{:03d}_pos.gt'.format(N))
    worst_g.save('fas_worst_{:03d}_pos.gt'.format(N))
    cc.draw_clustering(best_g, filename='fas_best_{:03d}_pos.pdf'.format(N))
    cc.draw_clustering(worst_g, filename='fas_worst_{:03d}_pos.pdf'.format(N))
    import sys
    sys.exit()
    N, k = 33, 4
    best_g = None
    best_d = N*N
    for _ in range(6):
        g = make_rings(N, k)
        t, c, d = run_one_experiment(g)
        if d < best_d:
            best_g, best_d = g.copy(), d
    cc.draw_clustering(best_g, filename='ring_{:03d}.pdf'.format(N))
    from operator import itemgetter
    Ns = np.linspace(6, 150, 6)
    K = 100
    for n in map(int, Ns):
        this_run = []
        for _ in range(K):
            this_run.append(run_one_experiment(make_circle(n)))
        res = {'time': map(itemgetter(0), this_run),
               'nb_cluster': map(itemgetter(1), this_run),
               'nb_error': map(itemgetter(2), this_run)}
        p.save_var('circle_{}.my'.format(n), res)
