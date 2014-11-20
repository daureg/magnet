#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""."""
from timeit import default_timer
import numpy as np
import densify
import cc_pivot as cc
import random as r
import graph_tool as gt


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


def make_rings(size, nb_ring, shared_sign=True):
    # TODO add ring_size_ratio, the length ratio between the small half rings
    # and long half rings. So 1 should yield equal size rings summing up to N.
    graph = gt.Graph(directed=False)
    graph.ep['fake'] = graph.new_edge_property('bool')
    edge_is_positive = graph.new_edge_property("bool")
    graph.ep['sign'] = edge_is_positive
    cluster_index = graph.new_vertex_property("int")
    graph.vp['cluster'] = cluster_index

    v1, v2 = graph.add_vertex(), graph.add_vertex()
    shared = graph.add_edge(v1, v2)
    edge_is_positive[shared] = shared_sign

    def add_cycle(length):
        start = graph.vertex(1)
        negative_index = r.randint(0, length-1)
        for i in range(length-1):
            end = graph.add_vertex()
            e = graph.add_edge(start, end)
            edge_is_positive[e] = i != negative_index
            start = end
        e = graph.add_edge(end, 0)
        edge_is_positive[e] = (length - 1) != negative_index

    for _ in range(nb_ring):
        add_cycle(size/nb_ring)

    edge_tuple = lambda e: (min(map(int, e)), max(map(int, e)))
    densify.N = graph.num_vertices()
    densify.EDGES_SIGN = {edge_tuple(e): bool(edge_is_positive[e])
                          for e in graph.edges()}
    return graph


def run_one_experiment(graph):
    start = default_timer()
    densify.complete_graph(graph)
    elapsed = default_timer() - start
    cc.cc_pivot(graph)
    nb_cluster = np.unique(graph.vp['cluster'].a).size
    disagreements = cc.count_disagreements(graph)
    return elapsed, nb_cluster, disagreements.a.sum().ravel()[0]


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
    N, k = 33, 4
    best_g = None
    best_d = N*N
    for _ in xrange(6):
        g = make_rings(N, k)
        t, c, d = run_one_experiment(g)
        if d < best_d:
            best_g, best_d = g.copy(), d
    cc.draw_clustering(best_g, filename='ring_{:03d}.pdf'.format(N))
    import sys
    sys.exit()
    import persistent as p
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
