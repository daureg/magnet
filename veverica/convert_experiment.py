#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Convert graph_tool graph to dict of sets and back within the redensify
module."""
import redensify
import graph_tool as gt
import random as r


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
    current_cluster_index = 0
    clustered = set()
    cluster = {}

    def add_to_current_cluster(node):
        cluster[node] = current_cluster_index
        clustered.add(node)

    N = redensify.N
    still_unclustered = list(range(N))
    while still_unclustered:
        pivot = r.choice(still_unclustered)
        add_to_current_cluster(pivot)
        for n in redensify.G[pivot]:
            edge = (n, pivot) if n < pivot else (pivot, n)
            positive_neighbor = redensify.EDGES_SIGN[edge]
            if positive_neighbor:
                add_to_current_cluster(n)
        current_cluster_index += 1
        still_unclustered = list(set(range(N)).difference(clustered))
    return cluster
