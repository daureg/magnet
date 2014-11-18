#! /usr/bin/python2
# vim: set fileencoding=utf-8
import graph_tool.generation as gtgeneration
import graph_tool.draw as gtdraw
import prettyplotlib as ppl
COLORS = ppl.colors.set1
import random as r
"""Implementation of Quick Pivot algorithm for correlation clustering.
Ailon, N., Charikar, M., & Newman, A. (2008). Aggregating inconsistent
information. Journal of the ACM, 55(5), 1–27. doi:10.1145/1411509.1411513"""


def cc_pivot(graph):
    """Fill g's cluster_index according to Ailon algorithm"""
    current_cluster_index = 0
    clustered = graph.new_vertex_property('bool')
    graph.set_vertex_filter(clustered, inverted=True)

    def add_to_current_cluster(node):
        graph.vp['cluster'][node] = current_cluster_index
        clustered[node] = True

    still_unclustered = list(graph.vertices())
    while len(still_unclustered) > 0:
        pivot = r.choice(still_unclustered)
        add_to_current_cluster(pivot)
        for e in pivot.out_edges():
            positive_neighbor = graph.ep['sign'][e]
            if positive_neighbor:
                add_to_current_cluster(e.target())
        current_cluster_index += 1
        still_unclustered = list(graph.vertices())
    graph.set_vertex_filter(None)


def count_disagreements(g):
    """Return a boolean edge map of disagreement with current clustering"""
    disagree = g.new_edge_property('bool')
    cluster = lambda v: g.vp['cluster'][v]
    positive = lambda e: g.ep['sign'][e]
    negative = lambda e: not positive(e)
    for e in g.edges():
        disagree[e] = (cluster(e.source()) == cluster(e.target()) and
                       negative(e)) or (
                           cluster(e.source()) != cluster(e.target()) and
                           positive(e))
    return disagree


def make_signed_graph(graph):
    """Add sign and cluster information to a graph"""
    edge_is_positive = graph.new_edge_property("bool")
    cluster_index = graph.new_vertex_property("int")
    graph.ep['sign'] = edge_is_positive
    for e in graph.edges():
        if r.random() > .7:
            graph.ep['sign'][e] = True
    graph.vp['cluster'] = cluster_index
    return graph


def add_cluster_name_and_color(graph):
    cluster_color = graph.new_vertex_property('vector<float>')
    cluster_name = graph.new_vertex_property('string')
    for v in graph.vertices():
        cluster_color[v] = list(COLORS[graph.vp['cluster'][v]])+[0.9, ]
        cluster_name[v] = '{:01d}'.format(graph.vp['cluster'][v])
    return {'fill_color': cluster_color,
            'text': cluster_name}


def add_edge_sign_color(graph):
    ecolors = [[.8, .2, .2, .8], [.2, .8, .2, .8]]
    edge_color = graph.new_edge_property('vector<float>')
    for e in graph.edges():
        edge_color[e] = ecolors[graph.ep['sign'][e]]
    return {'color': edge_color}


def add_edge_disagreement_size(graph, disagreement):
    edge_width = graph.new_edge_property('float')
    for e in graph.edges():
        edge_width[e] = 8 if disagreement[e] else 4
    return {'pen_width': edge_width}


def draw_clustering(graph, filename=None, pos=None, vmore=None):
    pos = pos or gtdraw.sfdp_layout(graph, cooling_step=0.95, epsilon=5e-2)
    vertex_options = {'pen_width': 0}
    if vmore:
        vertex_options.update(vmore)
    vertex_options.update(add_cluster_name_and_color(graph))
    name = graph.new_vertex_property('string')
    for i, v in enumerate(graph.vertices()):
        name[v] = str(i)
    vertex_options['text'] = name
    d = count_disagreements(graph)
    print(str(d.a.sum().ravel()[0]) + ' disagreements')
    edge_options = {'pen_width': 3}
    edge_options.update(add_edge_sign_color(graph))
    # edge_options.update(add_edge_disagreement_size(graph, d))

    gtdraw.graph_draw(graph, pos=pos, vprops=vertex_options,
                      eprops=edge_options, output=filename)

if __name__ == '__main__':
    N = 10
    c = make_signed_graph(gtgeneration.complete_graph(N))
    cc_pivot(c)
    draw_clustering(c, "complete_10.pdf")
