#! /usr/bin/python2
# vim: set fileencoding=utf-8
import graph_tool.generation as gtgeneration
import graph_tool.draw as gtdraw
import numpy as np
import seaborn as sns
import random as r
from collections import defaultdict
"""Implementation of Quick Pivot algorithm for correlation clustering.
Ailon, N., Charikar, M., & Newman, A. (2008). Aggregating inconsistent
information. Journal of the ACM, 55(5), 1–27. doi:10.1145/1411509.1411513"""


def cc_pivot(graph, pivot_seq=None):
    """Fill g's cluster_index according to Ailon algorithm"""
    current_cluster_index = 0
    clustered = graph.new_vertex_property('bool')
    graph.set_vertex_filter(clustered, inverted=True)

    def add_to_current_cluster(node):
        graph.vp['cluster'][node] = current_cluster_index
        clustered[node] = True
        # print('\tadd {} to cluster {}'.format(int(node),
        #                                       current_cluster_index))

    still_unclustered = list(graph.vertices())
    pivots = (graph.vertex(_) for _ in pivot_seq) if pivot_seq else None
    # print(pivots)
    while len(still_unclustered) > 0:
        pivot = r.choice(still_unclustered)
        if pivots:
            try:
                pivot = next(pivots)
                # print('pick '+int(pivot))
                while pivot not in still_unclustered:
                    pivot = next(pivots)
            except StopIteration:
                pivot = r.choice(still_unclustered)
        # print('choose {} as pivot'.format(int(pivot)))
        add_to_current_cluster(pivot)
        for e in pivot.out_edges():
            positive_neighbor = graph.ep['sign'][e]
            if positive_neighbor:
                add_to_current_cluster(e.target())
        current_cluster_index += 1
        still_unclustered = list(graph.vertices())
    graph.set_vertex_filter(None)


def count_disagreements(g, alt_index=False):
    """Return a boolean edge map of disagreement with current clustering"""
    if not alt_index:
        if 'fake' in g.ep:
            g.set_edge_filter(g.ep['fake'], inverted=True)
        cluster_index_name = 'cluster'
    else:
        cluster_index_name = alt_index
    disagree = g.new_edge_property('bool')
    cluster = lambda v: g.vp[cluster_index_name][v]
    positive = lambda e: g.ep['sign'][e]
    negative = lambda e: not positive(e)
    for e in g.edges():
        disagree[e] = (cluster(e.source()) == cluster(e.target()) and
                       negative(e)) or (
                           cluster(e.source()) != cluster(e.target()) and
                           positive(e))
    g.set_edge_filter(None)
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


def add_cluster_name_and_color(graph, cluster_prop='cluster'):
    cluster_color = graph.new_vertex_property('vector<float>')
    cluster_name = graph.new_vertex_property('string')
    nb_cluster = np.unique(graph.vp[cluster_prop].a).size
    colors = sns.color_palette("Set1", nb_cluster)
    for v in graph.vertices():
        cluster_color[v] = list(colors[graph.vp[cluster_prop][v]])+[0.9, ]
        cluster_name[v] = '{:01d}'.format(graph.vp[cluster_prop][v])
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


def lined_up_cluster(graph, cluster_index_name):
    """Return nodes positions where members of a given cluster are vertically
    aligned."""
    pos = graph.new_vertex_property('vector<float>')
    coord = []
    counter = defaultdict(int)
    for c in graph.vp[cluster_index_name].a:
        coord.append((5*c+.5*r.choice(np.linspace(-3, 3, 6)), counter[c]))
        counter[c] += 2
    pos.set_2d_array(np.array(list(zip(*tmp))))
    return pos


def draw_clustering(graph, filename=None, pos=None, vmore=None,
                    emore=None, show_filling=False,
                    cluster_index_name='cluster'):
    graph.set_edge_filter(graph.ep['fake'], inverted=True)
    pos = pos or gtdraw.sfdp_layout(graph)
    vertex_options = {'pen_width': 0}
    if vmore:
        vertex_options.update(vmore)
    vertex_options.update(add_cluster_name_and_color(graph,
                                                     cluster_index_name))
    name = graph.new_vertex_property('string')
    for i, v in enumerate(graph.vertices()):
        name[v] = str(i)
    if np.unique(graph.vp[cluster_index_name].a).size < 2:
        vertex_options['text'] = name
    d = count_disagreements(graph, alt_index=cluster_index_name)
    if not show_filling:
        graph.set_edge_filter(graph.ep['fake'], inverted=True)
    print(str(d.a.sum().ravel()[0]) + ' disagreements')
    if not show_filling:
        edge_options = {'pen_width': 2}
    else:
        edge_width = graph.new_edge_property('float')
        for e in graph.edges():
            if not graph.ep['fake'][e]:
                edge_width[e] = 6
            else:
                edge_width[e] = 3  # if graph.ep['sign'][e] else 1
        edge_options = {'pen_width': edge_width}
    edge_options.update(add_edge_sign_color(graph))
    # edge_options.update(add_edge_disagreement_size(graph, d))
    if emore:
        edge_options.update(emore)

    gtdraw.graph_draw(graph, pos=pos, vprops=vertex_options,
                      eprops=edge_options, output=filename, fit_view=True,
                      output_size=(600, 600))
    graph.set_edge_filter(None)

if __name__ == '__main__':
    N = 10
    c = make_signed_graph(gtgeneration.complete_graph(N))
    cc_pivot(c)
    draw_clustering(c, "complete_10.pdf")
