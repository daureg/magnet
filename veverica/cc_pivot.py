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


def cc_pivot(g):
    """Fill g's cluster_index according to Aion algorithm"""
    current_cluster_index = 0
    clustered = g.new_vertex_property('bool')
    c.set_vertex_filter(clustered, inverted=True)

    def add_to_current_cluster(node):
        c.vp['cluster'][node] = current_cluster_index
        clustered[node] = True

    still_unclustered = list(c.vertices())
    while len(still_unclustered) > 0:
        pivot = r.choice(still_unclustered)
        add_to_current_cluster(pivot)
        for e in pivot.out_edges():
            positive_neighbor = c.ep['sign'][e]
            if positive_neighbor:
                add_to_current_cluster(e.target())
        current_cluster_index += 1
        still_unclustered = list(c.vertices())
    c.set_vertex_filter(None)


def count_disagrements(g):
    """Return a boolean edge map of disagrement with current clustering"""
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


def build_graph():
    """Build a complete with random sign"""
    graph = gtgeneration.complete_graph(N)
    edge_is_positive = graph.new_edge_property("bool")
    cluster_index = graph.new_vertex_property("int")
    graph.ep['sign'] = edge_is_positive
    for e in graph.edges():
        if r.random() > .7:
            graph.ep['sign'][e] = True
    graph.vp['cluster'] = cluster_index
    return graph


def draw_clustering(c, filename):
    pos = gtdraw.sfdp_layout(c, cooling_step=0.95, epsilon=5e-2)
    cluster_color = c.new_vertex_property('vector<float>')
    cluster_name = c.new_vertex_property('string')
    for v in c.vertices():
        cluster_color[v] = list(COLORS[c.vp['cluster'][v]])+[0.9, ]
        cluster_name[v] = '{:01d}'.format(c.vp['cluster'][v])
    # plain, dash = [1.0, 0.0, .0], [0.1, 0.1, 0]
    # line_styles = [dash, plain]
    ecolors = [[.8, .2, .2, .8], [.2, .8, .2, .8]]
    # dash_pattern = c.new_edge_property('vector<float>')
    edge_color = c.new_edge_property('vector<float>')
    for e in c.edges():
        # dash_pattern[e] = line_styles[c.ep['sign'][e]]
        edge_color[e] = ecolors[c.ep['sign'][e]]

    d = count_disagrements(c)
    edge_width = c.new_edge_property('float')
    for e in c.edges():
        edge_width[e] = 8 if d[e] else 4
    gtdraw.graph_draw(c, pos=pos, edge_color=edge_color,
                      vertex_fill_color=cluster_color, vertex_pen_width=0,
                      vertex_text=cluster_name,
                      # edge_pen_width=edge_width,
                      # edge_dash_style=dash_pattern,
                      edge_pen_width=edge_width,
                      output=filename)

if __name__ == '__main__':
    N = 10
    c = build_graph()
    cc_pivot(c)
    draw_clustering(c, "complete_10.pdf")
