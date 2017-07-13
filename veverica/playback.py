#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Draw graph at a given step of completion."""
import graph_tool.draw as gtdraw
import numpy as np


def draw_state(n_iter, graph, history, pos):
    """draw one state"""
    edge_tuple = lambda e: (min(map(int, e)), max(map(int, e)))
    n = graph.num_vertices()
    pivot, d_edges = history[n_iter]

    vpen = graph.new_vertex_property('int')
    vpen.a = 4*(np.arange(n) == pivot)
    name = graph.new_vertex_property('string')
    name = graph.new_vertex_property('string')
    for i, v in enumerate(graph.vertices()):
        name[v] = str(i)
    vertex_options = {'size': 26, 'color': 'blue', 'pen_width': vpen,
                      'text': name}

    old_enough = graph.new_edge_property('bool')
    old_enough.a = graph.ep['depth'].a <= n_iter//2

    ecolors = [[.8, .2, .2, .8], [.2, .8, .2, .8]]
    edge_color = graph.new_edge_property('vector<float>')
    edge_width = graph.new_edge_property('int')
    for e in graph.edges():
        if edge_tuple(e) in d_edges:
            edge_width[e] = 5
            old_enough[e] = True
            if n_iter % 2 == 0:
                edge_color[e] = [.6, .6, .6, .8]
            else:
                edge_color[e] = ecolors[graph.ep['sign'][e]]
        else:
            edge_width[e] = 3
            edge_color[e] = ecolors[graph.ep['sign'][e]]

    edge_options = {'pen_width': edge_width, 'color': edge_color}

    graph.set_edge_filter(old_enough)
    gtdraw.graph_draw(graph, pos=pos, vprops=vertex_options,
                      eprops=edge_options, output=None, fit_view=True,
                      output_size=(600, 600))
    graph.set_edge_filter(None)
