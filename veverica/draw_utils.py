#! /usr/bin/python
# vim: set fileencoding=utf-8
"""Utilities to draw graphs and trees."""
import numpy as np


def map_from_list_of_edges(graph, edges):
    """Return an edge map true on `edges`"""
    tree_map = graph.new_edge_property('boolean')
    graph.set_edge_filter(None)
    for e in graph.edges():
        u, v = e
        u, v = int(u), int(v)
        if (u, v) in edges:
            tree_map[e] = True
    return tree_map

good_edge = [ 3/255, 169/255, 244/255, .8 ]
bad_edge = [ 1, 193/255, 7/255, .8 ]
red, green = [.8, .2, .2, .5], [.2, .8, .2, .5]
black, light_gray = [.05, .05, .05, .9], [.8, .85, .9, .4]
bfs_col = [ 1, 193/255, 7/255, .8 ]


def prop_to_size(data, mi=0, ma=5, power=0.5, log=False):
    minx, maxx = min(data), max(data)
    if log:
        minx, maxx = np.log(minx), np.log(maxx)
    drange = maxx-minx
    return lambda x: mi + (ma-mi)*(((np.log(x) if log else x) - minx)/drange)**power


def color_graph(g, tree, stretch=None):
    ecol, esize = g.new_edge_property('vector<double>'), g.new_edge_property('float')
    tree_size = 2.3
    g.set_edge_filter(None)
    if stretch:
        size = prop_to_size(stretch.values(), mi=1, ma=4, power=.8)
    for e in g.edges():
        u, v = e
        u, v = int(u), int(v)
        if (u, v) in tree:
            ecol[e] = black
            esize[e] = tree_size
            continue
        esize[e] = 0.7 if stretch is None else size(stretch[(u, v)])
        ecol[e] = light_gray
    return ecol, esize
