#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Create Delaunay triangulation of random points in the plane."""
from timeit import default_timer as clock
import graph_tool.generation as gen
import persistent
import numpy as np
import sys


def to_python_graph(graph):
    """represents `graph` by two dictionaries"""
    G = {int(u): {int(v) for v in u.out_neighbours()}
         for u in graph.vertices()}
    E = {(int(e.source()), int(e.target())): True for e in graph.edges()}
    return G, E


if __name__ == '__main__':
    # pylint: disable=C0103
    n = int(sys.argv[1])
    start = clock()
    points = np.random.random((n, 2))*(n//50+1)
    g, _ = gen.triangulation(points, type="delaunay")
    persistent.save_var('belgrade/triangle_{}.my'.format(n),
                        to_python_graph(g))
    print('create {} edges in {:.3f} seconds'.format(g.num_edges(),
                                                     clock() - start))
