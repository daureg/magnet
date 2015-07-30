#! /usr/bin/env python
# vim: set fileencoding=utf-8
# import msgpack_pypy as msgpack
import msgpack
from timeit import default_timer as clock
from collections import defaultdict


def read_text_graph(filename):
    """return a graph representation from `filename`"""
    graph = defaultdict(list)
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue
            u, v, _ = [int(_) for _ in line.split('\t')]
            graph[u].append(v)
            graph[v].append(u)
    return [(k, tuple(v)) for k, v in graph.iteritems()]


def save_packed_graph(graph_as_list, filename):
    with open(filename, 'w+b') as outfile:
        msgpack.pack(graph_as_list, outfile)


def load_packed_graph(filename):
    with open(filename, 'r+b') as packfile:
        return msgpack.unpack(packfile, use_list=False)


def build_graph(graph_as_list):
    return {node: adj for node, adj in graph_as_list}


def load_graph(filename):
    return build_graph(load_packed_graph(filename))

if __name__ == '__main__':
    # do it once
    # save_packed_graph(read_text_graph('soc-sign-epinions.txt'), 'epi.pack')
    start = clock()
    G = load_graph('epi.pack')
    end = clock() - start
    nb_nodes = len(G)
    nb_edges = sum((len(adj) for adj in G.values())) // 2
    print('loaded {} nodes and {} edges in {:.3f} seconds'.format(nb_nodes,
                                                                  nb_edges,
                                                                  end))
