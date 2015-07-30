#! /usr/bin/env python
# vim: set fileencoding=utf-8
import msgpack
from timeit import default_timer as clock
from collections import defaultdict
from itertools import zip_longest as izip


def read_text_graph(filename):
    """return a flat list of edges in `filename`"""
    edges = []
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue
            u, v, _ = [int(_) for _ in line.split('\t')]
            edges.extend([u, v])
    return edges


def save_packed_edges(edges, filename):
    with open(filename, 'w+b') as outfile:
        msgpack.pack(list(edges), outfile)


def load_packed_edges(filename):
    with open(filename, 'r+b') as packfile:
        return msgpack.unpack(packfile, use_list=False)


def build_graph(edges):
    graph = defaultdict(list)
    for u, v in izip(edges[::2], edges[1::2]):
        graph[u].append(v)
        graph[v].append(u)
    return graph


def load_graph(filename):
    return build_graph(load_packed_edges(filename))

if __name__ == '__main__':
    # do it once
    # save_packed_edges(read_text_graph('soc-sign-epinions.txt'), 'epi.pack')
    start = clock()
    G = load_graph('epi.pack')
    end = clock() - start
    nb_nodes = len(G)
    nb_edges = sum((len(adj) for adj in G.values())) // 2
    print('loaded {} nodes and {} edges in {:.3f} seconds'.format(nb_nodes,
                                                                  nb_edges,
                                                                  end))
