#! /usr/bin/env python
# vim: set fileencoding=utf-8
# import msgpack_pypy as msgpack
from collections import defaultdict
from timeit import default_timer as clock

import msgpack


def add_edge(tree, u, v):
    """Update adjacency list `tree` with the (u, v) edge"""
    if u in tree:
        tree[u].add(v)
    else:
        tree[u] = set([v])
    if v in tree:
        tree[v].add(u)
    else:
        tree[v] = set([u])


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
    return {node: set(adj) for node, adj in graph_as_list}


def load_graph(filename):
    return build_graph(load_packed_graph(filename))


def build_directed_signed_graph(graph_as_list):
    """return directed edges but undirected graph."""
    G, E = {}, {}
    for (u, v, s) in graph_as_list:
        E[(u, v)] = bool(s)
        add_edge(G, u, v)
    return G, E


def load_directed_signed_graph(filename):
    return build_directed_signed_graph(load_packed_graph(filename))

if __name__ == '__main__':
    import persistent as p
    # start = clock()
    # G, _ = p.load_var('twitter_triangle_graph.my')
    # print(clock()-start)
    # save_packed_graph([(k, tuple(v)) for k, v in G.items()], 'twitter.pack')
    # import sys
    # sys.exit()
    # do it once
    # save_packed_graph(read_text_graph('soc-sign-epinions.txt'), 'epi.pack')
    start = clock()
    G = load_graph('twitter.pack')
    nb_nodes = len(G)
    nb_edges = sum((len(adj) for adj in G.values())) // 2
    end = clock() - start
    print('loaded {} nodes and {} edges in {:.3f} seconds'.format(nb_nodes,
                                                                  nb_edges,
                                                                  end))
    start = clock()
    G, E = p.load_var('twitter_triangle_graph.my')
    end = clock() - start
    nb_nodes = len(G)
    nb_edges = sum((len(adj) for adj in G.values())) // 2
    print('loaded {} nodes and {} edges in {:.3f} seconds'.format(nb_nodes,
                                                                  nb_edges,
                                                                  end))
