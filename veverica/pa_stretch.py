#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Compute BFS and GTX tree on increasingly large PA graph to see whether they
.differ with respect to test edge stetch"""
from timeit import default_timer as clock

import numpy as np
from graph_tool.topology import label_largest_component, shortest_distance

import convert_experiment as cexp
import new_galaxy as ng
import pred_on_tree as pot


def load_wiki():
    import graph_tool as gt
    import real_world as rw
    graph_file = 'wiki_simple.gt'
    ds_file = 'wiki_dst.npy'
    k = gt.load_graph(graph_file)
    dst_mat = np.load(ds_file)
    lcc = label_largest_component(k)
    k.set_vertex_filter(lcc)
    lcc_nodes = np.where(lcc.a)[0]
    rw.read_original_graph('soc-wiki.txt')
    cexp.redensify.G = rw.G
    cexp.redensify.N = len(rw.G)
    cexp.redensify.EDGES_SIGN = rw.EDGE_SIGN
    return k, lcc_nodes, dst_mat


def make_graph(n):
    start = clock()
    cexp.preferential_attachment(n, m=3, gamma=1.05, c=.4,
                                 bonus_neighbor_prob=.13)
    k = cexp.to_graph_tool()
    lcc = label_largest_component(k)
    k.set_vertex_filter(lcc)
    lcc_nodes = np.where(lcc.a)[0]
    full_dst = shortest_distance(k, dense=False)
    full_mat = np.zeros((n, n), dtype=np.uint8)
    for v in k.vertices():
        full_mat[int(v), :] = full_dst[v].a.astype(np.uint8)
    del full_dst
    print('make_graph {:.3f}'.format(clock() - start))
    return k, lcc_nodes, full_mat


def compute_stretch(k, dst, edges, lcc_nodes):
    """Compute the stretch of all edges of `k` but those in the graph spanned
    by `edges`"""
    test_graph = {}
    slcc = set(lcc_nodes)
    k.set_vertex_filter(None)
    k.set_edge_filter(None)
    n = k.num_vertices()
    bfsmap = k.new_edge_property('boolean')
    for e in k.edges():
        u, v = int(e.source()), int(e.target())
        if (u, v) in edges:
            bfsmap[e] = True
        else:
            bfsmap[e] = False
            if u in slcc:
                pot.add_edge_to_tree(test_graph, u, v)
    k.set_edge_filter(bfsmap)

    tree_dst = shortest_distance(k, dense=False)
    tree_mat = np.zeros((n, n), dtype=np.uint8)
    for v in k.vertices():
        tree_mat[int(v), :] = tree_dst[v].a.astype(np.uint8)

    edge_paths, paths = [], []
    for i, v in enumerate(lcc_nodes):
        graph_distance = dst[v, lcc_nodes[i+1:]]
        tree_distance = tree_mat[v, lcc_nodes[i+1:]]
        if v in test_graph:
            edge_paths.extend(tree_mat[v, sorted(test_graph[v])].ravel())
        ratio = (tree_distance/graph_distance)
        paths.extend(ratio.ravel())
    prct = list(np.percentile(edge_paths, [25, 50, 75]))
    return prct + [np.mean(edge_paths)]


def process_graph(n):
    n_rep = 8
    k, lcc_nodes, full_mat = make_graph(n)
    # k, lcc_nodes, full_mat = load_wiki()
    degrees = sorted(((node, len(adj))
                      for node, adj in cexp.redensify.G.items()),
                     key=lambda x: x[1])
    roots = [_[0] for _ in reversed(degrees[-n_rep:])]
    stats = np.zeros((n_rep, 12))
    for i, root in enumerate(roots):
        start = clock()
        gtx, _ = ng.galaxy_maker(cexp.redensify.G, 10, short=True)
        stats[i, 4:8] = compute_stretch(k, full_mat, gtx, lcc_nodes)
        bfs = pot.get_bfs_tree(cexp.redensify.G, root)
        stats[i, :4] = compute_stretch(k, full_mat, bfs, lcc_nodes)
        dfs = pot.get_dfs_tree(cexp.redensify.G, root)
        stats[i, 8:] = compute_stretch(k, full_mat, dfs, lcc_nodes)
        print('one_rep {:.3f}'.format(clock() - start))
    return stats


if __name__ == '__main__':
    # pylint: disable=C0103
    from multiprocessing import Pool
    num_threads = 14
    per_thread = 1
    sizes = [2**i for i in range(9, 15)]
    for n in sizes:
        jobs = [n for task in range(num_threads*per_thread)]
        pool = Pool(num_threads)
        runs = list(pool.imap(process_graph, jobs, chunksize=per_thread))
        pool.close()
        pool.join()
        np.save('altexp/pa_stretch_{}'.format(n), np.vstack(runs))
