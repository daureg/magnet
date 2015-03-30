#! /usr/bin/python
# vim: set fileencoding=utf-8
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
import numpy as np
import redensify
import convert_experiment as cexp
from graph_tool.topology import shortest_distance
import graph_tool
import os
import pred_on_tree as pot
# import cc_pivot as cc
ROOTS = [12, 34, 39, 42, 43, 47, 95, 128, 136, 169, 170, 209, 225, 251, 280,
         280, 317, 349, 351, 369, 399, 406, 433, 458, 476, 484, 498, 499, 503,
         506, 543, 553, 561, 565, 579, 621, 666, 676, 687, 739, 789, 795, 843,
         847, 854, 875, 883, 907, 928, 975]


def get_graph():
    if os.path.isfile('universe/noise.gt'):
        g = graph_tool.load_graph('universe/noise.gt')
        dst_mat = np.load('universe/noise_dst.npy')
        cexp.to_python_graph(g)
        return g, dst_mat
    cexp.random_signed_communities(2, 500, 13, 11.5/500, .0, .0)
    g = cexp.to_graph_tool()
    n = g.num_vertices()
    dst = shortest_distance(g, dense=False)
    dst_mat = np.zeros((n, n), dtype=np.uint8)
    for v in g.vertices():
        dst_mat[int(v), :] = dst[v].a.astype(np.uint8)
    g.save('universe/noise.gt')
    np.save('universe/noise_dst', dst_mat)


def compute_trees():
    # import galaxy
    # galaxy.galaxy_maker_clean(redensify.G, 5, outname='universe/noise')
    assert len(redensify.G.keys()) == 1000
    trees = []
    for root in ROOTS:
        bfs_edges = pot.get_bfs_tree(redensify.G, root)
        tree_adjacency = {}
        for u, v in bfs_edges:
            pot.add_edge_to_tree(tree_adjacency, u, v)
        trees.append((tree_adjacency, bfs_edges))
    return trees


def compute_stretch(gt_graph, dst_mat, spanner_edges):
    n = gt_graph.num_vertices()
    train_edges = {(u, v) for u, v in spanner_edges}
    test_graph = {}
    for e in redensify.EDGES_SIGN.keys():
        if e not in train_edges:
            pot.add_edge_to_tree(test_graph, e[0], e[1])
    spannermap = gt_graph.new_edge_property('boolean')
    num_edges = 0
    for e in gt_graph.edges():
        u, v = int(e.source()), int(e.target())
        spannermap[e] = (u, v) in spanner_edges
        num_edges += 1
    print('{}, {:.1f}\% &'.format(len(spanner_edges),
                                  100*len(spanner_edges) / num_edges))
    gt_graph.set_edge_filter(spannermap)
    spanner_dst = shortest_distance(gt_graph, dense=False)
    gt_graph.set_edge_filter(None)
    spanner_mat = np.zeros((n, n), dtype=np.uint8)
    for v in gt_graph.vertices():
        spanner_mat[int(v), :] = spanner_dst[v].a.astype(np.uint8)
    tsum, tsize, esum, esize = 0, 0, 0, 0
    for v in range(n):
        graph_distance = dst_mat[v, v+1:]
        tree_distance = spanner_mat[v, v+1:]
        if v in test_graph:
            esum += spanner_mat[v, sorted(test_graph[v])].sum()
            esize += len(test_graph[v])
        ratio = (tree_distance/graph_distance)
        tsum += ratio.sum()
        tsize += ratio.shape[0]
    path_stretch = tsum/tsize
    edge_stretch = esum/esize
    return path_stretch, edge_stretch


def compute_prediction_galaxy(k, edge_signs):
    gold, pred, brute_pred = pot.predict_edges('universe/noise_'+str(k),
                                               all_signs=edge_signs,
                                               use_brute=False)
    # res = []
    # for p in pred, brute_pred:
    #     res.append((accuracy_score(gold, pred), f1_score(gold, pred),
    #                 matthews_corrcoef(gold, pred)))
    return (accuracy_score(gold, pred), f1_score(gold, pred),
            matthews_corrcoef(gold, pred))


def compute_prediction_bfs(tree, edge_signs):
    edge_binary = {e: 2*int(s)-1 for e, s in edge_signs.items()}
    tags = pot.dfs_tagging(tree, edge_binary, ROOTS[i])
    gold, pred = pot.make_pred(tree, tags, edge_binary)
    acc = accuracy_score(gold, pred)
    return acc, f1_score(gold, pred), matthews_corrcoef(gold, pred)


if __name__ == '__main__':
    import random
    gt_graph, dst_mat = get_graph()
    BFS = compute_trees()
    # bfs_stretch = np.zeros((len(BFS), 2))
    # for i, tree in enumerate(BFS):
    #     bfs_stretch[i, :] = compute_stretch(gt_graph, dst_mat, tree[1])
    # print(' & '.join(['{:.3f} ({:.3f})'.format(*_)
    #                   for _ in zip(np.mean(bfs_stretch, 0),
    #                                np.std(bfs_stretch, 0))]))
    # for k in range(3):
    #     basename = 'universe/noise_' + str(k)
    #     spanner_edges, _, _, _ = pot.read_spanner_from_file(basename)
    #     res = compute_stretch(gt_graph, dst_mat, spanner_edges)
    #     print(' & '.join(['{:.3f}'.format(m) for m in res]))
    # for e, s in redensify.EDGES_SIGN.items():
    #     redensify.EDGES_SIGN[e] = 1 if s else -1

    noise_level = [-1, 2, 4, 7, 10, 15, 20, 30, 40]
    # noise_level = [2, 40]
    for p in noise_level:
        p /= 100
        print(p)
        edge_signs = {}
        bfs_res = np.zeros((len(BFS), 3))
        for e, s in redensify.EDGES_SIGN.items():
            edge_signs[e] = not s if random.random() < p else s
        for i, tree in enumerate(BFS):
            bfs_res[i, :] = compute_prediction_bfs(tree[0], edge_signs)
        print(' & '.join(['{:.3f} ({:.3f})'.format(*l)
                          for l in zip(np.mean(bfs_res, 0),
                                       np.std(bfs_res, 0))]))
        for k in range(3):
            res = compute_prediction_galaxy(k, edge_signs)
            print(' & '.join(['{:.3f}'.format(o) for o in res]))



# 999, 13.5\%  & 2.004 (0.023) & 5.597 (0.068)
# 4174, 56.3\% & 1.243         & 3.497
# 1012, 13.6\% & 2.638         & 7.303
# 999, 13.5\%  & 2.678         & 7.417

# 1.000 (0.000) & 1.000 (0.000) & 1.000 (0.000)
# 1.000         & 1.000         & 1.000
# 1.000         & 1.000         & 1.000
# 1.000         & 1.000         & 1.000
# 0.02
# 0.872 (0.070) & 0.912 (0.052) & 0.681 (0.147)
# 0.913         & 0.944         & 0.744
# 0.876         & 0.915         & 0.689
# 0.876         & 0.915         & 0.689
# 0.04
# 0.762 (0.063) & 0.830 (0.051) & 0.448 (0.117)
# 0.799         & 0.865         & 0.486
# 0.730         & 0.803         & 0.393
# 0.729         & 0.803         & 0.392
# 0.07
# 0.664 (0.061) & 0.749 (0.053) & 0.265 (0.106)
# 0.739         & 0.817         & 0.374
# 0.532         & 0.621         & 0.059
# 0.532         & 0.622         & 0.060
# 0.1
# 0.614 (0.053) & 0.702 (0.049) & 0.179 (0.087)
# 0.677         & 0.766         & 0.260
# 0.643         & 0.728         & 0.232
# 0.643         & 0.728         & 0.231
# 0.15
# 0.552 (0.032) & 0.635 (0.033) & 0.077 (0.049)
# 0.583         & 0.680         & 0.096
# 0.523         & 0.603         & 0.035
# 0.524         & 0.604         & 0.036
# 0.2
# 0.521 (0.019) & 0.596 (0.021) & 0.029 (0.029)
# 0.540         & 0.627         & 0.050
# 0.503         & 0.579         & -0.007
# 0.503         & 0.579         & -0.007
# 0.3
# 0.503 (0.005) & 0.551 (0.006) & 0.006 (0.010)
# 0.510         & 0.565         & 0.013
# 0.511         & 0.553         & 0.023
# 0.510         & 0.553         & 0.022
# 0.4
# 0.499 (0.006) & 0.529 (0.007) & -0.001 (0.013)
# 0.492         & 0.518         & -0.014
# 0.500         & 0.523         & 0.002
# 0.500         & 0.523         & 0.002
