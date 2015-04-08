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


def compute_trees(seed=None):
    if seed:
        import galaxy
        galaxy.galaxy_maker_clean(redensify.G, 5,
                                  outname='universe/noise_{}'.format(seed))
        return None
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


def compute_prediction_galaxy(k, edge_signs, seed=None):
    basename = 'universe/noise_{}_{}'.format(seed, k)
    gold, pred, brute_pred = pot.predict_edges(basename,
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
    import real_world as rw
    import sys
    noise = int(sys.argv[1])
    # gt_graph, dst_mat = get_graph()

    def shuffle_nodes(seed):
        random.seed(seed)
        rperm = list(redensify.G.keys())
        random.shuffle(rperm)
        rperm = {i: v for i, v in enumerate(rperm)}
        _ = rw.reindex_nodes(redensify.G, redensify.EDGES_SIGN, rperm)
        redensify.G, redensify.EDGES_SIGN = _

    # BFS = compute_trees()
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

    # noise_level = [-1, 2, 4, 7, 10, 15, 20, 30, 40]
    n_rep = 50
    noise_level = [noise]
    seeds = [100*s + 57 for s in range(n_rep)]
    for p in noise_level:
        p /= 100
        print(p)
        edge_signs = {}
        # bfs_res = np.zeros((len(BFS), 3))
        for e, s in redensify.EDGES_SIGN.items():
            edge_signs[e] = not s if random.random() < p else s
        # for i, tree in enumerate(BFS):
        #     bfs_res[i, :] = compute_prediction_bfs(tree[0], edge_signs)
        # print(' & '.join(['{:.3f} ({:.3f})'.format(*l)
        #                   for l in zip(np.mean(bfs_res, 0),
        #                                np.std(bfs_res, 0))]))
        for s in seeds:
            get_graph()
            shuffle_nodes(s)
            compute_trees(s)
        for k in range(3):
            gtx_res = np.zeros((n_rep, 3))
            for i, s in enumerate(seeds):
                get_graph()
                shuffle_nodes(s)
                edge_signs = {}
                for e, sign in redensify.EDGES_SIGN.items():
                    edge_signs[e] = not sign if random.random() < p else sign
                gtx_res[i, :] = compute_prediction_galaxy(k, edge_signs, s)
            txt_res = (' & '.join(['{:.3f} ({:.3f})'.format(*l)
                                   for l in zip(np.mean(gtx_res, 0),
                                                np.std(gtx_res, 0))]))
            print('& $k={}$ & & {} & & \\\\'.format(k, txt_res))
