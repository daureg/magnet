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
PA = True
ROOTS_ER = [12, 34, 39, 42, 43, 47, 95, 128, 136, 169, 170, 209, 225, 251, 280,
            280, 317, 349, 351, 369, 399, 406, 433, 458, 476, 484, 498, 499,
            503, 506, 543, 553, 561, 565, 579, 621, 666, 676, 687, 739, 789,
            795, 843, 847, 854, 875, 883, 907, 928, 975]
ROOTS_PA = [71, 94, 76, 89, 113, 93, 41, 75, 157, 35, 95, 56, 36, 45, 25, 58,
            38, 53, 69, 79, 21, 33, 52, 43, 34, 26, 46, 47, 66, 31, 40, 28, 7,
            32, 29, 30, 22, 27, 39, 18, 17, 24, 5, 23, 20, 15, 16, 19, 14, 13]
BASENAME = 'universe/noise'
NUM_TRAIN_EDGES = 0
if PA:
    BASENAME += 'PA'
    ROOTS = ROOTS_PA
    K = 2
else:
    ROOTS = ROOTS_ER
    K = 3


def get_graph(balanced=False):
    """Load the graph from BASENAME and optionally remove positive edges to
    balance the graph. NOTE: this only modify redensify structure and not
    graph_tool & its distance matrix"""
    if balanced:
        import persistent
    if os.path.isfile(BASENAME+'.gt'):
        g = graph_tool.load_graph(BASENAME+'.gt')
        dst_mat = np.load(BASENAME+'_dst.npy')
        cexp.to_python_graph(g)
        if balanced:
            to_delete = persistent.load_var(BASENAME+'_balance.my')
            for edge in to_delete:
                pot.delete_edge(redensify.G, edge, redensify.EDGES_SIGN)
        return g, dst_mat
    if not PA:
        cexp.random_signed_communities(2, 500, 13, 11.5/500, .0, .0)
        g = cexp.to_graph_tool()
    else:
        cexp.preferential_attachment(1000, gamma=1.4, m=12)
        cexp.turn_into_signed_graph_by_propagation(2)
        DEGREES = sorted(((node, len(adj))
                          for node, adj in cexp.redensify.G.items()),
                         key=lambda x: x[1])
        u, v = DEGREES[-1][0], DEGREES[-2][0]
        u, v = v, u if u > v else u, v
        del cexp.redensify.EDGES_SIGN[(u, v)]
        cexp.redensify.G[u].remove(v)
        cexp.redensify.G[v].remove(u)
    n = g.num_vertices()
    dst = shortest_distance(g, dense=False)
    dst_mat = np.zeros((n, n), dtype=np.uint8)
    for v in g.vertices():
        dst_mat[int(v), :] = dst[v].a.astype(np.uint8)
    g.save(BASENAME+'.gt')
    np.save(BASENAME+'_dst', dst_mat)


def compute_trees(seed=None):
    if seed:
        import galaxy
        galaxy.galaxy_maker_clean(redensify.G, 5,
                                  outname=BASENAME+'_{}'.format(seed))
        return None
    assert len(redensify.G.keys()) == 1000, len(redensify.G.keys())
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
    global NUM_TRAIN_EDGES
    basename = BASENAME+'_{}_{}'.format(seed, k)
    spanner_edges, _, _, _ = pot.read_spanner_from_file(basename)
    train_edges = {(u, v) for u, v in spanner_edges}
    NUM_TRAIN_EDGES = len(train_edges)
    gold, pred, brute_pred = pot.predict_edges(basename,
                                               all_signs=edge_signs,
                                               use_brute=False)
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
    from copy import deepcopy
    noise = int(sys.argv[1])
    balanced = True
    gt_graph, dst_mat = get_graph(balanced=balanced)
    orig_g = deepcopy(redensify.G)
    orig_es = deepcopy(redensify.EDGES_SIGN)

    def shuffle_nodes(seed):
        random.seed(seed)
        rperm = list(orig_g.keys())
        random.shuffle(rperm)
        rperm = {i: v for i, v in enumerate(rperm)}
        _ = rw.reindex_nodes(orig_g, orig_es, rperm)
        redensify.G, redensify.EDGES_SIGN = _
        return rperm

    BFS = compute_trees()
    # bfs_stretch = np.zeros((len(BFS), 2))
    # for i, tree in enumerate(BFS):
    #     bfs_stretch[i, :] = compute_stretch(gt_graph, dst_mat, tree[1])
    # print(' & '.join(['{:.3f} ({:.3f})'.format(*_)
    #                   for _ in zip(np.mean(bfs_stretch, 0),
    #                                np.std(bfs_stretch, 0))]))
    # for k in range(3):
    #     basename = BASENAME+'_' + str(k)
    #     spanner_edges, _, _, _ = pot.read_spanner_from_file(basename)
    #     res = compute_stretch(gt_graph, dst_mat, spanner_edges)
    #     print(' & '.join(['{:.3f}'.format(m) for m in res]))
    # for e, s in redensify.EDGES_SIGN.items():
    #     redensify.EDGES_SIGN[e] = 1 if s else -1

    # noise_level = [-1, 2, 4, 7, 10, 15, 20, 30, 40]
    n_rep = 50
    n_noise_bfs_rep = 20
    n_noise_gtx_rep = n_noise_bfs_rep
    assert n_noise_bfs_rep == n_noise_gtx_rep
    edge_noises = []
    for i in range(n_noise_bfs_rep):
        p = noise/100
        edge_signs = {}
        for e, sign in orig_es.items():
            edge_signs[e] = not sign if random.random() < p else sign
        edge_noises.append(edge_signs)
    noise_level = [noise, ]
    seeds = [100*s + 57 for s in range(n_rep)]
    name = '\multirow{{4}}{{*}}{{{:.2f}}}'.format(noise/100)
    for p in noise_level:
        p /= 100
        print(p)
        bfs_res = np.zeros((n_noise_bfs_rep*len(BFS), 3))
        for j in range(n_noise_bfs_rep):
            edge_signs = edge_noises[j]
            for i, tree in enumerate(BFS):
                idx = j*len(BFS)+i
                bfs_res[idx, :] = compute_prediction_bfs(tree[0], edge_signs)
        txt_res = ' & '.join(['{:.3f} ({:.3f})'.format(*l)
                              for l in zip(np.mean(bfs_res, 0),
                                           np.std(bfs_res, 0))])
        print('{} & BFS & & {} & & \\\\'.format(name, txt_res))
        # continue
        nodes_mappings = []
        for s in seeds:
            get_graph(balanced=balanced)
            nodes_mappings.append(shuffle_nodes(s))
            compute_trees(s)
        for k in range(K):
            gtx_res = np.zeros((n_rep*n_noise_gtx_rep, 3))
            for i, s in enumerate(seeds):
                get_graph(balanced=balanced)
                _ = shuffle_nodes(s)
                assert _ == nodes_mappings[i], i
                for j in range(n_noise_gtx_rep):
                    _, edge_signs = rw.reindex_nodes({}, edge_noises[j],
                                                     nodes_mappings[i])
                    score = compute_prediction_galaxy(k, edge_signs, s)
                    idx = j*n_rep + i
                    gtx_res[idx, :] = score
            txt_res = (' & '.join(['{:.3f} ({:.3f})'.format(*l)
                                   for l in zip(np.mean(gtx_res, 0),
                                                np.std(gtx_res, 0))]))
            print('& $k={}$ & & {} & & \\\\'.format(k, txt_res))
            if noise == 2:
                fraction = NUM_TRAIN_EDGES/len(redensify.EDGES_SIGN)
                print('{}, {:.1f}%'.format(NUM_TRAIN_EDGES, fraction))
