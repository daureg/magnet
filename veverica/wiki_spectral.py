#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Compare Asym trained on a random set of edges with Asym trained on various
kind of trees"""
import real_world as rw
import numpy as np
import spectral_prediction as sp
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import pred_on_tree as pot
import persistent
import sys
SEEDS = list(range(6000, 6014))

if __name__ == '__main__':
    # pylint: disable=C0103
    acc = np.zeros((12, 14))
    f1 = np.zeros((12, 14))
    mcc = np.zeros((12, 14))
    outname = 'universe/wik{}{}_{}{}'.format
    for i, seed in enumerate(SEEDS):
        if i < 4:
            continue
        print(seed)
        rw.read_original_graph('soc-wiki.txt', seed=seed)
        sp.rw.G, sp.rw.EDGE_SIGN = rw.G, rw.EDGE_SIGN
        all_lcc_edges = {}
        lcc_tree = pot.get_bfs_tree(rw.G, rw.DEGREES[-1][0])
        assert all((e in rw.EDGE_SIGN for e in lcc_tree))
        heads, tails = zip(*lcc_tree)
        slcc = sorted(set(heads).union(set(tails)))
        mapping = {v: i for i, v in enumerate(slcc)}
        for e, s in rw.EDGE_SIGN.items():
            u, v = e
            if u not in slcc:
                continue
            all_lcc_edges[(u, v)] = s

        print('BFS')
        edge_binary = {e: 2*int(s)-1 for e, s in rw.EDGE_SIGN.items()}
        assert all((e in edge_binary for e in lcc_tree))
        bfs_tree = {}
        for u, v in lcc_tree:
            pot.add_edge_to_tree(bfs_tree, u, v)
        tags = pot.dfs_tagging(bfs_tree, edge_binary, rw.DEGREES[-1][0])
        gold, pred = pot.make_pred(bfs_tree, tags, edge_binary)
        a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
                   matthews_corrcoef(gold, pred))
        acc[0, i], f1[0, i], mcc[0, i] = a, f, m

        print('Asym BFS')
        adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                                 tree_edges=lcc_tree)
        a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
        print(a, f, m)
        acc[1, i], f1[1, i], mcc[1, i] = a, f, m

        print('utree')
        basename = outname('', '', seed, '_2')
        gold, pred, _ = pot.predict_edges(basename, all_lcc_edges, slcc)
        a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
                   matthews_corrcoef(gold, pred))
        acc[2, i], f1[2, i], mcc[2, i] = a, f, m

        print('Asym utree')
        _, gtx_tree = pot.read_tree(basename+'.edges')
        adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                                 tree_edges=gtx_tree)
        a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
        acc[3, i], f1[3, i], mcc[3, i] = a, f, m

        print('utree short')
        basename = outname('_short', '', seed, '_2')
        gold, pred, _ = pot.predict_edges(basename, all_lcc_edges, slcc)
        a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
                   matthews_corrcoef(gold, pred))
        acc[4, i], f1[4, i], mcc[4, i] = a, f, m

        print('Asym utree short')
        _, gtx_tree = pot.read_tree(basename+'.edges')
        adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                                 tree_edges=gtx_tree)
        a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
        acc[5, i], f1[5, i], mcc[5, i] = a, f, m

        pr = len(gtx_tree)/len(rw.EDGE_SIGN)
        print('Asym random {:.1f}'.format(pr))
        adj, test_edges = sp.get_training_matrix(pr, mapping, slcc)
        a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
        acc[6, i], f1[6, i], mcc[6, i] = a, f, m

        print('stree')
        basename = outname('', '_safe', seed, '')
        gold, pred = persistent.load_var(basename + '_res.my')
        a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
                   matthews_corrcoef(gold, pred))
        acc[7, i], f1[7, i], mcc[7, i] = a, f, m

        print('Asym stree')
        basename = outname('', '_safe', seed, '_2')
        _, gtx_tree = pot.read_tree(basename+'.edges')
        adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                                 tree_edges=gtx_tree)
        a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
        acc[8, i], f1[8, i], mcc[8, i] = a, f, m

        print('stree short')
        basename = outname('_short', '_safe', seed, '')
        gold, pred = persistent.load_var(basename + '_res.my')
        a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
                   matthews_corrcoef(gold, pred))
        acc[9, i], f1[9, i], mcc[9, i] = a, f, m

        print('Asym stree short')
        basename = outname('_short', '_safe', seed, '_2')
        _, gtx_tree = pot.read_tree(basename+'.edges')
        adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                                 tree_edges=gtx_tree)
        a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
        acc[10, i], f1[10, i], mcc[10, i] = a, f, m

        pr = len(gtx_tree)/len(rw.EDGE_SIGN)
        print('Asym random {:.1f}'.format(pr))
        adj, test_edges = sp.get_training_matrix(pr, mapping, slcc)
        a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
        acc[11, i], f1[11, i], mcc[11, i] = a, f, m

        np.savez('wiki_spectral_2', acc=acc, f1=f1, mcc=mcc)
