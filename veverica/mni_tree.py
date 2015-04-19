#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Compare performance of all our kind of tree with BFS"""
import real_world as rw
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import pred_on_tree as pot
import persistent
import redensify
import args_experiments as ae
SEEDS = list(range(6000, 6100))

if __name__ == '__main__':
    # pylint: disable=C0103
    parser = ae.get_parser()
    args = parser.parse_args()
    balanced = args.balanced
    num_k = 4 if balanced else 4
    num_method = 2 + 2 + 2*num_k
    acc = np.zeros((num_method, len(SEEDS)))
    f1 = np.zeros_like(acc)
    mcc = np.zeros_like(acc)
    data = "wik"
    outname = 'universe/wik{}{}_{}{}'.format
    if balanced:
        outname = 'universe/wik_bal{}{}_{}{}'.format
    for i, seed in enumerate(SEEDS):
        args.seed = seed
        print(seed)
        rw.read_original_graph('soc-wiki.txt', seed=seed, balanced=balanced)
        # ae.load_raw('universe/noiseLP', redensify, args)
        # rw.G, rw.EDGE_SIGN = redensify.G, redensify.EDGES_SIGN
        # if rw.DEGREES is None:
        #     rw.DEGREES = sorted(((node, len(adj))
        #                          for node, adj in rw.G.items()),
        #                         key=lambda x: x[1])
        num_e = len(rw.EDGE_SIGN)
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

        edge_binary = {e: 2*int(s)-1 for e, s in rw.EDGE_SIGN.items()}
        bfs_edges = pot.get_bfs_tree(rw.G, rw.DEGREES[-(i+1)][0])
        print('BFS {:.1f}%'.format(100*(len(bfs_edges)/num_e)))
        bfs_tree = {}
        for u, v in bfs_edges:
            pot.add_edge_to_tree(bfs_tree, u, v)
        tags = pot.dfs_tagging(bfs_tree, edge_binary, rw.DEGREES[-1][0])
        gold, pred = pot.make_pred(bfs_tree, tags, edge_binary)
        a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
                   matthews_corrcoef(gold, pred))
        acc[0, i], f1[0, i], mcc[0, i] = a, f, m


        dfs_edges = pot.get_dfs_tree(rw.G, rw.DEGREES[-(i+1)][0])
        print('DFS {:.1f}%'.format(100*(len(dfs_edges)/num_e)))
        dfs_tree = {}
        for u, v in dfs_edges:
            pot.add_edge_to_tree(dfs_tree, u, v)
        tags = pot.dfs_tagging(dfs_tree, edge_binary, rw.DEGREES[-1][0])
        gold, pred = pot.make_pred(dfs_tree, tags, edge_binary)
        a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
                   matthews_corrcoef(gold, pred))
        acc[-1, i], f1[-1, i], mcc[-1, i] = a, f, m
        continue

        basename = outname('', '_safe', seed, '')
        gold, pred = persistent.load_var(basename + '_res.my')
        print('stree {:.1f}%'.format(100*(1-len(pred)/num_e)))
        a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
                   matthews_corrcoef(gold, pred))
        acc[1, i], f1[1, i], mcc[1, i] = a, f, m

        basename = outname('_short', '_safe', seed, '')
        gold, pred = persistent.load_var(basename + '_res.my')
        print('stree short {:.1f}%'.format(100*(1-len(pred)/num_e)))
        a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
                   matthews_corrcoef(gold, pred))
        acc[2, i], f1[2, i], mcc[2, i] = a, f, m

        for k in range(num_k):
            basename = outname('', '', seed, '_'+str(k))
            gold, pred, _ = pot.predict_edges(basename, all_lcc_edges, slcc)
            print('utree {} {:.1f}%'.format(k+1, 100*(1-len(pred)/num_e)))
            a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
                       matthews_corrcoef(gold, pred))
            acc[3+2*k, i], f1[3+2*k, i], mcc[3+2*k, i] = a, f, m

            basename = outname('_short', '', seed, '_'+str(k))
            gold, pred, _ = pot.predict_edges(basename, all_lcc_edges, slcc)
            print('utree short {} {:.1f}%'.format(k+1,
                                                  100*(1-len(pred)/num_e)))
            a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
                       matthews_corrcoef(gold, pred))
            acc[3+2*k+1, i], f1[3+2*k+1, i], mcc[3+2*k+1, i] = a, f, m
        np.savez('{}{}_tree_classic'.format(data, '_bal' if balanced else ''),
                 acc=acc, f1=f1, mcc=mcc)
