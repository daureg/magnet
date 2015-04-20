#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Compare performance of all our kind of tree with BFS"""
import real_world as rw
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from copy import deepcopy
import pred_on_tree as pot
import persistent
import redensify
import args_experiments as ae
SEEDS = list(range(6000, 6100))


def compute_one_seed(args):
    balanced = args.balanced
    num_k = 4
    num_method = 2 + 2 + 2*num_k
    acc = np.zeros((num_method))
    f1 = np.zeros_like(acc)
    mcc = np.zeros_like(acc)
    outname = 'universe/lp{}{}_{}{}'.format
    if balanced:
        outname = 'universe/lp_bal{}{}_{}{}'.format
    seed = args.seed
    print(seed)
    # rw.read_original_graph('soc-wiki.txt', seed=seed, balanced=balanced)
    ae.load_raw('universe/noiseLP', redensify, args)
    rw.G, rw.EDGE_SIGN = redensify.G, redensify.EDGES_SIGN
    if rw.DEGREES is None:
        rw.DEGREES = sorted(((node, len(adj))
                             for node, adj in rw.G.items()),
                            key=lambda x: x[1])
    num_e = len(rw.EDGE_SIGN)
    all_lcc_edges = {}
    lcc_tree = pot.get_bfs_tree(rw.G, rw.DEGREES[-1][0])
    assert all((e in rw.EDGE_SIGN for e in lcc_tree))
    heads, tails = zip(*lcc_tree)
    slcc = sorted(set(heads).union(set(tails)))
    for e, s in rw.EDGE_SIGN.items():
        u, v = e
        if u not in slcc:
            continue
        all_lcc_edges[(u, v)] = s

    edge_binary = {e: 2*int(s)-1 for e, s in rw.EDGE_SIGN.items()}
    bfs_edges = pot.get_bfs_tree(rw.G, rw.DEGREES[-1][0])
    print('BFS {:.1f}%'.format(100*(len(bfs_edges)/num_e)))
    bfs_tree = {}
    for u, v in bfs_edges:
        pot.add_edge_to_tree(bfs_tree, u, v)
    tags = pot.dfs_tagging(bfs_tree, edge_binary, rw.DEGREES[-1][0])
    gold, pred = pot.make_pred(bfs_tree, tags, edge_binary)
    a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
               matthews_corrcoef(gold, pred))
    acc[0], f1[0], mcc[0] = a, f, m

    basename = outname('', '_safe', seed, '')
    gold, pred = persistent.load_var(basename + '_res.my')
    print('stree {:.1f}%'.format(100*(1-len(pred)/num_e)))
    a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
               matthews_corrcoef(gold, pred))
    acc[1], f1[1], mcc[1] = a, f, m

    basename = outname('_short', '_safe', seed, '')
    gold, pred = persistent.load_var(basename + '_res.my')
    print('stree short {:.1f}%'.format(100*(1-len(pred)/num_e)))
    a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
               matthews_corrcoef(gold, pred))
    acc[2], f1[2], mcc[2] = a, f, m

    for k in range(num_k):
        basename = outname('', '', seed, '_'+str(k))
        gold, pred, _ = pot.predict_edges(basename, all_lcc_edges, slcc)
        print('utree {} {:.1f}%'.format(k+1, 100*(1-len(pred)/num_e)))
        a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
                   matthews_corrcoef(gold, pred))
        acc[3+2*k], f1[3+2*k], mcc[3+2*k] = a, f, m

        basename = outname('_short', '', seed, '_'+str(k))
        gold, pred, _ = pot.predict_edges(basename, all_lcc_edges, slcc)
        print('utree short {} {:.1f}%'.format(k+1,
                                              100*(1-len(pred)/num_e)))
        a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
                   matthews_corrcoef(gold, pred))
        acc[3+2*k+1], f1[3+2*k+1], mcc[3+2*k+1] = a, f, m

    dfs_edges = pot.get_dfs_tree(rw.G, rw.DEGREES[-1][0])
    print('DFS {:.1f}%'.format(100*(len(dfs_edges)/num_e)))
    dfs_tree = {}
    for u, v in dfs_edges:
        pot.add_edge_to_tree(dfs_tree, u, v)
    tags = pot.dfs_tagging(dfs_tree, edge_binary, rw.DEGREES[-1][0])
    gold, pred = pot.make_pred(dfs_tree, tags, edge_binary)
    a, f, m = (accuracy_score(gold, pred), f1_score(gold, pred),
               matthews_corrcoef(gold, pred))
    acc[-1], f1[-1], mcc[-1] = a, f, m

    return acc, f1, mcc

if __name__ == '__main__':
    # pylint: disable=C0103
    from multiprocessing import Pool
    from operator import itemgetter
    parser = ae.get_parser()
    args = parser.parse_args()
    balanced = args.balanced
    num_k = 4 if balanced else 4
    num_method = 1 + 2 + 2*num_k
    data = "lp"
    outname = 'universe/lp{}{}_{}{}'.format
    seeded_args = []
    for seed in SEEDS:
        targs = deepcopy(args)
        targs.seed = seed
        seeded_args.append(targs)
    if balanced:
        outname = 'universe/lp_bal{}{}_{}{}'.format

    num_threads = 2
    per_thread = len(SEEDS) // num_threads
    pool = Pool(num_threads)
    runs = list(pool.imap(compute_one_seed, seeded_args, chunksize=per_thread))
    pool.close()
    pool.join()
    acc = np.vstack(list(map(itemgetter(0), runs)))
    f1 = np.vstack(list(map(itemgetter(1), runs)))
    mcc = np.vstack(list(map(itemgetter(2), runs)))
    np.savez('{}{}_tree'.format(data, '_bal' if balanced else ''),
             acc=acc, f1=f1, mcc=mcc)
