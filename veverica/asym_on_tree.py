#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Compare performance of all our kind of tree as predicted by Asym"""
import real_world as rw
import numpy as np
import spectral_prediction as sp
from copy import deepcopy
import pred_on_tree as pot
import redensify
import args_experiments as ae
SEEDS = list(range(6000, 6090))


def compute_one_seed(args):
    balanced = args.balanced
    data = args.data.lower()
    dataname = ae.further_parsing(args)[0]
    last_k = 4 if data in ['lp'] else 3
    suffix = '_{}'.format(last_k-1)
    num_method = 2 + 2 + 2*2
    acc = np.zeros((num_method))
    f1 = np.zeros_like(acc)
    mcc = np.zeros_like(acc)
    names = []
    outname = ('universe/'+data+'{}{}_{}{}').format
    if balanced:
        outname = ('universe/'+data+'_bal{}{}_{}{}').format
    seed = args.seed
    print(seed)
    if dataname.startswith('soc'):
        rw.read_original_graph(dataname, seed=seed, balanced=balanced)
    if data == 'lp':
        ae.load_raw('universe/noiseLP', redensify, args)
        rw.G, rw.EDGE_SIGN = redensify.G, redensify.EDGES_SIGN
    if data == 'lr':
        pass
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
    mapping = {v: i for i, v in enumerate(slcc)}
    for e, s in rw.EDGE_SIGN.items():
        u, v = e
        if u not in slcc:
            continue
        all_lcc_edges[(u, v)] = s

    root = rw.DEGREES[-rw.r.randint(1, 100)][0]

    bfs_edges = pot.get_bfs_tree(rw.G, root)
    name = 'BFS {:.1f}%'.format(100*(len(bfs_edges)/num_e))
    names.append(name)
    adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                             tree_edges=bfs_edges)
    a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
    acc[0], f1[0], mcc[0] = a, f, m

    basename = outname('', '_safe', seed, suffix)
    _, gtx_tree = pot.read_tree(basename+'.edges')
    name = 'stree {:.1f}%'.format(100*len(gtx_tree)/num_e)
    names.append(name)
    adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                             tree_edges=gtx_tree)
    a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
    acc[1], f1[1], mcc[1] = a, f, m

    basename = outname('_short', '_safe', seed, suffix)
    _, gtx_tree = pot.read_tree(basename+'.edges')
    name = 'stree short {:.1f}%'.format(100*len(gtx_tree)/num_e)
    names.append(name)
    adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                             tree_edges=gtx_tree)
    a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
    acc[2], f1[2], mcc[2] = a, f, m

    for i, k in enumerate([1, last_k]):
        basename = outname('', '', seed, '_'+str(k-1))
        _, gtx_tree = pot.read_tree(basename+'.edges')
        name = 'utree {} {:.1f}%'.format(k, 100*len(gtx_tree)/num_e)
        names.append(name)
        adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                                 tree_edges=gtx_tree)
        a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
        acc[3+2*i], f1[3+2*i], mcc[3+2*i] = a, f, m

        basename = outname('_short', '', seed, '_'+str(k-1))
        _, gtx_tree = pot.read_tree(basename+'.edges')
        name = 'utree short {} {:.1f}%'.format(k, 100*len(gtx_tree)/num_e)
        names.append(name)
        adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                                 tree_edges=gtx_tree)
        a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
        acc[3+2*i+1], f1[3+2*i+1], mcc[3+2*i+1] = a, f, m

    dfs_edges = pot.get_dfs_tree(rw.G, root)
    name = 'DFS {:.1f}%'.format(100*(len(dfs_edges)/num_e))
    names.append(name)
    adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                             tree_edges=dfs_edges)
    a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
    acc[-1], f1[-1], mcc[-1] = a, f, m

    return acc, f1, mcc, names

if __name__ == '__main__':
    # pylint: disable=C0103
    from multiprocessing import Pool
    from operator import itemgetter
    parser = ae.get_parser()
    args = parser.parse_args()
    balanced = args.balanced
    data = args.data.lower()
    seeded_args = []
    for seed in SEEDS:
        targs = deepcopy(args)
        targs.seed = seed
        seeded_args.append(targs)

    num_threads = 14
    per_thread = len(SEEDS) // num_threads
    pool = Pool(num_threads)
    runs = list(pool.imap(compute_one_seed, seeded_args, chunksize=per_thread))
    pool.close()
    pool.join()
    acc = np.vstack(list(map(itemgetter(0), runs)))
    f1 = np.vstack(list(map(itemgetter(1), runs)))
    mcc = np.vstack(list(map(itemgetter(2), runs)))
    np.savez('altexp/{}{}_asym'.format(data,
                                       '_bal' if balanced else ''),
             acc=acc, f1=f1, mcc=mcc)
    names = runs[-1][-1]
    print('\n'.join(['{}{:.3f} ({:.3f})'.format(n.ljust(40),
                                                np.mean(mcc, 0)[i],
                                                np.std(mcc, 0)[i])
                     for i, n in enumerate(names)]))
