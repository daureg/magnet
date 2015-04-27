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
from glob import glob
SEEDS = list(range(6000, 6090))


def find_tree_filename(outname, kinds):
    """return the `outname` parametrized by `kinds` with the maximum number of
    iteration"""
    is_short, is_safe, seed = kinds
    suffix = "_*.edges"
    pattern = outname(is_short, is_safe, seed, suffix)
    candidates = sorted(glob(pattern))
    return candidates[-1][:-6]


def compute_one_seed(args):
    balanced = args.balanced
    only_random = args.random
    data = args.data.lower()
    dataname = ae.further_parsing(args)[0]
    num_method = 3 if only_random else 8
    acc = np.zeros((num_method))
    f1 = np.zeros_like(acc)
    mcc = np.zeros_like(acc)
    names = []
    outname = ('lp10/'+data+'{}{}_{}{}').format
    if balanced:
        outname = ('lp10/'+data+'_bal{}{}_{}{}').format
    seed = args.seed
    print(seed)
    if dataname.startswith('soc'):
        rw.read_original_graph(dataname, seed=seed, balanced=balanced)
    if data in ['lp', 'lr']:
        ae.load_raw('universe/noise'+data.upper(), redensify, args)
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
    mapping = {v: i for i, v in enumerate(slcc)}
    for e, s in rw.EDGE_SIGN.items():
        u, v = e
        if u not in slcc:
            continue
        all_lcc_edges[(u, v)] = s

    root = rw.DEGREES[-rw.r.randint(1, 100)][0]

    if not only_random:
        bfs_edges = pot.get_bfs_tree(rw.G, root)
        name = 'BFS {:.1f}%'.format(100*(len(bfs_edges)/num_e))
        names.append(name)
        adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                                 tree_edges=bfs_edges)
        a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
        acc[0], f1[0], mcc[0] = a, f, m

    if args.data in ['SLA', 'EPI', 'LR']:
        names.append('stree')
        if not only_random:
            names.append('stree short')
    else:
        basename = find_tree_filename(outname, ('', '_safe', seed))
        _, gtx_tree = pot.read_tree(basename+'.edges')
        fraction = len(gtx_tree)/num_e
        name = '{} {:.1f}%'.format('Asym' if only_random else 'stree',
                                   100*fraction)
        names.append(name)
        if only_random:
            adj, test_edges = sp.get_training_matrix(fraction, mapping, slcc)
        else:
            adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                                     tree_edges=gtx_tree)
        a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
        index = 0 if only_random else 1
        acc[index], f1[index], mcc[index] = a, f, m

        if not only_random:
            basename = find_tree_filename(outname, ('_short', '_safe', seed))
            _, gtx_tree = pot.read_tree(basename+'.edges')
            name = 'stree short {:.1f}%'.format(100*len(gtx_tree)/num_e)
            names.append(name)
            adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                                     tree_edges=gtx_tree)
            a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
            acc[2], f1[2], mcc[2] = a, f, m

    for i, k in enumerate(['1', 'last']):
        if i == 0:
            basename = outname('', '', seed, '_0')
        else:
            basename = find_tree_filename(outname, ('', '', seed))
        _, gtx_tree = pot.read_tree(basename+'.edges')
        fraction = len(gtx_tree)/num_e
        fname = 'Asym' if only_random else 'utree ' + k
        name = '{} {:.1f}%'.format(fname, 100*fraction)
        names.append(name)
        if only_random:
            adj, test_edges = sp.get_training_matrix(fraction, mapping, slcc)
        else:
            adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                                     tree_edges=gtx_tree)
        a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
        index = i+1 if only_random else (3 + 2*i)
        acc[index], f1[index], mcc[index] = a, f, m

        if only_random:
            continue

        if i == 0:
            basename = outname('_short', '', seed, '_0')
        else:
            basename = find_tree_filename(outname, ('_short', '', seed))
        _, gtx_tree = pot.read_tree(basename+'.edges')
        name = 'utree short {} {:.1f}%'.format(k, 100*len(gtx_tree)/num_e)
        names.append(name)
        adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                                 tree_edges=gtx_tree)
        a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
        acc[3+2*i+1], f1[3+2*i+1], mcc[3+2*i+1] = a, f, m

    if not only_random:
        dfs_edges = pot.get_dfs_tree(rw.G, root)
        name = 'DFS {:.1f}%'.format(100*(len(dfs_edges)/num_e))
        names.append(name)
        adj, test_edges = sp.get_training_matrix(-5, mapping, slcc,
                                                 tree_edges=dfs_edges)
        a, f, m = sp.predict_edges(adj, 15, mapping, test_edges)
        acc[7], f1[7], mcc[7] = a, f, m

    return acc, f1, mcc, names

if __name__ == '__main__':
    # pylint: disable=C0103
    from multiprocessing import Pool
    from operator import itemgetter
    parser = ae.get_parser()
    parser.add_argument("-r", "--random", action='store_true',
                        help="only compute Asym on random edges")
    args = parser.parse_args()
    only_random = args.random
    balanced = args.balanced
    data = args.data.lower()
    seeded_args = []
    upper = 6 if data == 'epi' and not only_random else 90
    for seed in SEEDS[:upper]:
        targs = deepcopy(args)
        targs.seed = seed
        seeded_args.append(targs)

    num_threads = 15
    per_thread = len(SEEDS[:upper]) // num_threads
    pool = Pool(num_threads)
    runs = list(pool.imap(compute_one_seed, seeded_args, chunksize=per_thread))
    pool.close()
    pool.join()
    acc = np.vstack(list(map(itemgetter(0), runs)))
    f1 = np.vstack(list(map(itemgetter(1), runs)))
    mcc = np.vstack(list(map(itemgetter(2), runs)))
    np.savez('altexp10/{}{}_asym_{}'.format(data,
                                            '_bal' if balanced else '',
                                            '_part' if only_random else ''),
             acc=acc, f1=f1, mcc=mcc)
    names = runs[-1][-1]
    print('\n'.join(['{}{:.3f} ({:.3f})'.format(n.ljust(40),
                                                np.mean(mcc, 0)[i],
                                                np.std(mcc, 0)[i])
                     for i, n in enumerate(names)]))
