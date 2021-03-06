# vim: set fileencoding=utf-8
"""Perform various experiments using the shazoo code."""
import logging
import sys
import time
from collections import defaultdict, deque
from itertools import product, repeat
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm, trange

import persistent
import shazoo as sz
from grid_stretch import add_edge
from nxmst import kruskal_mst_edges
from shazoo_precomputed_randomness import DataProvider
from wta import predict_signs as wta_predict
from wta import linearize_tree

if sz.USE_SCIPY or sys.version_info.major == 3:
    from shazoo_scipy import run_labprop, get_weight_matrix
else:
    def run_labprop(gold_signs, test_set, *args):
        return np.sign(2*np.random.random(len(test_set))-1)

    def get_weight_matrix(*args):
        return None, None
NUM_THREADS = 15
PARAMS_ABC = None


def get_perturb_proba(degrees, p0):
    """Assign a probability of being pertubed proportional to degree in original graph"""
    if p0 > 1:
        p0 /= 100
    if p0 < 1e-5:
        return np.zeros(degrees.shape, dtype=float)
    low_deg = degrees <= degrees.mean()
    high_deg = np.logical_not(low_deg)
    pi = np.zeros_like(degrees, dtype=float)
    pi[low_deg] = p0*((degrees[low_deg] - 0)/(degrees[low_deg].max() - 0))
    d_range = degrees[high_deg].max() - degrees[high_deg].min()
    pi[high_deg] = p0*((degrees[high_deg] - degrees[high_deg].min())/(d_range))+p0
    return pi * (p0/pi.mean())


def compute_phi(edge_weight, gold):
    return sum(1 for u, v in edge_weight if gold[u] != gold[v])


def make_graph(n, tree=True, p=.04):
    """Get a synthetic binary classification problem.

    It returns a preferential attachment graph with `n` nodes (that can be a
    `tree`) and assign sign by propagating labels from a few seeds (flipping
    the results with probability `p`)
    """
    res = sz.make_graph(n, tree, p)
    adj, _, ew, _, signs, gold = res[0]
    return adj, ew, gold, res[1]


def load_real_graph(dataset='citeseer', main_class=None):
    default_main_class = {'citeseer': 1,
                          'cora': 2,
                          'pubmed_core': 1,
                          'usps4500': 4,
                          'rcv1': 2,
                          'imdb': 0,
                          }
    main_class = main_class if main_class is not None else default_main_class[dataset]
    ew, y = persistent.load_var('{}_lcc.my'.format(dataset))
    adj = {}
    for u, v in ew:
        add_edge(adj, u, v)
    gold = {i: 1 if v == main_class else -1 for i, v in enumerate(y)}
    return adj, ew, gold, compute_phi(ew, gold)


def online_repetition_exps(num_rep=2, num_run=13):
    """Check whether majority vote reduces mistakes."""
    exp_start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60
    res_file = 'shazoo_run_{}.npz'.format(exp_start)
    res = np.zeros((num_rep, num_run+1, 3))
    for i in range(num_rep):
        keep_preds = defaultdict(list)
        adj, ew, gold_signs, num_phi = make_graph(600)
        start = sz.clock()
        for k in range(num_run):
            gold, preds = sz.threeway_batch_shazoo(adj, ew, {}, gold_signs)
            for j, (method, pred) in enumerate(sorted(preds.items(), key=lambda x: x[0])):
                mistakes = sum((1 for g, p in zip(gold, pred) if p != g))
                res[i, k, j] = mistakes
                keep_preds[method].append(list(pred))
                print('{} made {} mistakes'.format(method.ljust(6), mistakes))
                np.savez_compressed(res_file, res=res)
        time_elapsed = sz.clock() - start
        print(time_elapsed/num_run)
        for j, (method, preds) in enumerate(sorted(keep_preds.items(), key=lambda x: x[0])):
            pred = sz.majority_vote(preds)
            mistakes = sum((1 for g, p in zip(gold, pred) if p != g))
            res[i, -1, j] = mistakes
            print('{} made {} mistakes'.format(method.ljust(6), mistakes))
            np.savez_compressed(res_file, res=res)


def gamma_rate(dataset, part, machines):
    exp_start = (int(time.time()-(2017-1970)*365.25*24*60*60))//60
    res_file = 'tlinear_shazoo_{}_{}_{}.npz'.format(dataset, exp_start, part)
    dp = DataProvider(dataset, part, machines, max_trees=2)
    gold_signs = dp.gold_signs
    chunk_size = 2
    num_order = NUM_THREADS*chunk_size
    train_fraction = .1
    n_train_sets, n_trees, n_flips = 10, 1, 1
    n_gammas = 20
    gamma_mul = np.linspace(0.4, 1.25, n_gammas)
    test_size = gold_signs.size - int(train_fraction*dp.gold.size)
    rta_mst = np.zeros((n_train_sets, n_gammas, num_order, test_size), dtype=int)
    shz_mst = np.zeros((n_train_sets, n_gammas, 1, test_size), dtype=int)
    golds = np.zeros((n_train_sets, test_size))
    lprop = np.zeros_like(golds)
    weights, inv_degree = get_weight_matrix(None, dataset)
    pool = Pool(NUM_THREADS)
    train_gen = dp.training_sets(train_fraction, count=n_train_sets, split_across_machines=False)
    mst_gen = dp.mst_trees(count=n_trees, split_across_machines=False)
    flip_gen = dp.perturbed_sets(35, count=n_flips, split_across_machines=False)
    for i, (trains, trees, flips) in enumerate(tqdm(product(train_gen, mst_gen, flip_gen),
                                                    desc='params', unit='params', total=n_train_sets)):
        sorted_train_set, sorted_test_set, sorted_gold = trains
        batch_orders = []
        for bo in dp.batch_sequences(train_fraction, count=num_order, split_across_machines=True):
            batch_orders.append([sorted_train_set[u] for u in bo])
        adj, ew = trees
        perturbed_gold = flips
        pertubed_test_gold = [perturbed_gold[u] for u in sorted_test_set]
        golds[i, :] = pertubed_test_gold
        lprop_pred = run_labprop(perturbed_gold, sorted_test_set, sorted_train_set, weights,
                                 inv_degree)
        lprop[i, :] = lprop_pred
        for j, gm in enumerate(tqdm(gamma_mul, desc='gamma', unit='gamma')):
            args = zip(repeat(adj, num_order), batch_orders, repeat(ew, num_order),
                       repeat(gold_signs, num_order), repeat(perturbed_gold, num_order),
                       repeat(sorted_test_set, num_order), repeat(gm, num_order))
            runs = pool.imap_unordered(run_once, args, chunk_size)
            for k, lres in enumerate(runs):
                rta_mst[i, j, k, :] = lres[1][2]
                shz_mst[i, j, 0, :] = lres[2][2]
            np.savez_compressed(res_file, golds=golds, lprop=lprop, rta_mst=rta_mst,
                                shz_mst=shz_mst)
    pool.close()
    pool.join()


def single_tree(dataset, source, part, machines):
    exp_start = (int(time.time()-(2017-1970)*365.25*24*60*60))//60
    res_file = 'var_{}_shazoo_{}_{}_{}.npz'.format(source, dataset, exp_start, part)
    perturbations = np.array([0, 2.5, 5, 10, 20]) / 100
    dp = DataProvider(dataset, part, machines, max_trees=10)
    gold_signs = dp.gold_signs
    chunk_size = 3
    num_order = NUM_THREADS*chunk_size
    train_fraction = .1
    n_train_sets, n_trees, n_flips = 1, 1, 1
    if source == 'train':
        n_train_sets = 10
    if source == 'tree':
        n_trees = 10
    if source == 'flip':
        n_flips = 10
    test_size = gold_signs.size - int(train_fraction*dp.gold.size)
    rta = np.zeros((len(perturbations), n_train_sets, n_trees, n_flips, num_order, test_size),
                   dtype=int)
    rta_mst = np.zeros_like(rta)
    shz = np.zeros((len(perturbations), n_train_sets, n_trees, n_flips, 1, test_size), dtype=int)
    shz_mst = np.zeros_like(shz)
    wta = np.zeros_like(shz)
    wta_mst = np.zeros_like(shz)
    golds = np.zeros_like(shz)
    lprop = np.zeros_like(shz)
    weights, inv_degree = get_weight_matrix(None, dataset)
    pool = Pool(NUM_THREADS)
    total = n_train_sets*n_trees*n_flips
    for i, pfrac in enumerate(tqdm(perturbations, desc='perturbation', unit='flip')):
        train_gen = dp.training_sets(train_fraction, count=n_train_sets, split_across_machines=False)
        rst_gen = dp.rst_trees(count=n_trees, split_across_machines=False)
        flip_gen = dp.perturbed_sets(pfrac, count=n_flips, split_across_machines=False)
        for j, (trains, trees, flips) in enumerate(tqdm(product(train_gen, rst_gen, flip_gen),
                                                        desc='params', unit='params', total=total)):
            indices = [i, j if source == 'train' else 0,
                       j if source == 'tree' else 0,
                       j if source == 'flip' else 0, 0, slice(None)]
            cindices = tuple(indices)
            sorted_train_set, sorted_test_set, sorted_gold = trains
            batch_orders = []
            for bo in dp.batch_sequences(train_fraction, count=num_order, split_across_machines=True):
                batch_orders.append([sorted_train_set[u] for u in bo])
            adj, ew = trees
            perturbed_gold = flips
            pertubed_test_gold = [perturbed_gold[u] for u in sorted_test_set]
            wta_training_set = {u: perturbed_gold[u] for u in sorted_train_set}
            golds[cindices] = pertubed_test_gold
            # irrelevant over trees but ok it doesn't matter
            lprop_pred = run_labprop(perturbed_gold, sorted_test_set, sorted_train_set, weights,
                                     inv_degree)
            lprop[cindices] = lprop_pred
            args = zip(repeat(adj, num_order), batch_orders, repeat(ew, num_order),
                       repeat(gold_signs, num_order), repeat(perturbed_gold, num_order),
                       repeat(sorted_test_set, num_order), repeat(None, num_order))
            nodes_line, line_weight = linearize_tree(adj, ew, dp.bfs_root)
            pred = wta_predict(nodes_line, line_weight, wta_training_set)
            wta[cindices] = [pred[u] for u in sorted_test_set]
            runs = pool.imap_unordered(run_once, args, chunk_size)
            for k, lres in enumerate(runs):
                indices[4] = k
                rta[tuple(indices)] = lres[1][2]
                shz[cindices] = lres[2][2]
            np.savez_compressed(res_file, golds=golds, lprop=lprop, rta=rta, rta_mst=rta_mst,
                                shz=shz, shz_mst=shz_mst, wta=wta, wta_mst=wta_mst)

    for i, pfrac in enumerate(tqdm(perturbations, desc='perturbation', unit='flip')):
        train_gen = dp.training_sets(train_fraction, count=n_train_sets, split_across_machines=False)
        mst_gen = dp.mst_trees(count=n_trees, split_across_machines=False)
        flip_gen = dp.perturbed_sets(pfrac, count=n_flips, split_across_machines=False)
        for j, (trains, trees, flips) in enumerate(tqdm(product(train_gen, mst_gen, flip_gen),
                                                        desc='params', unit='params', total=total)):
            indices = [i, j if source == 'train' else 0,
                       j if source == 'tree' else 0,
                       j if source == 'flip' else 0, 0, slice(None)]
            cindices = tuple(indices)
            sorted_train_set, sorted_test_set, sorted_gold = trains
            batch_orders = []
            for bo in dp.batch_sequences(train_fraction, count=num_order, split_across_machines=True):
                batch_orders.append([sorted_train_set[u] for u in bo])
            adj, ew = trees
            perturbed_gold = flips
            pertubed_test_gold = [perturbed_gold[u] for u in sorted_test_set]
            wta_training_set = {u: perturbed_gold[u] for u in sorted_train_set}
            args = zip(repeat(adj, num_order), batch_orders, repeat(ew, num_order),
                       repeat(gold_signs, num_order), repeat(perturbed_gold, num_order),
                       repeat(sorted_test_set, num_order), repeat(None, num_order))
            nodes_line, line_weight = linearize_tree(adj, ew, dp.bfs_root)
            pred = wta_predict(nodes_line, line_weight, wta_training_set)
            wta_mst[cindices] = [pred[u] for u in sorted_test_set]
            runs = pool.imap_unordered(run_once, args, chunk_size)
            for k, lres in enumerate(runs):
                indices[4] = k
                rta_mst[tuple(indices)] = lres[1][2]
                shz_mst[cindices] = lres[2][2]
            np.savez_compressed(res_file, golds=golds, lprop=lprop, rta=rta, rta_mst=rta_mst,
                                shz=shz, shz_mst=shz_mst, wta=wta, wta_mst=wta_mst)
    pool.close()
    pool.join()


def lots_of_tree(dataset='imdb', part=0):
    exp_start = (int(time.time()-(2017-1970)*365.25*24*60*60))//60
    res_file = 'tree_shazoo_{}_{}_{}.npz'.format(dataset, exp_start, part)
    g_adj, g_ew, gold_signs, phi = load_real_graph(dataset)
    n = len(g_adj)
    train_fraction = .1
    num_trees = 28
    nodes_id = set(range(n))
    pool = Pool(NUM_THREADS)
    z = list(range(n))
    sz.random.shuffle(z)
    train_set = {u: gold_signs[u] for u in z[:int(train_fraction * n)]}
    test_set = nodes_id - set(train_set)
    sorted_test_set = sorted(test_set)
    sorted_train_set = sorted(train_set)
    sorted_gold = [gold_signs[u] for u in sorted_test_set]
    batch_order = []
    chunk_size = 9
    num_batch_order = NUM_THREADS*chunk_size
    res = np.zeros((num_trees, num_batch_order, len(sorted_test_set)), dtype=int)
    z = list(range(len(train_set)))
    for _ in range(num_batch_order):
        sz.random.shuffle(z)
        batch_order.append([sorted_train_set[u] for u in z])
    sz.GRAPH, sz.EWEIGHTS = g_adj, g_ew
    p = .1
    probas = (p/100.0)*np.ones(n)
    perturbed_gold = {u: (1 if sz.random.random() >= probas[u] else -1)*s
                      for u, s in sz.iteritems(gold_signs)}
    sorted_perturbed_gold = [perturbed_gold[u] for u in sorted_test_set]
    num_order = len(batch_order)
    for j in trange(num_trees, desc='tree', unit='tree', unit_scale=True):
        sz.get_rst(None)
        adj, ew = sz.TREE_ADJ, sz.TWEIGHTS
        args = zip(repeat(adj, num_order), batch_order, repeat(ew, num_order),
                   repeat(gold_signs, num_order), repeat(perturbed_gold, num_order),
                   repeat(sorted_test_set, num_order), repeat(None, num_order))
        runs = pool.imap_unordered(run_once, args, chunk_size)
        for i, lres in enumerate(runs):
            res[j, i, :] = lres[1][2]
        np.savez_compressed(res_file, res=res, gold=sorted_gold)
    pool.close()
    pool.join()


def linear_rta(dataset='imdb', part=0):
    global PARAMS_ABC
    exp_start = (int(time.time()-(2017-1970)*365.25*24*60*60))//60
    res_file = 'linear_rta_{}_{}_{}.npz'.format(dataset, exp_start, part)
    g_adj, g_ew, gold_signs, phi = load_real_graph(dataset)
    n = len(g_adj)
    train_fraction = .1
    num_trees, num_exps = 9, 85
    nodes_id = set(range(n))
    pool = Pool(NUM_THREADS)
    z = list(range(n))
    sz.random.shuffle(z)
    train_set = {u: gold_signs[u] for u in z[:int(train_fraction * n)]}
    test_set = nodes_id - set(train_set)
    sorted_test_set = sorted(test_set)
    sorted_train_set = sorted(train_set)
    sorted_gold = [gold_signs[u] for u in sorted_test_set]
    batch_order = []
    num_batch_order = NUM_THREADS
    res = np.zeros((num_exps, 4))
    gen = np.random.uniform
    # params = [(gen(1/4, 9/4), gen(1/4, 9/4), gen(-1/4, 1/4)) for _ in range(num_exps)]
    params = [(.6, 1.7, -.1) for _ in range(num_exps)]
    methods = ['l2cost', 'rta', 'shazoo']
    z = list(range(len(train_set)))
    for _ in range(num_batch_order):
        sz.random.shuffle(z)
        batch_order.append([sorted_train_set[u] for u in z])
    sz.GRAPH, sz.EWEIGHTS = g_adj, g_ew
    p = .1
    probas = (p/100.0)*np.ones(n)
    perturbed_gold = {u: (1 if sz.random.random() >= probas[u] else -1)*s
                      for u, s in sz.iteritems(gold_signs)}
    sorted_perturbed_gold = [perturbed_gold[u] for u in sorted_test_set]
    for i, row in enumerate(tqdm(params, desc='Pabc', unit='params')):
        PARAMS_ABC = tuple(row)
        lres, w = aggregate_trees(batch_order, (g_adj, g_ew, None), gold_signs, methods,
                                  num_trees, perturbed_gold, pool, sorted_gold,
                                  sorted_perturbed_gold, sorted_test_set, run_wta=False)
        res[i, :] = lres[1] + w
        np.savez_compressed(res_file, res=res, params=params)
    pool.close()
    pool.join()


def real_exps(num_tree=2, num_batch_order=15, train_fraction=.2, dataset='citeseer', part=0):
    exp_start = (int(time.time()-(2017-1970)*365.25*24*60*60))//60
    dbg_fmt = '%(asctime)s - %(relativeCreated)d:%(filename)s.%(funcName)s.%(threadName)s:%(lineno)d(%(levelname)s):%(message)s '
    # logging.basicConfig(filename='shazoo_{}.log'.format(exp_start), level=logging.DEBUG, format=dbg_fmt)
    logging.info('Started')
    res_file = 'shazoo_{}_{}_{}.npz'.format(dataset, exp_start, part)
    # perturbations = [0, 2.5, 5, 10, 20]
    perturbations = [2.5, 5]
    # train_size = np.array([2.5, 5, 10, 20, 40]) / 100
    train_size = np.array([2.5,]) / 100
    nrep = 30
    res = np.zeros((len(train_size), len(perturbations), nrep, 3, 2))
    res_mst = np.zeros_like(res)
    phis = np.zeros((len(train_size), len(perturbations), nrep))
    lprop_res = np.zeros((len(train_size), len(perturbations), nrep, 2))
    wta_res = np.zeros_like(lprop_res)
    wta_res_mst = np.zeros_like(wta_res)
    g_adj, g_ew, gold_signs, phi = load_real_graph(dataset)
    weights, inv_degree = get_weight_matrix(g_ew, dataset)
    n = len(g_adj)
    degrees = np.array([len(g_adj[u]) for u in range(n)])
    bfs_root = max(g_adj.items(), key=lambda x: len(x[1]))[0]
    nodes_id = set(range(n))
    methods = sorted(['shazoo', 'rta', 'l2cost'])
    pool = Pool(NUM_THREADS)
    for rep_id in trange(nrep, desc='rep.', unit='rep.'):
        logging.info('Starting rep %d', rep_id+1)
        for j, train_fraction in enumerate(tqdm(train_size, desc='tr.size', unit='trained')):
            logging.info('Starting train_fraction %d %%', train_fraction*100)
            z = list(range(n))
            sz.random.shuffle(z)
            train_set = {u: gold_signs[u] for u in z[:int(train_fraction * n)]}
            test_set = nodes_id - set(train_set)
            sorted_test_set = sorted(test_set)
            sorted_train_set = sorted(train_set)
            sorted_gold = [gold_signs[u] for u in sorted_test_set]
            batch_order = []
            z = list(range(len(train_set)))
            for _ in range(num_batch_order):
                sz.random.shuffle(z)
                batch_order.append([sorted_train_set[u] for u in z])
            for ip, p in enumerate(tqdm(perturbations, desc='perturbation', unit='flip')):
                logging.info('Starting perturbation %d %%', p)
                # probas = get_perturb_proba(degrees, p/100.0)
                probas = (p/100.0)*np.ones(degrees.size)
                perturbed_gold = {u: (1 if sz.random.random() >= probas[u] else -1)*s
                                  for u, s in sz.iteritems(gold_signs)}
                sorted_perturbed_gold = [perturbed_gold[u] for u in sorted_test_set]
                phis[j, ip, rep_id] = compute_phi(g_ew, perturbed_gold)
                logging.info('There are %d phi edges (%f)', phis[j, ip, rep_id], phis[j, ip, rep_id]/len(g_adj))
                lprop_pred = run_labprop(perturbed_gold, sorted_test_set,
                                         sorted_train_set, weights, inv_degree)
                mistakes = sum((1 for g, p in zip(sorted_gold, lprop_pred) if p != g))
                p_mistakes = sum((1 for g, p in zip(sorted_perturbed_gold, lprop_pred) if p != g))
                logging.info('Lprop made %d and %d mistakes', p_mistakes, mistakes)
                lprop_res[j, ip, rep_id, :] = (p_mistakes, mistakes)
                lres, wta = aggregate_trees(batch_order, (g_adj, g_ew, bfs_root), gold_signs,
                                            methods, num_tree, perturbed_gold, pool, sorted_gold,
                                            sorted_perturbed_gold, sorted_test_set)
                res[j, ip, rep_id, :, :] = lres
                wta_res[j, ip, rep_id, :] = wta
                # lres, wta = aggregate_trees(batch_order, (g_adj, g_ew, bfs_root), gold_signs,
                #                             methods, 1, perturbed_gold, pool, sorted_gold,
                #                             sorted_perturbed_gold, sorted_test_set)
                # res_mst[j, ip, rep_id, :, :] = lres
                # wta_res_mst[j, ip, rep_id, :] = wta
                np.savez_compressed(res_file, res=res, wta_res=wta_res, phis=phis, lprop=lprop_res,
                                    res_mst=res_mst, wta_res_mst=wta_res_mst)
    pool.close()
    pool.join()
    logging.info('Finished')


def get_mst(edge_weight):
    tree_edges = kruskal_mst_edges(edge_weight)
    tree_adj = {}
    for u, v in tree_edges:
        add_edge(tree_adj, u, v)
    return tree_adj, {e: 1/edge_weight[e] for e in tree_edges}


def aggregate_trees(batch_order, graph, gold_signs, methods, num_tree, perturbed_gold, pool,
                    sorted_gold, sorted_perturbed_gold, sorted_test_set, run_wta=True):
    keep_preds = defaultdict(list)
    wta_preds = []
    wta_training_set = {u: perturbed_gold[u] for u in batch_order[0]}
    g_adj, g_ew, bfs_root = graph
    inv_ew = {e: 1/w for e, w in sz.iteritems(g_ew)}
    ranging = range(1)
    if num_tree > 1:
        ranging = trange(num_tree, desc='tree', unit='tree', unit_scale=True)
    for i in ranging:
        if num_tree > 1:
            sz.GRAPH, sz.EWEIGHTS = g_adj, g_ew
            sz.get_rst(None)
            adj, ew = sz.TREE_ADJ, sz.TWEIGHTS
            logging.debug('drawn BFS tree number %d', i+1)
        else:
            adj, ew = get_mst(inv_ew)

        if run_wta:
            nodes_line, line_weight = linearize_tree(adj, ew, bfs_root)
            pred = wta_predict(nodes_line, line_weight, wta_training_set)
            wta_preds.append([pred[u] for u in sorted_test_set])

        num_order = len(batch_order)
        args = zip(repeat(adj, num_order), batch_order, repeat(ew, num_order),
                   repeat(gold_signs, num_order), repeat(perturbed_gold, num_order),
                   repeat(sorted_test_set, num_order), repeat(PARAMS_ABC, num_order))
        runs = list(pool.imap_unordered(run_once, args))
        for lres in runs:
            for method, data in zip(methods, lres):
                keep_preds[method].append(data[2])
        if not run_wta:
            args = zip(repeat(adj, num_order), batch_order, repeat(ew, num_order),
                       repeat(gold_signs, num_order), repeat(perturbed_gold, num_order),
                       repeat(sorted_test_set, num_order), repeat(None, num_order))
            runs = list(pool.imap_unordered(run_once, args))
            for lres in runs:
                wta_preds.append(lres[1][2])
        logging.debug('All threads finished for tree %d', i+1)
    res = []
    for j, (method, preds) in enumerate(sorted(keep_preds.items(), key=lambda x: x[0])):
        # TODO compute variance of individual prediction mistakes (splitting by
        # tree, since I know NUM_THREADS and num_tree)
        pred = sz.majority_vote(preds)
        mistakes = sum((1 for g, p in zip(sorted_gold, pred) if p != g))
        p_mistakes = sum((1 for g, p in zip(sorted_perturbed_gold, pred) if p != g))
        logging.debug('Aggregated over %d trees, %s made %d mistakes', num_tree, method, p_mistakes)
        res.append((p_mistakes, mistakes))
    mistakes, p_mistakes = 0, 0
    # if run_wta:
    pred = sz.majority_vote(wta_preds)
    mistakes = sum((1 for g, p in zip(sorted_gold, pred) if p != g))
    p_mistakes = sum((1 for g, p in zip(sorted_perturbed_gold, pred) if p != g))
    return res, (p_mistakes, mistakes)


def run_once(args):
    tree_adj, batch_order, ew, gold_signs, perturbed_gold, sorted_test_set, pabc = args
    logging.debug('Starting the online phase for one the batch order')
    node_vals = sz.threeway_batch_shazoo(tree_adj, ew, {}, perturbed_gold, order=batch_order,
                                         return_gammas=True, gamma_mul=pabc)[-1]
    logging.debug('Starting the batch phase for one the batch order')
    # bpreds = sz.batch_predict(tree_adj, node_vals, ew)
    bpreds = simple_offline_shazoo(tree_adj, ew, (node_vals[0], node_vals[2]))
    local_res = [(0, 0, np.sign(2*np.random.random(len(sorted_test_set))-1)), ]
    for j, (method, dpred) in enumerate(sorted(bpreds.items(), key=lambda x: x[0])):
        mistakes = sum((1 for node, p in sz.iteritems(dpred) if p != gold_signs[node]))
        p_mistakes = sum((1 for node, p in sz.iteritems(dpred) if p != perturbed_gold[node]))
        logging.debug('in one the batch order, %s made %d mistakes', method, p_mistakes)
        local_res.append((p_mistakes, mistakes, [dpred[u] for u in sorted_test_set]))
    return local_res


def simple_offline_shazoo(tree_adj, edge_weights, seen_signs):
    shazoo_signs, rta_signs = seen_signs
    test_vertices = set(tree_adj) - set(shazoo_signs)
    preds = {}
    for node in test_vertices:
        if node in preds:
            continue
        cuts = double_offline_cut_computation(tree_adj, seen_signs, edge_weights, node)
        new_pred = {n: (1 if cut[2] < cut[3] else -1,
                        1 if cut[0] < cut[1] else -1)
                    for n, cut in cuts.items()}
        assert all([n not in preds for n in new_pred])
        preds.update(new_pred)
    pred = defaultdict(dict)
    assert set(preds) == test_vertices
    for node, signs in sorted(preds.items()):
        pred['rta'][node] = signs[0]
        pred['shazoo'][node] = signs[1]
    return pred


def double_offline_cut_computation(tree_adj, nodes_sign, edge_weight, root):
    _, rooted_cut, _ = sz.flep(tree_adj, nodes_sign, edge_weight, root,
                               return_fullcut_info=True)
    discovered = defaultdict(bool)
    discovered[root] = True
    queue = deque()
    queue.append(root)
    while queue:
        v = queue.popleft()
        if v in nodes_sign[0]:
            continue
        for u in tree_adj[v]:
            if not discovered[u]:
                queue.append(u)
                discovered[u] = True
                if u in nodes_sign[0]:
                    continue
                ew = edge_weight[(u, v) if u < v else (v, u)]
                u_cp, u_cn, u_cp_, u_cn_ = rooted_cut[u]
                p_cp, p_cn, p_cp_, p_cn_ = rooted_cut[v]
                no_child_cp = p_cp - min(u_cp, u_cn + ew)
                no_child_cn = p_cn - min(u_cn, u_cp + ew)
                no_child_cp_ = p_cp_ - min(u_cp_, u_cn_ + ew)
                no_child_cn_ = p_cn_ - min(u_cn_, u_cp_ + ew)
                rooted_cut[u] = (u_cp + min(no_child_cp, no_child_cn + ew),
                                 u_cn + min(no_child_cn, no_child_cp + ew),
                                 u_cp_ + min(no_child_cp_, no_child_cn_ + ew),
                                 u_cn_ + min(no_child_cn_, no_child_cp_ + ew),)
    return rooted_cut


def star_exps(n, num_rep=5, p=.1):
    for i in range(num_rep):
        adj, ew, gold_signs, num_phi = make_star(n, p)
        print('phi edges: {}'.format(num_phi))
        start = sz.clock()
        gold, preds = sz.threeway_batch_shazoo(adj, ew, {}, gold_signs, list(range(n)))
        # gold, preds = sz.threeway_batch_shazoo(adj, ew, {}, gold_signs, list(range(1, n))+[0, ])
        for j, (method, pred) in enumerate(sorted(preds.items(), key=lambda x: x[0])):
            mistakes = sum((1 for g, p in zip(gold, pred) if p != g))
            print('{} made {} mistakes'.format(method.ljust(6), mistakes))
        time_elapsed = sz.clock() - start
        print(time_elapsed)


def make_star(n, p=.1):
    num_neg = int(p*n)
    E, tree_adj = {}, {}
    center = 0
    gold = {0: -1}
    for u in range(1, n):
        gold[u] = (-1 if u <= num_neg else 1)
        E[(center, u)] = 1#2*sz.random.random()
        sz.add_edge(tree_adj, center, u)
    return tree_adj, E, gold, n - num_neg


def benchmark(dataset='citeseer', num_run=10, train_fraction=.2):
    g_adj, g_ew, gold_signs, phi = load_real_graph(dataset)
    bfs_root = max(g_adj.items(), key=lambda x: len(x[1]))[0]
    n = len(g_adj)
    nodes_id = set(range(n))
    z = list(range(n))
    sz.random.shuffle(z)
    z = z[:int(train_fraction*n)]
    train_set = {u: gold_signs[u] for u in z}
    test_set = nodes_id - set(train_set)
    sorted_test_set = sorted(test_set)
    sorted_train_set = sorted(train_set)
    z = list(range(len(train_set)))
    sz.GRAPH, sz.EWEIGHTS = g_adj, g_ew
    sz.get_rst(bfs_root)
    adj, ew = sz.TREE_ADJ, sz.TWEIGHTS
    print(len(adj), len(ew))
    persistent.save_var('{}_rst.my'.format(dataset), (adj, ew, gold_signs), 2)
    for _ in range(num_run):
        sz.random.shuffle(z)
        batch_order = [sorted_train_set[u] for u in z]
        run_once((adj, batch_order, ew, gold_signs, gold_signs, sorted_test_set))


if __name__ == '__main__':
    # pylint: disable=C0103
    import socket
    try:
        part = int(socket.gethostname()[-1])-1
    except ValueError:
        part = 0
    sz.random.seed(123576 + part)
    # online_repetition_exps(num_rep=1, num_run=9)
    # star_exps(400, 1, .02)
    dataset = 'citeseer' if len(sys.argv) <= 1 else sys.argv[1]
    # real_exps(num_tree=15, num_batch_order=NUM_THREADS, dataset=dataset, part=part+1)
    # linear_rta(dataset, part+1)
    # for source in ['tree', 'train', 'flip']:
    #     single_tree(dataset, source, part+1, [1, 3, 4])
    gamma_rate(dataset, part+1, [1, 2, 3, 4])
    # benchmark('citeseer', num_run=1)
