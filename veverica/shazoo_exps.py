# vim: set fileencoding=utf-8
"""Perform various experiments using the shazoo code."""
from collections import defaultdict
from grid_stretch import add_edge
from tqdm import trange, tqdm
import persistent
import shazoo as sz
import numpy as np
import time
from itertools import repeat
from multiprocessing import Pool
import sys
import logging
if sz.USE_SCIPY or sys.version_info.major == 3:
    from shazoo_scipy import run_labprop, get_weight_matrix
else:
    def run_labprop(gold_signs, test_set, *args):
        return np.sign(2*np.random.random(len(test_set))-1)

    def get_weight_matrix(*args):
        return None, None
NUM_THREADS = 11


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


def real_exps(num_tree=2, num_batch_order=15, train_fraction=.2, dataset='citeseer', part=0):
    exp_start = (int(time.time()-(2017-1970)*365.25*24*60*60))//60
    dbg_fmt = '%(asctime)s - %(relativeCreated)d:%(filename)s.%(funcName)s.%(threadName)s:%(lineno)d(%(levelname)s):%(message)s '
    logging.basicConfig(filename='shazoo_{}.log'.format(exp_start), level=logging.DEBUG, format=dbg_fmt)
    logging.info('Started')
    res_file = 'shazoo_{}_{}_{}.npz'.format(dataset, exp_start, part)
    perturbations = [0, 2.5, 5, 10, 20]
    train_size = np.array([2.5, 5, 10, 20, 40]) / 100
    nrep = 3
    res = np.zeros((len(train_size), len(perturbations), nrep, 3, 2))
    phis = np.zeros((len(train_size), len(perturbations), nrep))
    lprop_res = np.zeros((len(train_size), len(perturbations), nrep, 2))
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
                probas = get_perturb_proba(degrees, p/100.0)
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
                lres = aggregate_trees(batch_order, (g_adj, g_ew, bfs_root), gold_signs, methods,
                                       num_tree, perturbed_gold, pool, sorted_gold,
                                       sorted_perturbed_gold, sorted_test_set)
                res[j, ip, rep_id, :, :] = lres
                np.savez_compressed(res_file, res=res, phis=phis, lprop=lprop_res)
    pool.close()
    pool.join()
    logging.info('Finished')


def aggregate_trees(batch_order, graph, gold_signs, methods, num_tree, perturbed_gold, pool,
                    sorted_gold, sorted_perturbed_gold, sorted_test_set):
    keep_preds = defaultdict(list)
    g_adj, g_ew, bfs_root = graph
    for i in trange(num_tree, desc='tree', unit='tree', unit_scale=True):
        sz.GRAPH, sz.EWEIGHTS = g_adj, g_ew
        sz.get_bfs(bfs_root)
        adj, ew = sz.TREE_ADJ, sz.TWEIGHTS
        logging.debug('drawn BFS tree number %d', i+1)
        args = zip(repeat(adj, num_tree), batch_order, repeat(ew, num_tree),
                   repeat(gold_signs, num_tree), repeat(perturbed_gold, num_tree),
                   repeat(sorted_test_set, num_tree))
        runs = list(pool.imap_unordered(run_once, args))
        for lres in runs:
            for method, data in zip(methods, lres):
                keep_preds[method].append(data[2])
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
    return res


def run_once(args):
    tree_adj, batch_order, ew, gold_signs, perturbed_gold, sorted_test_set = args
    logging.debug('Starting the online phase for one the batch order')
    node_vals = sz.threeway_batch_shazoo(tree_adj, ew, {}, perturbed_gold,
                                         order=batch_order, return_gammas=True)[-1]
    logging.debug('Starting the batch phase for one the batch order')
    bpreds = sz.batch_predict(tree_adj, node_vals, ew)
    local_res = []
    for j, (method, dpred) in enumerate(sorted(bpreds.items(), key=lambda x: x[0])):
        mistakes = sum((1 for node, p in sz.iteritems(dpred) if p != gold_signs[node]))
        p_mistakes = sum((1 for node, p in sz.iteritems(dpred) if p != perturbed_gold[node]))
        if method == 'l2cost' and not sz.USE_SCIPY:
            local_res.append((0, 0, np.sign(2*np.random.random(len(sorted_test_set))-1)))
        else:
            logging.debug('in one the batch order, %s made %d mistakes', method, p_mistakes)
            local_res.append((p_mistakes, mistakes, [dpred[u] for u in sorted_test_set]))
    return local_res


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
    sz.random.seed(123460 + part)
    # online_repetition_exps(num_rep=1, num_run=9)
    # star_exps(400, 1, .02)
    dataset = 'citeseer' if len(sys.argv) <= 1 else sys.argv[1]
    real_exps(num_tree=17, num_batch_order=NUM_THREADS, dataset=dataset, part=part)
    # benchmark('citeseer', num_run=1)
