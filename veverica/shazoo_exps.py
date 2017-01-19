# vim: set fileencoding=utf-8
"""Perform various experiments using the shazoo code."""
from collections import defaultdict
from grid_stretch import add_edge
from tqdm import trange, tqdm
import persistent
import shazoo as sz
import numpy as np
import time


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


def load_citeseer(main_class=1):
    ew, y = persistent.load_var('citeseer_lcc.my')
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


def real_exps(num_tree=2, num_run=15, train_fraction=.2):
    exp_start = (int(time.time()-(2017-1970)*365.25*24*60*60))//60
    res_file = 'shazoo_citeseer_{}.npz'.format(exp_start)
    perturbations = [0, 2.5, 5, 10, 20, 30]
    res = np.zeros((len(perturbations), num_tree, num_run+2, 3, 2))
    phis = np.zeros(len(perturbations))
    g_adj, g_ew, gold_signs, phi = load_citeseer()
    n = len(g_adj)
    bfs_root = max(g_adj.items(), key=lambda x: len(x[1]))[0]
    nodes_id = set(range(n))
    z = list(range(n))
    sz.random.shuffle(z)
    z = z[:int(train_fraction*n)]
    train_set = {u: gold_signs[u] for u in z}
    test_set = nodes_id - set(train_set)
    sorted_test_set = sorted(test_set)
    sorted_train_set = sorted(train_set)
    sorted_gold = [gold_signs[u] for u in sorted_test_set]
    batch_order = []
    z = list(range(len(train_set)))
    for _ in range(num_run):
        sz.random.shuffle(z)
        batch_order.append([sorted_train_set[u] for u in z])
    for ip, p in enumerate(tqdm(perturbations, desc='perturbation', unit='flip', unit_scale=True)):
        tqdm.write('starting')
        perturbed_gold = {u: (1 if sz.random.random() >= p else -1)*s
                          for u, s in sz.iteritems(gold_signs)}
        sorted_perturbed_gold = [perturbed_gold[u] for u in sorted_test_set]
        phis[ip] = compute_phi(g_ew, perturbed_gold)
        for i in trange(num_tree, desc='tree', unit='tree', unit_scale=True):
            keep_preds = defaultdict(list)
            sz.GRAPH, sz.EWEIGHTS = g_adj, g_ew
            sz.get_bfs(bfs_root)
            adj, ew = sz.TREE_ADJ, sz.TWEIGHTS
            start = sz.clock()
            for k in range(num_run):
                _, _, node_vals = sz.threeway_batch_shazoo(adj, ew, {}, perturbed_gold,
                                                           order=batch_order[k], return_gammas=True)
                bpreds = sz.batch_predict(adj, node_vals, ew)
                for j, (method, dpred) in enumerate(sorted(bpreds.items(), key=lambda x: x[0])):
                    mistakes = sum((1 for node, p in sz.iteritems(dpred) if p != gold_signs[node]))
                    p_mistakes = sum((1 for node, p in sz.iteritems(dpred) if p != perturbed_gold[node]))
                    res[ip, i, k, j, :] = (p_mistakes, mistakes)
                    keep_preds[method].append([dpred[u] for u in sorted_test_set])
                    np.savez_compressed(res_file, res=res)
            # time_elapsed = sz.clock() - start
            # print(time_elapsed/num_run)
            for j, (method, preds) in enumerate(sorted(keep_preds.items(), key=lambda x: x[0])):
                pred = sz.majority_vote(preds[:5])
                mistakes = sum((1 for g, p in zip(sorted_gold, pred) if p != g))
                p_mistakes = sum((1 for g, p in zip(sorted_perturbed_gold, pred) if p != g))
                res[ip, i, -2, j, :] = (p_mistakes, mistakes)
                pred = sz.majority_vote(preds)
                mistakes = sum((1 for g, p in zip(sorted_gold, pred) if p != g))
                p_mistakes = sum((1 for g, p in zip(sorted_perturbed_gold, pred) if p != g))
                res[ip, i, -1, j, :] = (p_mistakes, mistakes)
                tqdm.write('{} made {} mistakes'.format(method.ljust(6), mistakes))
                np.savez_compressed(res_file, res=res)


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


if __name__ == '__main__':
    # pylint: disable=C0103
    sz.random.seed(123458)
    # online_repetition_exps(num_rep=1, num_run=9)
    # star_exps(400, 1, .02)
    real_exps(num_tree=10, num_run=17, train_fraction=.2)