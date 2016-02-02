#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""sign prediction in signed network based on node classification."""
from args_experiments import get_parser
from cmp_tree_features import confusion_matrix
from collections import defaultdict, Counter
from grid_stretch import add_edge
import random
import time


def balance_signs(G, E, seed=1489):
    # TODO: should I return the largest connected component?
    random.seed(seed)
    num_neg = Counter(E.values())[False]
    p = num_neg/(len(E)-num_neg)
    nG, nE = {}, {}
    for (u, v), s in E.items():
        if s is False or random.random()<=p:
            nE[(u, v)] = s
            add_edge(nG, u, v)
    random.seed(int(time.time()))
    return nG, nE


def select_edges(G, E, alpha, strategy, directed=False, sampling=None,
                 with_replacement=False):
    """if provided, sampling is a function which take the degree of the node
    and return an integer: number of edge to query"""
    res = {}
    def prune_training_set():
        num_unseen = len(E) - len(res)
        if num_unseen < 1000:
            to_delete = set()
            for e, i in zip(res, range(1000-num_unseen)):
                to_delete.add(e)
            for e in to_delete:
                del res[e]
    if strategy == 'random':
        res = dict(random.sample(list(E.items()), int(alpha*len(E))))
        prune_training_set()
        return res
    if sampling is None:
        sampling = lambda d: int(alpha*d)
    for u, adj in G.items():
        num_to_sample = max(min(len(adj), 1), sampling(len(adj)))
        num_to_sample = min(len(adj), num_to_sample)
        ladj = list(adj)
        if with_replacement:
            nei = [random.choice(ladj) for _ in range(num_to_sample)]
        else:
            nei = random.sample(ladj, num_to_sample)
        if directed:
            edges = {(u, v) if (u, v) in E else (v, u) for v in nei}
        else:
            edges = {(u, v) if u < v else (v, u) for v in nei}
        res.update({e: E[e] for e in edges})
    prune_training_set()
    return res


def predict_signs(E, known_signs, threshold):
    deg, ndeg = defaultdict(int), defaultdict(int)
    for (u, v), positive in known_signs.items():
        deg[u] += 1
        deg[v] += 1
        if not positive:
            ndeg[u] += 1
            ndeg[v] += 1
    ratio_dic = {u: ndeg[u]/deg[u] for u in ndeg}

    gold = []
    pred = []
    p = Counter(E.values())[True]/len(E)
    for (u, v), positive in E.items():
        # TODO: skip known_signs
        if (u, v) in known_signs:
            continue
        gold.append(int(positive))
        if u in ratio_dic and v in ratio_dic:
            # if u or v is a troll, then the edge is negative
            pred.append(1-int(ratio_dic[u] >= threshold or
                              ratio_dic[v] >= threshold))
        else:
            pred.append(1 if random.random() <= p else 0)
    return gold, pred


def evaluate_pred(gold, pred):
    from math import sqrt
    tp, tn, fp, fn = confusion_matrix(gold, pred, 1, 0)
    acc = (tp + tn)/len(gold)
    f1 = 2*tp/(2*tp+fn+fp)
    mcc = (tp*tn - fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return [acc, f1, mcc, fp/len(pred), 0]


def full_pipeline(G, E, alpha, strategy, threshold):
    known_signs = select_edges(G, E, alpha, strategy)
    gold, pred = predict_signs(E, known_signs, threshold)
    return evaluate_pred(gold, pred)+[len(known_signs)/len(E)]


if __name__ == '__main__':
    # pylint: disable=C0103
    from multiprocessing import Pool
    import persistent as p
    from itertools import product
    num_threads = 15
    num_rep = 1*num_threads
    cmd_args = get_parser().parse_args()
    G, E = p.load_var({'WIK': 'wikipedia_lcc.my',
                       'EPI': 'epinion_lcc.my',
                       'SLA': 'slashdot_lcc.my'}[cmd_args.data])
    if cmd_args.balanced:
        G, E = balance_signs(G, E)
    res = []
    alphas = range(5, 100, 10)
    strategies = ['random', 'uniform']
    params = list(product(alphas, strategies))
    # pool = Pool(processes=num_threads)
    with Pool(processes=num_threads) as pool:
        for alpha, strategy in params:
            args = []
            for _ in range(num_rep):
                args.append((G, E, alpha/100, strategy, .5))
            res.append(list(pool.starmap(full_pipeline, args,
                                         len(args)//num_threads)))
    # pool.close()
    # pool.join()
    p.save_var('{}_{}_{}.my'.format(cmd_args.data, int(time.time()),
                                    cmd_args.balanced),
               (params, res))
