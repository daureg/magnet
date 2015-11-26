#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""."""
import real_world as rw
import pred_on_tree as pot
from math import sqrt
from multiprocessing import Pool


def confusion_matrix(gold, pred, pos_label=1, neg_label=-1):
    tp, tn, fp, fn = 0, 0, 0, 0
    for g, p in zip(gold, pred):
        if g == pos_label == p:
            tp += 1
        if g == pos_label != p:
            fn += 1
        if g == neg_label == p:
            tn += 1
        if g == neg_label != p:
            fp += 1
    return tp, tn, fp, fn


def compute_mcc(gold, pred, pos_label=1, neg_label=-1):
    tp, tn, fp, fn = confusion_matrix(gold, pred, pos_label, neg_label)
    return (tp*tn - fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))


def do_it(r):
    bfs_edges = pot.get_bfs_tree(rw.G, r)
    tree = {}
    for u, v in bfs_edges:
        pot.add_edge_to_tree(tree, u, v)
    root_degree = len(tree[r])
    branching_factors = [len(tree[n]) for n in tree]
    gold, pred, stretch = [], [], []
    positive_fraction = 0
    path_lengths = [[], [],
                    [], []]
    for e, s in edge_binary.items():
        if e in bfs_edges:
            positive_fraction += 1 if s == 1 else 0
            continue
        if e[0] not in tree:
            continue
        gold.append(s)
        spred, slen = pot.brute_parity(e[0], e[1], tree, edge_binary)
        pred.append(spred)
        stretch.append(slen)
        if s == -1 == spred:
            path_lengths[0].append(slen)
        if s == -1 != spred:
            path_lengths[1].append(slen)
        if s == 1 != spred:
            path_lengths[2].append(slen)
        if s == 1 == spred:
            path_lengths[3].append(slen)
    # acc = accuracy_score(gold, pred)
    # f1, mcc = f1_score(gold, pred), matthews_corrcoef(gold, pred)
    mcc = compute_mcc(gold, pred)
    return (root_degree, branching_factors, positive_fraction, path_lengths,
            mcc)

if __name__ == '__main__':
    # pylint: disable=C0103
    num_threads, per_thread = 13, 6
    tasks = (num_threads*per_thread)
    rw.read_original_graph('soc-wiki.txt')
    roots = [_[0] for _ in rw.DEGREES[-tasks:]]
    edge_binary = {e: 2*int(s)-1 for e, s in rw.EDGE_SIGN.items()}
    features = []
    target = []
    pool = Pool(num_threads)
    res = list(pool.imap_unordered(do_it, roots[:tasks],
                                   chunksize=per_thread))
    pool.close()
    pool.join()
    import persistent
    persistent.save_var('wik_feature.my', res)
