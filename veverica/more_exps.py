# vim: set fileencoding=utf-8
from timeit import default_timer as clock

from future.utils import iteritems

import shazoo_exps as se


def single(num_rep=10, dataset='citeseer'):
    se.sz.random.seed(123459)
    adj, ew, gold_signs = se.persistent.load_var('{}_tree.my'.format(dataset))
    n = len(adj)
    # n = 2484 #2110
    order = se.sz.random.sample(list(range(n)), int(.2*n))
    timings = []
    for i in range(num_rep):
        # gr, phi_edges = se.sz.make_graph(n)
        # gr = (adj, None, ew, gold_signs)
        gr = (adj, None, ew, [-1 for _ in range(n)])
        # gr = (gr[0], None, gr[2], [-1 for _ in range(n)])
        start = clock()
        gold, preds, _ = se.sz.threeway_batch_shazoo(gr[0], gr[2], {}, gr[-1],
                                                     order, return_gammas=True)
        timings.append(clock() - start)
        # for j, (method, pred) in enumerate(sorted(preds.items(), key=lambda x: x[0])):
        #     mistakes = sum((1 for g, p in zip(gold, pred) if p != g))
            # print(method, mistakes)
    print(timings)
    print(sum(timings)/num_rep)


def real(num_run=10, train_fraction=.2):
    adj, ew, gold_signs = se.persistent.load_var('citeseer_tree.my')
    n = len(adj)
    nodes_id = set(range(n))
    z = list(nodes_id)
    se.sz.random.shuffle(z)
    z = z[:int(train_fraction*n)]
    train_set = {u: gold_signs[u] for u in z}
    test_set = nodes_id - set(train_set)
    sorted_test_set = sorted(test_set)
    sorted_train_set = sorted(train_set)
    z = list(range(len(train_set)))
    for _ in range(num_run):
        se.sz.random.shuffle(z)
        batch_order = [sorted_train_set[u] for u in z]
        se.run_once((adj, batch_order, ew, gold_signs, gold_signs, sorted_test_set))


def flep_timing(num_run=10, dataset=None):
    if dataset:
        adj, ew, signs = se.persistent.load_var('{}_tree.my'.format(dataset))
        n = len(adj)
    else:
        n = 50000
        gr, phi_edges = se.sz.make_graph(n)
        adj, ew, signs = gr[0], gr[2], gr[-1]
    leaves = [u for u, a in iteritems(adj) if len(a) == 1]
    root = max(iteritems(adj), key=lambda x: len(x[1]))[0]
    timings = []
    for _ in range(num_run):
        nodes_sign = {u: signs[u] for u in se.sz.random.sample(leaves, 50)}
        start = clock()
        # se.sz.flep(adj, nodes_sign, ew, root)
        # se.sz.find_hinge_nodes(adj, ew, nodes_sign, root)
        se.sz.predict_one_node_three_methods(root, adj, ew, (nodes_sign, nodes_sign, nodes_sign))
        timings.append(clock() - start)
    print(' '.join(('{:.4g}'.format(_) for _ in timings)))
    print(sum(timings[1:])/(num_run-1))


if __name__ == '__main__':
    # single(2, 'cora')
    flep_timing(dataset='citeseer')
    flep_timing(dataset=None)
