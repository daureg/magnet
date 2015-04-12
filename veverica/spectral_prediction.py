# vim: set fileencoding=utf-8
"""Implement Asym_exp as in Kunegis, J., Lommatzsch, A., & Bauckhage, C.
(2009). The Slashdot Zoo: Mining a Social Network with Negative Edges. WWW
09. doi:10.1145/1526709.1526809."""
import numpy as np
import real_world as rw
import random
import scipy.sparse
import pred_on_tree as pot
import sklearn.metrics


def get_training_matrix(pr_in_train_set, mapping, slcc, tree_edges=None):
    """Build a sparse adjacency matrix, keeping only a fraction
    `pr_in_train_set` of the entry (or those in `tree_edges`). Return edges to
    be predicted."""
    n = len(rw.G)
    madj = np.zeros((n, n), dtype=np.int8)
    for u in rw.G.keys():
        if rw.G[u]:
            vs, signs = zip(*((v, rw.EDGE_SIGN[(u, v) if u < v else (v, u)])
                              for v in sorted(rw.G[u])))
            madj[u, vs] = [1 if sign else -1 for sign in signs]
    test_edges = set()
    if tree_edges:
        pr_in_train_set = 10
    total_edges = 0
    for u, v in np.argwhere(madj):
        if u > v or u not in mapping:
            continue
        total_edges += 1
        if tree_edges and (u, v) not in tree_edges or \
           random.random() > pr_in_train_set:
            madj[u, v], madj[v, u] = 0, 0
            test_edges.add((u, v))

    print(total_edges, len(test_edges))
    if tree_edges:
        msg = '{} = {}+{}'.format(total_edges, len(tree_edges),
                                  len(test_edges))
        assert len(test_edges) == total_edges - len(tree_edges), msg
    sadj = scipy.sparse.csc_matrix(madj[np.ix_(slcc, slcc)],
                                   dtype=np.double)
    return sadj, test_edges


def predict_edges(adjacency, nb_dim, mapping, test_edges):
    eigsv, U = scipy.sparse.linalg.eigsh(adjacency, k=nb_dim)
    D = np.diag(np.exp(eigsv))
    recover = np.dot(U, np.dot(D, U.T))
    gold, pred = [], []
    for u, v in test_edges:
        gold.append(1 if rw.EDGE_SIGN[(u, v)] else -1)
        pred.append(1 if recover[mapping[u], mapping[v]] > 0 else -1)
    return (sklearn.metrics.accuracy_score(gold, pred),
            sklearn.metrics.f1_score(gold, pred),
            sklearn.metrics.matthews_corrcoef(gold, pred))

if __name__ == '__main__':
    # pylint: disable=C0103
    import sys
    import graph_tool as gt
    import redensify
    import convert_experiment as cexp
    import pred_on_tree as pot
    from copy import deepcopy
    balanced = True
    rw.read_original_graph('soc-wiki.txt', balanced=balanced)
    noise = int(sys.argv[1])
    assert noise == 0 or noise > 1, 'give noise as a percentage'
    BASENAME = 'universe/noisePA'
    # G = gt.load_graph(BASENAME+'.gt')

    def add_cc_noise(noise, balanced=False):
        cexp.to_python_graph(G)
        if balanced:
            import persistent
            to_delete = persistent.load_var(BASENAME+'_balance.my')
            for edge in to_delete:
                pot.delete_edge(redensify.G, edge, redensify.EDGES_SIGN)
        cexp.add_noise(noise/100, noise/100)
        rw.G = deepcopy(redensify.G)
        rw.EDGE_SIGN = deepcopy(redensify.EDGES_SIGN)
        rw.DEGREES = sorted(((node, len(adj)) for node, adj in rw.G.items()),
                            key=lambda x: x[1])
        return get_largest_component()

    def get_largest_component():
        lcc_tree = pot.get_bfs_tree(rw.G, rw.DEGREES[-1][0])
        print(len(lcc_tree)/len(rw.EDGE_SIGN))
        heads, tails = zip(*lcc_tree)
        slcc = sorted(set(heads).union(set(tails)))
        # a mapping between original vertex indices and indices in the largest
        # component
        mapping = {v: i for i, v in enumerate(slcc)}
        return mapping, slcc

    # nb_dims = [5, 10, 15, 25]
    n_rep = 100
    nb_dims = n_rep*[15, ]
    # training_fraction = [5, 7.5, 11, 17, 27.1, 50]
    # training_fraction = [16.5, 36.1]
    training_fraction = [12.8, ]
    n_noise = 1
    acc = np.zeros((len(nb_dims)*n_noise, len(training_fraction)))
    f1 = np.zeros((len(nb_dims)*n_noise, len(training_fraction)))
    mcc = np.zeros((len(nb_dims)*n_noise, len(training_fraction)))
    print(noise/100)
    for k in range(n_noise):
        # mapping, slcc = add_cc_noise(noise, balanced=balanced)
        mapping, slcc = get_largest_component()
        for i, nb_dim in enumerate(nb_dims):
            for j, pr in enumerate(training_fraction):
                # print(nb_dim, pr)
                adj, test_edges = get_training_matrix(pr/100, mapping, slcc)
                res = predict_edges(adj, nb_dim, mapping, test_edges)
                # print(res)
                acc[i+k*n_rep, j] = res[0]
                f1[i+k*n_rep, j] = res[1]
                mcc[i+k*n_rep, j] = res[2]
    txt_res = ' & '.join(['{:.3f} ({:.3f})'.format(np.mean(_[:, 0], 0),
                                                   np.std(_[:, 0], 0))
                          for _ in [acc, f1, mcc]])
    print('& AsymExp $z={}$ & {:.1f}% & {} & & \\\\'.format(nb_dims[0],
                                                            training_fraction[0],
                                                            txt_res))
    # out_name = 'asym_{}_{}_{}'.format
    # np.save(out_name('noise', int(noise), 'acc'), acc)
    # np.save(out_name('noise', int(noise), 'f1'), f1)
    # np.save(out_name('noise', int(noise), 'mcc'), mcc)
