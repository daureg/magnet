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
import sys


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
    if tree_edges:
        msg = '{} = {}+{}'.format(total_edges, len(tree_edges),
                                  len(test_edges))
        real = len(tree_edges) + len(test_edges)
        assert .99*total_edges <= real <= 1.01*total_edges, msg
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
    import graph_tool as gt
    import redensify
    import convert_experiment as cexp
    from copy import deepcopy
    import noise_influence as nsi
    import glob
    from time import time
    import args_experiments as ae
    bfstrees = None
    training_fraction = None
    parser = ae.get_parser('Predict sign using a spectral method')
    args = parser.parse_args()
    a = ae.further_parsing(args)
    DATA = sys.argv[1].upper()
    BASENAME, SEEDS, SYNTHETIC_DATA, PREFIX, noise, BALANCED = a
    LAUNCH = int(time() - 1427846400)

    def load_graph(seed=None):
        if BASENAME.startswith('soc'):
            rw.read_original_graph(BASENAME, seed=seed, balanced=BALANCED)
            redensify.G = deepcopy(rw.G)
            redensify.EDGES_SIGN = deepcopy(rw.EDGE_SIGN)
        else:
            G = gt.load_graph(BASENAME+'.gt')
            cexp.to_python_graph(G)

    load_graph()
    orig_g = deepcopy(redensify.G)
    orig_es = deepcopy(redensify.EDGES_SIGN)

    def add_cc_noise(noise, balanced=False, seed=None):
        global bfstrees
        redensify.G = deepcopy(orig_g)
        redensify.EDGES_SIGN = deepcopy(orig_es)
        if balanced and SYNTHETIC_DATA:
            import persistent
            to_delete = persistent.load_var(BASENAME+'_balance.my')
            for edge in to_delete:
                pot.delete_edge(redensify.G, edge, redensify.EDGES_SIGN)
        if seed is not None:
            random.seed(seed)
            rperm = list(redensify.G.keys())
            random.shuffle(rperm)
            rperm = {i: v for i, v in enumerate(rperm)}
            _ = rw.reindex_nodes(redensify.G, redensify.EDGES_SIGN, rperm)
            redensify.G, redensify.EDGES_SIGN = _
        cexp.add_noise(noise/100, noise/100)
        rw.G = deepcopy(redensify.G)
        rw.EDGE_SIGN = deepcopy(redensify.EDGES_SIGN)
        rw.DEGREES = sorted(((node, len(adj)) for node, adj in rw.G.items()),
                            key=lambda x: x[1])
        if not bfstrees:
            if SYNTHETIC_DATA:
                bfstrees = [t[1] for t in nsi.compute_trees()]
            else:
                bfstrees = []
                for root in (u[0] for u in rw.DEGREES[-50:]):
                    bfstrees.append(pot.get_bfs_tree(rw.G, root))

        return get_largest_component()

    def get_largest_component():
        global training_fraction
        lcc_tree = pot.get_bfs_tree(rw.G, rw.DEGREES[-1][0])
        if not training_fraction:
            training_fraction = len(lcc_tree)/len(rw.EDGE_SIGN)
        heads, tails = zip(*lcc_tree)
        slcc = sorted(set(heads).union(set(tails)))
        # a mapping between original vertex indices and indices in the largest
        # component
        mapping = {v: i for i, v in enumerate(slcc)}
        return mapping, slcc

    bfs_rep = 50
    SEEDS = SEEDS[:bfs_rep]
    NB_DIM = 15
    n_noise = 20 if SYNTHETIC_DATA else 1
    bfs_res = np.zeros((bfs_rep*n_noise, 3))
    random_res = np.zeros((bfs_rep*n_noise, 3))
    gtx_res = np.zeros((len(SEEDS)*n_noise, 3))
    for k in range(n_noise):
        mapping, slcc = add_cc_noise(noise, balanced=BALANCED, seed=None)
        for i, tree in enumerate(bfstrees[:bfs_rep]):
            adj, test_edges = get_training_matrix(-5, mapping, slcc,
                                                  tree_edges=tree)
            res = predict_edges(adj, NB_DIM, mapping, test_edges)
            bfs_res[i+k*bfs_rep, :] = res

        pr = training_fraction
        for i in range(bfs_rep):
            mapping, slcc = add_cc_noise(noise, balanced=BALANCED, seed=None)
            adj, test_edges = get_training_matrix(pr, mapping, slcc)
            res = predict_edges(adj, NB_DIM, mapping, test_edges)
            random_res[i+k*bfs_rep, :] = res

        for i, seed in enumerate(SEEDS):
            mapping, slcc = add_cc_noise(noise, balanced=BALANCED,
                                         seed=seed)
            file_pattern = 'universe/{}_{}_*.edges'.format(PREFIX, seed)
            gtx_trees = sorted(glob.glob(file_pattern))
            basename = gtx_trees[-1][:-6]
            spanner_edges, _, _, _ = pot.read_spanner_from_file(basename)
            gtx_tree = {(u, v) for u, v in spanner_edges}
            adj, test_edges = get_training_matrix(-5, mapping, slcc,
                                                  tree_edges=gtx_tree)
            res = predict_edges(adj, NB_DIM, mapping, test_edges)
            gtx_res[i+k*len(SEEDS), :] = res
    res_name = '{}_{}_{:.1f}_{}'.format(DATA, BALANCED, noise, LAUNCH)
    for kind, arr in zip(['random', 'bfs', 'gtx'],
                         [random_res, bfs_res, gtx_res]):
        txt_res = (' & '.join(['{:.3f} ({:.3f})'.format(*l)
                               for l in zip(np.mean(arr, 0),
                                            np.std(arr, 0))]))
        np.save('altexp/{}_{}.npy'.format(res_name, kind), arr)
        params = (kind, 100*training_fraction, txt_res)
        print('& AsymExp {} & {:.1f}% & {} & & \\\\'.format(*params))
