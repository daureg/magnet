# vim: set fileencoding=utf-8
"""Implement Asym_exp as in Kunegis, J., Lommatzsch, A., & Bauckhage, C.
(2009). The Slashdot Zoo: Mining a Social Network with Negative Edges. WWW
09. doi:10.1145/1526709.1526809."""
import numpy as np
import real_world as rw
import random
import scipy.sparse as sps
import sklearn.metrics


def get_training_matrix(pr_in_train_set, mapping, slcc, tree_edges=None,
                        G=rw.G, EDGE_SIGN=rw.EDGE_SIGN):
    """Build a sparse adjacency matrix, keeping only a fraction
    `pr_in_train_set` of the entry (or those in `tree_edges`). Return edges to
    be predicted."""
    n = len(slcc)
    data, row, col = [], [], []
    for u, mu in mapping.items():
        neighbors = [v for v in G[u].intersection(slcc) if mu < mapping[v]]
        signs = [1 if EDGE_SIGN[(u, v) if u < v else (v, u)] else -1
                 for v in neighbors]
        row.extend((mu for _ in signs))
        col.extend((mapping[v] for v in neighbors))
        data.extend(signs)
    test_edges = set()
    if tree_edges:
        pr_in_train_set = 10
        mapped_edges = set()
        for u, v in tree_edges:
            if u not in mapping:
                continue
            if mapping[u] < mapping[v]:
                mapped_edges.add((mapping[u], mapping[v]))
            else:
                mapped_edges.add((mapping[v], mapping[u]))
        tree_edges = mapped_edges
    total_edges = 0
    for i, (u, v) in enumerate(zip(row, col)):
        total_edges += 1
        if tree_edges and (u, v) not in tree_edges or \
           random.random() > pr_in_train_set:
            data[i] = 0
            test_edges.add((u, v))
    if tree_edges:
        msg = '{} = {}+{}'.format(total_edges, len(tree_edges),
                                  len(test_edges))
        real = len(tree_edges) + len(test_edges)
        assert real == total_edges, msg

    sadj = sps.csc_matrix(sps.coo_matrix((data, (row, col)), shape=(n, n)),
                          dtype=np.double)
    sadj += sps.csc_matrix(sps.coo_matrix((data, (col, row)), shape=(n, n)),
                           dtype=np.double)
    return sadj, test_edges


def predict_edges(adjacency, nb_dim, mapping, test_edges, G=rw.G,
                  EDGE_SIGN=rw.EDGE_SIGN, bk=9000):
    eigsv, U = sps.linalg.eigsh(adjacency, k=nb_dim)
    U = U.astype(np.float32)
    D = np.diag(np.exp(eigsv)).astype(np.float32)
    partial = np.dot(D, U.T)
    n = adjacency.shape[0]
    gold, pred = [], []
    nmapping = {v: k for k, v in mapping.items()}
    # avoid using more than 45Gb of memory
    bk = int(127e6*45/n)
    if n < bk:
        recover = np.dot(U, partial)
        for u, v in test_edges:
            gold.append(1 if EDGE_SIGN[(nmapping[u], nmapping[v])] else -1)
            pred.append(1 if recover[u, v] > 0 else -1)
    else:
        # num_thread copies of recover takes too much space in memory
        # so we need to predict by blocks of bk nodes
        test_blocks = [[] for _ in range(n//bk + 1)]
        for u, v in test_edges:
            test_blocks[u//bk].append((u, v))
        for i, test_edges in enumerate(test_blocks):
            recover = np.dot(U[i*bk:(i+1)*bk, :], partial)
            for u, v in test_edges:
                real_edge = (nmapping[u], nmapping[v])
                gold.append(1 if EDGE_SIGN[real_edge] else -1)
                pred.append(1 if recover[u-i*bk, v] > 0 else -1)
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
    import persistent
    import pred_on_tree as pot
    import sys
    bfstrees = None
    parser = ae.get_parser('Predict sign using a spectral method')
    args = parser.parse_args()
    a = ae.further_parsing(args)
    DATA = sys.argv[1].upper()
    BASENAME, SEEDS, SYNTHETIC_DATA, PREFIX, noise, BALANCED = a
    LAUNCH = int(time() - 1427846400)
    training_fraction = 0.375 if DATA == 'LP' else None

    def load_graph(seed=None):
        if BASENAME.startswith('soc'):
            rw.read_original_graph(BASENAME, seed=seed, balanced=BALANCED)
            redensify.G = deepcopy(rw.G)
            redensify.EDGES_SIGN = deepcopy(rw.EDGE_SIGN)
        elif DATA == 'LP':
            _ = persistent.load_var(BASENAME+'.my')
            redensify.G, redensify.EDGES_SIGN = _
            return
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
            to_delete = persistent.load_var(BASENAME+'_balance.my')
            for edge in to_delete:
                pot.delete_edge(redensify.G, edge, redensify.EDGES_SIGN)
        cexp.add_noise(noise/100, noise/100)
        if seed is not None:
            random.seed(seed)
            rperm = list(redensify.G.keys())
            random.shuffle(rperm)
            rperm = {i: v for i, v in enumerate(rperm)}
            _ = rw.reindex_nodes(redensify.G, redensify.EDGES_SIGN, rperm)
            redensify.G, redensify.EDGES_SIGN = _
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

    bfs_rep = 18
    SEEDS = SEEDS[:bfs_rep]
    NB_DIM = 15
    n_noise = 11 if SYNTHETIC_DATA else 1
    bfs_res = np.zeros((bfs_rep*n_noise, 3))
    random_res = np.zeros((bfs_rep*n_noise, 3))
    gtx_res = np.zeros((len(SEEDS)*n_noise, 3))
    for k in range(n_noise):
        # mapping, slcc = add_cc_noise(noise, balanced=BALANCED, seed=None)
        # for i, tree in enumerate(bfstrees[:bfs_rep]):
        #     adj, test_edges = get_training_matrix(-5, mapping, slcc,
        #                                           tree_edges=tree)
        #     res = predict_edges(adj, NB_DIM, mapping, test_edges)
        #     bfs_res[i+k*bfs_rep, :] = res

        # for i, seed in enumerate(SEEDS):
        #     mapping, slcc = add_cc_noise(noise, balanced=BALANCED,
        #                                  seed=seed)
        #     file_pattern = 'universe/{}_{}_*.edges'.format(PREFIX, seed)
        #     gtx_trees = sorted(glob.glob(file_pattern))
        #     basename = gtx_trees[-1][:-6]
        #     print(basename)
        #     _, spanner_edges = pot.read_tree(basename+'.edges')
        #     gtx_tree = {(u, v) for u, v in spanner_edges if u in slcc}
        #     adj, test_edges = get_training_matrix(-5, mapping, slcc,
        #                                           tree_edges=gtx_tree)
        #     res = predict_edges(adj, NB_DIM, mapping, test_edges)
        #     gtx_res[i+k*len(SEEDS), :] = res
        #     training_fraction = len(spanner_edges) / len(rw.EDGE_SIGN)

        pr = training_fraction
        for i in range(bfs_rep):
            print('{}/{}: {}/{}'.format(k, n_noise, i, bfs_rep))
            mapping, slcc = add_cc_noise(noise, balanced=BALANCED, seed=None)
            adj, test_edges = get_training_matrix(pr, mapping, slcc)
            res = predict_edges(adj, NB_DIM, mapping, test_edges)
            random_res[i+k*bfs_rep, :] = res

    res_name = '{}_{}_{:.1f}_{}'.format(DATA, BALANCED, noise, LAUNCH)
    for kind, arr in zip(['random', 'bfs', 'gtx'],
                         [random_res, bfs_res, gtx_res]):
        txt_res = (' & '.join(['{:.3f} ({:.3f})'.format(*l)
                               for l in zip(np.mean(arr, 0),
                                            np.std(arr, 0))]))
        np.save('altexp10/{}_{}.npy'.format(res_name, kind), arr)
        params = (kind, 100*training_fraction, txt_res)
        print('& AsymExp {} & {:.1f}% & {} & & \\\\'.format(*params))
