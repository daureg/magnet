import random

import msgpack
import numpy as np

import persistent
from grid_stretch import add_edge

train_size = np.array([2.5, 5, 10, 20, 40]) / 100
pertubations = np.array([0, 2.5, 5, 10, 20, 35]) / 100


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


def compute_phi(edge_weight, gold):
    return sum(1 for u, v in edge_weight if gold[u] != gold[v])


def process_edges_into_tree(tree_edges, edge_weights):
    tree_adj = dict()
    for u, v in tree_edges:
        add_edge(tree_adj, u, v)
    return (tree_adj, {e: edge_weights[e] for e in tree_edges})


class DataProvider(object):
    def __init__(self, dataset, machine_id, all_machines, max_trees=None):
        with open('{}.random'.format(dataset), 'r+b') as packfile:
            data = msgpack.unpack(packfile, use_list=False)
        g_adj, g_ew, gold_signs, phi = load_real_graph(dataset)
        self.bfs_root = max(g_adj.items(), key=lambda x: len(x[1]))[0]
        nodes = list((range(len(g_adj))))
        self.gold_signs = np.array([gold_signs[u] for u in nodes])
        self.gold = np.array([gold_signs[u] for u in nodes])

        batch_orders_raw = np.array(data[b'batch_order'])
        self.batch_orders = {}
        for bo, tf in zip(batch_orders_raw, train_size):
            self.batch_orders[int(1000*tf)] = np.array([list(_) for _ in bo])

        with open('{}_extra_000.random'.format(dataset), 'r+b') as packfile:
            pertub_35 = np.array(msgpack.unpack(packfile, use_list=False))
        changed_signs = data[b'changed_signs']
        self.changed_signs = np.vstack((changed_signs, pertub_35[np.newaxis, :, :]))

        if max_trees is not None:
            max_trees *= len(all_machines)
        mst_raw = data[b'mst']
        self.mst_processed = []
        for tree_edges in mst_raw[:max_trees]:
            self.mst_processed.append(process_edges_into_tree(tree_edges, g_ew))

        rst_raw = data[b'rst']
        self.rst_processed = []
        for tree_edges in rst_raw[:max_trees]:
            self.rst_processed.append(process_edges_into_tree(tree_edges, g_ew))

        self.nodes_order = np.array(data[b'nodes_order'])

        self.mmapping = {v: i for i, v in enumerate(sorted(all_machines))}
        self.this_machine = self.mmapping[machine_id]

        self.sorted_training_set = None
        self.sorted_testing_set = None

    def _get_indices(self, split, count):
        start = self.this_machine if split else 0
        step = len(self.mmapping) if split else 1
        stop = None if count is None else start + count*step
        return slice(start, stop, step)

    def training_sets(self, fraction, count=None, split_across_machines=True):
        if fraction > 1:
            fraction /= 100
        n = len(self.nodes_order[0])
        sep = int(fraction*n)
        for node_order in self.nodes_order[self._get_indices(split_across_machines, count)]:
            self.sorted_training_set = sorted(node_order[:sep])
            self.sorted_testing_set = sorted(node_order[sep:])
            gold = self.gold[self.sorted_testing_set]
            yield self.sorted_training_set, self.sorted_testing_set, gold

    def batch_sequences(self, fraction, count=None, split_across_machines=False):
        if fraction > 1:
            fraction /= 100
        indices = self._get_indices(split_across_machines, count)
        for batch_order in self.batch_orders[int(1000*fraction)][indices]:
            # yield [self.sorted_training_set[u] for u in batch_order]
            yield batch_order

    def perturbed_sets(self, fraction, count=None, split_across_machines=True):
        if fraction > 1:
            fraction /= 100
        level = {int(1000*v): i for i, v in enumerate(pertubations)}[int(1000*fraction)]
        for pgold in self.changed_signs[level][self._get_indices(split_across_machines, count)]:
            gold_dict = {i: g for i, g in enumerate(pgold)}
            yield gold_dict
            # yield pgold[self.sorted_training_set], pgold[self.sorted_testing_set], gold_dict

    def rst_trees(self, count=None, split_across_machines=False):
        indices = self._get_indices(split_across_machines, count)
        for tree_adj, ew in self.rst_processed[indices]:
            yield (tree_adj, ew)

    def mst_trees(self, count=None, split_across_machines=False):
        indices = self._get_indices(split_across_machines, count)
        for tree_adj, ew in self.mst_processed[indices]:
            yield (tree_adj, ew)


def create_random_data(dataset, n_rep=300):
    import shazoo_exps as se
    data = {}
    g_adj, g_ew, gold_signs, phi = se.load_real_graph(dataset)
    g_adj = {int(u): {int(v) for v in adj} for u, adj in g_adj.items()}
    n = len(g_adj)
    nodes = list((range(n)))
    gold = np.array([gold_signs[u] for u in nodes])
    inv_ew = {(int(e[0]), int(e[1])): 1/w for e, w in se.sz.iteritems(g_ew)}
    g_ew = {(int(e[0]), int(e[1])): w for e, w in se.sz.iteritems(g_ew)}

    rst = []
    for _ in range(n_rep):
        se.sz.GRAPH, se.sz.EWEIGHTS = g_adj, g_ew
        se.sz.get_rst(None)
        adj, ew = se.sz.TREE_ADJ, se.sz.TWEIGHTS
        rst.append(tuple(set(ew)))
    data['rst'] = tuple(rst)

    trees = []
    for i in range(n_rep):
        adj, ew = se.get_mst(inv_ew)
        trees.append(tuple(set(ew)))
        if i == 2:
            res = []
            for j, s in enumerate(trees):
                for t in trees[j+1:]:
                    res.append(set(s) == set(t))
            if any(res):
                break
    data['mst'] = tuple(trees)

    nodes_order = []
    for _ in range(n_rep):
        random.shuffle(nodes)
        nodes_order.append(tuple(nodes))
    data['nodes_order'] = tuple(nodes_order)

    batch_order = []
    for ts in train_size:
        level = []
        max_index = int(ts*n)
        indices = list(range(max_index))
        for _ in range(n_rep):
            random.shuffle(indices)
            level.append(tuple(indices))
        batch_order.append(tuple(level))
    data['batch_order'] = tuple(batch_order)

    ones = np.ones(n, dtype=int)
    changed_signs = []
    for ts in pertubations:
        changed_signs.append(create_random_perturbations(ts, ones, nodes, gold, n_rep))
    data['changed_signs'] = tuple(changed_signs)

    with open('{}.random'.format(dataset), 'w+b') as outfile:
        msgpack.pack(data, outfile)


def create_random_perturbations(ts, ones, nodes, gold, n_rep):
    level = []
    num_modif = int(ts*len(nodes))
    for _ in range(n_rep):
        changed_idx = random.sample(nodes, num_modif)
        ones[changed_idx] *= -1
        level.append(tuple((gold * ones).tolist()))
        ones[changed_idx] *= -1
    return tuple(level)


if __name__ == "__main__":
    from sys import argv
    from timeit import default_timer as clock
    random.seed(123456)
    dataset = 'imdb' if len(argv) <= 1 else argv[1]
    start = clock()
    import shazoo_exps as se
    g_adj, g_ew, gold_signs, phi = se.load_real_graph(dataset)
    g_adj = {int(u): {int(v) for v in adj} for u, adj in g_adj.items()}
    n = len(g_adj)
    nodes = list((range(n)))
    gold = np.array([gold_signs[u] for u in nodes])
    ones = np.ones(n, dtype=int)
    ts = .35
    data = create_random_perturbations(ts, ones, nodes, gold, n_rep=300)
    with open('{}_extra_000.random'.format(dataset), 'w+b') as outfile:
        msgpack.pack(data, outfile)
    # create_random_data(dataset)
    print('{:.4g}'.format(clock() - start))
