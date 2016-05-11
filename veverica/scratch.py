from collections import Counter
from convert_pbm_images import build_graph, read_img
from copy import deepcopy
from math import sqrt
from new_galaxy import galaxy_maker
from socket import gethostname as hostname
import convert_experiment as cexp
import grid_stretch as gs
import persistent
import pred_on_tree as pot
import random
import os


def confusion_number(gold, pred):
    tp, tn, fp, fn = 0, 0, 0, 0
    for g, p in zip(gold, pred):
        if g == 1 == p:
            tp += 1
        if g == 1 != p:
            fn += 1
        if g == -1 == p:
            tn += 1
        if g == -1 != p:
            fp += 1
    return tp, tn, fp, fn


def mcc(tp, tn, fp, fn):
    return (tp*tn - fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))


def accuracy(tp, tn, fp, fn):
    return (tp + tn)/(tp + tn + fp + fn)


def f1_score(tp, tn, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2*(precision*recall)/(precision + recall)


def merge_into_2_clusters(edges, clusters):
    largest_clusters = sorted(Counter(clusters).items(), reverse=True,
                              key=lambda x: x[1])
    size1, size2 = largest_clusters[0][1], 0
    group1, group2 = set([largest_clusters[0][0]]), set()
    for c_idx, c_size in largest_clusters[1:]:
        if size2 < size1:
            group2.add(c_idx)
            size2 += c_size
        else:
            group1.add(c_idx)
            size1 += c_size
    new_cluster = [int(_ in group1) for _ in clusters]
    for i, j in edges.keys():
        edges[(i, j)] = new_cluster[i] == new_cluster[j]
    return edges, new_cluster


def load_graph(name, size=None):
    is_triangle = False
    if name.endswith('.my'):
        G, E = persistent.load_var(name)
        if name.find('triangle') < 0:
            return G, E
        else:
            is_triangle = True
            cexp.redensify.G = G
            cexp.redensify.N = len(G)
            cexp.redensify.EDGES_SIGN = E
    if name.endswith('.pbm'):
        return build_graph(*read_img(name))
    assert is_triangle or name in ['PA', 'grid']
    assert is_triangle or size > 10
    if name == 'PA':
        cexp.fast_preferential_attachment(size, 3, .13)
    if name == 'grid':
        G, E_keys = gs.make_grid(size)
        cexp.redensify.G = G
        cexp.redensify.N = len(G)
        cexp.redensify.EDGES_SIGN = {e: True for e in E_keys}
    n = cexp.redensify.N
    nb_cluster = int(2*sqrt(n))
    ci = cexp.turn_into_signed_graph_by_propagation(nb_cluster,
                                                    infected_fraction=.9)
    G = deepcopy(cexp.redensify.G)
    E = deepcopy(cexp.redensify.EDGES_SIGN)
    merge_into_2_clusters(E, ci)
    return G, E


def average_strech(edges, tree):
    tree_adj = {}
    for u, v in tree:
        gs.add_edge(tree_adj, u, v)
    prt = gs.ancestor_info(tree_adj, 42)
    test_edges = edges.difference(tree)
    return sum((gs.tree_path(u, v, prt)
                for u, v in test_edges))/len(test_edges)


def add_noise(E, noise_level):
    assert noise_level == 0 or noise_level >= 1
    if noise_level == 0:
        return E
    noise_level /= 100
    rand = random.random
    return {e: not s if rand() < noise_level else s for e, s in E.items()}


def run_asym(G, E, tree_edges):
    import spectral_prediction as sp
    # assume that G is connected
    mapping, slcc = {k: k for k in G}, set(G.keys())
    training_fraction = len(G) / len(E)
    adj, test_edges = sp.get_training_matrix(training_fraction, mapping, slcc,
                                             tree_edges, G, E)
    return sp.predict_edges(adj, 15, mapping, test_edges, G, E)


def process_graph(G, E, noise, outname, asym=False):
    root = max(G.items(), key=lambda x: len(x[1]))[0]
    if not outname.startswith('belgrade/'):
        outname = 'belgrade/' + outname
    basename = '{}_{}'.format(outname, hostname())
    suffix = '.asymres' if asym else '.myres'
    if os.path.isfile(basename+'_perf'+suffix):
        return
    bfs = gs.perturbed_bfs(G, root)
    gtx, _ = galaxy_maker(G, 50, short=True, output_name=outname)
    stretch = None

    binary_signs = {e: (1 if s else -1) for e, s in E.items()}
    perf = []
    for train_edges in [bfs, gtx]:
        if asym:
            perf.extend(run_asym(G, E, train_edges))
        else:
            tree = {}
            for u, v in train_edges:
                gs.add_edge(tree, u, v)
            tags = pot.dfs_tagging(tree, binary_signs, root)
            gold, pred = pot.make_pred(tree, tags, binary_signs)
            tp, tn, fp, fn = confusion_number(gold, pred)
            perf.extend([accuracy(tp, tn, fp, fn), f1_score(tp, tn, fp, fn),
                         mcc(tp, tn, fp, fn)])
    if asym:
        _, edges = pot.read_tree(outname+'_0.edges')
        perf.extend(run_asym(G, E, edges))
        perf.extend(run_asym(G, E, tree_edges=None))
    else:
        gold, pred, _ = pot.predict_edges(outname+'_0', all_signs=E,
                                          degrees={root: 5})
        tp, tn, fp, fn = confusion_number(gold, pred)
        perf.extend([accuracy(tp, tn, fp, fn), f1_score(tp, tn, fp, fn),
                     mcc(tp, tn, fp, fn)])

    if noise == 0 and not asym:
        print(basename)
        bfsst = average_strech(set(E.keys()), bfs)
        persistent.save_var(basename+'_bfsst'+suffix, bfsst)
        gtxst = average_strech(set(E.keys()), gtx)
        persistent.save_var(basename+'_gtxst'+suffix, gtxst)
        stretch = [bfsst, gtxst]

    persistent.save_var(basename+'_perf'+suffix, perf)
    return perf, stretch
