# vim: set fileencoding=utf-8
"""Implement a merging post processing on top of Ailon algorithm"""
import sys
import os
sys.path.append(os.path.expanduser('~/venvs/34/lib/python3.4/site-packages/'))
from itertools import repeat, combinations, product
from operator import itemgetter
import convert_experiment as cexp
import redensify
import random as r
from collections import deque
NUM_THREADS = 14


def merge_gain(cv1, cv2, cluster):
    """Compute the number of disagreements edge that merging cluster `cv1` and
    `cv2` would create"""
    nodes1 = {k for k, v in cluster.items() if v == cv1}
    nodes2 = {k for k, v in cluster.items() if v == cv2}
    across_edges = {(u, v) if u < v else (v, u)
                    for u, v in product(nodes1, nodes2)}
    gain = 0
    for edge in across_edges:
        if edge not in redensify.EDGES_SIGN:
            continue
        if redensify.EDGES_SIGN[edge]:
            gain += 1
        else:
            gain -= 1
    return gain


# http://stackoverflow.com/a/12343826
def dict_argmax(dictionary):
    """Return the largest value of `dictionary`"""
    vals = list(dictionary.values())
    keys = list(dictionary.keys())
    return keys[vals.index(max(vals))]


def merge_two_clusters(cv1, cv2, orig_gains, orig_cluster):
    """return orig_cluster with clusters `cv1` and `cv2` merged, along with an
    updated version of `orig_gains`"""
    if cv1 > cv2:
        cv1, cv2 = cv2, cv1
    for k, cluster_id in orig_cluster.items():
        if cluster_id == cv2:
            orig_cluster[k] = cv1
        if cluster_id > cv2:
            orig_cluster[k] -= 1
    new_gains = {}
    for pair, gain in orig_gains.items():
        i, j = pair
        assert i < j
        newi, newj = i, j
        if i == cv1 and j == cv2:
            continue
        if i >= cv2:
            newi = cv1 if i == cv2 else i-1
        if j >= cv2:
            newj = cv1 if j == cv2 else j-1
        if newi > newj:
            newi, newj = newj, newi
        need_to_recompute = i == cv1 or i == cv2 or j == cv1 or j == cv2
        if need_to_recompute:
            new_gains[newi, newj] = merge_gain(newi, newj, orig_cluster)
        else:
            new_gains[newi, newj] = gain
    return new_gains, orig_cluster


def greedy_general():
    """Find a initial clustering through Ailon algo and greedily improve it by
    merging pair of clusters."""
    nb_disa, cluster = 1e6, None
    for _ in range(100):
        # TODO compare with cc_pivot
        cluster_ = cc_general_pivot()
        current_disa = sum(cexp.count_disagreements(cluster_))
        if current_disa < nb_disa:
            cluster = cluster_
            nb_disa = current_disa

    nb_cluster = len(set(cluster.values()))
    gains = {(a, b): merge_gain(a, b, cluster)
             for a, b in combinations(range(nb_cluster), 2)}
    nb_iter = 0
    while nb_iter < 10 * nb_cluster:
        if not gains:
            break
        i, j = dict_argmax(gains)
        if not gains[(i, j)] >= 0:
            break
        gains, cluster = merge_two_clusters(i, j, gains, cluster)
        nb_iter += 1
    return cluster


def cc_general_pivot():
    """Fill g's cluster_index according to a tweaked version of Ailon
    algorithm working (?) on general graph"""
    N = redensify.N
    clustered = set()
    unclustered = set(range(N))
    cluster = {}
    current_cluster_index = 0

    def add_to_current_cluster(node):
        cluster[node] = current_cluster_index
        clustered.add(node)
        unclustered.remove(node)

    def get_neighbors_with_sign(src):
        """return a list of (neighbor, sign of src->neighbor)"""
        res = []
        for n in [_ for _ in redensify.G[src] if _ in unclustered]:
            edge = (n, src) if n < src else (src, n)
            sign = redensify.EDGES_SIGN[edge]
            res.append((n, sign))
        return res

    while unclustered:
        pivot = r.choice(list(unclustered))
        add_to_current_cluster(pivot)
        positive_neighbors = deque([_[0]
                                    for _ in get_neighbors_with_sign(pivot)
                                    if _[1]])
        while positive_neighbors:
            u = positive_neighbors.popleft()
            add_to_current_cluster(u)
            neighbors_of_neighbor = get_neighbors_with_sign(u)
            signs = [_[1] for _ in neighbors_of_neighbor]
            if all(signs):
                neighbors_index = [_[0] for _ in neighbors_of_neighbor
                                   if _[0] not in positive_neighbors]
                positive_neighbors.extend(neighbors_index)
        current_cluster_index += 1
    return cluster


def process_planted(kwargs):
    true_cluster, _ = cexp.planted_clusters(kwargs['ball_size'],
                                            kwargs['nb_balls'])
    delta = sum(cexp.count_disagreements(true_cluster))
    times, _, errors = run_general_experiment()
    return [times, delta, errors]


def run_planted_experiment(ball_size, nb_balls, n_rep=100):
    args = repeat({"ball_size": ball_size, "nb_balls": nb_balls}, n_rep)
    runs = list(pool.imap_unordered(process_planted, args,
                                    chunksize=n_rep//NUM_THREADS))
    res = {'time': list(map(itemgetter(0), runs)),
           'delta': list(map(itemgetter(1), runs)),
           'nb_error': list(map(itemgetter(2), runs))}
    cexp.p.save_var(cexp.savefile_name('planted', [ball_size, nb_balls],
                                       redensify.PivotSelection.Uniform,
                                       True), res)


def run_general_experiment():
    start = cexp.default_timer()
    clusters = greedy_general()
    elapsed = cexp.default_timer() - start
    disagreements = cexp.count_disagreements(clusters)
    nb_cluster = len(set(clusters.values()))
    return elapsed, nb_cluster, sum(disagreements)


if __name__ == '__main__':
    # pylint: disable=C0103,E0611
    from multiprocessing import Pool
    pool = Pool(NUM_THREADS)
    for params in [(15, 5), (25, 10), (6, 30), (12, 20), (35, 15), (20, 2),
                   (40, 2), (65, 2), (100, 2)]:
        run_planted_experiment(params[0], params[1], n_rep=3*NUM_THREADS)
    pool.close()
    pool.join()
