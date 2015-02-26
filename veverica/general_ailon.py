# vim: set fileencoding=utf-8
"""Implement a merging post processing on top of Ailon algorithm"""
import convert_experiment as cexp
import redensify


def merge_gain(cv1, cv2, cluster):
    """Compute the number of disagreements edge that merging cluster `cv1` and
    `cv2` would create"""
    nodes1 = {k for k, v in cluster.items() if v == cv1}
    nodes2 = {k for k, v in cluster.items() if v == cv2}
    across_edges = {(u, v) if u < v else (v, u)
                    for u, v in cexp.product(nodes1, nodes2)}
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
        cluster_ = cexp.cc_general_pivot()
        current_disa = sum(cexp.count_disagreements(cluster_))
        if current_disa < nb_disa:
            cluster = cluster_
            nb_disa = current_disa

    nb_cluster = len(set(cluster.values()))
    gains = {(a, b): merge_gain(a, b, cluster)
             for a, b in cexp.combinations(range(nb_cluster), 2)}
    nb_iter = 0
    while nb_iter < 10 * nb_cluster:
        i, j = dict_argmax(gains)
        if not gains[(i, j)] >= 0:
            break
        gains, cluster = merge_two_clusters(i, j, gains, cluster)
        nb_iter += 1
    return cluster
