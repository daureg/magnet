# vim: set fileencoding=utf-8
"""Try to build a spanner giving accurate answers knowing all the signs
already."""
from collections import deque
from sklearn.metrics import matthews_corrcoef
import real_world as rw
from copy import deepcopy
rw.read_original_graph('soc-sign-Slashdot090221.txt')
ADJACENCY, EDGE_SIGNS = deepcopy(rw.G), deepcopy(rw.EDGE_SIGN)


def consistent_bfs(adjacency, edge_signs, root):
    """Return a set of edges forming a tree rooted at `root` in which all
    internal path have no more than one negative edges. Also compute its score
    based on internal edges not part of the tree and outgoing edges."""
    tree = set()
    q = deque()
    discovered = {k: (False, 0, 0) for k in adjacency.keys()}
    q.append(root)
    discovered[root] = (True, 0, 0)
    tree_nodes = set()
    nb_iter = 0
    total_path_length, nb_paths = 0, 0
    while q and nb_iter < len(discovered):
        nb_iter += 1
        v = q.popleft()
        tree_nodes.add(v)
        negativity = discovered[v][1]
        dst_from_root = discovered[v][2]
        for w in adjacency[v]:
            if not discovered[w][0]:
                e = (v, w) if v < w else (w, v)
                sign = edge_signs[e]
                w_negativity = negativity + {False: 1, True: 0}[sign]
                discovered[w] = (True, w_negativity, dst_from_root+1)
                if w_negativity <= 1:
                    q.append(w)
                    tree.add(e)
                else:
                    total_path_length += dst_from_root
                    nb_paths += 1
    within_tree, outside_edges, one_neg_edges = 0, 0, 0
    gold, pred = [], []
    for node in tree_nodes:
        negativity = discovered[node][1]
        for endpoint in adjacency[node]:
            if endpoint < node:
                continue
            e = (node, endpoint)
            if endpoint in tree_nodes:
                if e not in tree:
                    within_tree += 1
                    number_of_neg_edges = discovered[endpoint][1] + negativity
                    bad = number_of_neg_edges > 1
                    one_neg_edges += 1 - int(bad)
                    if not bad:
                        pred.append(1-number_of_neg_edges)
                        gold.append(int(edge_signs[e]))
            else:
                outside_edges += 1
    matthews_score = -1
    if len(gold) > 0:
        matthews_score = matthews_corrcoef(gold, pred)
    if within_tree == 0 or nb_paths == 0:
        return (root, -5, -5, -5, -5)
    return (root, outside_edges/len(tree_nodes), one_neg_edges/within_tree,
            matthews_score, total_path_length/nb_paths)


def tree_score(inside_edges, outside_edges):
    return inside_edges - outside_edges


def merge_trees(list_of_tree):
    list_of_tree = sorted(list_of_tree, key=lambda x: x[1])


def cbfs(root):
    return consistent_bfs(ADJACENCY, EDGE_SIGNS, root)

if __name__ == '__main__':
    # pylint: disable=C0103
    import persistent as p
    from multiprocessing import pool
    import random
    pool = pool.Pool(13)
    roots = random.sample(list(ADJACENCY.keys()), 10000)
    res = pool.imap_unordered(cbfs, roots, chunksize=len(roots)//13)
    pool.close()
    pool.join()
    p.save_var('cbfs_val.my', list(res))
