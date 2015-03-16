#! /usr/bin/python
# vim: set fileencoding=utf-8
"""."""
from collections import defaultdict
from graph_tool.topology import random_spanning_tree, label_largest_component
from graph_tool.topology import min_spanning_tree
from sklearn.metrics import f1_score, matthews_corrcoef
import real_world as rw
from graph_tool.search import bfs_search, BFSVisitor
from collections import deque
import numpy as np


class BFSTree(BFSVisitor):
    tree = None

    def __init__(self):
        self.tree = []

    def tree_edge(self, e):
        self.tree.append((int(e.source()), int(e.target())))


def read_tree(filename):
    """Read a list of edge and return adjacency list"""
    tree = {}
    with open(filename) as f:
        for line in f.readlines():
            u, v = [int(_) for _ in line.strip().split(',')]
            if u in tree:
                tree[u].add(v)
            else:
                tree[u] = set([v])
            if v in tree:
                tree[v].add(u)
            else:
                tree[v] = set([u])
    return tree


def dfs_tagging(tree, edges, root):
    """Tag each nodes in `tree` by the parity of its path (in `edges`) from the
    root"""
    tags = defaultdict(int)

    def _dfs_tagging(node, sign):
        tags[node] = sign
        for child in tree[node]:
            if tags[child] != 0:
                continue
            e = (child, node) if child < node else (node, child)
            _dfs_tagging(child, sign*edges[e])
    _dfs_tagging(root, sign=1)
    return tags


def make_pred(tree, tags):
    """predict sign of all edges not in tree according to nodes tags"""
    gold, pred = [], []
    vertices = set(tree.keys())
    for e, s in rw.EDGE_SIGN.items():
        u, v = e
        if u not in vertices or v not in vertices or e[1] in tree[e[0]]:
            continue
        gold.append(s)
        # FIXME
        # assert u in tags, e
        # assert v in tags, e
        parity = tags[u]*tags[v]
        pred.append(parity or 1)
    return gold, pred


def read_in_memory_tree(graph, tree_map):
    """Return adjacency list from a `tree_map`"""
    tree = {}
    for e in graph.edges():
        if not tree_map[e]:
            continue
        u, v = int(e.source()), int(e.target())
        if u in tree:
            tree[u].add(v)
        else:
            tree[u] = set([v])
        if v in tree:
            tree[v].add(u)
        else:
            tree[v] = set([u])
    return tree


def assess_tree_fitness(tree):
    highest_degree_node = rw.DEGREES[-1][0]
    tags = dfs_tagging(tree, rw.EDGE_SIGN, highest_degree_node)
    gold, pred = make_pred(tree, tags)
    return f1_score(gold, pred), matthews_corrcoef(gold, pred)


def get_bfs_tree(G, root):
    tree = []
    q = deque()
    discovered = [False for _ in range(len(G))]
    q.append(root)
    discovered[root] = True
    while q:
        v = q.popleft()
        for w in G[v]:
            if not discovered[w]:
                q.append(w)
                discovered[w] = True
                e = v, w if v < w else w, v
                tree.append(e)
    return tree
if __name__ == '__main__':
    # pylint: disable=C0103
    import sys
    import graph_tool as gt
    from timeit import default_timer as clock
    import galaxy as gx
    args = sys.argv
    full_graph, tree_edges = args[1], args[2]
    gt_graph = {'soc-sign-Slashdot090221.txt': 'slashdot_simple.gt',
                'soc-sign-epinions.txt': 'epinion.gt'}[full_graph]
    start = clock()

    def print_diag(msg):
        global start
        info = '{}{:.2f} seconds'.format
        print(info(msg.ljust(60), clock() - start))
        start = clock()
    rw.read_original_graph(full_graph)
    for e, s in rw.EDGE_SIGN.items():
        rw.EDGE_SIGN[e] = 1 if s else -1
    print_diag('Read graph')
    try:
        k = gt.load_graph(gt_graph)
    except:
        k = gx.to_graph_tool_simple(rw.G)
    print_diag('Convert graph')
    lcc = label_largest_component(k)
    k.set_vertex_filter(lcc)
    print_diag('Extract largest component')
    tree = read_tree(tree_edges)
    print_diag('Read galaxy tree')
    f1, mc = assess_tree_fitness(tree)
    print_diag('Predict with galaxy tree')
    print('{}{:.3f}'.format('F1-score'.ljust(60), f1))
    print('{}{:.3f}'.format('Matthews correlation coefficient'.ljust(60), mc))
    lcc_nodes = list(np.where(lcc.a)[0])
    import random
    import persistent
    f1s, mcs = [], []
    roots = random.sample(lcc_nodes, 100) + [_[0] for _ in rw.DEGREES[-100:]]
    roots = list(set(roots))
    print(len(roots))
    for root in roots:
        bfst = get_bfs_tree(rw.G, root)
        with open('__.dat', 'w') as f:
            f.write('\n'.join(('{}, {}'.format(*e) for e in bfst)))
        tree = read_tree('__.dat')
        f1, mc = assess_tree_fitness(tree)
        print_diag('Predict with BFS {}'.format(root))
        print('{}{:.3f}'.format('F1-score'.ljust(60), f1))
        print('{}{:.3f}'.format('Matthews correlation coefficient'.ljust(60), mc))
        f1s.append(f1)
        mcs.append(mc)
    persistent.save_var('epi_f1', f1s)
    persistent.save_var('epi_mc', mcs)
    persistent.save_var('epi_roots', roots)
    # rst = random_spanning_tree(k)
    # rtree = read_in_memory_tree(k, rst)
    # print_diag('Make random spanning tree')
    # f1, mc = assess_tree_fitness(rtree)
    # print_diag('Predict with random spanning tree')
    # print('{}{:.3f}'.format('F1-score'.ljust(60), f1))
    # print('{}{:.3f}'.format('Matthews correlation coefficient'.ljust(60), mc))
    # mst = min_spanning_tree(k)
    # mtree = read_in_memory_tree(k, mst)
    # print_diag('Make minimum spanning tree')
    # f1, mc = assess_tree_fitness(mtree)
    # print_diag('Predict with MS tree')
    # print('{}{:.3f}'.format('F1-score'.ljust(60), f1))
    # print('{}{:.3f}'.format('Matthews correlation coefficient'.ljust(60), mc))
