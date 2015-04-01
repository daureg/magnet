#! /usr/bin/python
# vim: set fileencoding=utf-8
"""."""
from collections import defaultdict
from graph_tool.topology import random_spanning_tree, label_largest_component
from graph_tool.topology import min_spanning_tree
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
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
    edges = []
    with open(filename) as f:
        for line in f.readlines():
            u, v = [int(_) for _ in line.strip().split(',')]
            add_edge_to_tree(tree, u, v)
            edges.append((u, v))
    return tree, edges


def dfs_tagging(tree, edges, root):
    """Tag each nodes in `tree` by the parity of its path (in `edges`) from the
    root"""
    tags = defaultdict(int)

    def _dfs_tagging(node, sign):
        assert sign in [-1, 1], 'edges sign should not be boolean'
        tags[node] = sign
        for child in tree[node]:
            if tags[child] != 0:
                continue
            e = (child, node) if child < node else (node, child)
            _dfs_tagging(child, sign*edges[e])
    _dfs_tagging(root, sign=1)
    return tags


def make_pred(tree, tags, edge_signs=None):
    """predict sign of all edges not in tree according to nodes tags"""
    gold, pred = [], []
    vertices = set(tree.keys())
    edge_signs = edge_signs or rw.EDGE_SIGN
    for e, s in edge_signs.items():
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


def add_edge_to_tree(tree, u, v):
    """Update adjacency list `tree` with the (u, v) edge"""
    if u in tree:
        tree[u].add(v)
    else:
        tree[u] = set([v])
    if v in tree:
        tree[v].add(u)
    else:
        tree[v] = set([u])


def read_in_memory_tree(graph, tree_map):
    """Return adjacency list from a `tree_map`"""
    tree = {}
    for e in graph.edges():
        if not tree_map[e]:
            continue
        u, v = int(e.source()), int(e.target())
        add_edge_to_tree(tree, u, v)
    return tree


def assess_tree_fitness(tree):
    highest_degree_node = rw.DEGREES[-1][0]
    tags = dfs_tagging(tree, rw.EDGE_SIGN, highest_degree_node)
    gold, pred = make_pred(tree, tags)
    acc = accuracy_score(gold, pred)
    return acc, f1_score(gold, pred), matthews_corrcoef(gold, pred)


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
                e = (v, w) if v < w else (w, v)
                tree.append(e)
    return tree

def read_spanner_from_file(basename):
    """Reparse information written by galaxy.export_spanner"""
    adjacency, spanner = read_tree(basename+'.edges')
    with open(basename+'.sm') as f:
        raw_star_membership = f.readlines()
    star_membership = {}
    for line in raw_star_membership:
        orig_node, star_idx = [int(_) for _ in line.split()]
        star_membership[orig_node] = star_idx
    with open(basename+'.lle') as f:
        raw_lle = f.readlines()
    low_level_edges = {}
    for line in raw_lle:
        top_edge, orig_edge = line.split('\t')
        te0, te1 = [int(_) for _ in top_edge.split(', ')]
        te0, te1 = (te0, te1) if te0 < te1 else (te1, te0)
        oe0, oe1 = [int(_) for _ in orig_edge.split(', ')]
        low_level_edges[(te0, te1)] = (oe0, oe1)
    return spanner, star_membership, low_level_edges, adjacency


def split_into_trees(spanner, star_membership):
    """Build a tree adjacency for each star based on the edges in `spanner`"""
    trees = defaultdict(dict)
    for u, v in spanner:
        head_star = star_membership[u]
        tail_star = star_membership[v]
        if head_star == tail_star:
            add_edge_to_tree(trees[head_star], u, v)
    return trees


def parity_in_star(star_idx, i, j, tags, star_membership):
    """Return the parity of the i->j path within star numbered `star_idx`"""
    err = (i, star_membership[i], star_idx, j, star_membership[j])
    assert star_membership[i] == star_idx == star_membership[j], err
    return tags[star_idx][i] * tags[star_idx][j]


def orient_path(stars_path, low_level_edges, star_membership):
    """Given a connected list of stars, return a list of edges in the original
    graph, each oriented from one star to the next one in the path."""
    i, j = stars_path
    real_edge = low_level_edges[(i, j) if i < j else (j, i)]
    if star_membership[real_edge[0]] == i:
        return [real_edge]
    return [(real_edge[1], real_edge[0])]


def parity(i, j, spanner, edge_sign, tags, star_membership, low_level_edges):
    """Return the parity of the i->j path"""
    i_star = star_membership[i]
    j_star = star_membership[j]
    if i_star == j_star:
        return parity_in_star(i_star, i, j, tags, star_membership)
    if i_star > j_star:
        i, j = j, i
        i_star, j_star = j_star, i_star
    star_path = orient_path([i_star, j_star], low_level_edges, star_membership)
    parity = parity_in_star(i_star, i, star_path[0][0], tags, star_membership)
    u, v = star_path[-1]
    canonical_edge = (u, v) if u < v else (v, u)
    parity *= edge_sign[canonical_edge]
    parity *= parity_in_star(j_star, star_path[-1][1], j, tags,
                             star_membership)
    return parity


def brute_parity(i, j, low_level_graph, edge_sign):
    """Compute directly the shortest_path between i and j in low_level_graph
    and multiply edges sign"""
    parity = 1
    path = shortest_path(low_level_graph, i, j)
    for u, v in zip(path, path[1:]):
        parity *= edge_sign[(u, v) if u < v else (v, u)]
    return parity


def shortest_path(graph, src, dst):#vertices):
    """Return a shortest_path between vertices[0] and vertices[1]"""
    # src, dst = vertices
    q = deque()
    discovered = {_: False for _ in graph.keys()}
    q.append(src)
    discovered[src] = True
    predecessor = {}
    while q:
        v = q.popleft()
        for w in graph[v]:
            if not discovered[w]:
                predecessor[w] = v
                q.append(w)
                discovered[w] = True
    path = [dst]
    next_step = predecessor[dst]
    while next_step != src:
        path.append(predecessor[path[-1]])
        next_step = predecessor[path[-1]]
    path.append(src)
    # TODO compute parity here (in reversed order, doesn't matter) and return
    # only [path[-1], path[-2], path[1], path[0]] (special case if there is
    # only one edge (ie two vertices)) plus the parity of the path in the
    # middle (or 1 if nothing)
    return list(reversed(path))


def predict_edges(basename, all_signs=None, lcc_nodes=None, use_brute=False):
    # import redensify
    edge_signs = {}
    # all_signs = all_signs or redensify.EDGES_SIGN
    for e, s in all_signs.items():
        edge_signs[e] = 1 if s else -1
    _ = read_spanner_from_file(basename)
    spanner, star_membership, low_level_edges, low_level_graph = _
    # if not lcc_nodes:
    #     lcc_nodes = set(star_membership.keys())
    top_graph = {}
    for u, v in low_level_edges.keys():
        add_edge_to_tree(top_graph, u, v)
    train_edges = {(u, v) for u, v in spanner
                   if not lcc_nodes or u in lcc_nodes}
    test_edges = set(edge_signs.keys()) - train_edges
    trees = split_into_trees(train_edges, star_membership)
    tags = {}
    for star, tree in trees.items():
        tags[star] = dfs_tagging(tree, edge_signs,
                                 root=next(iter(tree.keys())))
    # Since we build trees and tags based on edges of the span, stars
    # consisting of a single node do not appear. Thus we create their tag
    # structure manually
    for node in top_graph.keys():
        if node not in tags:
            tags[node] = defaultdict(lambda: 1)
    gold, pred, brute_pred = [], [], []
    for e in test_edges:
        s = edge_signs[e]
        u, v = e
        gold.append(s)
        uv_sign = parity(u, v, top_graph, edge_signs, tags, star_membership,
                         low_level_edges)
        pred.append(uv_sign)
        if use_brute:
            brute_pred.append(brute_parity(u, v, low_level_graph, edge_signs))
    return gold, pred, brute_pred

if __name__ == '__main__':
    # pylint: disable=C0103
    import sys
    import graph_tool as gt
    from timeit import default_timer as clock
    import galaxy as gx
    args = sys.argv
    full_graph, tree_edges, seed = args[1], args[2], int(args[3])
    gt_graph = 'xx'
    # gt_graph = {'soc-sign-Slashdot090221.txt': 'slashdot_simple.gt',
    #             'soc-wiki.txt': 'wiki_simple.gt',
    #             'soc-sign-epinions.txt': 'epinion.gt'}[full_graph]
    SLA = full_graph == 'soc-sign-Slashdot090221.txt'
    start = clock()

    def print_diag(msg):
        global start
        info = '{}{:.2f} seconds'.format
        print(info(msg.ljust(60), clock() - start))
        start = clock()
    rw.read_original_graph(full_graph, seed=seed)
    for e, s in rw.EDGE_SIGN.items():
        rw.EDGE_SIGN[e] = 1 if s else -1
    # print_diag('Read graph')
    # try:
    #     k = gt.load_graph(gt_graph)
    # except:
    #     k = gx.to_graph_tool_simple(rw.G)
    # print_diag('Convert graph')
    # lcc = label_largest_component(k)
    # lcc_nodes = list(np.where(lcc.a)[0])
    # print('LCC: {}'.format(len(set(lcc_nodes))))
    # k.set_vertex_filter(lcc)
    # print_diag('Extract largest component')
    tree, _ = read_tree(tree_edges)
    print_diag('Read galaxy tree')
    acc, f1, mc = assess_tree_fitness(tree)
    print_diag('Predict with galaxy tree')
    print('{}{:.3f}'.format('Accuracy'.ljust(60), acc))
    print('{}{:.3f}'.format('F1-score'.ljust(60), f1))
    print('{}{:.3f}'.format('Matthews correlation coefficient'.ljust(60), mc))
    with open('wiki_res.dat', 'a') as f:
        f.write('\t'.join(map(str, [acc, f1, mc])))
    sys.exit()
    import random
    import persistent
    accs, f1s, mcs = [], [], []
    roots = random.sample(lcc_nodes, 100) + [_[0] for _ in rw.DEGREES[-100:]]
    roots = list(set(roots))
    roots = [_[0] for _ in rw.DEGREES[-150:]]
    print(len(roots))
    for root in roots:
        bfst = get_bfs_tree(rw.G, root)
        with open('__.dat', 'w') as f:
            f.write('\n'.join(('{}, {}'.format(*e) for e in bfst)))
        tree, _ = read_tree('__.dat')
        acc, f1, mc = assess_tree_fitness(tree)
        print_diag('Predict with BFS {}'.format(root))
        print('{}{:.3f}'.format('F1-score'.ljust(60), f1))
        print('{}{:.3f}'.format('Matthews correlation coefficient'.ljust(60), mc))
        accs.append(acc)
        f1s.append(f1)
        mcs.append(mc)
    prefix = 'wik'
    persistent.save_var('{}_acc'.format(prefix), accs)
    persistent.save_var('{}_f1'.format(prefix), f1s)
    persistent.save_var('{}_mc'.format(prefix), mcs)
    persistent.save_var('{}_roots'.format(prefix), roots)
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
