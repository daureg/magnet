# vim: set fileencoding=utf-8
# pylint: disable=C0103
import persistent as p
from timeit import default_timer as clock
import pred_on_tree as pot
import real_world as rw
import graph_tool as gt
import numpy as np
import sys
from graph_tool.topology import label_largest_component, shortest_distance
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
start = clock()
GRAPH = 'slash'
if GRAPH == 'slash':
    SLASH = True
    graph_file = 'slashdot_simple.gt'
    ds_file = 'slashdot_dst.npy'
    orig_file = 'soc-sign-Slashdot090221.txt'
    basename = 'universe/slashdot_'
    prefix = 'sla_gtx'
    size = 82140
elif GRAPH == 'epinion':
    SLASH = False
    graph_file = 'epinion.gt'
    ds_file = 'epi_graph_dst.npy'
    basename = 'universe/epinion_'
    orig_file = 'soc-sign-epinions.txt'
    prefix = 'epi_gtx'
    size = 131580
elif GRAPH == 'wiki':
    SLASH = False
    graph_file = 'wiki_simple.gt'
    ds_file = 'wiki_dst.npy'
    basename = 'universe/wiki_'
    orig_file = 'soc-wiki.txt'
    prefix = 'wik_gtx'
    size = 7115

n = size
idx = int(sys.argv[1])
seed = None if len(sys.argv) <= 2 else int(sys.argv[2])
if seed is not None:
    basename += str(seed) + '_'


def print_diag(msg):
    global start, idx
    info = '{}{:.2f} seconds\n'.format
    with open('{}_out.{}'.format(prefix, idx), 'a') as f:
        f.write(info(msg.ljust(60), clock() - start))
    start = clock()

# k = gt.load_graph(graph_file)
# dst_mat = np.load(ds_file)
# lcc = label_largest_component(k)
# k.set_vertex_filter(lcc)
# lcc_nodes = np.where(lcc.a)[0]
# slcc = set(lcc_nodes)
all_lcc_edges = {}
rw.read_original_graph(orig_file, seed=seed)
lcc_tree = pot.get_bfs_tree(rw.G, rw.DEGREES[-1][0])
heads, tails = zip(*lcc_tree)
slcc = set(heads).union(set(tails))
for e, s in rw.EDGE_SIGN.items():
    u, v = e
    if u not in slcc:
        continue
    all_lcc_edges[(u, v)] = s
print_diag('load graph')
gold, pred, _ = pot.predict_edges(basename+str(idx), all_lcc_edges, slcc)
print(accuracy_score(gold, pred), f1_score(gold, pred),
      matthews_corrcoef(gold, pred))
_ = pot.read_spanner_from_file(basename+str(idx))
spanner, star_membership, low_level_edges, low_level_graph = _
train_edges = {(u, v) for u, v in spanner if u in slcc}
# print('Active set size {}, {:.1f}'.format(len(train_edges),
#                                           100*len(train_edges)/len(all_lcc_edges)))
sys.exit()
bfs_tree = train_edges
test_edges = set(all_lcc_edges.keys()) - bfs_tree
test_graph = {}
for u, v in test_edges:
    pot.add_edge_to_tree(test_graph, u, v)
bfsmap = k.new_edge_property('boolean')
for e in k.edges():
    u, v = int(e.source()), int(e.target())
    if (u, v) in bfs_tree:
        bfsmap[e] = True
    else:
        bfsmap[e] = False
k.set_vertex_filter(None)
k.set_edge_filter(bfsmap)
print_diag('build tree {}, {} test edges'.format(idx, len(test_edges)))
bfs_dst = shortest_distance(k, dense=False)
bfs_mat = np.zeros((n, n), dtype=np.uint8)
for v in k.vertices():
    bfs_mat[int(v), :] = bfs_dst[v].a.astype(np.uint8)
print_diag('computed pairwise distance')
bsum = 0
bsize = 0
esum = 0
for i, v in enumerate(lcc_nodes):
    graph_distance = dst_mat[v, lcc_nodes[i+1:]]
    tree_distance = bfs_mat[v, lcc_nodes[i+1:]]
    if v in test_graph:
        esum += bfs_mat[v, sorted(test_graph[v])].sum()
    ratio = (tree_distance/graph_distance)
    bsum += ratio.sum()
    bsize += ratio.shape[0]
path_stretch = bsum/bsize
edge_stretch = (esum/2)/len(test_edges)
print_diag('computed stats')
print(idx, idx, path_stretch, edge_stretch)
p.save_var('{}_out_{}.my'.format(prefix, idx),
           (idx, idx, path_stretch, edge_stretch))
