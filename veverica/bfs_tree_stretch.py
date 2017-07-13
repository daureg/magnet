# pylint: disable=C0103
import sys
from timeit import default_timer as clock

import graph_tool as gt
import numpy as np
from graph_tool.topology import label_largest_component, shortest_distance

import persistent as p
import pred_on_tree as pot
import real_world as rw

start = clock()
SLASH = False
if SLASH:
    graph_file = 'slashdot_simple.gt'
    ds_file = 'slashdot_dst.npy'
    orig_file = 'soc-sign-Slashdot090221.txt'
    prefix = 'sla'
    size = 82140
else:
    graph_file = 'epinion.gt'
    ds_file = 'epi_graph_dst.npy'
    orig_file = 'soc-sign-epinions.txt'
    prefix = 'epi'
    size = 131580

n = size
idx = int(sys.argv[1])

def print_diag(msg):
    global start, idx
    info = '{}{:.2f} seconds\n'.format
    with open('{}_out.{}'.format(prefix, idx), 'a') as f:
        f.write(info(msg.ljust(60), clock() - start))
    start = clock()

k = gt.load_graph(graph_file)
dst_mat = np.load(ds_file)
lcc = label_largest_component(k)
k.set_vertex_filter(lcc)
lcc_nodes = np.where(lcc.a)[0]
slcc = set(lcc_nodes)
all_lcc_edges = {(int(u), int(v)) for u, v in k.edges() if int(u) in slcc}
rw.read_original_graph(orig_file)
high_degree = [_[0] for _ in rw.DEGREES[-200:][::-1]]
for e, s in rw.EDGE_SIGN.items():
    rw.EDGE_SIGN[e] = 1 if s else -1
print_diag('load graph')
root = high_degree[idx]
bfs_tree = set(pot.get_bfs_tree(rw.G, root))
test_edges = all_lcc_edges - bfs_tree
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
print_diag('build tree {}, {} test edges'.format(root, len(test_edges)))
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
print(idx, root, path_stretch, edge_stretch)
p.save_var('{}_out_{}.my'.format(prefix, idx),
           (idx, root, path_stretch, edge_stretch))
