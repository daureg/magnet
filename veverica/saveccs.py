import sys

import networkx as nx

import pack_graph as pg
import persistent

pref = sys.argv[1]
G,E = pg.load_directed_signed_graph('directed_{}.pack'.format(pref))
ng=nx.Graph()
n,m=len(G),len(E)
ng.add_nodes_from(range(m+2*n))
sorted_edges = {i: e for i, e in enumerate(sorted(E))}
edges=[]
for ve, (p, q) in sorted_edges.items():
    vp = p + m
    vq = q + m + n
    edges.extend(((vp, ve, {'capacity': 1}), (ve, vq, {'capacity': 1})))    
ng.add_edges_from(edges)
ccs = [c for c in nx.connected_component_subgraphs(ng) if len(c)>3]
persistent.save_var('{}_cc.my'.format(pref), (ccs, sorted_edges), 2)
